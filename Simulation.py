"""Graphical MuJoCo simulation for the parallel wheel-legged robot."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
import math
import os
from pathlib import Path
import threading
import time

import glfw
import mujoco
import numpy as np

try:
    from OpenGL import GL
except ImportError:  # pragma: no cover - HUD still has a MuJoCo overlay fallback.
    GL = None

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - graphical HUD falls back to MuJoCo text.
    Image = None
    ImageDraw = None
    ImageFont = None

from controller import ParallelLegController


ROOT = Path(__file__).resolve().parent
FORWARD_SPEED = 0.50
TURN_RATE = 0.75
DEFAULT_RENDER_HZ = 45.0
MAX_CATCHUP_SECONDS = 0.05
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 900
CAMERA_LOOKAHEAD = 1.7
HUD_FONT_SCALE = mujoco.mjtFontScale.mjFONTSCALE_50
CURVE_HISTORY_SECONDS = 12.0
CURVE_MAX_POINTS = min(600, mujoco.mjMAXLINEPNT)
CURVE_PANEL_WIDTH = 430
CURVE_PANEL_HEIGHT = 150
CURVE_PANEL_MIN_HEIGHT = 64
CURVE_PANEL_MARGIN = 12
CURVE_PANEL_GAP = 8
CURVE_RANGE_PADDING = 0.03
CURVE_LEGEND_FONT_SIZE = 9
CURVE_LEGEND_PADDING = 3
CURVE_LEGEND_SWATCH_WIDTH = 10
CURVE_LEGEND_SWATCH_HEIGHT = 5
CURVE_LEGEND_ITEM_GAP = 5
CURVE_LEGEND_ROW_GAP = 1
HUD_BITMAP_FONT_SIZE = 14
HUD_BITMAP_MIN_FONT_SIZE = 11
HUD_BITMAP_PADDING = 8
HUD_BITMAP_MARGIN = 8
HUD_BITMAP_COLUMN_GAP = 24
HUD_BITMAP_LINE_GAP = 3
HUD_CJK_FONT_PATHS = (
    "/mnt/c/Windows/Fonts/msyh.ttc",
    "/mnt/c/Windows/Fonts/simhei.ttf",
    "/mnt/c/Windows/Fonts/simsun.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
)
CURVE_LABEL_TRANSLATIONS = {
    "cmd v": "指令速度",
    "target v": "目标速度",
    "actual v": "实际速度",
    "cmd yaw": "指令偏航",
    "target yaw": "目标偏航",
    "actual yaw": "实际偏航",
    "pitch phi": "俯仰角",
    "roll gamma": "滚转角",
    "target L": "目标腿长",
    "left L": "左腿长",
    "right L": "右腿长",
    "target theta": "目标腿摆角",
    "mean theta": "平均腿摆角",
    "left theta": "左腿摆角",
    "right theta": "右腿摆角",
    "wheel L": "左轮力矩",
    "wheel R": "右轮力矩",
    "wheel avg": "轮平均力矩",
    "hip L": "左髋力矩",
    "hip R": "右髋力矩",
    "yaw assist": "偏航辅助",
}


KEY_LABELS = {
    32: "Space",
    262: "Right",
    263: "Left",
    264: "Down",
    265: "Up",
    266: "PageUp",
    267: "PageDown",
    256: "Esc",
}


class KeyboardCommandState:
    """Tracks held movement keys and applies momentary drive commands."""

    FORWARD_KEYS = {ord("W"), 265}  # GLFW_KEY_UP
    REVERSE_KEYS = {ord("S"), 264}  # GLFW_KEY_DOWN
    LEFT_KEYS = {ord("A"), 263}  # GLFW_KEY_LEFT
    RIGHT_KEYS = {ord("D"), 262}  # GLFW_KEY_RIGHT

    def __init__(self):
        self.held_keys: set[int] = set()
        self.last_speed = 0.0
        self.last_yaw = 0.0
        self.last_event = "none"

    @staticmethod
    def _canonical_key(keycode: int) -> int:
        key = chr(keycode).upper() if 0 <= keycode < 256 else ""
        return ord(key) if key else keycode

    def _movement_command(self) -> tuple[float, float]:
        forward = any(key in self.held_keys for key in self.FORWARD_KEYS)
        reverse = any(key in self.held_keys for key in self.REVERSE_KEYS)
        left = any(key in self.held_keys for key in self.LEFT_KEYS)
        right = any(key in self.held_keys for key in self.RIGHT_KEYS)

        speed = 0.0
        if forward != reverse:
            speed = FORWARD_SPEED if forward else -FORWARD_SPEED

        yaw = 0.0
        if left != right:
            yaw = TURN_RATE if left else -TURN_RATE
        return speed, yaw

    @staticmethod
    def key_label(keycode: int) -> str:
        if keycode in KEY_LABELS:
            return KEY_LABELS[keycode]
        if 0 <= keycode < 256 and chr(keycode).isprintable():
            return chr(keycode).upper()
        return f"key_{keycode}"

    def held_key_summary(self) -> str:
        if not self.held_keys:
            return "none"
        return "+".join(self.key_label(key) for key in sorted(self.held_keys))

    def apply(self, controller: ParallelLegController) -> None:
        speed, yaw = self._movement_command()
        if (
            speed == 0.0
            and yaw == 0.0
            and (self.last_speed != 0.0 or self.last_yaw != 0.0)
        ):
            controller.stop_motion()
        else:
            controller.command.speed = speed
            controller.command.yaw = yaw
        self.last_speed = speed
        self.last_yaw = yaw

    def clear_motion(self, controller: ParallelLegController) -> None:
        self.held_keys.difference_update(
            self.FORWARD_KEYS | self.REVERSE_KEYS | self.LEFT_KEYS | self.RIGHT_KEYS
        )
        self.last_speed = 0.0
        self.last_yaw = 0.0
        self.last_event = "press X"
        controller.stop_motion()

    def reset(self) -> None:
        self.held_keys.clear()
        self.last_speed = 0.0
        self.last_yaw = 0.0
        self.last_event = "press R"

    def handle_event(
        self,
        controller: ParallelLegController,
        now: float,
        keycode: int,
        action: int,
    ) -> str | None:
        key = self._canonical_key(keycode)
        motion_keys = (
            self.FORWARD_KEYS | self.REVERSE_KEYS | self.LEFT_KEYS | self.RIGHT_KEYS
        )

        if action == glfw.PRESS:
            self.last_event = f"press {self.key_label(key)}"
            if key in motion_keys:
                self.held_keys.add(key)
                self.apply(controller)
                return self._status()
            return handle_key(controller, now, keycode, self)

        if action == glfw.RELEASE and key in motion_keys:
            self.last_event = f"release {self.key_label(key)}"
            self.held_keys.discard(key)
            self.apply(controller)
            return self._status()

        if action == glfw.REPEAT and key in motion_keys:
            self.last_event = f"hold {self.key_label(key)}"
            self.held_keys.add(key)
            self.apply(controller)
            return None

        return None

    def _status(self) -> str | None:
        speed, yaw = self._movement_command()
        if speed > 0.0 and yaw > 0.0:
            return "forward + turn left"
        if speed > 0.0 and yaw < 0.0:
            return "forward + turn right"
        if speed < 0.0 and yaw > 0.0:
            return "reverse + turn left"
        if speed < 0.0 and yaw < 0.0:
            return "reverse + turn right"
        if speed > 0.0:
            return "forward"
        if speed < 0.0:
            return "reverse"
        if yaw > 0.0:
            return "turn left"
        if yaw < 0.0:
            return "turn right"
        return "stop"


def step_physics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller: ParallelLegController,
    steps: int = 1,
) -> None:
    """Advance fixed-step control and physics independently of rendering."""
    for _ in range(steps):
        controller.update(data.time)
        mujoco.mj_step(model, data)


def sync_render_data(
    model: mujoco.MjModel, physics_data: mujoco.MjData, render_data: mujoco.MjData
) -> None:
    """Copy a physics snapshot into data owned exclusively by the viewer."""
    render_data.time = physics_data.time
    render_data.qpos[:] = physics_data.qpos
    render_data.qvel[:] = physics_data.qvel
    render_data.act[:] = physics_data.act
    render_data.ctrl[:] = physics_data.ctrl
    render_data.mocap_pos[:] = physics_data.mocap_pos
    render_data.mocap_quat[:] = physics_data.mocap_quat
    mujoco.mj_forward(model, render_data)


def reset_simulation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller: ParallelLegController,
    keyboard: KeyboardCommandState | None = None,
    render_data: mujoco.MjData | None = None,
) -> None:
    """Reset MuJoCo data, controller memory, keyboard state, and render snapshot."""
    controller.initialize_pose()
    if keyboard is not None:
        keyboard.reset()
    mujoco.mj_forward(model, data)
    if render_data is not None:
        sync_render_data(model, data, render_data)


@dataclass(frozen=True)
class ControlCurveSample:
    """One rendered sample for the in-view control-history curves."""

    time: float
    command_speed: float
    target_speed: float
    actual_speed: float
    command_yaw_rate: float
    target_yaw_rate: float
    actual_yaw_rate: float
    pitch: float
    roll: float
    target_leg_swing: float
    mean_leg_angle: float
    left_leg_angle: float
    right_leg_angle: float
    target_leg_length: float
    left_leg_length: float
    right_leg_length: float
    left_wheel_torque: float
    right_wheel_torque: float
    left_hip_torque: float
    right_hip_torque: float
    yaw_assist_torque: float


class ControlCurveHistory:
    """Small ring buffer that converts control samples into MuJoCo figures."""

    def __init__(
        self,
        history_seconds: float = CURVE_HISTORY_SECONDS,
        max_points: int = CURVE_MAX_POINTS,
    ):
        self.history_seconds = history_seconds
        self.max_points = max(1, min(max_points, mujoco.mjMAXLINEPNT))
        self.samples: deque[ControlCurveSample] = deque()

    def clear(self) -> None:
        self.samples.clear()

    def append(self, sample: ControlCurveSample) -> None:
        if self.samples and sample.time < self.samples[-1].time:
            self.samples.clear()
        self.samples.append(sample)
        min_time = sample.time - self.history_seconds
        while self.samples and self.samples[0].time < min_time:
            self.samples.popleft()
        while len(self.samples) > self.max_points:
            self.samples.popleft()

    def _figure_specs(self):
        return (
            (
                "Forward speed v (m/s)",
                (-0.8, 0.8),
                0.035,
                (
                    ("cmd v", (0.15, 0.65, 1.0), lambda s: s.command_speed),
                    ("target v", (1.0, 0.72, 0.18), lambda s: s.target_speed),
                    ("actual v", (0.55, 0.95, 0.42), lambda s: s.actual_speed),
                ),
            ),
            (
                "Yaw rate omega (rad/s)",
                (-1.0, 1.0),
                0.030,
                (
                    ("cmd yaw", (0.15, 0.65, 1.0), lambda s: s.command_yaw_rate),
                    ("target yaw", (1.0, 0.72, 0.18), lambda s: s.target_yaw_rate),
                    ("actual yaw", (0.55, 0.95, 0.42), lambda s: s.actual_yaw_rate),
                ),
            ),
            (
                "Body angles (rad)",
                (-0.8, 0.8),
                0.012,
                (
                    ("pitch phi", (1.0, 0.34, 0.28), lambda s: s.pitch),
                    ("roll gamma", (0.40, 0.82, 1.0), lambda s: s.roll),
                ),
            ),
            (
                "Leg length L (m)",
                (0.095, 0.215),
                0.0025,
                (
                    ("target L", (1.0, 0.76, 0.18), lambda s: s.target_leg_length),
                    ("left L", (0.20, 0.76, 1.0), lambda s: s.left_leg_length),
                    ("right L", (0.40, 0.95, 0.52), lambda s: s.right_leg_length),
                ),
            ),
            (
                "Leg swing theta (rad)",
                (-1.2, 0.4),
                0.018,
                (
                    ("target theta", (1.0, 0.76, 0.18), lambda s: s.target_leg_swing),
                    ("mean theta", (0.88, 0.72, 1.0), lambda s: s.mean_leg_angle),
                    ("left theta", (0.20, 0.76, 1.0), lambda s: s.left_leg_angle),
                    ("right theta", (0.40, 0.95, 0.52), lambda s: s.right_leg_angle),
                ),
            ),
            (
                "Wheel torque tau_w (Nm)",
                (-4.5, 4.5),
                0.15,
                (
                    ("wheel L", (0.15, 0.65, 1.0), lambda s: s.left_wheel_torque),
                    ("wheel R", (0.40, 0.95, 0.52), lambda s: s.right_wheel_torque),
                    (
                        "wheel avg",
                        (1.0, 0.72, 0.18),
                        lambda s: 0.5
                        * (s.left_wheel_torque + s.right_wheel_torque),
                    ),
                ),
            ),
            (
                "Hip torque tau_h (Nm)",
                (-8.0, 8.0),
                0.18,
                (
                    ("hip L", (1.0, 0.34, 0.28), lambda s: s.left_hip_torque),
                    ("hip R", (0.88, 0.72, 1.0), lambda s: s.right_hip_torque),
                ),
            ),
            (
                "Yaw assist torque (Nm)",
                (-6.0, 6.0),
                0.12,
                (
                    ("yaw assist", (1.0, 0.76, 0.18), lambda s: s.yaw_assist_torque),
                ),
            ),
        )

    def curve_legends(self) -> tuple[tuple[tuple[str, tuple[float, float, float]], ...], ...]:
        latest_sample = self.samples[-1] if self.samples else None
        return tuple(
            tuple(
                (
                    self._legend_label(
                        name,
                        None if latest_sample is None else getter(latest_sample),
                    ),
                    color,
                )
                for name, color, getter in lines
            )
            for _, _, _, lines in self._figure_specs()
        )

    @staticmethod
    def _legend_label(name: str, value: float | None = None) -> str:
        translation = CURVE_LABEL_TRANSLATIONS.get(name)
        label = f"{name}({translation})" if translation else name
        if value is None:
            return label
        return f"{label}={ControlCurveHistory._format_curve_value(name, value)}"

    @staticmethod
    def _format_curve_value(name: str, value: float) -> str:
        value = float(value)
        if not math.isfinite(value):
            value = 0.0
        if name in {"target L", "left L", "right L"}:
            return f"{value:+.3f}"
        if "theta" in name or "phi" in name or "gamma" in name or "yaw" in name:
            return f"{value:+.3f}"
        return f"{value:+.2f}"

    def build_figures(self) -> tuple[mujoco.MjvFigure, ...]:
        return tuple(
            self._build_figure(title, y_range, min_span, lines)
            for title, y_range, min_span, lines in self._figure_specs()
        )

    def _build_figure(
        self,
        title: str,
        y_range,
        min_span: float,
        lines,
    ) -> mujoco.MjvFigure:
        figure = mujoco.MjvFigure()
        figure.title = title
        figure.xlabel = ""
        figure.flg_legend = 0
        figure.flg_ticklabel = 0
        figure.flg_ticklabel[1] = 1
        figure.flg_extend = 0
        figure.range[0][0] = -self.history_seconds
        figure.range[0][1] = 0.0
        figure.figurergba[:] = (0.02, 0.02, 0.02, 0.72)
        figure.panergba[:] = (0.0, 0.0, 0.0, 0.42)
        figure.gridrgb[:] = (0.36, 0.36, 0.36)
        figure.textrgb[:] = (0.92, 0.92, 0.92)

        samples = list(self.samples)[-self.max_points :]
        dynamic_y_range = self._dynamic_y_range(samples, lines, y_range, min_span)
        figure.range[1][0] = dynamic_y_range[0]
        figure.range[1][1] = dynamic_y_range[1]
        if not samples:
            return figure

        latest_time = samples[-1].time
        point_count = len(samples)
        for line_index, (name, color, getter) in enumerate(lines):
            figure.linename[line_index] = name
            figure.linergb[line_index] = color
            figure.linepnt[line_index] = point_count
            for point_index, sample in enumerate(samples):
                value = float(getter(sample))
                if not math.isfinite(value):
                    value = 0.0
                figure.linedata[line_index, 2 * point_index] = (
                    sample.time - latest_time
                )
                figure.linedata[line_index, 2 * point_index + 1] = value
        return figure

    @staticmethod
    def _dynamic_y_range(samples, lines, limit_range, min_span: float) -> tuple[float, float]:
        values: list[float] = []
        for sample in samples:
            for _, _, getter in lines:
                value = float(getter(sample))
                if math.isfinite(value):
                    values.append(value)
        if not values:
            return limit_range

        low = min(values)
        high = max(values)
        center = 0.5 * (low + high)
        span = max(high - low, min_span)
        span *= 1.0 + CURVE_RANGE_PADDING
        low = center - 0.5 * span
        high = center + 0.5 * span

        limit_low, limit_high = limit_range
        if high - low >= limit_high - limit_low:
            return limit_range
        if low < limit_low:
            high += limit_low - low
            low = limit_low
        if high > limit_high:
            low -= high - limit_high
            high = limit_high
        low = max(low, limit_low)
        high = min(high, limit_high)
        return low, high


def collect_control_curve_sample(
    controller: ParallelLegController,
    sim_time: float,
    pitch: float,
    roll: float,
) -> ControlCurveSample:
    """Copy controller state into a renderer-owned curve sample."""
    left_leg = controller.last_legs["left"]
    right_leg = controller.last_legs["right"]
    actuator = controller.last_actuator_commands
    hip_lf = actuator.get("Left_front_joint_actuator", 0.0)
    hip_lr = actuator.get("Left_rear_joint_actuator", 0.0)
    hip_rf = actuator.get("Right_front_joint_actuator", 0.0)
    hip_rr = actuator.get("Right_rear_joint_actuator", 0.0)
    return ControlCurveSample(
        time=sim_time,
        command_speed=controller.command.speed,
        target_speed=controller.target_speed,
        actual_speed=float(controller.robot_velocity[3]),
        command_yaw_rate=controller.command.yaw,
        target_yaw_rate=controller.target_yaw_rate,
        actual_yaw_rate=controller._joint_velocity("root_yaw"),
        pitch=pitch,
        roll=roll,
        target_leg_swing=controller.command.leg_swing,
        mean_leg_angle=0.5 * (left_leg.angle + right_leg.angle),
        left_leg_angle=left_leg.angle,
        right_leg_angle=right_leg.angle,
        target_leg_length=controller.target_leg_length,
        left_leg_length=left_leg.length,
        right_leg_length=right_leg.length,
        left_wheel_torque=actuator.get("Left_Wheel_joint_actuator", 0.0),
        right_wheel_torque=actuator.get("Right_Wheel_joint_actuator", 0.0),
        left_hip_torque=0.5 * (hip_lf + hip_lr),
        right_hip_torque=0.5 * (hip_rf + hip_rr),
        yaw_assist_torque=actuator.get("Yaw_assist_actuator", 0.0),
    )


def render_control_curves(
    history: ControlCurveHistory,
    viewport: mujoco.MjrRect,
    context: mujoco.MjrContext,
) -> None:
    """Render compact real-time control figures in the lower-right viewport."""
    figures = history.build_figures()
    legends = history.curve_legends()
    if not figures:
        return

    available_height = max(
        1,
        viewport.height - 2 * CURVE_PANEL_MARGIN - CURVE_PANEL_GAP * (len(figures) - 1),
    )
    panel_height = max(
        CURVE_PANEL_MIN_HEIGHT,
        min(CURVE_PANEL_HEIGHT, available_height // len(figures)),
    )
    panel_width = max(
        260,
        min(CURVE_PANEL_WIDTH, viewport.width - 2 * CURVE_PANEL_MARGIN),
    )
    left = viewport.width - panel_width - CURVE_PANEL_MARGIN

    for index, figure in enumerate(figures):
        bottom = CURVE_PANEL_MARGIN + (
            len(figures) - 1 - index
        ) * (panel_height + CURVE_PANEL_GAP)
        rect = mujoco.MjrRect(left, bottom, panel_width, panel_height)
        mujoco.mjr_figure(rect, figure, context)
        render_curve_legend(legends[index], rect, context)


def _activity_label(active: bool) -> str:
    return "active/激活" if active else "ready/就绪"


def _display_none(value: str) -> str:
    return "none/无" if value == "none" else value


def _translate_key_event(event: str) -> str:
    if event == "none":
        return "none/无"
    action, _, key = event.partition(" ")
    action_cn = {
        "press": "按下",
        "release": "释放",
        "hold": "长按",
    }.get(action)
    if action_cn is None or not key:
        return event
    return f"{event}/{action_cn} {key}"


def build_control_overlay(
    controller: ParallelLegController,
    sim_time: float,
    pitch: float,
    roll: float,
    yaw: float,
    robot_xyz,
    held_keys: str = "none",
    last_key_event: str = "none",
) -> tuple[str, str]:
    """Build control and body-state text columns for the graphical overlay."""
    mode = "airborne" if controller.is_airborne else "grounded"
    mode_cn = "离地" if controller.is_airborne else "接地"
    if controller.wheels_on_jump_terrain:
        terrain = "jump terrain"
        terrain_cn = "跳台区域"
    elif controller.wheels_on_terrain:
        terrain = "test terrain"
        terrain_cn = "测试地形"
    else:
        terrain = "floor"
        terrain_cn = "地面"

    left_leg = controller.last_legs["left"]
    right_leg = controller.last_legs["right"]
    actuator = controller.last_actuator_commands
    wheel_l = actuator.get("Left_Wheel_joint_actuator", 0.0)
    wheel_r = actuator.get("Right_Wheel_joint_actuator", 0.0)
    hip_lf = actuator.get("Left_front_joint_actuator", 0.0)
    hip_lr = actuator.get("Left_rear_joint_actuator", 0.0)
    hip_rf = actuator.get("Right_front_joint_actuator", 0.0)
    hip_rr = actuator.get("Right_rear_joint_actuator", 0.0)

    control_lines = [
        "Control data (控制数据)",
        f"controller(控制器): {controller.REVISION}",
        f"sim(仿真时间): {sim_time:5.2f} s",
        f"keys held(当前按键): {_display_none(held_keys)}",
        f"last key(最近按键): {_translate_key_event(last_key_event)}",
        "cmd speed/yaw(指令速度/偏航): "
        f"{controller.command.speed:+.2f} m/s, {controller.command.yaw:+.2f} rad/s",
        "target speed/yaw(目标速度/偏航): "
        f"{controller.target_speed:+.2f} m/s, {controller.target_yaw_rate:+.2f} rad/s",
        f"target distance(目标距离): {controller.target_distance:+.3f} m",
        "leg target(腿长目标): "
        f"{controller.target_leg_length:.3f} m, swing(摆角) {controller.command.leg_swing:+.3f} rad",
        f"wheel torque L/R(轮力矩左/右): {wheel_l:+.2f}, {wheel_r:+.2f} Nm",
        f"hip torque LF/LR(左髋前/后): {hip_lf:+.2f}, {hip_lr:+.2f} Nm",
        f"hip torque RF/RR(右髋前/后): {hip_rf:+.2f}, {hip_rr:+.2f} Nm",
        f"yaw assist(偏航辅助): {actuator.get('Yaw_assist_actuator', 0.0):+.2f} Nm",
    ]
    body_lines = [
        "Body and leg state (机体与腿部状态)",
        f"xyz(位置): {robot_xyz[0]:+.2f}, {robot_xyz[1]:+.2f}, {robot_xyz[2]:+.2f} m",
        f"rpy(滚转/俯仰/偏航): {roll:+.3f}, {pitch:+.3f}, {yaw:+.3f} rad",
        f"contact(接触): {mode}/{mode_cn} on {terrain}/{terrain_cn}",
        "rough terrain(起伏顺应): "
        f"{_activity_label(controller.rough_terrain_active)}, "
        f"h={controller.rough_terrain_height_estimate:.3f} m",
        f"left leg(左腿): L={left_leg.length:.3f} m, a={left_leg.angle:+.3f} rad",
        f"right leg(右腿): L={right_leg.length:.3f} m, a={right_leg.angle:+.3f} rad",
        "leg rates L/R(腿长速度左/右): "
        f"{left_leg.length_rate:+.3f}, {right_leg.length_rate:+.3f} m/s",
        f"airborne time(离地时间): {controller.airborne_time:.3f} s",
        f"stall timer(卡滞计时): {controller.stall_timer:.3f} s",
        f"stall pose(卡滞姿态): {_activity_label(controller.stall_pose_recovery_active)}",
        f"tip recovery(倾倒自救): {_activity_label(controller.stop_tip_recovery_active)}",
        f"stair climb(上台阶): {_activity_label(controller.stair_climb_active)}",
        f"edge climb(贴边爬升): {_activity_label(controller.stair_edge_climb_active)}",
        f"jump(跳跃): {_activity_label(controller.jump_time is not None)}",
    ]
    return "\n".join(control_lines), "\n".join(body_lines)


def build_viewer_overlay(
    controller: ParallelLegController,
    sim_time: float,
    pitch: float,
    roll: float,
    yaw: float,
    robot_xyz,
    held_keys: str = "none",
    last_key_event: str = "none",
) -> tuple[str, str]:
    """Build the fixed on-screen control and body-state HUD."""
    return build_control_overlay(
        controller,
        sim_time,
        pitch,
        roll,
        yaw,
        robot_xyz,
        held_keys,
        last_key_event,
    )


@lru_cache(maxsize=8)
def _load_hud_font(font_size: int):
    if ImageFont is None:
        return None
    for font_path in HUD_CJK_FONT_PATHS:
        if not Path(font_path).exists():
            continue
        try:
            return ImageFont.truetype(font_path, font_size)
        except OSError:
            continue
    for font_name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(font_name, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def _make_hud_overlay_image(left_text: str, right_text: str, font_size: int):
    if Image is None or ImageDraw is None:
        return None
    font = _load_hud_font(font_size)
    if font is None:
        return None

    left_lines = left_text.splitlines()
    right_lines = right_text.splitlines()
    probe = Image.new("RGB", (1, 1))
    probe_draw = ImageDraw.Draw(probe)

    def line_width(line: str) -> int:
        bbox = probe_draw.textbbox((0, 0), line or " ", font=font)
        return bbox[2] - bbox[0]

    line_bbox = probe_draw.textbbox((0, 0), "Hg控制数据", font=font)
    line_height = max(font_size + 4, line_bbox[3] - line_bbox[1])
    left_width = max((line_width(line) for line in left_lines), default=1)
    right_width = max((line_width(line) for line in right_lines), default=1)
    line_count = max(len(left_lines), len(right_lines), 1)
    width = (
        2 * HUD_BITMAP_PADDING
        + left_width
        + HUD_BITMAP_COLUMN_GAP
        + right_width
    )
    height = (
        2 * HUD_BITMAP_PADDING
        + line_count * line_height
        + max(0, line_count - 1) * HUD_BITMAP_LINE_GAP
    )
    image = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(image)

    def draw_column(lines: list[str], x: int) -> None:
        y = HUD_BITMAP_PADDING
        for index, line in enumerate(lines):
            fill = (255, 255, 255) if index == 0 else (226, 226, 226)
            draw.text((x, y), line, font=font, fill=fill)
            y += line_height + HUD_BITMAP_LINE_GAP

    draw_column(left_lines, HUD_BITMAP_PADDING)
    draw_column(right_lines, HUD_BITMAP_PADDING + left_width + HUD_BITMAP_COLUMN_GAP)
    return image


def build_hud_overlay_bitmap(
    left_text: str,
    right_text: str,
    max_width: int | None = None,
) -> np.ndarray | None:
    """Rasterize bilingual HUD text so CJK glyphs render reliably."""
    for font_size in range(HUD_BITMAP_FONT_SIZE, HUD_BITMAP_MIN_FONT_SIZE - 1, -1):
        image = _make_hud_overlay_image(left_text, right_text, font_size)
        if image is None:
            return None
        if max_width is None or image.width <= max_width or font_size == HUD_BITMAP_MIN_FONT_SIZE:
            if max_width is not None and image.width > max_width:
                image = image.crop((0, 0, max_width, image.height))
            return np.ascontiguousarray(np.flipud(np.asarray(image, dtype=np.uint8)))
    return None


def mujoco_rgb_pixel_buffer(bitmap: np.ndarray) -> np.ndarray:
    """Convert an RGB image into the flat column buffer required by mjr_drawPixels."""
    return np.ascontiguousarray(bitmap.reshape(-1, 1))


def _rgb255(color: tuple[float, float, float]) -> tuple[int, int, int]:
    return tuple(int(255 * max(0.0, min(1.0, channel))) for channel in color)


def build_curve_legend_bitmap(
    legend_items: tuple[tuple[str, tuple[float, float, float]], ...],
    max_width: int,
) -> np.ndarray | None:
    """Rasterize a compact colored legend for one MuJoCo figure panel."""
    if Image is None or ImageDraw is None:
        return None
    font = _load_hud_font(CURVE_LEGEND_FONT_SIZE)
    if font is None or not legend_items:
        return None

    max_width = max(80, max_width)
    probe = Image.new("RGB", (1, 1))
    probe_draw = ImageDraw.Draw(probe)
    line_bbox = probe_draw.textbbox((0, 0), "Hg", font=font)
    row_height = max(
        CURVE_LEGEND_FONT_SIZE + 3,
        line_bbox[3] - line_bbox[1],
        CURVE_LEGEND_SWATCH_HEIGHT,
    )
    available_width = max(1, max_width - 2 * CURVE_LEGEND_PADDING)
    rows: list[list[tuple[str, tuple[float, float, float], int]]] = [[]]
    row_widths = [0]

    for name, color in legend_items:
        text_bbox = probe_draw.textbbox((0, 0), name, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        item_width = (
            CURVE_LEGEND_SWATCH_WIDTH
            + 4
            + text_width
            + CURVE_LEGEND_ITEM_GAP
        )
        if rows[-1] and row_widths[-1] + item_width > available_width:
            rows.append([])
            row_widths.append(0)
        rows[-1].append((name, color, text_width))
        row_widths[-1] += item_width

    width = min(max_width, max(row_widths, default=1) + 2 * CURVE_LEGEND_PADDING)
    height = (
        2 * CURVE_LEGEND_PADDING
        + len(rows) * row_height
        + max(0, len(rows) - 1) * CURVE_LEGEND_ROW_GAP
    )
    image = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(image)

    y = CURVE_LEGEND_PADDING
    for row in rows:
        x = CURVE_LEGEND_PADDING
        for name, color, text_width in row:
            rgb = _rgb255(color)
            swatch_top = y + max(0, (row_height - CURVE_LEGEND_SWATCH_HEIGHT) // 2)
            draw.rectangle(
                (
                    x,
                    swatch_top,
                    x + CURVE_LEGEND_SWATCH_WIDTH,
                    swatch_top + CURVE_LEGEND_SWATCH_HEIGHT,
                ),
                fill=rgb,
            )
            draw.text((x + CURVE_LEGEND_SWATCH_WIDTH + 4, y - 1), name, font=font, fill=rgb)
            x += (
                CURVE_LEGEND_SWATCH_WIDTH
                + 4
                + text_width
                + CURVE_LEGEND_ITEM_GAP
            )
        y += row_height + CURVE_LEGEND_ROW_GAP

    return np.ascontiguousarray(np.flipud(np.asarray(image, dtype=np.uint8)))


def render_curve_legend(
    legend_items: tuple[tuple[str, tuple[float, float, float]], ...],
    panel_rect: mujoco.MjrRect,
    context: mujoco.MjrContext,
) -> bool:
    """Draw a colored legend above the figure content."""
    bitmap = build_curve_legend_bitmap(
        legend_items,
        max_width=max(1, panel_rect.width - 12),
    )
    if bitmap is None:
        return False
    height, width = bitmap.shape[:2]
    left = panel_rect.left + panel_rect.width - width - 6
    bottom = panel_rect.bottom + panel_rect.height - height - 6
    rect = mujoco.MjrRect(left, bottom, width, height)
    try:
        mujoco.mjr_rectangle(rect, 0.0, 0.0, 0.0, 0.88)
        mujoco.mjr_drawPixels(mujoco_rgb_pixel_buffer(bitmap), None, rect, context)
    except Exception:
        return False
    return True


def render_hud_overlay(
    left_text: str,
    right_text: str,
    viewport: mujoco.MjrRect,
    context: mujoco.MjrContext,
) -> bool:
    """Draw the bilingual HUD as a bitmap; return False if fallback is needed."""
    max_width = max(1, viewport.width - 2 * HUD_BITMAP_MARGIN)
    bitmap = build_hud_overlay_bitmap(left_text, right_text, max_width=max_width)
    if bitmap is None:
        return False
    height, width = bitmap.shape[:2]
    bottom = max(0, viewport.height - height - HUD_BITMAP_MARGIN)
    rect = mujoco.MjrRect(HUD_BITMAP_MARGIN, bottom, width, height)
    depth_was_enabled = False
    blend_was_enabled = False
    try:
        if GL is not None:
            depth_was_enabled = bool(GL.glIsEnabled(GL.GL_DEPTH_TEST))
            blend_was_enabled = bool(GL.glIsEnabled(GL.GL_BLEND))
            GL.glDisable(GL.GL_DEPTH_TEST)
            GL.glDisable(GL.GL_BLEND)
        mujoco.mjr_rectangle(rect, 0.0, 0.0, 0.0, 0.94)
        mujoco.mjr_drawPixels(mujoco_rgb_pixel_buffer(bitmap), None, rect, context)
    except Exception:
        return False
    finally:
        if GL is not None:
            if depth_was_enabled:
                GL.glEnable(GL.GL_DEPTH_TEST)
            if blend_was_enabled:
                GL.glEnable(GL.GL_BLEND)
    return True


def run_realtime_physics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller: ParallelLegController,
    stop_event: threading.Event,
    state_lock: threading.Lock,
) -> None:
    """Run fixed-step control on a thread that cannot be stalled by rendering."""
    dt = model.opt.timestep
    max_catchup_steps = max(1, int(MAX_CATCHUP_SECONDS / dt))
    next_physics = time.perf_counter()
    while not stop_event.is_set():
        now = time.perf_counter()
        if now < next_physics:
            stop_event.wait(next_physics - now)
            continue

        physics_steps = 0
        with state_lock:
            while now >= next_physics and physics_steps < max_catchup_steps:
                step_physics(model, data, controller)
                next_physics += dt
                physics_steps += 1

        if physics_steps == max_catchup_steps and now >= next_physics:
            next_physics = now + dt


def handle_key(
    controller: ParallelLegController,
    now: float,
    keycode: int,
    keyboard: KeyboardCommandState | None = None,
) -> str | None:
    """Apply a non-momentary keyboard command and return an optional status."""
    key = chr(keycode).lower() if 0 <= keycode < 256 else ""
    if keycode == 265 or key == "w":  # GLFW_KEY_UP
        return "forward"
    if keycode == 264 or key == "s":  # GLFW_KEY_DOWN
        return "reverse"
    if keycode == 263 or key == "a":  # GLFW_KEY_LEFT
        return "turn left"
    if keycode == 262 or key == "d":  # GLFW_KEY_RIGHT
        return "turn right"
    if key == "x":
        if keyboard is None:
            controller.stop_motion()
        else:
            keyboard.clear_motion(controller)
        return "stop"
    if key == "r":
        return "reset"
    if key == " ":
        controller.start_jump(now)
        return "jump"
    if keycode == 266:  # GLFW_KEY_PAGE_UP
        controller.adjust_length(0.008)
        return "leg up"
    if keycode == 267:  # GLFW_KEY_PAGE_DOWN
        controller.adjust_length(-0.008)
        return "leg down"
    if key == "[":
        controller.adjust_leg_swing(-0.04)
        return "leg swing backward"
    if key == "]":
        controller.adjust_leg_swing(0.04)
        return "leg swing forward"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="run a short non-graphical validation")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument(
        "--render-hz",
        type=float,
        default=DEFAULT_RENDER_HZ,
        help="graphical refresh rate; physics remains fixed-step",
    )
    parser.add_argument(
        "--status-interval",
        type=float,
        default=2.0,
        help="seconds between graphical real-time status messages; 0 disables",
    )
    parser.add_argument(
        "--detailed-meshes",
        action="store_true",
        help="render the high-resolution STEP meshes instead of fast primitive visuals",
    )
    parser.add_argument(
        "--hide-control-hud",
        action="store_true",
        help="hide the control and body-state overlay in the graphical viewer",
    )
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(ROOT / "MJCF" / "scene.xml"))
    data = mujoco.MjData(model)
    controller = ParallelLegController(model, data)
    controller.initialize_pose()
    dt = model.opt.timestep
    if args.headless:
        steps = int(args.seconds / dt)
        for step in range(steps):
            if step > steps // 3:
                controller.command.speed = FORWARD_SPEED
            if step > 2 * steps // 3:
                controller.command.yaw = 0.10
            step_physics(model, data, controller)
        robot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot")
        yaw_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_yaw")
        yaw = data.qpos[model.jnt_qposadr[yaw_id]]
        pitch_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_pitch")
        pitch = data.qpos[model.jnt_qposadr[pitch_id]]
        roll_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_roll")
        roll = data.qpos[model.jnt_qposadr[roll_id]]
        print(
            f"headless validation complete: time={data.time:.2f}s "
            f"robot_xyz={data.xpos[robot_id].round(3).tolist()} "
            f"roll={roll:.3f} pitch={pitch:.3f} yaw={yaw:.3f}"
        )
        return

    help_text = (
        "Hold W/S or Arrow Up/Down: forward/reverse | Hold A/D or Arrow Left/Right: turn | "
        "PageUp/PageDown: leg length | [ ]: leg swing | Space: jump | X: stop | R: reset | Esc: exit"
    )
    print(help_text)
    render_hz = max(args.render_hz, 1.0)
    print(
        f"loaded controller={controller.REVISION} | "
        f"physics={1.0 / dt:.0f} Hz | control={1.0 / controller.CONTROL_DT:.0f} Hz | "
        f"render={render_hz:.0f} Hz | pid={os.getpid()}"
    )
    print("physics runs on an independent real-time thread; restart this process after code changes")

    render_data = mujoco.MjData(model)
    sync_render_data(model, data, render_data)
    state_lock = threading.Lock()
    stop_event = threading.Event()
    keyboard = KeyboardCommandState()

    if not glfw.init():
        raise RuntimeError("could not initialize GLFW")
    window = glfw.create_window(
        WINDOW_WIDTH,
        WINDOW_HEIGHT,
        "Parallel Wheel-Leg MuJoCo Simulation",
        None,
        None,
    )
    if window is None:
        glfw.terminate()
        raise RuntimeError("could not create GLFW window")

    glfw.make_context_current(window)
    glfw.swap_interval(1)

    cam = mujoco.MjvCamera()
    cam.distance = 5.6
    cam.azimuth = 45
    cam.elevation = -24
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = args.detailed_meshes
    opt.geomgroup[2] = not args.detailed_meshes
    scene = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(model, HUD_FONT_SCALE)

    robot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot")
    status_wall_time = time.perf_counter()
    status_sim_time = data.time
    control_curves = ControlCurveHistory()

    def key_callback(_window, keycode: int, _scancode: int, action: int, _mods: int) -> None:
        nonlocal status_wall_time, status_sim_time
        if keycode == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)
            return
        if action not in (glfw.PRESS, glfw.RELEASE, glfw.REPEAT):
            return
        with state_lock:
            status = keyboard.handle_event(controller, data.time, keycode, action)
            if status == "reset":
                reset_simulation(model, data, controller, keyboard, render_data)
                control_curves.clear()
                status_wall_time = time.perf_counter()
                status_sim_time = data.time
        if status:
            print(
                f"{status}: speed={controller.command.speed:.2f} m/s "
                f"yaw_rate={controller.command.yaw:.2f} rad/s"
            )

    glfw.set_key_callback(window, key_callback)

    physics_thread = threading.Thread(
        target=run_realtime_physics,
        args=(model, data, controller, stop_event, state_lock),
        name="mujoco-physics",
        daemon=True,
    )
    physics_thread.start()
    try:
        render_dt = 1.0 / render_hz
        while not glfw.window_should_close(window):
            started = time.perf_counter()
            with state_lock:
                sync_render_data(model, data, render_data)
                sim_time = data.time
                pitch = controller._joint_position("root_pitch")
                roll = controller._joint_position("root_roll")
                yaw = controller._joint_position("root_yaw")
                robot_xyz = render_data.xpos[robot_id].copy()
                held_keys = keyboard.held_key_summary()
                last_key_event = keyboard.last_event
                curve_sample = collect_control_curve_sample(
                    controller,
                    sim_time,
                    pitch,
                    roll,
                )
                if not args.hide_control_hud:
                    overlay_left, overlay_right = build_viewer_overlay(
                        controller,
                        sim_time,
                        pitch,
                        roll,
                        yaw,
                        robot_xyz,
                        held_keys,
                        last_key_event,
                    )

            cam.lookat[:] = render_data.xpos[robot_id] + (
                CAMERA_LOOKAHEAD,
                0.0,
                0.04,
            )
            width, height = glfw.get_framebuffer_size(window)
            viewport = mujoco.MjrRect(0, 0, width, height)
            mujoco.mjv_updateScene(
                model,
                render_data,
                opt,
                None,
                cam,
                mujoco.mjtCatBit.mjCAT_ALL,
                scene,
            )
            mujoco.mjr_render(viewport, scene, context)
            if not args.hide_control_hud:
                control_curves.append(curve_sample)
                render_control_curves(control_curves, viewport, context)
                if not render_hud_overlay(overlay_left, overlay_right, viewport, context):
                    fallback_width = min(760, max(1, viewport.width - 2 * HUD_BITMAP_MARGIN))
                    fallback_height = min(330, max(1, viewport.height - 2 * HUD_BITMAP_MARGIN))
                    fallback_rect = mujoco.MjrRect(
                        HUD_BITMAP_MARGIN,
                        max(0, viewport.height - fallback_height - HUD_BITMAP_MARGIN),
                        fallback_width,
                        fallback_height,
                    )
                    mujoco.mjr_rectangle(fallback_rect, 0.0, 0.0, 0.0, 0.88)
                    mujoco.mjr_overlay(
                        mujoco.mjtFont.mjFONT_NORMAL,
                        mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        viewport,
                        overlay_left,
                        overlay_right,
                        context,
                    )
            glfw.swap_buffers(window)
            glfw.poll_events()

            remaining = render_dt - (time.perf_counter() - started)
            if remaining > 0:
                time.sleep(remaining)

            now = time.perf_counter()
            if args.status_interval > 0 and now - status_wall_time >= args.status_interval:
                real_time_factor = (sim_time - status_sim_time) / (
                    now - status_wall_time
                )
                print(
                    f"runtime controller={controller.REVISION} "
                    f"sim={sim_time:.1f}s real_time={real_time_factor:.2f}x "
                    f"pitch={pitch:.3f} roll={roll:.3f} yaw={yaw:.3f} "
                    f"target_speed={controller.target_speed:.2f} "
                    f"leg_target={controller.target_leg_length:.3f}"
                )
                status_wall_time = now
                status_sim_time = sim_time
    finally:
        stop_event.set()
        if physics_thread.is_alive():
            physics_thread.join()
        context.free()
        glfw.destroy_window(window)
        glfw.terminate()


if __name__ == "__main__":
    main()
