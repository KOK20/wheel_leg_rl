"""VMC + 六状态LQR控制器（用于STEP风格的轮腿模型）。"""

from __future__ import annotations

from dataclasses import dataclass
import math

import mujoco
import numpy as np

from kinematics import FiveBarKinematics


@dataclass
class Command:
    """操作员指令（速度、偏航、腿长、摆角）。"""
    speed: float = 0.0
    yaw: float = 0.0
    leg_length: float = 0.165
    leg_swing: float = 0.0


@dataclass
class LegState:
    """一条腿的观测状态。"""
    length: float = 0.165
    angle: float = 0.0
    length_rate: float = 0.0
    angle_rate: float = 0.0


class ParallelLegController:
    """将原始VMC/LQR结构转换为MuJoCo力矩控制。"""

    REVISION = "smooth-vmc-v16"
    MIN_LENGTH = 0.105
    MAX_LENGTH = 0.205
    MAX_LEG_SWING = 0.35
    WHEEL_RADIUS = 0.0625
    CONTROL_DT = 0.001
    MAX_ACCELERATION = 0.30
    STOP_ACCELERATION = 1.2
    REVERSAL_ACCELERATION = 0.55
    SPEED_COMMAND_DEADBAND = 0.02
    SPEED_REVERSAL_DEADBAND = 0.04
    STOP_TARGET_SPEED = 0.03
    STOP_POSITION_RELAX_RATE = 0.10
    MAX_YAW_ACCELERATION = 1.5
    YAW_COMMAND_DEADBAND = 0.02
    MAX_POSITION_ERROR = 0.30
    MAX_YAW_ERROR = 0.45
    YAW_RATE_MODE_THRESHOLD = 0.02
    LEG_FORCE_KP = 450.0
    LEG_FORCE_KD = 55.0
    LOW_LENGTH_BLEND_START = 0.145
    LOW_LENGTH_WHEEL_SCALE = 0.42
    LOW_LENGTH_HIP_SCALE = 0.72
    LOW_LENGTH_LEG_KP_SCALE = 0.62
    LOW_LENGTH_LEG_KD_SCALE = 1.18
    TURN_LEG_DAMPING = 100.0
    MAX_AXIAL_FORCE = 180.0
    SUPPORT_FEEDFORWARD = 0.75
    MAX_ROLL_FORCE = 30.0
    MAX_TURN_TORQUE = 0.28
    MAX_YAW_ASSIST_TORQUE = 6.0
    MAX_WHEEL_TORQUE = 4.0
    MAX_HIP_TORQUE = 8.0
    BODY_PITCH_TP_KP = 18.0
    BODY_PITCH_TP_KD = 3.0
    WHEEL_TORQUE_RATE = 700.0
    HIP_TORQUE_RATE = 400.0
    ROLL_KP = 180.0
    ROLL_KD = 25.0
    YAW_KP = 1.5
    YAW_KD = 2.5
    YAW_KI = 6.0
    MAX_YAW_INTEGRAL = 1.5
    YAW_ASSIST_TORQUE_RATE = 45.0
    TURN_DRIFT_COMPENSATION = 0.0
    LEG_SYNC_KP = 12.0
    LEG_SYNC_KD = 1.8
    LEG_RATE_FILTER_TAU = 0.04
    LEG_COMMAND_RATE = 0.10
    ROUGH_TERRAIN_LEG_COMMAND_RATE = 0.24
    ROUGH_TERRAIN_APPROACH_MARGIN = 0.35
    ROUGH_TERRAIN_EXIT_MARGIN = 0.30
    ROUGH_TERRAIN_LATERAL_MARGIN = 0.18
    ROUGH_TERRAIN_SPEED_LIMIT = 0.46
    ROUGH_TERRAIN_HEIGHT_GAIN = 0.70
    ROUGH_TERRAIN_HEIGHT_CAP = 0.060
    ROUGH_TERRAIN_MIN_LEG_LENGTH = 0.130
    ROUGH_TERRAIN_LEG_KP_SCALE = 0.94
    ROUGH_TERRAIN_LEG_KD_SCALE = 1.20
    ROUGH_TERRAIN_PITCH_TP_KP = 7.5
    ROUGH_TERRAIN_PITCH_TP_KD = 1.2
    AIRBORNE_DEBOUNCE = 0.03
    JUMP_MOTION_HOLD = 1.0
    JUMP_LANDING_RECOVERY = 0.25
    MOVING_JUMP_SPEED_START = 0.25
    MOVING_JUMP_SPEED_FULL = 0.65
    MOVING_JUMP_MIN_FORCE_SCALE = 0.50
    MOVING_JUMP_MIN_EXTEND_LENGTH = 0.182
    JUMP_COURSE_FORCE_SCALE = 1.25
    JUMP_COURSE_COMPRESS_TIME = 0.18
    JUMP_COURSE_EXTEND_TIME = 0.46
    JUMP_COURSE_RECOVER_TIME = 0.82
    JUMP_COURSE_TUCK_LENGTH = 0.155
    STAIR_CLIMB_SPEED_LIMIT = 0.50
    STAIR_CLIMB_TRIGGER_MIN_SPEED = 0.24
    STAIR_CLIMB_TRIGGER_MIN_DISTANCE = 0.22
    STAIR_CLIMB_TRIGGER_MAX_DISTANCE = 0.25
    STAIR_SIDE_TRIGGER_MAX_DISTANCE = 0.18
    STAIR_TRIGGER_WINDOW = 0.03
    STAIR_CLIMB_CRAWL_SPEED = 0.18
    STAIR_CONTACT_ASSIST_COOLDOWN = 1.15
    STAIR_CONTACT_NEAR_MARGIN = 0.38
    STAIR_EDGE_CRAWL_FRONT_MARGIN = 0.12
    STAIR_EDGE_CRAWL_REAR_MARGIN = 0.24
    STAIR_EDGE_CRAWL_PITCH = 0.08
    STAIR_EDGE_CRAWL_SPEED = 0.30
    STAIR_EDGE_CRAWL_LEG_LENGTH = 0.200
    STAIR_EDGE_CRAWL_WHEEL_TORQUE = 1.35
    STAIR_EDGE_PRELOAD_DISTANCE = 0.045
    STAIR_EDGE_CLIMB_LIFT_FORCE = 40.0
    STAIR_EDGE_CLIMB_PUSH_FORCE = 10.0
    STAIR_EDGE_YAW_ALIGN_MARGIN = 0.18
    STAIR_EDGE_YAW_ALIGN_LIMIT = 0.70
    STAIR_EDGE_TURN_TORQUE_LIMIT = 0.20
    STAIR_EDGE_YAW_ASSIST_LIMIT = 3.0
    STAIR_EDGE_CRAWL_MAX_TIME = 2.40
    STAIR_EDGE_ESCAPE_PITCH = 0.85
    STAIR_EDGE_ESCAPE_ROLL = 0.65
    STAIR_EDGE_ESCAPE_TIME = 0.75
    STAIR_EDGE_ESCAPE_SPEED = 0.16
    STAIR_CLIMB_MAX_WHEEL_TORQUE = 2.20
    STAIR_EDGE_CRAWL_PITCH_TP_KP = 12.0
    STAIR_EDGE_CRAWL_PITCH_TP_KD = 2.0
    STAIR_CLIMB_WHEEL_FEEDFORWARD = 0.10
    STAIR_CLIMB_PITCH_TP_KP = 4.0
    STAIR_CLIMB_PITCH_TP_KD = 0.7
    TERRAIN_DRIVE_JUMP_FORCE_SCALE = 0.0
    JUMP_TERRAIN_APPROACH_MARGIN = 0.90
    JUMP_TERRAIN_EXIT_MARGIN = 0.45
    JUMP_TERRAIN_LATERAL_MARGIN = 0.15
    STALL_COMMAND_THRESHOLD = 0.20
    STALL_SPEED_THRESHOLD = 0.14
    STALL_PITCH_THRESHOLD = 0.18
    STALL_DETECT_TIME = 0.16
    STALL_RECOVERY_TIME = 0.80
    STALL_RECOVERY_SPEED = 0.22
    STALL_RECOVERY_FORCE = 55.0
    STALL_RECOVERY_LEG_LENGTH = 0.120
    STALL_POSE_PITCH_THRESHOLD = 0.35
    STALL_POSE_LOW_SPEED_TARGET = 0.08
    STALL_POSE_WHEEL_TORQUE = 1.60
    STALL_FORWARD_ESCAPE_WHEEL_TORQUE = 1.60
    STALL_REVERSE_ESCAPE_WHEEL_TORQUE = 3.00
    STOP_TIP_PITCH_THRESHOLD = 0.45
    STOP_TIP_RELEASE_PITCH = 0.28
    STOP_TIP_DETECT_TIME = 0.12
    STOP_TIP_RIGHTING_LEG_LENGTH = 0.185
    STOP_TIP_RIGHTING_FORCE = 35.0
    STOP_TIP_RIGHTING_TP_KP = 30.0
    STOP_TIP_RIGHTING_TP_KD = 5.0
    STOP_TIP_RIGHTING_WHEEL_TORQUE = 0.85

    # State order: leg angle, leg angular rate, x error, velocity error,
    # chassis pitch, chassis pitch rate. Two rows output wheel torque and Tp.
    LQR_K = np.array(
        [
            [0.0, 0.0, -8.0, -18.0, -67.5, -13.5],
            [4.0, 1.6, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        # ---- 机器人模型与状态初始化 ----
        self.model = model
        self.data = data
        self.kinematics = FiveBarKinematics()
        self.command = Command()
        self.jump_time: float | None = None
        self.forward_distance = 0.0
        self.target_distance = 0.0
        self.target_speed = 0.0
        self.target_yaw = 0.0
        self.target_yaw_rate = 0.0
        self.yaw_rate_integral = 0.0
        self.last_speed_command = 0.0
        self.last_yaw_command = 0.0
        self.target_leg_length = self.command.leg_length
        self.jump_motion_hold_until = 0.0
        self.last_control_time = -self.CONTROL_DT
        self.last_legs = {"left": LegState(), "right": LegState()}
        self.last_actuator_commands: dict[str, float] = {}
        self.airborne_time = 0.0
        self.is_airborne = False
        self.wheels_on_terrain = False
        self.wheels_on_jump_terrain = False
        self.wheels_on_climb_terrain = False
        self.wheels_on_rough_terrain = False
        self.chassis_on_support = False
        self.chassis_on_jump_terrain = False
        self.chassis_on_climb_terrain = False
        self.rough_terrain_active = False
        self.rough_terrain_height_estimate = 0.0
        self.jump_landing_recovery_pending = False
        self.jump_course_active = False
        self.stair_climb_active = False
        self.stair_edge_climb_active = False
        self.stair_climb_edge_index: int | None = None
        self.completed_stair_climb_edges: set[int] = set()
        self.last_stair_contact_assist_time = -math.inf
        self.stair_edge_climb_started_at: float | None = None
        self.stair_edge_escape_until = 0.0
        self.stair_edge_escape_speed = 0.0
        self.stall_timer = 0.0
        self.stall_recovery_until = 0.0
        self.stall_recovery_speed = 0.0
        self.stall_blocked_direction = 0.0
        self.stall_pose_recovery_active = False
        self.stop_tip_timer = 0.0
        self.stop_tip_recovery_active = False

        self.actuator_ids = {
            name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in (
                "Left_front_joint_actuator",
                "Left_rear_joint_actuator",
                "Right_front_joint_actuator",
                "Right_rear_joint_actuator",
                "Left_Wheel_joint_actuator",
                "Right_Wheel_joint_actuator",
                "Yaw_assist_actuator",
            )
        }
        self.joint_ids = {
            name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in (
                "root_x",
                "root_y",
                "root_z",
                "root_yaw",
                "root_roll",
                "root_pitch",
                "left_front_hip",
                "left_front_knee",
                "left_rear_hip",
                "left_rear_knee",
                "right_front_hip",
                "right_front_knee",
                "right_rear_hip",
                "right_rear_knee",
            )
        }
        self.reference = {
            "front_crank": math.atan2(0.042005, 0.072743),
            "front_rod": math.atan2(-0.109412, -0.116743),
            "rear_crank": math.atan2(0.042005, -0.072743),
            "rear_rod": math.atan2(-0.109412, 0.116743),
        }
        self.root_x_dof_id = self.model.jnt_dofadr[self.joint_ids["root_x"]]
        self.root_y_dof_id = self.model.jnt_dofadr[self.joint_ids["root_y"]]
        self.root_z_dof_id = self.model.jnt_dofadr[self.joint_ids["root_z"]]
        self.robot_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "robot"
        )
        self.robot_velocity = np.zeros(6)
        self.floor_geom_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )
        self.chassis_geom_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "chassis_collision"
        )
        self.support_geom_ids = {self.floor_geom_id}
        self.rough_terrain_geom_ids: set[int] = set()
        self.rough_terrain_x_min = math.inf
        self.rough_terrain_x_max = -math.inf
        self.rough_terrain_y_half = 0.0
        self.climb_terrain_geom_ids: set[int] = set()
        self.climb_terrain_boxes: list[tuple[float, float, float, float]] = []
        self.jump_terrain_geom_ids: set[int] = set()
        self.jump_terrain_x_min = math.inf
        self.jump_terrain_x_max = -math.inf
        self.jump_terrain_y_half = 0.0
        self.jump_terrain_edges: list[tuple[float, float]] = []
        self.jump_terrain_boxes: list[tuple[float, float, float, float]] = []
        for geom_id in range(model.ngeom):
            geom_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
            )
            if geom_name and geom_name.startswith("terrain_"):
                self.support_geom_ids.add(geom_id)
                geom_pos = model.geom_pos[geom_id]
                geom_size = model.geom_size[geom_id]
                climbable_terrain = (
                    model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX
                    and not geom_name.startswith("terrain_jump_lip_")
                    and "slope" not in geom_name
                    and "lead_in" not in geom_name
                )
                if climbable_terrain:
                    self.climb_terrain_geom_ids.add(geom_id)
                    self.climb_terrain_boxes.append(
                        (
                            float(geom_pos[0] - geom_size[0]),
                            float(geom_pos[0] + geom_size[0]),
                            float(geom_pos[1] - geom_size[1]),
                            float(geom_pos[1] + geom_size[1]),
                        )
                    )
                if geom_name.startswith("terrain_jump_lip_"):
                    self.jump_terrain_geom_ids.add(geom_id)
                elif geom_name.startswith("terrain_jump_"):
                    self.jump_terrain_geom_ids.add(geom_id)
                    self.jump_terrain_x_min = min(
                        self.jump_terrain_x_min, float(geom_pos[0] - geom_size[0])
                    )
                    self.jump_terrain_x_max = max(
                        self.jump_terrain_x_max, float(geom_pos[0] + geom_size[0])
                    )
                    self.jump_terrain_y_half = max(
                        self.jump_terrain_y_half, float(geom_size[1])
                    )
                    self.jump_terrain_edges.append(
                        (
                            float(geom_pos[0] - geom_size[0]),
                            float(geom_pos[0] + geom_size[0]),
                        )
                    )
                    self.jump_terrain_boxes.append(
                        (
                            float(geom_pos[0] - geom_size[0]),
                            float(geom_pos[0] + geom_size[0]),
                            float(geom_pos[1] - geom_size[1]),
                            float(geom_pos[1] + geom_size[1]),
                        )
                    )
                else:
                    self.rough_terrain_geom_ids.add(geom_id)
                    self.rough_terrain_x_min = min(
                        self.rough_terrain_x_min, float(geom_pos[0] - geom_size[0])
                    )
                    self.rough_terrain_x_max = max(
                        self.rough_terrain_x_max, float(geom_pos[0] + geom_size[0])
                    )
                    self.rough_terrain_y_half = max(
                        self.rough_terrain_y_half, float(geom_size[1])
                    )
        self.jump_terrain_edges.sort()
        self.wheel_geom_ids = {
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_wheel_collision"
            )
            for side in ("left", "right")
        }
        self.total_mass = float(np.sum(model.body_mass))
        self.last_actuator_commands = {
            name: 0.0 for name in self.actuator_ids
        }

    def _set_joint_position(self, name: str, value: float) -> None:
        joint_id = self.joint_ids[name]
        self.data.qpos[self.model.jnt_qposadr[joint_id]] = value

    def _joint_position(self, name: str) -> float:
        joint_id = self.joint_ids[name]
        return float(self.data.qpos[self.model.jnt_qposadr[joint_id]])

    def _joint_velocity(self, name: str) -> float:
        joint_id = self.joint_ids[name]
        return float(self.data.qvel[self.model.jnt_dofadr[joint_id]])

    @staticmethod
    def _move_towards(current: float, target: float, max_delta: float) -> float:
        return current + float(np.clip(target - current, -max_delta, max_delta))

    def _low_length_blend(self, length: float) -> float:
        span = self.LOW_LENGTH_BLEND_START - self.MIN_LENGTH
        if span <= 0.0:
            return 1.0
        return float(np.clip((length - self.MIN_LENGTH) / span, 0.0, 1.0))

    @staticmethod
    def _scheduled_scale(minimum: float, blend: float) -> float:
        return minimum + (1.0 - minimum) * blend

    def _set_actuator_torque(self, name: str, target: float, rate: float, dt: float) -> None:
        # 按速率斜坡变化，避免指令瞬变导致的尖峰
        previous = self.last_actuator_commands[name]
        command = self._move_towards(
            previous, target, rate * dt
        )
        self.last_actuator_commands[name] = command
        self.data.ctrl[self.actuator_ids[name]] = command

    def stop_motion(self) -> None:
        """Cancel operator motion commands and hold the current pose target."""
        # 按键松开或紧急停止时，清空速度/偏航指令，保留当前位姿目标
        self.command.speed = 0.0
        self.command.yaw = 0.0
        self.target_speed = 0.0
        self.target_yaw_rate = 0.0
        self.last_speed_command = 0.0
        self.last_yaw_command = 0.0
        self.target_distance = self.forward_distance
        self.target_yaw = self._joint_position("root_yaw")
        self.yaw_rate_integral = 0.0
        self.stall_timer = 0.0
        self.stall_recovery_until = 0.0
        self.stall_recovery_speed = 0.0
        self.stall_blocked_direction = 0.0
        self.stall_pose_recovery_active = False
        self.stop_tip_timer = 0.0
        self.stop_tip_recovery_active = False

    def _wheels_grounded(self) -> bool:
        # 扫描接触信息，判断轮子是否着地、是否在非地面地形或跳跃平台上
        grounded = False
        self.wheels_on_terrain = False
        self.wheels_on_jump_terrain = False
        self.wheels_on_climb_terrain = False
        self.wheels_on_rough_terrain = False
        self.chassis_on_support = False
        self.chassis_on_jump_terrain = False
        self.chassis_on_climb_terrain = False
        for contact in self.data.contact:
            pair = {contact.geom1, contact.geom2}
            support_contacts = pair & self.support_geom_ids
            if support_contacts and self.chassis_geom_id in pair:
                self.chassis_on_support = True
                if support_contacts & self.jump_terrain_geom_ids:
                    self.chassis_on_jump_terrain = True
                if support_contacts & self.climb_terrain_geom_ids:
                    self.chassis_on_climb_terrain = True
            if support_contacts and pair & self.wheel_geom_ids:
                grounded = True
                if any(geom_id != self.floor_geom_id for geom_id in support_contacts):
                    self.wheels_on_terrain = True
                if support_contacts & self.jump_terrain_geom_ids:
                    self.wheels_on_jump_terrain = True
                if support_contacts & self.climb_terrain_geom_ids:
                    self.wheels_on_climb_terrain = True
                if support_contacts & self.rough_terrain_geom_ids:
                    self.wheels_on_rough_terrain = True
        return grounded

    def _near_rough_terrain_course(self) -> bool:
        # 是否在可直接驶过的坡/小坎地形范围附近
        if not self.rough_terrain_geom_ids:
            return False
        root_x = self._joint_position("root_x")
        root_y = self._joint_position("root_y")
        return (
            self.rough_terrain_x_min - self.ROUGH_TERRAIN_APPROACH_MARGIN
            <= root_x
            <= self.rough_terrain_x_max + self.ROUGH_TERRAIN_EXIT_MARGIN
            and abs(root_y)
            <= self.rough_terrain_y_half + self.ROUGH_TERRAIN_LATERAL_MARGIN
        )

    def _rough_terrain_support_height(self) -> float:
        # 用机体高度和上一拍腿长估计轮下地形高度，驱动“主动悬架”腿长补偿
        mean_leg_length = 0.5 * (
            self.last_legs["left"].length + self.last_legs["right"].length
        )
        body_z = float(self.data.xpos[self.robot_body_id, 2])
        support_height = body_z - mean_leg_length - self.WHEEL_RADIUS + 0.053
        return float(np.clip(support_height, 0.0, self.ROUGH_TERRAIN_HEIGHT_CAP))

    def _rough_terrain_leg_length_target(self, support_height: float) -> float:
        relief = self.ROUGH_TERRAIN_HEIGHT_GAIN * support_height
        return float(
            np.clip(
                self.command.leg_length - relief,
                self.ROUGH_TERRAIN_MIN_LEG_LENGTH,
                self.command.leg_length,
            )
        )

    def _near_jump_terrain_course(self) -> bool:
        # 是否在跳跃地形的水平/侧向范围附近
        if not self.jump_terrain_geom_ids:
            return False
        root_x = self._joint_position("root_x")
        root_y = self._joint_position("root_y")
        return (
            self.jump_terrain_x_min - self.JUMP_TERRAIN_APPROACH_MARGIN
            <= root_x
            <= self.jump_terrain_x_max + self.JUMP_TERRAIN_EXIT_MARGIN
            and abs(root_y)
            <= self.jump_terrain_y_half + self.JUMP_TERRAIN_LATERAL_MARGIN
        )

    def _near_any_jump_terrain(self, margin: float) -> bool:
        if not self.jump_terrain_boxes:
            return False
        root_x = self._joint_position("root_x")
        root_y = self._joint_position("root_y")
        for x_min, x_max, y_min, y_max in self.jump_terrain_boxes:
            if (
                x_min - margin <= root_x <= x_max + margin
                and y_min - margin <= root_y <= y_max + margin
            ):
                return True
        return False

    def _near_any_climb_terrain(self, margin: float) -> bool:
        if not self.climb_terrain_boxes:
            return False
        root_x = self._joint_position("root_x")
        root_y = self._joint_position("root_y")
        for x_min, x_max, y_min, y_max in self.climb_terrain_boxes:
            if (
                x_min - margin <= root_x <= x_max + margin
                and y_min - margin <= root_y <= y_max + margin
            ):
                return True
        return False

    def _stair_climb_trigger_edge(self, speed_target: float) -> int | None:
        # 沿当前车体朝向寻找即将接触的跳台边缘，而不是只认世界x方向。
        if abs(speed_target) <= self.STAIR_CLIMB_TRIGGER_MIN_SPEED:
            return None
        root_x = self._joint_position("root_x")
        root_y = self._joint_position("root_y")
        yaw = self._joint_position("root_yaw")
        direction_sign = math.copysign(1.0, speed_target)
        dir_x = direction_sign * math.cos(yaw)
        dir_y = direction_sign * math.sin(yaw)
        axis_sum = abs(dir_x) + abs(dir_y)
        x_axis_blend = abs(dir_x) / axis_sum if axis_sum > 1e-6 else 1.0
        trigger_max = (
            self.STAIR_SIDE_TRIGGER_MAX_DISTANCE
            + (
                self.STAIR_CLIMB_TRIGGER_MAX_DISTANCE
                - self.STAIR_SIDE_TRIGGER_MAX_DISTANCE
            )
            * x_axis_blend
        )
        trigger_min = max(0.0, trigger_max - self.STAIR_TRIGGER_WINDOW)
        for index, (x_min, x_max, y_min, y_max) in enumerate(self.climb_terrain_boxes):
            if index in self.completed_stair_climb_edges:
                continue
            distance_to_edge = self._ray_box_entry_distance(
                root_x,
                root_y,
                dir_x,
                dir_y,
                x_min,
                x_max,
                y_min,
                y_max,
                0.0,
            )
            if (
                distance_to_edge is not None
                and trigger_min <= distance_to_edge <= trigger_max
            ):
                return index
        return None

    @staticmethod
    def _ray_box_entry_distance(
        origin_x: float,
        origin_y: float,
        dir_x: float,
        dir_y: float,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        margin: float,
    ) -> float | None:
        x_min -= margin
        x_max += margin
        y_min -= margin
        y_max += margin
        t_min = -math.inf
        t_max = math.inf
        for origin, direction, low, high in (
            (origin_x, dir_x, x_min, x_max),
            (origin_y, dir_y, y_min, y_max),
        ):
            if abs(direction) < 1e-6:
                if origin < low or origin > high:
                    return None
                continue
            t1 = (low - origin) / direction
            t2 = (high - origin) / direction
            t_near = min(t1, t2)
            t_far = max(t1, t2)
            t_min = max(t_min, t_near)
            t_max = min(t_max, t_far)
            if t_min > t_max:
                return None
        if t_max < 0.0:
            return None
        return max(t_min, 0.0)

    def _stair_climb_edge_completed(self, edge_index: int) -> bool:
        if edge_index >= len(self.climb_terrain_boxes):
            return False
        root_x = self._joint_position("root_x")
        root_y = self._joint_position("root_y")
        x_min, x_max, y_min, y_max = self.climb_terrain_boxes[edge_index]
        return (
            x_min - 0.08 <= root_x <= x_max + 0.08
            and y_min - 0.08 <= root_y <= y_max + 0.08
        )

    def _stair_edge_climb_candidate(
        self, speed_target: float, pitch: float, forward_velocity: float
    ) -> bool:
        # 轮子/机体搭到台阶边缘后，改用低速伸腿贴边爬升，避免大轮矩原地疯车。
        # 这里不自动触发跳跃：接近边缘时先降速贴住，再靠轮矩和虚拟腿支撑慢慢爬。
        if abs(speed_target) <= self.STAIR_CLIMB_TRIGGER_MIN_SPEED:
            return False
        if self.jump_time is not None:
            return False
        touching_climb_terrain = (
            self.wheels_on_climb_terrain or self.chassis_on_climb_terrain
        )
        if not touching_climb_terrain and not self._near_any_climb_terrain(
            self.STAIR_CONTACT_NEAR_MARGIN
        ):
            return False
        root_x = self._joint_position("root_x")
        root_y = self._joint_position("root_y")
        yaw = self._joint_position("root_yaw")
        direction_sign = math.copysign(1.0, speed_target)
        dir_x = direction_sign * math.cos(yaw)
        dir_y = direction_sign * math.sin(yaw)
        near_edge = touching_climb_terrain
        preload_edge_index: int | None = None
        for index, (x_min, x_max, y_min, y_max) in enumerate(self.climb_terrain_boxes):
            inside_box = x_min <= root_x <= x_max and y_min <= root_y <= y_max
            if not inside_box:
                distance_to_edge = self._ray_box_entry_distance(
                    root_x,
                    root_y,
                    dir_x,
                    dir_y,
                    x_min,
                    x_max,
                    y_min,
                    y_max,
                    0.0,
                )
                if (
                    distance_to_edge is not None
                    and distance_to_edge <= self.STAIR_EDGE_PRELOAD_DISTANCE
                ):
                    preload_edge_index = index
                    near_edge = True
            if (
                x_min - self.STAIR_EDGE_CRAWL_FRONT_MARGIN
                <= root_x
                <= x_max + self.STAIR_EDGE_CRAWL_REAR_MARGIN
                and y_min - self.STAIR_EDGE_CRAWL_FRONT_MARGIN
                <= root_y
                <= y_max + self.STAIR_EDGE_CRAWL_REAR_MARGIN
            ):
                near_edge = True
        if preload_edge_index is not None:
            self.stair_climb_edge_index = preload_edge_index
            return True
        if not near_edge:
            return False
        if abs(pitch) < self.STAIR_EDGE_CRAWL_PITCH:
            return False
        if abs(forward_velocity) > self.STAIR_EDGE_CRAWL_SPEED:
            return False
        return True

    def _stair_edge_alignment_yaw(self, speed_target: float) -> float:
        if not self.climb_terrain_boxes or abs(speed_target) <= self.SPEED_COMMAND_DEADBAND:
            return self._joint_position("root_yaw")
        root_x = self._joint_position("root_x")
        root_y = self._joint_position("root_y")
        yaw = self._joint_position("root_yaw")
        speed_sign = math.copysign(1.0, speed_target)
        motion_x = speed_sign * math.cos(yaw)
        motion_y = speed_sign * math.sin(yaw)
        nearest_axis: str | None = None
        nearest_distance = math.inf
        for x_min, x_max, y_min, y_max in self.climb_terrain_boxes:
            if (
                root_x < x_min - self.STAIR_EDGE_YAW_ALIGN_MARGIN
                or root_x > x_max + self.STAIR_EDGE_YAW_ALIGN_MARGIN
                or root_y < y_min - self.STAIR_EDGE_YAW_ALIGN_MARGIN
                or root_y > y_max + self.STAIR_EDGE_YAW_ALIGN_MARGIN
            ):
                continue
            edge_distances = (
                ("x", abs(root_x - x_min)),
                ("x", abs(root_x - x_max)),
                ("y", abs(root_y - y_min)),
                ("y", abs(root_y - y_max)),
            )
            axis, distance = min(edge_distances, key=lambda item: item[1])
            if distance < nearest_distance:
                nearest_axis = axis
                nearest_distance = distance
        if nearest_axis is None or nearest_distance > self.STAIR_EDGE_YAW_ALIGN_MARGIN:
            return yaw
        if nearest_axis == "x":
            motion_yaw = 0.0 if motion_x >= 0.0 else math.pi
        else:
            motion_yaw = math.pi / 2.0 if motion_y >= 0.0 else -math.pi / 2.0
        if speed_target < 0.0:
            motion_yaw += math.pi
        return motion_yaw

    def _stall_recovery_direction(self, speed_target: float) -> float:
        # 根据台阶边界，决定优先脱困方向（先尽量后退回去）
        if speed_target < -self.STALL_COMMAND_THRESHOLD:
            return -1.0
        root_x = self._joint_position("root_x")
        for front_edge, rear_edge in sorted(self.jump_terrain_edges):
            if front_edge - 0.10 <= root_x <= front_edge + 0.25:
                return -1.0
            if rear_edge - 0.25 <= root_x <= rear_edge + 0.10:
                return 1.0
        return -math.copysign(1.0, speed_target)

    def initialize_pose(self) -> None:
        """Assemble the five-bars at nominal length before enabling control."""
        # 重置到初始姿态并清零所有临时控制记忆
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        front, front_rod, rear, rear_rod = self.kinematics.inverse(self.command.leg_length)
        for side in ("left", "right"):
            front_motor_q = self.reference["rear_crank"] - front
            rear_motor_q = self.reference["front_crank"] - rear
            self._set_joint_position(f"{side}_front_hip", front_motor_q)
            self._set_joint_position(
                f"{side}_front_knee",
                (self.reference["rear_rod"] - front_rod) - front_motor_q,
            )
            self._set_joint_position(f"{side}_rear_hip", rear_motor_q)
            self._set_joint_position(
                f"{side}_rear_knee",
                (self.reference["front_rod"] - rear_rod) - rear_motor_q,
            )
        root_z = self.command.leg_length + self.WHEEL_RADIUS - 0.0765 - 0.053
        root_z_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "root_z")
        self.data.qpos[self.model.jnt_qposadr[root_z_id]] = root_z
        mujoco.mj_forward(self.model, self.data)
        self.forward_distance = 0.0
        self.target_distance = 0.0
        self.target_speed = 0.0
        self.jump_time = None
        self.target_yaw = self._joint_position("root_yaw")
        self.target_yaw_rate = 0.0
        self.yaw_rate_integral = 0.0
        self.last_speed_command = 0.0
        self.last_yaw_command = 0.0
        self.target_leg_length = self.command.leg_length
        self.jump_motion_hold_until = 0.0
        self.airborne_time = 0.0
        self.is_airborne = False
        self.wheels_on_terrain = False
        self.wheels_on_jump_terrain = False
        self.wheels_on_climb_terrain = False
        self.wheels_on_rough_terrain = False
        self.chassis_on_support = False
        self.chassis_on_jump_terrain = False
        self.chassis_on_climb_terrain = False
        self.rough_terrain_active = False
        self.rough_terrain_height_estimate = 0.0
        self.jump_landing_recovery_pending = False
        self.jump_course_active = False
        self.stair_climb_active = False
        self.stair_edge_climb_active = False
        self.stair_climb_edge_index = None
        self.completed_stair_climb_edges.clear()
        self.last_stair_contact_assist_time = -math.inf
        self.stair_edge_climb_started_at = None
        self.stair_edge_escape_until = 0.0
        self.stair_edge_escape_speed = 0.0
        self.stall_timer = 0.0
        self.stall_recovery_until = 0.0
        self.stall_recovery_speed = 0.0
        self.stall_blocked_direction = 0.0
        self.stall_pose_recovery_active = False
        self.stop_tip_timer = 0.0
        self.stop_tip_recovery_active = False
        for name in self.last_actuator_commands:
            self.last_actuator_commands[name] = 0.0
        for side in ("left", "right"):
            front, rear = self._absolute_hip_angles(side)
            point = self.kinematics.forward(front, rear)
            self.last_legs[side] = LegState(
                length=float(np.linalg.norm(point)),
                angle=math.atan2(point[0], -point[1]),
            )
        self.update(self.data.time, force=True)

    def adjust_length(self, delta: float) -> None:
        # 安全地调整目标腿长
        self.command.leg_length = float(
            np.clip(self.command.leg_length + delta, self.MIN_LENGTH, self.MAX_LENGTH)
        )

    def adjust_leg_swing(self, delta: float) -> None:
        # 安全地调整目标摆角
        self.command.leg_swing = float(
            np.clip(
                self.command.leg_swing + delta,
                -self.MAX_LEG_SWING,
                self.MAX_LEG_SWING,
            )
        )

    def start_jump(self, now: float, force_course: bool = False) -> None:
        # 触发跳跃时重置状态，避免旧的失速保护影响本次起跳
        if self.jump_time is None:
            self.jump_time = now
            self.jump_landing_recovery_pending = False
            self.stall_timer = 0.0
            self.stall_recovery_until = 0.0
            self.stall_recovery_speed = 0.0
            self.stall_blocked_direction = 0.0
            self.stall_pose_recovery_active = False
            self.stop_tip_timer = 0.0
            self.stop_tip_recovery_active = False
            self.jump_course_active = (
                force_course
                or self._near_jump_terrain_course()
                or self._near_any_climb_terrain(self.STAIR_CONTACT_NEAR_MARGIN)
                or self.wheels_on_jump_terrain
                or self.chassis_on_jump_terrain
                or self.wheels_on_climb_terrain
                or self.chassis_on_climb_terrain
            )
            self.stair_climb_active = self.jump_course_active
            self.stair_edge_climb_active = False
            self.stair_climb_edge_index = None
            jump_hold = 0.0 if self.jump_course_active else self.JUMP_MOTION_HOLD
            self.jump_motion_hold_until = max(
                self.jump_motion_hold_until, now + jump_hold
            )

    def _jump_profile(self, now: float) -> tuple[float, float]:
        # 跳跃曲线：先快速收腿->撑开->过渡回收
        if self.jump_time is None:
            return self.command.leg_length, 0.0
        elapsed = now - self.jump_time
        compress_time = (
            self.JUMP_COURSE_COMPRESS_TIME if self.jump_course_active else 0.22
        )
        extend_time = (
            self.JUMP_COURSE_EXTEND_TIME if self.jump_course_active else 0.38
        )
        recover_time = (
            self.JUMP_COURSE_RECOVER_TIME if self.jump_course_active else 0.72
        )
        tuck_length = (
            self.JUMP_COURSE_TUCK_LENGTH if self.jump_course_active else 0.145
        )
        jump_force = 130.0 * (
            self.JUMP_COURSE_FORCE_SCALE if self.jump_course_active else 1.0
        )
        if elapsed < compress_time:
            return self.MIN_LENGTH, 0.0
        if elapsed < extend_time:
            return self.MAX_LENGTH, jump_force
        if elapsed < recover_time:
            return tuck_length, 0.0
        if self.stair_climb_edge_index is not None:
            edge_index = self.stair_climb_edge_index
            if self._stair_climb_edge_completed(edge_index):
                self.completed_stair_climb_edges.add(edge_index)
        self.jump_time = None
        self.jump_course_active = False
        self.stair_climb_active = False
        self.stair_edge_climb_active = False
        self.stair_climb_edge_index = None
        return self.command.leg_length, 0.0

    def _absolute_hip_angles(self, side: str) -> tuple[float, float]:
        # 由关节编码值反算物理曲柄的绝对角度
        front = self.reference["rear_crank"] - self._joint_position(f"{side}_front_hip")
        rear = self.reference["front_crank"] - self._joint_position(f"{side}_rear_hip")
        return front, rear

    def _measure_leg(self, side: str, dt: float) -> LegState:
        # 利用正运动学得到当前腿的几何长度与朝向，并做一阶低通导数平滑
        front, rear = self._absolute_hip_angles(side)
        point = self.kinematics.forward(front, rear)
        length = float(np.linalg.norm(point))
        angle = math.atan2(point[0], -point[1])
        previous = self.last_legs[side]
        alpha = dt / (self.LEG_RATE_FILTER_TAU + dt)
        raw_length_rate = (length - previous.length) / dt
        angle_delta = math.atan2(
            math.sin(angle - previous.angle), math.cos(angle - previous.angle)
        )
        raw_angle_rate = angle_delta / dt
        return LegState(
            length=length,
            angle=angle,
            length_rate=previous.length_rate
            + alpha * (raw_length_rate - previous.length_rate),
            angle_rate=previous.angle_rate
            + alpha * (raw_angle_rate - previous.angle_rate),
        )

    def _vmc_torques(self, side: str, axial_force: float, hip_torque: float) -> np.ndarray:
        # VMC力映射：先从笛卡尔力再转到两关节力矩空间
        front, rear = self._absolute_hip_angles(side)
        point = self.kinematics.forward(front, rear)
        length = max(float(np.linalg.norm(point)), 1e-4)
        radial = point / length
        tangent = np.array([-radial[1], radial[0]])
        endpoint_force = axial_force * radial + (hip_torque / length) * tangent

        jacobian = self.kinematics.jacobian(front, rear)
        return jacobian.T @ endpoint_force

    def update(self, now: float, force: bool = False) -> None:
        # 主控制流程：状态采样 -> 接地/空中判定 -> 目标滤波 -> 力控闭环 -> 输出执行
        if not force and now - self.last_control_time < self.CONTROL_DT - 1e-9:
            return
        dt = max(now - self.last_control_time, self.CONTROL_DT)
        self.last_control_time = now

        pitch = self._joint_position("root_pitch")
        pitch_rate = self._joint_velocity("root_pitch")
        roll = self._joint_position("root_roll")
        roll_rate = self._joint_velocity("root_roll")
        yaw = self._joint_position("root_yaw")
        yaw_rate = self._joint_velocity("root_yaw")
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.robot_body_id,
            self.robot_velocity,
            1,
        )
        forward_velocity = float(self.robot_velocity[3])
        was_airborne = self.is_airborne
        wheels_grounded = self._wheels_grounded()
        # 将离地抖动去抖（连续0.03s才当作真正离地）
        if wheels_grounded or self.chassis_on_support:
            self.airborne_time = 0.0
        else:
            self.airborne_time += dt
        self.is_airborne = self.airborne_time >= self.AIRBORNE_DEBOUNCE
        if self.jump_time is not None and self.is_airborne:
            self.jump_landing_recovery_pending = True
        if (
            was_airborne
            and not self.is_airborne
            and self.jump_landing_recovery_pending
        ):
            self.jump_landing_recovery_pending = False
            self.jump_motion_hold_until = max(
                self.jump_motion_hold_until, now + self.JUMP_LANDING_RECOVERY
            )

        self.command.leg_length = float(
            np.clip(self.command.leg_length, self.MIN_LENGTH, self.MAX_LENGTH)
        )
        self.command.leg_swing = float(
            np.clip(self.command.leg_swing, -self.MAX_LEG_SWING, self.MAX_LEG_SWING)
        )
        holding_jump_motion = (
            now < self.jump_motion_hold_until
            or (self.is_airborne and self.jump_landing_recovery_pending)
        )
        yaw_command = 0.0 if holding_jump_motion else self.command.yaw
        speed_target = (
            0.0
            if holding_jump_motion
            else self.command.speed
            + self.TURN_DRIFT_COMPENSATION * abs(self.command.yaw)
        )
        # 如果速度很小，视为无方向，以避免误触发失速/切向逻辑
        command_direction = (
            math.copysign(1.0, speed_target)
            if abs(speed_target) > self.STALL_COMMAND_THRESHOLD
            else 0.0
        )
        if (
            command_direction == 0.0
            or self.jump_time is not None
            or (
                self.stall_blocked_direction != 0.0
                and command_direction != self.stall_blocked_direction
            )
        ):
            self.stall_blocked_direction = 0.0
        recovering_from_stall = now < self.stall_recovery_until
        if (
            recovering_from_stall
            and command_direction != 0.0
            and self.stall_recovery_speed * command_direction < 0.0
        ):
            self.stall_recovery_until = 0.0
            self.stall_recovery_speed = 0.0
            self.stall_blocked_direction = 0.0
            self.stall_timer = 0.0
            self.target_distance = self.forward_distance
            recovering_from_stall = False
        blocked_by_stall = (
            not recovering_from_stall
            and self.stall_blocked_direction != 0.0
            and command_direction == self.stall_blocked_direction
        )
        if blocked_by_stall:
            speed_target = 0.0
        stair_assist_enabled = self.STAIR_CLIMB_TRIGGER_MIN_SPEED < 1.0
        if not stair_assist_enabled:
            self.stair_edge_climb_started_at = None
            self.stair_edge_escape_until = 0.0
            self.stair_edge_escape_speed = 0.0
        stair_edge_escape_active = (
            stair_assist_enabled and now < self.stair_edge_escape_until
        )
        stair_edge_climb_active = (
            False
            if stair_edge_escape_active or not stair_assist_enabled
            else self._stair_edge_climb_candidate(speed_target, pitch, forward_velocity)
        )
        if stair_edge_climb_active:
            if self.stair_edge_climb_started_at is None:
                self.stair_edge_climb_started_at = now
            body_height = float(self.data.xpos[self.robot_body_id, 2])
            edge_crawl_unstable = (
                abs(pitch) >= self.STAIR_EDGE_ESCAPE_PITCH
                or abs(roll) >= self.STAIR_EDGE_ESCAPE_ROLL
            )
            edge_crawl_timed_out = (
                edge_crawl_unstable
                or (
                    now - self.stair_edge_climb_started_at
                    >= self.STAIR_EDGE_CRAWL_MAX_TIME
                    and abs(forward_velocity) < self.STALL_SPEED_THRESHOLD
                    and (abs(pitch) > 0.18 or body_height < 0.15)
                )
            )
            if edge_crawl_timed_out:
                escape_source = (
                    self.command.speed
                    if abs(self.command.speed) > self.SPEED_COMMAND_DEADBAND
                    else speed_target
                )
                escape_direction = -math.copysign(1.0, escape_source or 1.0)
                self.stair_edge_escape_speed = (
                    escape_direction * self.STAIR_EDGE_ESCAPE_SPEED
                )
                self.stair_edge_escape_until = now + self.STAIR_EDGE_ESCAPE_TIME
                self.stair_edge_climb_started_at = None
                self.jump_time = None
                self.jump_course_active = False
                self.stair_climb_active = False
                self.stair_climb_edge_index = None
                stair_edge_climb_active = False
                stair_edge_escape_active = True
        else:
            if (
                not self._near_any_climb_terrain(0.02)
                or abs(pitch) < self.STAIR_EDGE_CRAWL_PITCH
                or abs(forward_velocity) > self.STAIR_EDGE_CRAWL_SPEED
            ):
                self.stair_edge_climb_started_at = None
        if stair_edge_escape_active:
            self.stall_timer = 0.0
            self.stall_recovery_until = 0.0
            self.stall_recovery_speed = 0.0
            self.stall_blocked_direction = 0.0
            recovering_from_stall = False
            blocked_by_stall = False
            speed_target = self.stair_edge_escape_speed
        if stair_edge_climb_active:
            self.stall_timer = 0.0
            self.stall_recovery_until = 0.0
            self.stall_recovery_speed = 0.0
            self.stall_blocked_direction = 0.0
            recovering_from_stall = False
            blocked_by_stall = False
            stair_direction = math.copysign(
                1.0,
                self.command.speed
                if abs(self.command.speed) > self.SPEED_COMMAND_DEADBAND
                else speed_target,
            )
            speed_target = math.copysign(
                min(abs(self.command.speed), self.STAIR_CLIMB_CRAWL_SPEED),
                stair_direction,
            )
            self.stair_climb_active = True
        self.stair_edge_climb_active = stair_edge_climb_active
        if (
            not stair_edge_climb_active
            and not stair_edge_escape_active
            and self.jump_time is None
        ):
            self.stair_climb_active = False
        stalled_against_obstacle = (
            self.jump_time is None
            and not self.is_airborne
            and not recovering_from_stall
            and not blocked_by_stall
            and not stair_edge_climb_active
            and not stair_edge_escape_active
            and abs(speed_target) > self.STALL_COMMAND_THRESHOLD
            and abs(forward_velocity) < self.STALL_SPEED_THRESHOLD
            and abs(pitch) > self.STALL_PITCH_THRESHOLD
            and (
                self._near_jump_terrain_course()
                or self.wheels_on_jump_terrain
                or self.chassis_on_support
            )
        )
        if stalled_against_obstacle:
            self.stall_timer += dt
        else:
            self.stall_timer = 0.0
        if (
            self.stall_timer >= self.STALL_DETECT_TIME
            and now >= self.stall_recovery_until
        ):
            recovery_direction = self._stall_recovery_direction(speed_target)
            command_direction = math.copysign(1.0, speed_target)
            self.stall_recovery_speed = recovery_direction * self.STALL_RECOVERY_SPEED
            self.stall_recovery_until = now + self.STALL_RECOVERY_TIME
            self.stall_blocked_direction = (
                command_direction
                if recovery_direction != command_direction
                else 0.0
            )
            self.stall_timer = 0.0
            self.target_speed = self.stall_recovery_speed
            self.target_distance = self.forward_distance
            recovering_from_stall = True
            blocked_by_stall = False
        if recovering_from_stall:
            # 脱困状态下，临时接管速度目标，固定向后/回避方向推进
            speed_target = self.stall_recovery_speed
        # 不再基于距离自动起跳；台阶通过由手动Space或贴边爬升状态完成。
        if (
            self.stair_climb_active
            and abs(speed_target) > self.STAIR_CLIMB_SPEED_LIMIT
        ):
            speed_target = math.copysign(
                self.STAIR_CLIMB_SPEED_LIMIT, speed_target
            )
        rough_terrain_active = (
            self.jump_time is None
            and not recovering_from_stall
            and not blocked_by_stall
            and not stair_edge_climb_active
            and not stair_edge_escape_active
            and not self.wheels_on_jump_terrain
            and (
                self.wheels_on_rough_terrain
                or (
                    abs(speed_target) > self.SPEED_COMMAND_DEADBAND
                    and self._near_rough_terrain_course()
                )
            )
        )
        self.rough_terrain_active = rough_terrain_active
        self.rough_terrain_height_estimate = (
            self._rough_terrain_support_height() if rough_terrain_active else 0.0
        )
        if (
            rough_terrain_active
            and abs(speed_target) > self.ROUGH_TERRAIN_SPEED_LIMIT
        ):
            speed_target = math.copysign(
                self.ROUGH_TERRAIN_SPEED_LIMIT, speed_target
            )
        no_operator_motion = (
            abs(self.command.speed) <= self.SPEED_COMMAND_DEADBAND
            and abs(self.command.yaw) <= self.YAW_COMMAND_DEADBAND
        )
        previous_speed_command = self.last_speed_command
        speed_command_stopped = (
            abs(speed_target) <= self.SPEED_COMMAND_DEADBAND
            and abs(previous_speed_command) > self.SPEED_COMMAND_DEADBAND
        )
        speed_command_reversed = (
            abs(speed_target) > self.SPEED_REVERSAL_DEADBAND
            and abs(previous_speed_command) > self.SPEED_REVERSAL_DEADBAND
            and speed_target * previous_speed_command < 0.0
        )
        if abs(speed_target) <= self.SPEED_COMMAND_DEADBAND:
            acceleration_limit = self.STOP_ACCELERATION
        elif speed_command_reversed:
            acceleration_limit = self.REVERSAL_ACCELERATION
        else:
            acceleration_limit = self.MAX_ACCELERATION
        self.target_speed = self._move_towards(
            self.target_speed, speed_target, acceleration_limit * dt
        )
        previous_yaw_command = self.last_yaw_command
        yaw_command_stopped = (
            abs(yaw_command) <= self.YAW_COMMAND_DEADBAND
            and abs(previous_yaw_command) > self.YAW_COMMAND_DEADBAND
        )
        yaw_command_reversed = (
            abs(yaw_command) > self.YAW_COMMAND_DEADBAND
            and abs(previous_yaw_command) > self.YAW_COMMAND_DEADBAND
            and yaw_command * previous_yaw_command < 0.0
        )
        if yaw_command_stopped or yaw_command_reversed:
            self.yaw_rate_integral = 0.0
            self.target_yaw = yaw
            self.target_yaw_rate = 0.0
        self.target_yaw_rate = self._move_towards(
            self.target_yaw_rate, yaw_command, self.MAX_YAW_ACCELERATION * dt
        )
        self.forward_distance += forward_velocity * dt
        if speed_command_stopped or speed_command_reversed:
            # 用户停下或反向指令时，冻结行进基准，避免继续积分“往前跑”
            self.target_distance = self.forward_distance
        else:
            self.target_distance += self.target_speed * dt
            if (
                abs(speed_target) <= self.SPEED_COMMAND_DEADBAND
                and abs(self.target_speed) <= self.STOP_TARGET_SPEED
            ):
                self.target_distance = self._move_towards(
                    self.target_distance,
                    self.forward_distance,
                    self.STOP_POSITION_RELAX_RATE * dt,
                )
        if recovering_from_stall or blocked_by_stall or stair_edge_escape_active:
            self.target_distance = self.forward_distance
        self.data.qfrc_applied[self.root_x_dof_id] = 0.0
        self.data.qfrc_applied[self.root_y_dof_id] = 0.0
        self.data.qfrc_applied[self.root_z_dof_id] = 0.0
        if recovering_from_stall:
            self.data.qfrc_applied[self.root_x_dof_id] = math.copysign(
                self.STALL_RECOVERY_FORCE, self.stall_recovery_speed
            )
        elif stair_edge_climb_active and not self.is_airborne:
            # 贴边爬升时加入小幅虚拟支撑：上托机体、沿行进方向轻推，模拟轮子挂住台阶边缘。
            climb_direction = math.copysign(
                1.0,
                self.command.speed
                if abs(self.command.speed) > self.SPEED_COMMAND_DEADBAND
                else speed_target,
            )
            climb_push = climb_direction * self.STAIR_EDGE_CLIMB_PUSH_FORCE
            self.data.qfrc_applied[self.root_x_dof_id] = climb_push * math.cos(yaw)
            self.data.qfrc_applied[self.root_y_dof_id] = climb_push * math.sin(yaw)
            self.data.qfrc_applied[self.root_z_dof_id] = (
                self.STAIR_EDGE_CLIMB_LIFT_FORCE
            )
        self.last_speed_command = speed_target
        self.last_yaw_command = yaw_command
        stop_tip_candidate = (
            no_operator_motion
            and self.jump_time is None
            and not self.is_airborne
            and not recovering_from_stall
            and not blocked_by_stall
            and not stair_edge_climb_active
            and not stair_edge_escape_active
            and abs(self.target_speed) <= self.STOP_TARGET_SPEED
            and abs(self.target_yaw_rate) <= self.YAW_RATE_MODE_THRESHOLD
            and abs(pitch) >= self.STOP_TIP_PITCH_THRESHOLD
        )
        if stop_tip_candidate:
            self.stop_tip_timer += dt
        elif (
            not no_operator_motion
            or self.jump_time is not None
            or abs(pitch) <= self.STOP_TIP_RELEASE_PITCH
        ):
            self.stop_tip_timer = 0.0
            self.stop_tip_recovery_active = False
        if self.stop_tip_timer >= self.STOP_TIP_DETECT_TIME:
            self.stop_tip_recovery_active = True
        if self.stop_tip_recovery_active:
            self.target_speed = 0.0
            self.target_yaw_rate = 0.0
            self.target_distance = self.forward_distance
            self.target_yaw = yaw
        stall_pose_recovery_active = (
            not self.is_airborne
            and self.jump_time is None
            and not stair_edge_climb_active
            and not stair_edge_escape_active
            and abs(pitch) >= self.STALL_POSE_PITCH_THRESHOLD
            and (
                blocked_by_stall
                or (
                    self._near_jump_terrain_course()
                    and abs(forward_velocity) < self.STALL_SPEED_THRESHOLD
                    and abs(speed_target) > self.STALL_COMMAND_THRESHOLD
                    and (
                        abs(yaw_command) > self.YAW_COMMAND_DEADBAND
                        or (
                            speed_target < -self.STALL_COMMAND_THRESHOLD
                            and abs(yaw_command) > self.YAW_COMMAND_DEADBAND
                        )
                    )
                )
                or (
                    abs(self.target_speed) <= self.STALL_POSE_LOW_SPEED_TARGET
                    and (
                        abs(yaw_command) > self.YAW_COMMAND_DEADBAND
                        or abs(self.target_yaw_rate) > self.YAW_RATE_MODE_THRESHOLD
                        or speed_target > self.STALL_COMMAND_THRESHOLD
                    )
                )
            )
        )
        stall_reverse_escape_active = (
            not self.is_airborne
            and self.jump_time is None
            and not stall_pose_recovery_active
            and self._near_jump_terrain_course()
            and speed_target < -self.STALL_COMMAND_THRESHOLD
            and abs(pitch) >= self.STALL_POSE_PITCH_THRESHOLD
        )
        self.stall_pose_recovery_active = stall_pose_recovery_active
        if stall_pose_recovery_active:
            self.target_yaw_rate = 0.0
            self.target_yaw = yaw
            self.yaw_rate_integral = 0.0
            self.target_distance = self.forward_distance
        if stair_edge_climb_active:
            edge_yaw = self._stair_edge_alignment_yaw(speed_target)
            edge_yaw_error = math.atan2(
                math.sin(edge_yaw - yaw), math.cos(edge_yaw - yaw)
            )
            self.target_yaw_rate = 0.0
            self.target_yaw = yaw + float(
                np.clip(
                    edge_yaw_error,
                    -self.STAIR_EDGE_YAW_ALIGN_LIMIT,
                    self.STAIR_EDGE_YAW_ALIGN_LIMIT,
                )
            )
            self.yaw_rate_integral = 0.0
            self.target_distance = self.forward_distance
        if stair_edge_escape_active:
            self.target_yaw_rate = 0.0
            self.target_yaw = yaw
            self.yaw_rate_integral = 0.0
            self.target_distance = self.forward_distance
        x_error = float(
            np.clip(
                self.forward_distance - self.target_distance,
                -self.MAX_POSITION_ERROR,
                self.MAX_POSITION_ERROR,
            )
        )
        self.target_distance = self.forward_distance - x_error
        if self.is_airborne:
            x_error = 0.0
            self.target_distance = self.forward_distance

        if abs(self.target_yaw_rate) > self.YAW_RATE_MODE_THRESHOLD:
            self.target_yaw = yaw
            raw_yaw_error = 0.0
        else:
            raw_yaw_error = math.atan2(
                math.sin(self.target_yaw - yaw), math.cos(self.target_yaw - yaw)
            )
        yaw_error = float(
            np.clip(raw_yaw_error, -self.MAX_YAW_ERROR, self.MAX_YAW_ERROR)
        )
        self.target_yaw = yaw + yaw_error

        jumping = self.jump_time is not None
        target_length, jump_force = self._jump_profile(now)
        if jumping and jump_force > 0.0:
            if not self.jump_course_active:
                driving_on_terrain = (
                    self.wheels_on_terrain
                    and not self.wheels_on_jump_terrain
                    and abs(self.command.speed) > self.SPEED_COMMAND_DEADBAND
                )
                if driving_on_terrain:
                    jump_force *= self.TERRAIN_DRIVE_JUMP_FORCE_SCALE
                    target_length = min(target_length, self.command.leg_length)
                else:
                    speed_span = (
                        self.MOVING_JUMP_SPEED_FULL - self.MOVING_JUMP_SPEED_START
                    )
                    moving_jump_blend = float(
                        np.clip(
                            (abs(forward_velocity) - self.MOVING_JUMP_SPEED_START)
                            / speed_span,
                            0.0,
                            1.0,
                        )
                    )
                    force_scale = 1.0 - (
                        1.0 - self.MOVING_JUMP_MIN_FORCE_SCALE
                    ) * moving_jump_blend
                    max_moving_length = self.MAX_LENGTH - (
                        self.MAX_LENGTH - self.MOVING_JUMP_MIN_EXTEND_LENGTH
                    ) * moving_jump_blend
                    jump_force *= force_scale
                    target_length = min(target_length, max_moving_length)
        if not jumping and (recovering_from_stall or blocked_by_stall):
            # 脱困/卡住时先收腿，减少与台阶前沿的机械卡位几率
            target_length = min(target_length, self.STALL_RECOVERY_LEG_LENGTH)
        if (
            rough_terrain_active
            and not jumping
            and not self.stop_tip_recovery_active
            and not stall_pose_recovery_active
            and not stair_edge_climb_active
        ):
            target_length = min(
                target_length,
                self._rough_terrain_leg_length_target(
                    self.rough_terrain_height_estimate
                ),
            )
        if stair_edge_climb_active:
            target_length = max(target_length, self.STAIR_EDGE_CRAWL_LEG_LENGTH)
        if stall_pose_recovery_active:
            target_length = max(target_length, self.STOP_TIP_RIGHTING_LEG_LENGTH)
        if self.stop_tip_recovery_active:
            target_length = max(target_length, self.STOP_TIP_RIGHTING_LEG_LENGTH)
        if jumping:
            self.target_leg_length = target_length
        else:
            leg_command_rate = (
                self.ROUGH_TERRAIN_LEG_COMMAND_RATE
                if rough_terrain_active or stair_edge_climb_active
                else self.LEG_COMMAND_RATE
            )
            self.target_leg_length = self._move_towards(
                self.target_leg_length, target_length, leg_command_rate * dt
            )
            target_length = self.target_leg_length
        low_length_blend = self._low_length_blend(target_length)
        wheel_scale = self._scheduled_scale(
            self.LOW_LENGTH_WHEEL_SCALE, low_length_blend
        )
        hip_scale = self._scheduled_scale(self.LOW_LENGTH_HIP_SCALE, low_length_blend)
        leg_kp_scale = self._scheduled_scale(
            self.LOW_LENGTH_LEG_KP_SCALE, low_length_blend
        )
        leg_kd_scale = self._scheduled_scale(
            self.LOW_LENGTH_LEG_KD_SCALE, low_length_blend
        )
        if rough_terrain_active:
            leg_kp_scale *= self.ROUGH_TERRAIN_LEG_KP_SCALE
            leg_kd_scale *= self.ROUGH_TERRAIN_LEG_KD_SCALE
        support = (
            0.0
            if self.is_airborne
            else self.SUPPORT_FEEDFORWARD * self.total_mass * 9.81 / 2.0
        )
        # Positive roll raises the right side, so support must move to the left.
        roll_force = np.clip(
            -(self.ROLL_KP * roll + self.ROLL_KD * roll_rate),
            -self.MAX_ROLL_FORCE,
            self.MAX_ROLL_FORCE,
        )
        yaw_rate_error = self.target_yaw_rate - yaw_rate
        if abs(self.target_yaw_rate) > self.YAW_RATE_MODE_THRESHOLD and not self.is_airborne:
            self.yaw_rate_integral = float(
                np.clip(
                    self.yaw_rate_integral + yaw_rate_error * dt,
                    -self.MAX_YAW_INTEGRAL,
                    self.MAX_YAW_INTEGRAL,
                )
            )
        else:
            self.yaw_rate_integral = self._move_towards(
                self.yaw_rate_integral, 0.0, 2.0 * dt
            )
        yaw_assist_torque = float(np.clip(
            self.YAW_KP * yaw_error
            + self.YAW_KD * yaw_rate_error
            + self.YAW_KI * self.yaw_rate_integral,
            -self.MAX_YAW_ASSIST_TORQUE,
            self.MAX_YAW_ASSIST_TORQUE,
        ))
        turn_torque = np.clip(
            0.35 * yaw_rate_error,
            -self.MAX_TURN_TORQUE,
            self.MAX_TURN_TORQUE,
        )
        if self.is_airborne:
            roll_force = 0.0
            turn_torque = 0.0
            yaw_assist_torque = 0.0
            jump_force = 0.0

        legs = {
            side: self._measure_leg(side, dt) for side in ("left", "right")
        }
        self.last_legs.update(legs)
        mean_angle = 0.5 * (legs["left"].angle + legs["right"].angle)
        mean_angle_rate = 0.5 * (
            legs["left"].angle_rate + legs["right"].angle_rate
        )
        state = np.array(
            [
                mean_angle - self.command.leg_swing,
                mean_angle_rate,
                x_error,
                forward_velocity - self.target_speed,
                pitch,
                pitch_rate,
            ]
        )
        wheel_torque, common_hip_torque = -(self.LQR_K @ state)
        common_hip_torque -= (
            self.BODY_PITCH_TP_KP * pitch
            + self.BODY_PITCH_TP_KD * pitch_rate
        )
        if rough_terrain_active:
            common_hip_torque -= (
                self.ROUGH_TERRAIN_PITCH_TP_KP * pitch
                + self.ROUGH_TERRAIN_PITCH_TP_KD * pitch_rate
            )
        if self.stair_climb_active:
            common_hip_torque -= (
                self.STAIR_CLIMB_PITCH_TP_KP * pitch
                + self.STAIR_CLIMB_PITCH_TP_KD * pitch_rate
            )
        if stair_edge_climb_active:
            common_hip_torque -= (
                self.STAIR_EDGE_CRAWL_PITCH_TP_KP * pitch
                + self.STAIR_EDGE_CRAWL_PITCH_TP_KD * pitch_rate
            )
        if self.stop_tip_recovery_active or stall_pose_recovery_active:
            common_hip_torque -= (
                self.STOP_TIP_RIGHTING_TP_KP * pitch
                + self.STOP_TIP_RIGHTING_TP_KD * pitch_rate
            )
        wheel_torque *= wheel_scale
        if (
            self.stair_climb_active
            and not self.is_airborne
            and abs(self.command.speed) > self.SPEED_COMMAND_DEADBAND
        ):
            wheel_torque += math.copysign(
                self.STAIR_CLIMB_WHEEL_FEEDFORWARD, self.command.speed
            )
        common_hip_torque *= hip_scale
        if self.is_airborne:
            wheel_torque = 0.0
        if stair_edge_climb_active and not self.is_airborne:
            wheel_torque = math.copysign(
                self.STAIR_EDGE_CRAWL_WHEEL_TORQUE, self.command.speed
            )
            turn_torque = float(
                np.clip(
                    turn_torque,
                    -self.STAIR_EDGE_TURN_TORQUE_LIMIT,
                    self.STAIR_EDGE_TURN_TORQUE_LIMIT,
                )
            )
            yaw_assist_torque = float(
                np.clip(
                    yaw_assist_torque,
                    -self.STAIR_EDGE_YAW_ASSIST_LIMIT,
                    self.STAIR_EDGE_YAW_ASSIST_LIMIT,
                )
            )
        if stair_edge_escape_active and not self.is_airborne:
            wheel_torque = math.copysign(
                self.STAIR_EDGE_CRAWL_WHEEL_TORQUE,
                self.stair_edge_escape_speed or -self.command.speed or -1.0,
            )
            turn_torque = 0.0
            yaw_assist_torque = 0.0
        if self.stop_tip_recovery_active or stall_pose_recovery_active:
            x_error = 0.0
            self.target_distance = self.forward_distance
            if self.stop_tip_recovery_active:
                wheel_torque = float(
                    np.clip(
                        wheel_torque,
                        -self.STOP_TIP_RIGHTING_WHEEL_TORQUE,
                        self.STOP_TIP_RIGHTING_WHEEL_TORQUE,
                    )
                )
            elif speed_target < -self.STALL_COMMAND_THRESHOLD:
                wheel_torque = -self.STALL_REVERSE_ESCAPE_WHEEL_TORQUE
            elif speed_target > self.STALL_COMMAND_THRESHOLD:
                wheel_torque = -self.STALL_FORWARD_ESCAPE_WHEEL_TORQUE
            elif (
                abs(yaw_command) > self.YAW_COMMAND_DEADBAND
                or abs(self.target_yaw_rate) > self.YAW_RATE_MODE_THRESHOLD
            ):
                wheel_torque = 0.0
            else:
                wheel_torque = float(
                    np.clip(
                        wheel_torque,
                        -self.STALL_POSE_WHEEL_TORQUE,
                        self.STALL_POSE_WHEEL_TORQUE,
                    )
                )
            turn_torque = 0.0
            yaw_assist_torque = 0.0
        elif stall_reverse_escape_active:
            wheel_torque = -self.STALL_REVERSE_ESCAPE_WHEEL_TORQUE
            turn_torque = 0.0
            yaw_assist_torque = 0.0

        side_outputs: dict[str, np.ndarray] = {}
        for side, roll_sign in (("left", -1.0), ("right", 1.0)):
            leg = legs[side]
            hip_torque = common_hip_torque - (
                self.LEG_SYNC_KP * (leg.angle - mean_angle)
                + self.LEG_SYNC_KD * (leg.angle_rate - mean_angle_rate)
            )
            axial_force = float(
                np.clip(
                    support
                    + jump_force
                    + (
                        self.STOP_TIP_RIGHTING_FORCE
                        if self.stop_tip_recovery_active or stall_pose_recovery_active
                        else 0.0
                    )
                    + self.LEG_FORCE_KP
                    * leg_kp_scale
                    * (target_length - leg.length)
                    - (
                        self.LEG_FORCE_KD
                        * leg_kd_scale
                        + self.TURN_LEG_DAMPING * abs(self.target_yaw_rate)
                    )
                    * leg.length_rate
                    + roll_sign * roll_force,
                    -self.MAX_AXIAL_FORCE,
                    self.MAX_AXIAL_FORCE,
                )
            )
            hip_motor_torques = np.clip(
                self._vmc_torques(side, axial_force, hip_torque),
                -self.MAX_HIP_TORQUE,
                self.MAX_HIP_TORQUE,
            )
            side_outputs[side] = hip_motor_torques

        max_wheel_torque = self.MAX_WHEEL_TORQUE * wheel_scale
        climb_terrain_torque_guard = (
            self.stair_climb_active
            or stair_edge_climb_active
            or stair_edge_escape_active
            or (
                stair_assist_enabled
                and not recovering_from_stall
                and not stall_reverse_escape_active
                and (
                    self.wheels_on_climb_terrain
                    or self.chassis_on_climb_terrain
                    or self._near_any_climb_terrain(self.STAIR_CONTACT_NEAR_MARGIN)
                )
            )
        )
        if climb_terrain_torque_guard:
            max_wheel_torque = min(
                max_wheel_torque, self.STAIR_CLIMB_MAX_WHEEL_TORQUE
            )
        wheel_torque = float(
            np.clip(wheel_torque, -max_wheel_torque, max_wheel_torque)
        )
        turn_headroom = max(0.0, max_wheel_torque - abs(wheel_torque))
        turn_torque = float(np.clip(turn_torque, -turn_headroom, turn_headroom))
        wheel_torque_rate = self.WHEEL_TORQUE_RATE * wheel_scale

        for side, roll_sign in (("left", -1.0), ("right", 1.0)):
            hip_motor_torques = side_outputs[side]
            side_wheel_torque = float(
                np.clip(
                    wheel_torque - roll_sign * turn_torque,
                    -max_wheel_torque,
                    max_wheel_torque,
                )
            )
            prefix = "Left" if side == "left" else "Right"
            self._set_actuator_torque(
                f"{prefix}_front_joint_actuator",
                -hip_motor_torques[0],
                self.HIP_TORQUE_RATE,
                dt,
            )
            self._set_actuator_torque(
                f"{prefix}_rear_joint_actuator",
                -hip_motor_torques[1],
                self.HIP_TORQUE_RATE,
                dt,
            )
            wheel_name = f"{prefix}_Wheel_joint_actuator"
            if self.is_airborne:
                self.last_actuator_commands[wheel_name] = 0.0
                self.data.ctrl[self.actuator_ids[wheel_name]] = 0.0
            elif self.stop_tip_recovery_active or stall_pose_recovery_active:
                self.last_actuator_commands[wheel_name] = side_wheel_torque
                self.data.ctrl[self.actuator_ids[wheel_name]] = side_wheel_torque
            else:
                self._set_actuator_torque(
                    wheel_name,
                    side_wheel_torque,
                    wheel_torque_rate,
                    dt,
                )
        if self.is_airborne or self.stop_tip_recovery_active or stall_pose_recovery_active:
            self.last_actuator_commands["Yaw_assist_actuator"] = 0.0
            self.data.ctrl[self.actuator_ids["Yaw_assist_actuator"]] = 0.0
        else:
            self._set_actuator_torque(
                "Yaw_assist_actuator",
                yaw_assist_torque,
                self.YAW_ASSIST_TORQUE_RATE,
                dt,
            )
