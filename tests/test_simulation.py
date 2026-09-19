import glfw
import mujoco
import numpy as np

from controller import ParallelLegController
from Simulation import (
    CAMERA_LOOKAHEAD,
    FORWARD_SPEED,
    TURN_RATE,
    ControlCurveHistory,
    ControlCurveSample,
    KeyboardCommandState,
    build_curve_legend_bitmap,
    build_control_overlay,
    build_hud_overlay_bitmap,
    build_viewer_overlay,
    handle_key,
    mujoco_rgb_pixel_buffer,
    reset_simulation,
    step_physics,
    sync_render_data,
)


def make_controller():
    model = mujoco.MjModel.from_xml_path("MJCF/scene.xml")
    data = mujoco.MjData(model)
    controller = ParallelLegController(model, data)
    controller.initialize_pose()
    return data, controller


def test_direction_and_jump_keys():
    data, controller = make_controller()
    keyboard = KeyboardCommandState()
    assert keyboard.handle_event(controller, data.time, ord("w"), glfw.PRESS) == "forward"
    assert controller.command.speed == FORWARD_SPEED
    assert keyboard.held_key_summary() == "W"
    assert keyboard.last_event == "press W"
    assert keyboard.handle_event(controller, data.time, ord("a"), glfw.PRESS) == (
        "forward + turn left"
    )
    assert controller.command.yaw == TURN_RATE
    assert keyboard.held_key_summary() == "A+W"
    assert keyboard.handle_event(controller, data.time, ord("w"), glfw.RELEASE) == (
        "turn left"
    )
    assert controller.command.speed == 0.0
    assert keyboard.handle_event(controller, data.time, ord("a"), glfw.RELEASE) == "stop"
    assert controller.command.yaw == 0.0
    assert keyboard.held_key_summary() == "none"
    assert handle_key(controller, data.time, ord(" ")) == "jump"
    assert controller.jump_time == data.time
    assert keyboard.handle_event(controller, data.time, ord("s"), glfw.PRESS) == "reverse"
    assert keyboard.handle_event(controller, data.time, ord("x"), glfw.PRESS) == "stop"
    assert controller.command.speed == 0.0
    assert controller.command.yaw == 0.0


def test_graphical_and_headless_paths_use_the_same_fixed_step_control():
    data, controller = make_controller()
    start_time = data.time
    step_physics(controller.model, data, controller, steps=10)

    assert np.isclose(data.time, start_time + 10 * controller.model.opt.timestep)
    assert controller.last_control_time > start_time


def test_viewer_uses_an_independent_physics_snapshot():
    data, controller = make_controller()
    render_data = mujoco.MjData(controller.model)
    step_physics(controller.model, data, controller, steps=10)
    sync_render_data(controller.model, data, render_data)

    np.testing.assert_allclose(render_data.qpos, data.qpos)
    np.testing.assert_allclose(render_data.qvel, data.qvel)
    assert render_data.time == data.time


def test_default_viewer_camera_looks_ahead_to_terrain():
    assert CAMERA_LOOKAHEAD >= 1.5


def test_control_overlay_combines_control_and_body_state():
    data, controller = make_controller()
    robot_id = mujoco.mj_name2id(controller.model, mujoco.mjtObj.mjOBJ_BODY, "robot")
    overlay_left, overlay_right = build_control_overlay(
        controller,
        sim_time=data.time,
        pitch=controller._joint_position("root_pitch"),
        roll=controller._joint_position("root_roll"),
        yaw=controller._joint_position("root_yaw"),
        robot_xyz=data.xpos[robot_id],
        held_keys="A+W",
        last_key_event="press A",
    )

    assert "Control data (控制数据)" in overlay_left
    assert controller.REVISION in overlay_left
    assert "keys held(当前按键): A+W" in overlay_left
    assert "last key(最近按键): press A/按下 A" in overlay_left
    assert "cmd speed/yaw" in overlay_left
    assert "wheel torque L/R" in overlay_left
    assert "Body and leg state (机体与腿部状态)" in overlay_right
    assert "rpy" in overlay_right
    assert "left leg(左腿)" in overlay_right
    assert "contact(接触)" in overlay_right


def test_control_curve_history_builds_mujoco_figures():
    history = ControlCurveHistory(history_seconds=1.0, max_points=3)
    for index in range(5):
        value = float(index)
        history.append(
            ControlCurveSample(
                time=value * 0.4,
                command_speed=0.02 * value,
                target_speed=0.02 * value + 0.01,
                actual_speed=0.02 * value - 0.01,
                command_yaw_rate=0.12,
                target_yaw_rate=0.2,
                actual_yaw_rate=0.18,
                pitch=0.01 * value,
                roll=-0.01 * value,
                target_leg_swing=-0.18,
                mean_leg_angle=-0.2,
                left_leg_angle=-0.21,
                right_leg_angle=-0.19,
                target_leg_length=0.165,
                left_leg_length=0.160,
                right_leg_length=0.161,
                left_wheel_torque=1.0,
                right_wheel_torque=-1.0,
                left_hip_torque=2.0,
                right_hip_torque=-2.0,
                yaw_assist_torque=0.3,
            )
        )

    figures = history.build_figures()
    legends = history.curve_legends()

    assert len(figures) == 8
    assert len(legends[0]) == 3
    assert legends[0][0][0] == "cmd v(指令速度)=+0.08"
    assert legends[0][2][0] == "actual v(实际速度)=+0.07"
    assert [name for name, _color in legends[1]] == [
        "cmd yaw(指令偏航)=+0.120",
        "target yaw(目标偏航)=+0.200",
        "actual yaw(实际偏航)=+0.180",
    ]
    assert figures[0].title == "Forward speed v (m/s)"
    assert figures[0].flg_legend == 0
    assert figures[0].flg_ticklabel.tolist() == [0, 1]
    assert figures[0].xlabel == ""
    assert figures[0].linename[0] == b"cmd v"
    assert figures[0].linepnt[0] == 3
    assert figures[0].linedata[0, 0] < 0.0
    assert figures[0].linedata[0, 1] == 0.04
    assert figures[1].title == "Yaw rate omega (rad/s)"
    assert figures[2].title == "Body angles (rad)"
    assert figures[3].title == "Leg length L (m)"
    assert figures[4].title == "Leg swing theta (rad)"
    assert figures[5].title == "Wheel torque tau_w (Nm)"
    assert figures[6].title == "Hip torque tau_h (Nm)"
    assert figures[7].title == "Yaw assist torque (Nm)"
    assert figures[0].range[1][1] - figures[0].range[1][0] < 0.08


def test_curve_legend_bitmap_uses_all_line_items():
    bitmap = build_curve_legend_bitmap(
        (
            ("cmd v(指令速度)", (0.15, 0.65, 1.0)),
            ("target v(目标速度)", (1.0, 0.72, 0.18)),
            ("actual v(实际速度)", (0.55, 0.95, 0.42)),
        ),
        max_width=240,
    )
    if bitmap is None:
        return

    assert bitmap.ndim == 3
    assert bitmap.shape[2] == 3
    assert bitmap.shape[0] > 10
    assert bitmap.shape[1] > 120
    assert np.any(bitmap[:, :, 0] != bitmap[:, :, 1])


def test_bilingual_hud_overlay_can_be_rasterized_when_pillow_is_available():
    bitmap = build_hud_overlay_bitmap(
        "Control data (控制数据)\nkeys held(当前按键): W",
        "Body and leg state (机体与腿部状态)\ncontact(接触): grounded/接地",
        max_width=800,
    )
    if bitmap is None:
        return

    assert bitmap.ndim == 3
    assert bitmap.shape[2] == 3
    assert bitmap.shape[0] > 20
    assert bitmap.shape[1] > 100


def test_hud_bitmap_is_converted_to_mujoco_rgb_column_buffer():
    bitmap = np.zeros((4, 5, 3), dtype=np.uint8)
    buffer = mujoco_rgb_pixel_buffer(bitmap)

    assert buffer.shape == (4 * 5 * 3, 1)
    assert buffer.dtype == np.uint8
    assert buffer.flags.c_contiguous


def test_reset_simulation_restores_initial_pose_and_clears_keyboard_state():
    data, controller = make_controller()
    keyboard = KeyboardCommandState()
    render_data = mujoco.MjData(controller.model)
    root_x = mujoco.mj_name2id(controller.model, mujoco.mjtObj.mjOBJ_JOINT, "root_x")
    root_pitch = mujoco.mj_name2id(
        controller.model, mujoco.mjtObj.mjOBJ_JOINT, "root_pitch"
    )

    keyboard.handle_event(controller, data.time, ord("w"), glfw.PRESS)
    controller.start_jump(data.time)
    data.qpos[controller.model.jnt_qposadr[root_x]] = 4.0
    data.qpos[controller.model.jnt_qposadr[root_pitch]] = 1.0
    data.qvel[:] = 2.0
    controller.target_distance = 4.0
    controller.stall_pose_recovery_active = True
    mujoco.mj_forward(controller.model, data)

    reset_simulation(controller.model, data, controller, keyboard, render_data)

    assert data.time == 0.0
    assert controller._joint_position("root_x") == 0.0
    assert abs(controller._joint_position("root_pitch")) < 1e-9
    assert np.max(np.abs(data.qvel)) == 0.0
    assert abs(controller.target_distance) < 1e-6
    assert controller.jump_time is None
    assert not controller.stall_pose_recovery_active
    assert keyboard.held_key_summary() == "none"
    assert keyboard.last_event == "press R"
    np.testing.assert_allclose(render_data.qpos, data.qpos)
