from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from controller import ParallelLegController


ROOT = Path(__file__).resolve().parents[1]


def make_simulation():
    model = mujoco.MjModel.from_xml_path(str(ROOT / "MJCF" / "scene.xml"))
    data = mujoco.MjData(model)
    controller = ParallelLegController(model, data)
    controller.initialize_pose()
    return model, data, controller


def test_vmc_initializes_closed_loops():
    model, data, _ = make_simulation()
    for first, second in (
        ("left_wheel_center", "left_loop_end"),
        ("right_wheel_center", "right_loop_end"),
    ):
        first_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, first)
        second_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, second)
        error = data.site_xpos[first_id] - data.site_xpos[second_id]
        assert float(error @ error) < 3e-6


def test_step_link_meshes_follow_the_physical_four_bars():
    root = ET.parse(ROOT / "MJCF" / "scene.xml").getroot()

    def mesh_in_body(body_name):
        body = root.find(f".//body[@name='{body_name}']")
        return body.find("geom[@class='visual']").attrib["mesh"]

    assert mesh_in_body("left_front_drive") == "left_liangan3"
    assert mesh_in_body("left_front_rod") == "left_liangan1"
    assert mesh_in_body("left_rear_drive") == "left_liangan4"
    assert mesh_in_body("left_rear_rod") == "left_liangan2"
    assert mesh_in_body("right_front_drive") == "right_liangan4"
    assert mesh_in_body("right_front_rod") == "right_liangan2"
    assert mesh_in_body("right_rear_drive") == "right_liangan3"
    assert mesh_in_body("right_rear_rod") == "right_liangan1"


def test_leg_control_uses_only_torque_actuators():
    root = ET.parse(ROOT / "MJCF" / "scene.xml").getroot()
    actuators = list(root.find("actuator"))

    assert actuators
    assert all(actuator.tag == "motor" for actuator in actuators)


def test_detailed_mesh_toggle_keeps_floor_visible():
    root = ET.parse(ROOT / "MJCF" / "scene.xml").getroot()
    floor = root.find(".//geom[@name='floor']")
    fast_visual = root.find(".//default[@class='fast_visual']/geom")
    chassis_proxy = root.find(".//body[@name='robot']/geom[@type='box']")

    assert floor.attrib.get("group", "0") == "0"
    assert fast_visual.attrib["group"] == "2"
    assert chassis_proxy.attrib["group"] == "2"


def test_terrain_course_has_driveable_contact_surfaces():
    root = ET.parse(ROOT / "MJCF" / "scene.xml").getroot()
    expected = {
        "terrain_up_slope",
        "terrain_top_platform",
        "terrain_down_slope",
        "terrain_low_step_lead_in",
        "terrain_low_step",
        "terrain_mid_step_lead_in",
        "terrain_mid_step",
        "terrain_left_offset_step_lead_in",
        "terrain_left_offset_step",
        "terrain_right_offset_step_lead_in",
        "terrain_right_offset_step",
        "terrain_jump_level1",
        "terrain_jump_level2",
        "terrain_jump_landing",
    }
    terrain_geoms = {
        geom.attrib["name"]: geom
        for geom in root.findall(".//worldbody/geom")
        if geom.attrib.get("name", "").startswith("terrain_")
    }

    assert expected <= terrain_geoms.keys()
    for name in expected:
        geom = terrain_geoms[name]
        assert geom.attrib["type"] == "box"
        assert geom.attrib["class"] == "collision"
        assert geom.attrib["density"] == "0"
        assert geom.attrib.get("group", "0") == "0"

    assert terrain_geoms["terrain_up_slope"].attrib["euler"] == "0 -.12 0"
    assert terrain_geoms["terrain_down_slope"].attrib["euler"] == "0 .12 0"
    assert terrain_geoms["terrain_low_step_lead_in"].attrib["euler"] == "0 -.09 0"
    assert terrain_geoms["terrain_mid_step_lead_in"].attrib["euler"] == "0 -.13 0"
    assert not any(
        name.endswith("_lead_in")
        for name in terrain_geoms
        if name.startswith("terrain_jump_")
    )
    for name in (
        "terrain_jump_level1",
        "terrain_jump_level2",
        "terrain_jump_landing",
    ):
        assert "euler" not in terrain_geoms[name].attrib
    assert float(terrain_geoms["terrain_up_slope"].attrib["pos"].split()[0]) > 3.5


def test_robot_chassis_has_collision_shell_without_mass_change():
    root = ET.parse(ROOT / "MJCF" / "scene.xml").getroot()
    chassis_collision = root.find(".//geom[@name='chassis_collision']")

    assert chassis_collision is not None
    assert chassis_collision.attrib["type"] == "box"
    assert chassis_collision.attrib["class"] == "collision"
    assert chassis_collision.attrib["density"] == "0"

    _, _, controller = make_simulation()
    assert controller.total_mass == 8.0


def test_chassis_collision_keeps_tipped_robot_above_ground():
    model, data, controller = make_simulation()
    root_z = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_z")
    data.qpos[model.jnt_qposadr[root_z]] = 0.03
    controller._set_joint_position("root_pitch", 1.4)
    mujoco.mj_forward(model, data)
    chassis_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "chassis_collision"
    )
    min_body_height = 99.0
    chassis_contacts = 0
    for _ in range(3000):
        controller.update(data.time)
        mujoco.mj_step(model, data)
        min_body_height = min(
            min_body_height, float(data.xpos[controller.robot_body_id, 2])
        )
        for contact in data.contact:
            pair = {contact.geom1, contact.geom2}
            if chassis_id in pair and pair & controller.support_geom_ids:
                chassis_contacts += 1
                break

    assert chassis_contacts > 0
    assert min_body_height > 0.05
    assert not any(warning.number for warning in data.warning)


def test_controller_treats_terrain_as_ground_surfaces():
    model, _, controller = make_simulation()
    support_names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        for geom_id in controller.support_geom_ids
    }

    assert "floor" in support_names
    assert "terrain_up_slope" in support_names
    assert "terrain_top_platform" in support_names
    assert "terrain_down_slope" in support_names
    assert "terrain_low_step_lead_in" in support_names
    assert "terrain_jump_level1" in support_names
    assert "terrain_jump_level2" in support_names

    rough_names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        for geom_id in controller.rough_terrain_geom_ids
    }
    jump_names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        for geom_id in controller.jump_terrain_geom_ids
    }
    assert "terrain_low_step" in rough_names
    assert "terrain_jump_level1" not in rough_names
    assert "terrain_jump_level1" in jump_names


def test_front_motors_are_connected_to_front_linkage_roots():
    model, _, _ = make_simulation()
    for side in ("left", "right"):
        front_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_front_drive"
        )
        rear_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_rear_drive"
        )
        assert model.body_pos[front_id, 0] < model.body_pos[rear_id, 0]

        front_joint = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_front_hip"
        )
        rear_joint = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_rear_hip"
        )
        assert model.jnt_bodyid[front_joint] == front_id
        assert model.jnt_bodyid[rear_joint] == rear_id


def test_initialized_short_links_point_outward():
    model, data, _ = make_simulation()
    for side in ("left", "right"):
        front_hip = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_front_drive"
        )
        front_elbow = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_front_rod"
        )
        rear_hip = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_rear_drive"
        )
        rear_elbow = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_rear_rod"
        )

        assert data.xpos[front_elbow, 0] < data.xpos[front_hip, 0]
        assert data.xpos[rear_elbow, 0] > data.xpos[rear_hip, 0]


def test_leg_joints_have_hard_limits_but_wheels_remain_unlimited():
    model, _, _ = make_simulation()
    for side in ("left", "right"):
        for linkage in ("front", "rear"):
            for joint in ("hip", "knee"):
                joint_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{linkage}_{joint}"
                )
                assert model.jnt_limited[joint_id]

        wheel_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_wheel_joint"
        )
        assert not model.jnt_limited[wheel_id]


def test_leg_height_and_swing_commands_are_clamped():
    _, data, controller = make_simulation()
    controller.command.leg_length = 1.0
    controller.command.leg_swing = 1.0
    controller.update(data.time + controller.CONTROL_DT, force=True)
    assert controller.command.leg_length == controller.MAX_LENGTH
    assert controller.command.leg_swing == controller.MAX_LEG_SWING

    controller.adjust_length(-1.0)
    controller.adjust_leg_swing(-1.0)
    assert controller.command.leg_length == controller.MIN_LENGTH
    assert controller.command.leg_swing == -controller.MAX_LEG_SWING


def test_normal_leg_length_command_remains_compliant():
    _, data, controller = make_simulation()
    controller.command.leg_length = controller.MAX_LENGTH
    controller.update(data.time + controller.CONTROL_DT, force=True)

    assert controller.MIN_LENGTH < controller.target_leg_length < controller.MAX_LENGTH


def test_lqr_recovers_pitch_disturbance():
    model, data, controller = make_simulation()
    controller._set_joint_position("root_pitch", 0.12)
    mujoco.mj_forward(model, data)
    for _ in range(6000):
        controller.update(data.time)
        mujoco.mj_step(model, data)
    assert abs(controller._joint_position("root_pitch")) < 0.12
    assert not any(warning.number for warning in data.warning)


def test_zero_command_large_pitch_limits_wheels_instead_of_runaway():
    model, data, controller = make_simulation()
    root_z = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_z")

    max_wheel_after_recovery = 0.0
    active_steps = 0
    for _ in range(180):
        data.qpos[model.jnt_qposadr[root_z]] = 0.05
        controller._set_joint_position("root_pitch", 0.58)
        mujoco.mj_forward(model, data)
        controller.update(data.time, force=True)
        if controller.stop_tip_recovery_active:
            active_steps += 1
            max_wheel_after_recovery = max(
                max_wheel_after_recovery,
                abs(data.ctrl[controller.actuator_ids["Left_Wheel_joint_actuator"]]),
                abs(data.ctrl[controller.actuator_ids["Right_Wheel_joint_actuator"]]),
            )

    assert active_steps > 0
    assert max_wheel_after_recovery <= controller.STOP_TIP_RIGHTING_WHEEL_TORQUE
    assert abs(controller.forward_distance - controller.target_distance) < 1e-6
    assert controller.target_leg_length > controller.command.leg_length


def test_body_pitch_feedback_uses_virtual_leg_torque():
    nominal_model, nominal_data, nominal = make_simulation()

    model, data, controller = make_simulation()
    controller._set_joint_position("root_pitch", 0.12)
    mujoco.mj_forward(model, data)

    for _ in range(20):
        nominal.update(nominal_data.time)
        mujoco.mj_step(nominal_model, nominal_data)
        controller.update(data.time)
        mujoco.mj_step(model, data)

    front_name = "Left_front_joint_actuator"
    assert data.ctrl[controller.actuator_ids[front_name]] > (
        nominal_data.ctrl[nominal.actuator_ids[front_name]]
    )


def test_lqr_tracks_forward_speed_without_falling():
    model, data, controller = make_simulation()
    controller.command.speed = 0.25
    settled_velocities = []
    for _ in range(6000):
        controller.update(data.time)
        mujoco.mj_step(model, data)
        if data.time > 4.0:
            settled_velocities.append(float(controller.robot_velocity[3]))
    assert controller._joint_position("root_x") > 0.8
    assert abs(controller._joint_position("root_pitch")) < 0.2
    assert abs(controller._joint_position("root_roll")) < 0.2
    assert np.std(settled_velocities) < 0.02


def test_fast_forward_speed_remains_smooth():
    model, data, controller = make_simulation()
    controller.command.speed = 0.50
    settled_velocities = []
    wheel_commands = []
    max_pitch = 0.0
    for _ in range(7000):
        controller.update(data.time)
        mujoco.mj_step(model, data)
        max_pitch = max(max_pitch, abs(controller._joint_position("root_pitch")))
        if data.time > 5.0:
            settled_velocities.append(float(controller.robot_velocity[3]))
        wheel_commands.extend(
            [
                data.ctrl[controller.actuator_ids["Left_Wheel_joint_actuator"]],
                data.ctrl[controller.actuator_ids["Right_Wheel_joint_actuator"]],
            ]
        )

    assert np.mean(settled_velocities) > 0.48
    assert np.std(settled_velocities) < 0.01
    assert max_pitch < 0.04
    assert np.mean(np.abs(wheel_commands) > 3.9) == 0.0
    assert not any(warning.number for warning in data.warning)


def test_forward_start_damps_virtual_leg_swing():
    model, data, controller = make_simulation()
    controller.command.speed = 0.35
    mean_angles = []
    mean_rates = []
    pitches = []
    for _ in range(2000):
        controller.update(data.time)
        mujoco.mj_step(model, data)
        left = controller.last_legs["left"]
        right = controller.last_legs["right"]
        mean_angles.append(0.5 * (left.angle + right.angle))
        mean_rates.append(0.5 * (left.angle_rate + right.angle_rate))
        pitches.append(controller._joint_position("root_pitch"))

    assert max(map(abs, mean_angles)) < 0.16
    assert max(map(abs, mean_rates)) < 0.45
    assert max(map(abs, pitches)) < 0.06
    assert not any(warning.number for warning in data.warning)


def test_forward_command_pulses_do_not_whip_virtual_leg():
    model, data, controller = make_simulation()
    schedule = [
        (0.0, 0.35),
        (0.35, 0.0),
        (0.7, 0.35),
        (1.05, 0.0),
        (1.4, 0.35),
        (1.75, 0.0),
    ]
    next_command = 0
    mean_angles = []
    mean_rates = []
    pitches = []
    for _ in range(5000):
        while (
            next_command < len(schedule)
            and data.time >= schedule[next_command][0]
        ):
            controller.command.speed = schedule[next_command][1]
            next_command += 1
        controller.update(data.time)
        mujoco.mj_step(model, data)
        left = controller.last_legs["left"]
        right = controller.last_legs["right"]
        mean_angles.append(0.5 * (left.angle + right.angle))
        mean_rates.append(0.5 * (left.angle_rate + right.angle_rate))
        pitches.append(controller._joint_position("root_pitch"))

    assert max(map(abs, mean_angles)) < 0.07
    assert max(map(abs, mean_rates)) < 0.23
    assert max(map(abs, pitches)) < 0.04
    assert not any(warning.number for warning in data.warning)


def test_rapid_speed_reversals_settle_after_stop():
    model, data, controller = make_simulation()
    schedule = [
        (0.0, 0.35),
        (0.4, -0.35),
        (0.8, 0.35),
        (1.2, -0.35),
        (1.6, 0.35),
        (2.0, -0.35),
        (2.4, 0.0),
    ]
    next_command = 0
    for _ in range(8000):
        while (
            next_command < len(schedule)
            and data.time >= schedule[next_command][0]
        ):
            controller.command.speed = schedule[next_command][1]
            next_command += 1
        controller.update(data.time)
        mujoco.mj_step(model, data)

    controller.update(data.time, force=True)
    assert abs(controller.robot_velocity[3]) < 0.03
    assert abs(controller.forward_distance - controller.target_distance) < 0.01
    assert abs(controller._joint_position("root_pitch")) < 0.12
    assert not any(warning.number for warning in data.warning)


def test_combined_drive_and_turn_remains_level():
    model, data, controller = make_simulation()
    controller.command.speed = 0.25
    controller.command.yaw = 0.25
    max_pitch = 0.0
    max_roll = 0.0
    for _ in range(8000):
        controller.update(data.time)
        mujoco.mj_step(model, data)
        max_pitch = max(max_pitch, abs(controller._joint_position("root_pitch")))
        max_roll = max(max_roll, abs(controller._joint_position("root_roll")))

    assert max_pitch < 0.14
    assert max_roll < 0.05
    assert controller.forward_distance > 1.3
    assert controller._joint_position("root_yaw") > 0.25


def test_turning_keeps_virtual_legs_synced():
    model, data, controller = make_simulation()
    controller.command.yaw = 0.75
    leg_angle_deltas = []
    leg_rate_deltas = []
    for _ in range(5000):
        controller.update(data.time)
        mujoco.mj_step(model, data)
        left = controller.last_legs["left"]
        right = controller.last_legs["right"]
        leg_angle_deltas.append(left.angle - right.angle)
        leg_rate_deltas.append(left.angle_rate - right.angle_rate)

    assert np.max(np.abs(leg_angle_deltas)) < 0.04
    assert np.max(np.abs(leg_rate_deltas)) < 0.10
    assert abs(controller._joint_velocity("root_yaw")) > 0.70
    assert not any(warning.number for warning in data.warning)


def test_turn_in_place_limits_longitudinal_drift():
    model, data, controller = make_simulation()
    controller.command.yaw = 0.25
    for _ in range(8000):
        controller.update(data.time)
        mujoco.mj_step(model, data)

    assert abs(controller.forward_distance) < 0.25
    assert controller._joint_position("root_yaw") > 0.25


def test_yaw_command_zero_brakes_without_coasting():
    model, data, controller = make_simulation()
    controller.command.yaw = 0.75
    for _ in range(3000):
        controller.update(data.time)
        mujoco.mj_step(model, data)

    yaw_at_stop = controller._joint_position("root_yaw")
    controller.command.yaw = 0.0
    for _ in range(3000):
        controller.update(data.time)
        mujoco.mj_step(model, data)

    yaw_overshoot = controller._joint_position("root_yaw") - yaw_at_stop
    assert abs(yaw_overshoot) < 0.03
    assert abs(controller._joint_velocity("root_yaw")) < 0.01


def test_turning_wheel_torque_avoids_saturation_chatter():
    model, data, controller = make_simulation()
    controller.command.yaw = 0.25
    wheel_commands = []
    for _ in range(8000):
        controller.update(data.time)
        mujoco.mj_step(model, data)
        wheel_commands.append(
            [
                data.ctrl[controller.actuator_ids["Left_Wheel_joint_actuator"]],
                data.ctrl[controller.actuator_ids["Right_Wheel_joint_actuator"]],
            ]
        )

    commands = np.asarray(wheel_commands)
    assert np.mean(np.abs(commands) > 3.9) < 0.03
    assert np.max(np.abs(np.diff(commands, axis=0))) <= (
        controller.WHEEL_TORQUE_RATE * controller.CONTROL_DT + 1e-9
    )


def test_airborne_mode_releases_wheel_torque():
    model, data, controller = make_simulation()
    controller.start_jump(data.time)
    airborne_wheel_commands = []
    for _ in range(2000):
        controller.update(data.time)
        mujoco.mj_step(model, data)
        if controller.is_airborne:
            airborne_wheel_commands.extend(
                [
                    data.ctrl[controller.actuator_ids["Left_Wheel_joint_actuator"]],
                    data.ctrl[controller.actuator_ids["Right_Wheel_joint_actuator"]],
                ]
            )

    assert airborne_wheel_commands
    assert max(map(abs, airborne_wheel_commands)) < 1e-9


def test_forward_jump_lands_without_flipping():
    model, data, controller = make_simulation()
    controller.command.speed = 0.50
    jump_started = False
    max_pitch = 0.0
    min_height = 99.0
    airborne_steps = 0
    for _ in range(5000):
        if not jump_started and data.time >= 1.5:
            controller.start_jump(data.time)
            jump_started = True
        controller.update(data.time)
        mujoco.mj_step(model, data)
        max_pitch = max(max_pitch, abs(controller._joint_position("root_pitch")))
        min_height = min(min_height, float(data.xpos[controller.robot_body_id, 2]))
        airborne_steps += int(controller.is_airborne)

    assert airborne_steps > 0
    assert max_pitch < 0.12
    assert min_height > 0.10
    assert abs(controller._joint_position("root_roll")) < 0.05
    assert not any(warning.number for warning in data.warning)


def test_forward_jump_near_steps_uses_safe_landing_guard():
    model, data, controller = make_simulation()
    controller._set_joint_position("root_x", 7.25)
    mujoco.mj_forward(model, data)
    controller.forward_distance = 7.25
    controller.target_distance = 7.25
    controller.command.speed = 0.50
    terrain_step_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in (
            "terrain_low_step",
            "terrain_mid_step",
            "terrain_left_offset_step",
            "terrain_right_offset_step",
        )
    }
    jump_started = False
    terrain_contacts = 0
    max_pitch = 0.0
    min_height = 99.0
    for _ in range(4000):
        if not jump_started and data.time >= 0.8:
            controller.start_jump(data.time)
            jump_started = True
        controller.update(data.time)
        mujoco.mj_step(model, data)
        max_pitch = max(max_pitch, abs(controller._joint_position("root_pitch")))
        min_height = min(min_height, float(data.xpos[controller.robot_body_id, 2]))
        for contact in data.contact:
            pair = {contact.geom1, contact.geom2}
            if pair & terrain_step_ids and pair & controller.wheel_geom_ids:
                terrain_contacts += 1
                break

    assert terrain_contacts > 0
    assert max_pitch < 0.25
    assert min_height > 0.10
    assert abs(controller._joint_position("root_roll")) < 0.05
    assert not any(warning.number for warning in data.warning)


def test_step_lead_ins_are_driveable_without_flipping():
    model, data, controller = make_simulation()
    controller._set_joint_position("root_x", 7.25)
    mujoco.mj_forward(model, data)
    controller.forward_distance = 7.25
    controller.target_distance = 7.25
    controller.command.speed = 0.50
    max_pitch = 0.0
    max_roll = 0.0
    min_height = 99.0
    terrain_contacts = 0
    for _ in range(5000):
        controller.update(data.time)
        mujoco.mj_step(model, data)
        max_pitch = max(max_pitch, abs(controller._joint_position("root_pitch")))
        max_roll = max(max_roll, abs(controller._joint_position("root_roll")))
        min_height = min(min_height, float(data.xpos[controller.robot_body_id, 2]))
        for contact in data.contact:
            pair = {contact.geom1, contact.geom2}
            if (
                pair & controller.support_geom_ids
                and pair & controller.wheel_geom_ids
                and controller.floor_geom_id not in pair
            ):
                terrain_contacts += 1
                break

    assert terrain_contacts > 0
    assert controller._joint_position("root_x") > 9.0
    assert max_pitch < 0.08
    assert max_roll < 0.06
    assert min_height > 0.16
    assert not any(warning.number for warning in data.warning)


def test_rough_terrain_compliance_compresses_legs_on_small_steps():
    model, data, controller = make_simulation()
    controller._set_joint_position("root_x", 7.25)
    mujoco.mj_forward(model, data)
    controller.forward_distance = 7.25
    controller.target_distance = 7.25
    controller.command.speed = 0.50
    active_steps = 0
    max_height_estimate = 0.0
    min_target_leg_length = controller.command.leg_length

    for _ in range(3500):
        controller.update(data.time)
        mujoco.mj_step(model, data)
        active_steps += int(controller.rough_terrain_active)
        max_height_estimate = max(
            max_height_estimate, controller.rough_terrain_height_estimate
        )
        min_target_leg_length = min(
            min_target_leg_length, controller.target_leg_length
        )

    assert active_steps > 0
    assert max_height_estimate > 0.005
    assert min_target_leg_length < controller.command.leg_length - 0.006
    assert controller._joint_position("root_x") > 8.45
    assert not any(warning.number for warning in data.warning)


def test_two_stage_jump_terrain_can_be_mounted_without_flipping():
    model, data, controller = make_simulation()
    controller.STAIR_CLIMB_TRIGGER_MIN_SPEED = 999.0
    controller._set_joint_position("root_x", 10.0)
    mujoco.mj_forward(model, data)
    controller.forward_distance = 10.0
    controller.target_distance = 10.0
    controller.command.speed = 0.50
    next_edge = 0
    manual_jumps = 0
    terrain_level1 = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "terrain_jump_level1"
    )
    terrain_level2 = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "terrain_jump_level2"
    )
    terrain_landing = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "terrain_jump_landing"
    )
    level1_contacts = 0
    level2_contacts = 0
    landing_contacts = 0
    max_pitch = 0.0
    max_roll = 0.0
    min_height = 99.0
    for _ in range(12000):
        if next_edge < len(controller.jump_terrain_edges):
            front_edge, _ = controller.jump_terrain_edges[next_edge]
            ready_to_jump = (
                front_edge - controller._joint_position("root_x") <= 0.25
            )
        else:
            ready_to_jump = False
        if ready_to_jump and controller.jump_time is None:
            controller.start_jump(data.time)
            next_edge += 1
            manual_jumps += 1
        controller.update(data.time)
        mujoco.mj_step(model, data)
        max_pitch = max(max_pitch, abs(controller._joint_position("root_pitch")))
        max_roll = max(max_roll, abs(controller._joint_position("root_roll")))
        min_height = min(min_height, float(data.xpos[controller.robot_body_id, 2]))
        for contact in data.contact:
            pair = {contact.geom1, contact.geom2}
            if terrain_level1 in pair and pair & controller.wheel_geom_ids:
                level1_contacts += 1
            if terrain_level2 in pair and pair & controller.wheel_geom_ids:
                level2_contacts += 1
            if terrain_landing in pair and pair & controller.wheel_geom_ids:
                landing_contacts += 1

    assert manual_jumps >= 3
    assert level1_contacts > 0
    assert level2_contacts > 0
    assert landing_contacts > 0
    assert controller._joint_position("root_x") > 13.5
    assert max_pitch < 0.13
    assert max_roll < 0.05
    assert min_height > 0.10
    assert not any(warning.number for warning in data.warning)


def test_full_course_drive_auto_climbs_jump_terrain_from_w_command():
    model, data, controller = make_simulation()
    controller.command.speed = 0.50
    terrain_level1 = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "terrain_jump_level1"
    )
    terrain_level2 = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "terrain_jump_level2"
    )
    terrain_landing = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "terrain_jump_landing"
    )
    level1_contacts = 0
    level2_contacts = 0
    landing_contacts = 0
    climb_triggers = 0
    was_climbing = False
    max_pitch = 0.0
    max_roll = 0.0
    max_wheel_torque = 0.0
    min_height = 99.0
    for _ in range(36000):
        controller.update(data.time)
        if controller.stair_climb_active and not was_climbing:
            climb_triggers += 1
        was_climbing = controller.stair_climb_active
        mujoco.mj_step(model, data)
        max_pitch = max(max_pitch, abs(controller._joint_position("root_pitch")))
        max_roll = max(max_roll, abs(controller._joint_position("root_roll")))
        max_wheel_torque = max(
            max_wheel_torque,
            abs(controller.last_actuator_commands["Left_Wheel_joint_actuator"]),
            abs(controller.last_actuator_commands["Right_Wheel_joint_actuator"]),
        )
        min_height = min(min_height, float(data.xpos[controller.robot_body_id, 2]))
        for contact in data.contact:
            pair = {contact.geom1, contact.geom2}
            if terrain_level1 in pair and pair & controller.wheel_geom_ids:
                level1_contacts += 1
            if terrain_level2 in pair and pair & controller.wheel_geom_ids:
                level2_contacts += 1
            if terrain_landing in pair and pair & controller.wheel_geom_ids:
                landing_contacts += 1
        if controller._joint_position("root_x") > 13.6:
            break

    assert climb_triggers >= 3
    assert level1_contacts > 0
    assert level2_contacts > 0
    assert landing_contacts > 0
    assert controller._joint_position("root_x") > 13.5
    assert max_pitch < 0.12
    assert max_roll < 0.08
    assert max_wheel_torque <= controller.STAIR_CLIMB_MAX_WHEEL_TORQUE + 1e-9
    assert min_height > 0.12
    assert not any(warning.number for warning in data.warning)


def test_side_approach_can_mount_jump_terrain_without_runaway_wheels():
    model, data, controller = make_simulation()
    controller._set_joint_position("root_x", 11.35)
    controller._set_joint_position("root_y", -1.20)
    controller._set_joint_position("root_yaw", np.pi / 2.0)
    mujoco.mj_forward(model, data)
    controller.forward_distance = 0.0
    controller.target_distance = 0.0
    controller.target_yaw = np.pi / 2.0
    controller.command.speed = 0.42
    terrain_level1 = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "terrain_jump_level1"
    )
    level1_contacts = 0
    climb_triggers = 0
    was_climbing = False
    max_pitch = 0.0
    max_roll = 0.0
    max_wheel_torque = 0.0

    for _ in range(14000):
        controller.update(data.time, force=True)
        if controller.stair_climb_active and not was_climbing:
            climb_triggers += 1
        was_climbing = controller.stair_climb_active
        mujoco.mj_step(model, data)
        max_pitch = max(max_pitch, abs(controller._joint_position("root_pitch")))
        max_roll = max(max_roll, abs(controller._joint_position("root_roll")))
        max_wheel_torque = max(
            max_wheel_torque,
            abs(controller.last_actuator_commands["Left_Wheel_joint_actuator"]),
            abs(controller.last_actuator_commands["Right_Wheel_joint_actuator"]),
        )
        for contact in data.contact:
            pair = {contact.geom1, contact.geom2}
            if terrain_level1 in pair and pair & controller.wheel_geom_ids:
                level1_contacts += 1
        if (
            controller._joint_position("root_y") > -0.20
            and float(data.xpos[controller.robot_body_id, 2]) > 0.16
            and abs(controller._joint_position("root_pitch")) < 0.15
        ):
            break

    assert climb_triggers > 0
    assert level1_contacts > 0
    assert controller._joint_position("root_y") > -0.20
    assert float(data.xpos[controller.robot_body_id, 2]) > 0.16
    assert max_pitch < 0.12
    assert max_roll < 0.03
    assert max_wheel_torque <= controller.STAIR_CLIMB_MAX_WHEEL_TORQUE + 1e-9
    assert not any(warning.number for warning in data.warning)


def test_diagonal_jump_terrain_corner_contact_limits_wheels_and_backs_off():
    for label, start_y, yaw in (
        ("lower corner", -0.95, float(np.arctan2(0.95, 0.70))),
        ("upper corner", 0.95, float(np.arctan2(-0.95, 0.70))),
    ):
        model, data, controller = make_simulation()
        controller._set_joint_position("root_x", 10.55)
        controller._set_joint_position("root_y", start_y)
        controller._set_joint_position("root_yaw", yaw)
        mujoco.mj_forward(model, data)
        controller.forward_distance = 0.0
        controller.target_distance = 0.0
        controller.target_yaw = yaw
        controller.command.speed = 0.42
        climb_triggers = 0
        was_climbing = False
        escape_seen = False
        max_wheel_torque = 0.0

        for _ in range(12000):
            controller.update(data.time, force=True)
            if controller.stair_climb_active and not was_climbing:
                climb_triggers += 1
            was_climbing = controller.stair_climb_active
            escape_seen = escape_seen or data.time < controller.stair_edge_escape_until
            max_wheel_torque = max(
                max_wheel_torque,
                abs(controller.last_actuator_commands["Left_Wheel_joint_actuator"]),
                abs(controller.last_actuator_commands["Right_Wheel_joint_actuator"]),
            )
            mujoco.mj_step(model, data)

        assert climb_triggers > 0, label
        assert escape_seen, label
        assert (
            max_wheel_torque <= controller.STAIR_CLIMB_MAX_WHEEL_TORQUE + 1e-9
        ), label
        assert not any(warning.number for warning in data.warning), label


def test_two_stage_jump_terrain_auto_climbs_from_drive_command():
    model, data, controller = make_simulation()
    controller._set_joint_position("root_x", 10.0)
    mujoco.mj_forward(model, data)
    controller.forward_distance = 10.0
    controller.target_distance = 10.0
    controller.command.speed = 0.50
    terrain_level1 = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "terrain_jump_level1"
    )
    terrain_level2 = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "terrain_jump_level2"
    )
    terrain_landing = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "terrain_jump_landing"
    )
    level1_contacts = 0
    level2_contacts = 0
    landing_contacts = 0
    climb_triggers = 0
    was_climbing = False
    max_pitch = 0.0
    max_roll = 0.0
    min_height = 99.0
    for _ in range(14000):
        controller.update(data.time)
        if controller.stair_climb_active and not was_climbing:
            climb_triggers += 1
        was_climbing = controller.stair_climb_active
        mujoco.mj_step(model, data)
        max_pitch = max(max_pitch, abs(controller._joint_position("root_pitch")))
        max_roll = max(max_roll, abs(controller._joint_position("root_roll")))
        min_height = min(min_height, float(data.xpos[controller.robot_body_id, 2]))
        for contact in data.contact:
            pair = {contact.geom1, contact.geom2}
            if terrain_level1 in pair and pair & controller.wheel_geom_ids:
                level1_contacts += 1
            if terrain_level2 in pair and pair & controller.wheel_geom_ids:
                level2_contacts += 1
            if terrain_landing in pair and pair & controller.wheel_geom_ids:
                landing_contacts += 1

    assert climb_triggers >= 3
    assert level1_contacts > 0
    assert level2_contacts > 0
    assert landing_contacts > 0
    assert controller._joint_position("root_x") > 14.0
    assert max_pitch < 0.12
    assert max_roll < 0.06
    assert min_height > 0.12
    assert not any(warning.number for warning in data.warning)


def test_stall_recovery_allows_backing_away_from_jump_platform_edge():
    model, data, controller = make_simulation()
    controller.STAIR_CLIMB_TRIGGER_MIN_SPEED = 999.0
    controller._set_joint_position("root_x", 10.0)
    mujoco.mj_forward(model, data)
    controller.forward_distance = 10.0
    controller.target_distance = 10.0
    controller.command.speed = 0.50
    recovery_steps = 0
    min_x_after_reverse = 99.0
    max_pitch = 0.0
    min_target_leg_length = 99.0
    for _ in range(16000):
        if data.time >= 7.0:
            controller.command.speed = -0.50
            min_x_after_reverse = min(
                min_x_after_reverse, controller._joint_position("root_x")
            )
        controller.update(data.time)
        mujoco.mj_step(model, data)
        max_pitch = max(max_pitch, abs(controller._joint_position("root_pitch")))
        min_target_leg_length = min(
            min_target_leg_length, controller.target_leg_length
        )
        recovery_steps += int(data.time < controller.stall_recovery_until)

    assert recovery_steps > 0
    assert min_target_leg_length <= controller.STALL_RECOVERY_LEG_LENGTH + 0.005
    assert min_x_after_reverse < 9.5
    assert controller._joint_position("root_x") < 9.5
    assert max_pitch < 0.65
    assert not any(warning.number for warning in data.warning)


def test_jump_platform_stall_honors_reverse_command_and_limits_pose_torque():
    model, data, controller = make_simulation()
    controller._set_joint_position("root_x", 10.8)
    controller._set_joint_position("root_pitch", 0.48)
    mujoco.mj_forward(model, data)
    controller.forward_distance = 10.8
    controller.target_distance = 10.8
    controller.command.speed = -0.50
    controller.command.yaw = 0.75
    controller.stall_recovery_speed = controller.STALL_RECOVERY_SPEED
    controller.stall_recovery_until = data.time + 1.0

    controller.update(data.time, force=True)

    assert controller.stall_recovery_until == 0.0
    assert controller.stall_pose_recovery_active
    assert controller.target_speed < 0.0
    assert abs(controller.target_yaw_rate) < 1e-9
    assert abs(controller.forward_distance - controller.target_distance) < 1e-9
    assert controller.target_leg_length > controller.command.leg_length
    assert data.ctrl[controller.actuator_ids["Left_Wheel_joint_actuator"]] < 0.0
    assert data.ctrl[controller.actuator_ids["Right_Wheel_joint_actuator"]] < 0.0
    assert abs(data.ctrl[controller.actuator_ids["Left_Wheel_joint_actuator"]]) <= (
        controller.STALL_REVERSE_ESCAPE_WHEEL_TORQUE
    )
    assert abs(data.ctrl[controller.actuator_ids["Right_Wheel_joint_actuator"]]) <= (
        controller.STALL_REVERSE_ESCAPE_WHEEL_TORQUE
    )
    assert data.ctrl[controller.actuator_ids["Yaw_assist_actuator"]] == 0.0


def test_low_speed_large_pitch_with_turning_enters_pose_recovery():
    model, data, controller = make_simulation()
    controller._set_joint_position("root_pitch", 0.58)
    mujoco.mj_forward(model, data)
    controller.command.speed = 0.0
    controller.command.yaw = -0.75

    for _ in range(80):
        controller.update(data.time)
        mujoco.mj_step(model, data)

    assert controller.stall_pose_recovery_active
    assert abs(controller.target_yaw_rate) < 1e-9
    assert data.ctrl[controller.actuator_ids["Left_Wheel_joint_actuator"]] == 0.0
    assert data.ctrl[controller.actuator_ids["Right_Wheel_joint_actuator"]] == 0.0
    assert data.ctrl[controller.actuator_ids["Yaw_assist_actuator"]] == 0.0


def test_low_speed_large_pitch_forward_command_limits_runaway_wheels():
    model, data, controller = make_simulation()
    controller._set_joint_position("root_pitch", 0.38)
    mujoco.mj_forward(model, data)
    controller.command.speed = 0.50
    controller.target_speed = 0.03

    controller.update(data.time, force=True)

    assert controller.stall_pose_recovery_active
    assert data.ctrl[controller.actuator_ids["Left_Wheel_joint_actuator"]] < 0.0
    assert data.ctrl[controller.actuator_ids["Right_Wheel_joint_actuator"]] < 0.0
    assert abs(data.ctrl[controller.actuator_ids["Left_Wheel_joint_actuator"]]) <= (
        controller.STALL_FORWARD_ESCAPE_WHEEL_TORQUE
    )
    assert abs(data.ctrl[controller.actuator_ids["Right_Wheel_joint_actuator"]]) <= (
        controller.STALL_FORWARD_ESCAPE_WHEEL_TORQUE
    )
    assert abs(controller.forward_distance - controller.target_distance) < 1e-9


def test_jump_leaves_ground_and_lands_stably():
    model, data, controller = make_simulation()
    start_height = float(data.xpos[1, 2])
    controller.start_jump(data.time)
    heights = []
    contacts = []
    for _ in range(2000):
        controller.update(data.time)
        mujoco.mj_step(model, data)
        heights.append(float(data.xpos[1, 2]))
        contacts.append(data.ncon)
    assert max(heights) > start_height + 0.07
    assert min(contacts) == 0
    assert abs(controller._joint_position("root_pitch")) < 0.15
