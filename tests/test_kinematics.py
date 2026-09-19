import math

import numpy as np

from kinematics import FiveBarKinematics


def test_inverse_reaches_requested_virtual_leg_pose():
    kinematics = FiveBarKinematics()
    for length in np.linspace(0.08, 0.215, 8):
        for angle in np.linspace(-math.pi / 2.0 - 0.3, -math.pi / 2.0 + 0.3, 5):
            front, _, rear, _ = kinematics.inverse(float(length), float(angle))
            actual = kinematics.forward(front, rear)
            np.testing.assert_allclose(actual, kinematics.target_point(length, angle), atol=1e-8)


def test_inverse_keeps_l1_and_l4_outside_the_hip_pivots():
    kinematics = FiveBarKinematics()
    geometry = kinematics.g
    front, _, rear, _ = kinematics.inverse(0.165)

    front_elbow_x = -geometry.hip_spacing / 2.0 + geometry.front_crank * math.cos(front)
    rear_elbow_x = geometry.hip_spacing / 2.0 + geometry.rear_crank * math.cos(rear)

    assert front_elbow_x < -geometry.hip_spacing / 2.0
    assert rear_elbow_x > geometry.hip_spacing / 2.0


def test_analytic_jacobian_matches_finite_difference():
    kinematics = FiveBarKinematics()
    epsilon = 1e-7
    for length in np.linspace(0.105, 0.205, 6):
        front, _, rear, _ = kinematics.inverse(float(length))
        point = kinematics.forward(front, rear)
        finite_difference = np.column_stack(
            (
                (kinematics.forward(front + epsilon, rear) - point) / epsilon,
                (kinematics.forward(front, rear + epsilon) - point) / epsilon,
            )
        )
        np.testing.assert_allclose(
            kinematics.jacobian(front, rear), finite_difference, atol=1e-7
        )
