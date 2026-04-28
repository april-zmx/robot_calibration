import numpy as np

from robot_calibration.kinematics import DHParameter, calibrate_dh_offsets, forward_kinematics


def test_forward_kinematics_planar_2r_chain():
    params = [
        DHParameter(a=1.0, alpha=0.0, d=0.0, theta=0.0),
        DHParameter(a=0.5, alpha=0.0, d=0.0, theta=0.0),
    ]

    transform = forward_kinematics(params, np.array([np.pi / 2.0, 0.0]))

    np.testing.assert_allclose(transform[:3, 3], [0.0, 1.5, 0.0], atol=1e-12)


def test_calibrate_dh_offsets_recovers_link_lengths():
    true_params = [
        DHParameter(a=1.1, alpha=0.0, d=0.0, theta=0.0),
        DHParameter(a=0.7, alpha=0.0, d=0.0, theta=0.0),
    ]
    initial_params = [
        DHParameter(a=1.0, alpha=0.0, d=0.0, theta=0.0),
        DHParameter(a=0.6, alpha=0.0, d=0.0, theta=0.0),
    ]
    joint_positions = np.array(
        [
            [0.0, 0.0],
            [np.pi / 4.0, -np.pi / 6.0],
            [np.pi / 2.0, -np.pi / 4.0],
            [-np.pi / 3.0, np.pi / 5.0],
        ]
    )
    observations = np.array(
        [forward_kinematics(true_params, q)[:3, 3] for q in joint_positions]
    )

    result = calibrate_dh_offsets(
        initial_params,
        joint_positions,
        observations,
        estimate=("a",),
    )

    assert result.success
    np.testing.assert_allclose([p.a for p in result.parameters], [1.1, 0.7], atol=1e-6)
