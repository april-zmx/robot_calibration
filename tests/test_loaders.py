import numpy as np

from robot_calibration.data.loaders import load_csv, load_ur_csv


def test_load_csv_with_generic_joint_columns(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "time,q1,q2,qd1,qd2,tau1,tau2\n"
        "0.0,1.0,2.0,0.1,0.2,3.0,4.0\n"
        "0.1,1.1,2.1,0.2,0.3,3.1,4.1\n",
        encoding="utf-8",
    )

    dataset = load_csv(
        csv_path,
        position_columns=["q1", "q2"],
        velocity_columns=["qd1", "qd2"],
        torque_columns=["tau1", "tau2"],
    )

    np.testing.assert_allclose(dataset.time, [0.0, 0.1])
    np.testing.assert_allclose(dataset.positions, [[1.0, 2.0], [1.1, 2.1]])
    np.testing.assert_allclose(dataset.velocities, [[0.1, 0.2], [0.2, 0.3]])
    np.testing.assert_allclose(dataset.torques, [[3.0, 4.0], [3.1, 4.1]])


def test_load_ur_csv_parses_reference_column_layout(tmp_path):
    csv_path = tmp_path / "ur.csv"
    rows = []
    for sample in range(3):
        time = 10.0 + 0.1 * sample
        q = np.arange(1, 7, dtype=float) + sample
        qd = 10.0 + q
        currents = 20.0 + q
        desired_currents = 30.0 + q
        desired_torques = 40.0 + q
        rows.append(np.concatenate([[time], q, qd, currents, desired_currents, desired_torques]))
    np.savetxt(csv_path, np.vstack(rows), delimiter=",")

    dataset = load_ur_csv(csv_path, start_index=1, stop_index=3)

    np.testing.assert_allclose(dataset.time, [0.0, 0.1])
    np.testing.assert_allclose(dataset.positions[0], [2, 3, 4, 5, 6, 7])
    np.testing.assert_allclose(dataset.velocities[0], [12, 13, 14, 15, 16, 17])
    np.testing.assert_allclose(dataset.currents[0], [22, 23, 24, 25, 26, 27])
    np.testing.assert_allclose(dataset.desired_currents[0], [32, 33, 34, 35, 36, 37])
    np.testing.assert_allclose(dataset.desired_torques[0], [42, 43, 44, 45, 46, 47])
    assert dataset.metadata["format"] == "ur"
