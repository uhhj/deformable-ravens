import types

import numpy as np

from ravens.tasks.ccda_ohj_cable import OHJCablePhase0


def test_trace_separates_formal_wrench_diagnostics_and_oracle(monkeypatch):
    task = OHJCablePhase0()
    task.cable_bead_IDs = list(range(100, 132))
    task._layout = {"pull_direction": np.array([1.0, 0.0, 0.0]),
                    "reference_passive_xyz": np.zeros(3)}
    observation = {
        "suction_force_xyz": [1.0, 2.0, 3.0],
        "suction_torque_xyz": [4.0, 5.0, 6.0],
        "joint_motor_torque": [7.0] * 6,
        "joint_reaction_force_torque": [[8.0] * 6] * 6,
    }
    task._environment = types.SimpleNamespace(
        ur5=10, ee_tip_link=12,
        ccda_sensor_observation=lambda: observation)
    monkeypatch.setattr(task, "_bead_positions",
                        lambda: np.zeros((32, 3)))
    monkeypatch.setattr(task, "visible_keypoints",
                        lambda: np.zeros((16, 3)))
    monkeypatch.setattr(task, "statediff_state", lambda: np.zeros(51))
    monkeypatch.setattr(task, "formal_contact_sensor",
                        lambda: np.arange(1.0, 7.0))
    monkeypatch.setattr(
        task,
        "gripper_surface_tactile_patches",
        lambda: (
            np.arange(7.0, 19.0).reshape(4, 3),
            np.asarray([1, 2, 3, 4], dtype=np.int64),
        ),
    )
    monkeypatch.setattr(task, "_oracle_latch_contact", lambda: (9.0, [15, 16]))
    monkeypatch.setattr(
        task,
        "oracle_internal_cable_constraint_force_xyz",
        lambda: np.zeros((31, 3), dtype=np.float64),
    )
    monkeypatch.setattr(
        "ravens.tasks.ccda_ohj_cable.p.getLinkState",
        lambda *args, **kwargs: ((0.5, 0.0, 0.1),))
    row = task._trace_sample()
    assert len(row["statediff_state"]) == 51
    assert row["formal_wrench"] == list(np.arange(1.0, 7.0))
    expected_aggregate = np.arange(7.0, 19.0).reshape(4, 3).sum(axis=0)
    np.testing.assert_allclose(
        row["gripper_surface_tactile_force"], expected_aggregate)
    assert row["gripper_surface_contact_count"] == 10
    np.testing.assert_allclose(
        row["formal_sensor"],
        np.concatenate([np.arange(1.0, 7.0), expected_aggregate]))
    np.testing.assert_allclose(
        row["gripper_surface_tactile_patch_force"],
        np.arange(7.0, 19.0).reshape(4, 3))
    assert row["gripper_surface_tactile_patch_contact_count"] == [1, 2, 3, 4]
    assert len(row["formal_sensor_spatial"]) == 18
    np.testing.assert_allclose(
        row["formal_sensor_spatial"][6:], np.arange(7.0, 19.0))
    assert row["oracle_latch_contact_force"] == 9.0
    assert len(row["oracle_latch_contact_bead_mask"]) == 32
    assert row["oracle_latch_contact_bead_mask"][15] == 1
    assert row["oracle_latch_contact_bead_mask"][16] == 1
    assert np.asarray(
        row["oracle_internal_cable_constraint_force_xyz"]).shape == (31, 3)
    assert len(row["formal_sensor_spatial"]) == 18
    assert "oracle" not in "formal_wrench"
