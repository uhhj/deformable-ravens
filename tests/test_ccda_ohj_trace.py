import types

import numpy as np

from ravens.tasks.ccda_ohj_cable import OHJCablePhase0


def test_trace_separates_formal_wrench_diagnostics_and_oracle(monkeypatch):
    task = OHJCablePhase0()
    task.cable_bead_IDs = [1, 2]
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
                        lambda: np.zeros((2, 3)))
    monkeypatch.setattr(task, "visible_keypoints",
                        lambda: np.zeros((16, 3)))
    monkeypatch.setattr(task, "statediff_state", lambda: np.zeros(51))
    monkeypatch.setattr(task, "formal_contact_sensor",
                        lambda: np.arange(1.0, 7.0))
    monkeypatch.setattr(task, "_oracle_latch_contact", lambda: (9.0, [1]))
    monkeypatch.setattr(
        "ravens.tasks.ccda_ohj_cable.p.getLinkState",
        lambda *args, **kwargs: ((0.5, 0.0, 0.1),))
    row = task._trace_sample()
    assert len(row["statediff_state"]) == 51
    assert row["formal_wrench"] == list(np.arange(1.0, 7.0))
    assert row["oracle_latch_contact_force"] == 9.0
    assert "oracle" not in "formal_wrench"
