import types

import numpy as np

from ravens.tasks.ccda_ohj_cable import OHJCablePhase0
from ravens.tasks.ccda_ohj_geometry import VISIBLE_KEYPOINT_INDICES


def test_state_is_16_visible_keypoints_plus_ee(monkeypatch):
    task = OHJCablePhase0()
    task.cable_bead_IDs = list(range(32))
    task._layout = {"visible_keypoint_indices": np.asarray(
        VISIBLE_KEYPOINT_INDICES)}
    task._environment = types.SimpleNamespace(ur5=50, ee_tip_link=12)
    monkeypatch.setattr(
        task, "_bead_positions",
        lambda: np.arange(96, dtype=np.float64).reshape(32, 3))
    monkeypatch.setattr(
        "ravens.tasks.ccda_ohj_cable.p.getLinkState",
        lambda *args, **kwargs: ((1.0, 2.0, 3.0),))
    state = task.statediff_state()
    assert state.shape == (51,)
    np.testing.assert_array_equal(
        state[:48].reshape(16, 3),
        np.arange(96).reshape(32, 3)[list(VISIBLE_KEYPOINT_INDICES)])
    assert not any(value in state[:48] for value in np.arange(24, 72))


def test_formal_sensor_is_only_suction_force_and_torque():
    task = OHJCablePhase0()
    task._environment = types.SimpleNamespace(
        ccda_sensor_observation=lambda: {
            "suction_force_xyz": [1.0, 2.0, 3.0],
            "suction_torque_xyz": [4.0, 5.0, 6.0],
            "joint_motor_torque": [99.0] * 6,
            "joint_reaction_force_torque": [[88.0] * 6] * 6,
        })
    np.testing.assert_array_equal(
        task.formal_contact_sensor(), np.arange(1.0, 7.0))
