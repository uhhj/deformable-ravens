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


def test_gripper_surface_contacts_only_query_active_endpoint(monkeypatch):
    task = OHJCablePhase0()
    task.cable_bead_IDs = list(range(100, 132))
    task._layout = {"active_endpoint_index": 31}
    task._environment = types.SimpleNamespace(
        ee=types.SimpleNamespace(body=55))
    calls = []

    def fake_get_contact_points(**kwargs):
        calls.append(kwargs)
        return ()

    monkeypatch.setattr(
        "ravens.tasks.ccda_ohj_cable.p.getContactPoints",
        fake_get_contact_points,
    )
    task._gripper_surface_contacts()
    assert calls == [{
        "bodyA": 55,
        "bodyB": 131,
        "linkIndexA": 0,
    }]


def test_gripper_surface_tactile_uses_contact_manifold(monkeypatch):
    task = OHJCablePhase0()
    task._environment = types.SimpleNamespace(
        ee=types.SimpleNamespace(body=55))
    contact = [None] * 14
    contact[7] = [1.0, 0.0, 0.0]
    contact[9] = 2.0
    contact[10] = 0.5
    contact[11] = [0.0, 1.0, 0.0]
    contact[12] = -0.25
    contact[13] = [0.0, 0.0, 1.0]
    monkeypatch.setattr(
        task, "_gripper_surface_contacts", lambda: [tuple(contact)])
    monkeypatch.setattr(
        "ravens.tasks.ccda_ohj_cable.p.getLinkState",
        lambda *args, **kwargs: (
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ),
    )
    monkeypatch.setattr(
        "ravens.tasks.ccda_ohj_cable.p.getMatrixFromQuaternion",
        lambda quat: [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ],
    )
    force = task.gripper_surface_tactile_force()
    np.testing.assert_allclose(force, [2.0, 0.5, -0.25])


def test_combined_sensor_is_grasp_wrench_plus_surface_tactile(monkeypatch):
    task = OHJCablePhase0()
    monkeypatch.setattr(
        task, "formal_contact_sensor", lambda: np.arange(1.0, 7.0))
    monkeypatch.setattr(
        task, "gripper_surface_tactile_force",
        lambda: np.arange(7.0, 10.0))
    sensor = task.combined_contact_sensor()
    np.testing.assert_array_equal(sensor, np.arange(1.0, 10.0))
