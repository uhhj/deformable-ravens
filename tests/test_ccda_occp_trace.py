import types

import numpy as np

from ravens.tasks.ccda_occp_audit import OCCPAuditCable


def test_trace_separates_sensor_and_oracle_fields(monkeypatch):
    task = OCCPAuditCable()
    task.cable_bead_IDs = [10]
    task.pin_body_id = 20
    task._ee_target_position = [0.4, 0., 0.1]
    task._layout = {'visible_mask': np.asarray([1], dtype=np.int64)}
    task._environment = types.SimpleNamespace(
        ur5=30,
        joints=[1],
        ee_tip_link=2,
        ccda_sensor_observation=lambda: {
            'joint_motor_torque': [1.],
            'joint_reaction_force_torque': [[2.] * 6],
            'suction_force_xyz': [3., 0., 0.],
            'suction_torque_xyz': [0., 4., 0.],
            'grasp_active': 1,
            'constraint_available': 1,
        })
    monkeypatch.setattr(
        'ravens.tasks.ccda_occp_audit.p.getBasePositionAndOrientation',
        lambda body: ((0.4, 0., 0.1), (0., 0., 0., 1.)))
    monkeypatch.setattr(
        'ravens.tasks.ccda_occp_audit.p.getBaseVelocity',
        lambda body: ((0., 0., 0.), (0., 0., 0.)))
    monkeypatch.setattr(
        'ravens.tasks.ccda_occp_audit.p.getJointState',
        lambda body, joint: (0.1, 0.2, (0.,) * 6, 1.))
    monkeypatch.setattr(
        'ravens.tasks.ccda_occp_audit.p.getLinkState',
        lambda *args, **kwargs: (
            (0.4, 0., 0.1), (0., 0., 0., 1.), None, None, None, None,
            (0., 0., 0.), (0., 0., 0.)))
    monkeypatch.setattr(
        'ravens.tasks.ccda_occp_audit.p.getContactPoints',
        lambda *args: [tuple([0.] * 9 + [5., 1., 0., 2.])])
    monkeypatch.setattr(
        'ravens.tasks.ccda_occp_audit.p.getClosestPoints',
        lambda *args, **kwargs: [tuple([0.] * 8 + [0.01])])

    row = task._trace_sample()
    assert row['sensor_suction_force_xyz'] == [3., 0., 0.]
    assert row['oracle_pin_contact_force'] > 0
    assert not any('pin' in key for key in row if key.startswith('sensor_'))
    assert all(key.startswith('oracle_') for key in row if 'pin_contact' in key)
    assert row['visible_mask'] == [1]
    assert row['oracle_pin_min_signed_distance'] == 0.01
    assert not any('distance' in key for key in row if key.startswith('sensor_'))
