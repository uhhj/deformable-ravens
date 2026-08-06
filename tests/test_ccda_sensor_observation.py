import types

from ravens.environment import Environment


def test_sensor_reads_joint_and_constraint(monkeypatch):
    env = Environment.__new__(Environment)
    env.ur5 = 2
    env.joints = [2, 3]
    env.ee = types.SimpleNamespace(activated=True, contact_constraint=19)
    monkeypatch.setattr(
        'ravens.environment.p.getJointState',
        lambda body, joint: (0., 0., (joint,) * 6, float(joint)))
    monkeypatch.setattr(
        'ravens.environment.p.getConstraintState',
        lambda constraint: (1, 2, 3, 4, 5, 6))
    value = env.ccda_sensor_observation()
    assert value['joint_motor_torque'] == [2., 3.]
    assert value['joint_reaction_force_torque'][0] == [2.] * 6
    assert value['suction_force_xyz'] == [1., 2., 3.]
    assert value['suction_torque_xyz'] == [4., 5., 6.]
    assert value['grasp_active'] == 1
    assert value['constraint_available'] == 1
    assert 'constraint_id' not in value


def test_sensor_handles_no_constraint(monkeypatch):
    env = Environment.__new__(Environment)
    env.ur5 = 2
    env.joints = [2]
    env.ee = types.SimpleNamespace(activated=False, contact_constraint=None)
    monkeypatch.setattr(
        'ravens.environment.p.getJointState',
        lambda *args: (0., 0., (0.,) * 6, 0.))
    value = env.ccda_sensor_observation()
    assert value['suction_force_norm'] == 0.
    assert value['constraint_available'] == 0
