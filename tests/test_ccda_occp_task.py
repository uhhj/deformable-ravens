import types

import pybullet as p
import pytest

from ravens.tasks.ccda_occp_audit import OCCPAuditCable
from ravens.tasks.ccda_occp_geometry import compute_occp_layout
from ravens.environment import Environment


@pytest.fixture
def bullet_world():
    connection = p.connect(p.DIRECT)
    try:
        yield
    finally:
        p.disconnect(connection)


def make_geometry(task):
    env = types.SimpleNamespace(objects=[])
    layout = compute_occp_layout(task.geometry_config, 'free')
    task.object_points = {}
    task._IDs = {}
    task.cable_bead_IDs = []
    task.cable_constraint_ids = []
    task._create_cable(env, layout)
    task._create_hidden_pin(layout)
    task._create_occluder(layout)
    task._layout = layout
    return env, layout


def test_pin_is_collision_enabled_and_invisible_in_both_conditions(bullet_world):
    task = OCCPAuditCable()
    make_geometry(task)
    assert p.getCollisionShapeData(task.pin_body_id, -1)
    assert all(v[7][3] == 0.0 for v in p.getVisualShapeData(task.pin_body_id))
    for condition in ('free', 'right_hidden_jam'):
        task.arm_condition(condition)
        assert p.getCollisionShapeData(task.pin_body_id, -1)


def test_occluder_has_no_collision(bullet_world):
    task = OCCPAuditCable()
    make_geometry(task)
    assert not p.getCollisionShapeData(task.occluder_body_id, -1)
    assert p.getVisualShapeData(task.occluder_body_id)


def test_passive_endpoint_is_fixed(bullet_world):
    task = OCCPAuditCable()
    env, layout = make_geometry(task)
    assert len(env.objects) == layout['bead_positions'].shape[0]
    assert p.getDynamicsInfo(task.cable_bead_IDs[0], -1)[0] == 0.0
    assert p.getDynamicsInfo(task.cable_bead_IDs[-1], -1)[0] > 0.0


def test_active_stabilizer_can_be_released(bullet_world):
    task = OCCPAuditCable()
    make_geometry(task)
    constraint = task.active_endpoint_stabilizer_id
    assert constraint is not None
    assert p.getConstraintInfo(constraint)
    task.release_active_endpoint_stabilizer()
    assert task.active_endpoint_stabilizer_id is None


def test_arm_does_not_advance_physics(bullet_world):
    task = OCCPAuditCable()
    make_geometry(task)
    before = task.physics_step_count()
    result = task.arm_condition('right_hidden_jam')
    assert task.physics_step_count() == before
    assert result['arm_max_bead_jump'] == 0.0


def test_occp_environment_reset_smoke():
    env = Environment(disp=False, hz=480, deterministic=True)
    try:
        task = OCCPAuditCable()
        task.configure_audit(
            pair_id='smoke', seed=1, trace_stride=2, settle_seconds=0.01)
        env.reset(task)
        assert len(task.cable_bead_IDs) == 28
        assert task.physics_step_count() == 0
        assert task.active_endpoint_stabilizer_id is not None
        assert env.ccda_sensor_observation()['joint_motor_torque']
    finally:
        env.stop()
