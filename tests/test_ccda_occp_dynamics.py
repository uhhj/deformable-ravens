import types

import numpy as np
import pybullet as p
import pytest

from ravens.tasks.ccda_occp_audit import OCCPAuditCable
from ravens.tasks.ccda_occp_geometry import compute_occp_layout


@pytest.fixture
def built_task():
    p.connect(p.DIRECT)
    task = OCCPAuditCable()
    task.object_points = {}
    task._IDs = {}
    task.cable_bead_IDs = []
    task.cable_constraint_ids = []
    layout = compute_occp_layout(task.geometry_config, 'free')
    task._create_cable(types.SimpleNamespace(objects=[]), layout)
    task._create_hidden_pin(layout)
    task._layout = layout
    try:
        yield task, layout
    finally:
        p.disconnect()


def test_adjacent_constraints_use_midpoint_anchors(built_task):
    task, layout = built_task
    info = p.getConstraintInfo(task.cable_constraint_ids[0])
    parent_world = p.multiplyTransforms(
        *p.getBasePositionAndOrientation(info[0]), info[6], [0, 0, 0, 1])[0]
    child_world = p.multiplyTransforms(
        *p.getBasePositionAndOrientation(info[2]), info[7], [0, 0, 0, 1])[0]
    midpoint = 0.5 * (layout['bead_positions'][0] + layout['bead_positions'][1])
    assert np.allclose(parent_world, midpoint, atol=1e-7)
    assert np.allclose(child_world, midpoint, atol=1e-7)


def test_adjacent_collision_is_disabled_without_global_group_change(monkeypatch):
    p.connect(p.DIRECT)
    calls = []
    original = p.setCollisionFilterPair

    def record(body_a, body_b, link_a, link_b, enableCollision):
        calls.append((body_a, body_b, link_a, link_b, enableCollision))
        return original(body_a, body_b, link_a, link_b, enableCollision)

    monkeypatch.setattr(p, 'setCollisionFilterPair', record)
    task = OCCPAuditCable()
    task.object_points, task._IDs = {}, {}
    task.cable_bead_IDs, task.cable_constraint_ids = [], []
    layout = compute_occp_layout(task.geometry_config, 'free')
    try:
        task._create_cable(types.SimpleNamespace(objects=[]), layout)
        expected = set(zip(task.cable_bead_IDs[:-1], task.cable_bead_IDs[1:]))
        observed = set((row[0], row[1]) for row in calls)
        assert observed == expected
        assert all(row[2:] == (-1, -1, 0) for row in calls)
        assert (task.cable_bead_IDs[0], task.cable_bead_IDs[2]) not in observed
    finally:
        p.disconnect()


def test_collision_margin_and_friction_are_applied(built_task):
    task, _ = built_task
    bead = p.getDynamicsInfo(task.cable_bead_IDs[1], -1)
    pin = p.getDynamicsInfo(task.pin_body_id, -1)
    assert bead[1] == pytest.approx(task.geometry_config.bead_lateral_friction)
    assert pin[1] == pytest.approx(task.geometry_config.pin_lateral_friction)
    assert bead[11] == pytest.approx(task.geometry_config.collision_margin_m)
    assert pin[11] == pytest.approx(task.geometry_config.collision_margin_m)
