import types

import numpy as np
import pybullet as p
import pytest

from ravens.tasks.ccda_ohj_cable import OHJCablePhase0
from ravens.tasks.ccda_ohj_geometry import (
    OHJGeometryConfig, compute_ohj_layout)


@pytest.fixture
def built_task():
    p.connect(p.DIRECT)
    task = OHJCablePhase0()
    task.object_points, task._IDs = {}, {}
    task.cable_bead_IDs, task.cable_constraint_ids = [], []
    layout = compute_ohj_layout(task.geometry_config, "free")
    task._create_cable(types.SimpleNamespace(objects=[]), layout)
    task._create_hidden_latch(layout)
    task._create_occluder(layout)
    task._layout = layout
    try:
        yield task
    finally:
        p.disconnect()


def test_all_beads_are_dynamic_and_occluder_is_visual_only(built_task):
    assert len(built_task.cable_bead_IDs) == 32
    assert all(p.getDynamicsInfo(body, -1)[0] > 0
               for body in built_task.cable_bead_IDs)
    assert not p.getCollisionShapeData(built_task.occluder_body_id, -1)
    assert p.getVisualShapeData(built_task.occluder_body_id)


def test_arm_condition_only_moves_hidden_latch(built_task):
    before = built_task._bead_positions().copy()
    free_latch = np.asarray(p.getBasePositionAndOrientation(
        built_task.latch_body_id)[0])
    result = built_task.arm_condition("jam_right")
    after = built_task._bead_positions()
    jam_latch = np.asarray(p.getBasePositionAndOrientation(
        built_task.latch_body_id)[0])
    assert np.array_equal(before, after)
    assert result["arm_max_bead_jump"] == 0.0
    assert not np.array_equal(free_latch, jam_latch)


def test_dual_post_guide_arms_without_initial_penetration_or_cable_jump():
    p.connect(p.DIRECT)
    try:
        task = OHJCablePhase0()
        task.geometry_config = OHJGeometryConfig(
            jam_surface_clearance_m=0.00025,
            latch_topology="dual_post_directional_guide")
        task.object_points = {}
        task._IDs = {}
        task.cable_bead_IDs = []
        task.cable_constraint_ids = []
        free_layout = compute_ohj_layout(task.geometry_config, "free")
        task._create_cable(types.SimpleNamespace(objects=[]), free_layout)
        task._create_hidden_latch(free_layout)
        task._create_occluder(free_layout)
        task._layout = free_layout
        assert task.directional_guide_body_id is not None
        before = task._bead_positions().copy()
        result = task.arm_condition("jam_right")
        after = task._bead_positions()
        np.testing.assert_array_equal(before, after)
        assert result["arm_max_bead_jump"] == 0.0
        assert result["latch_topology"] == "dual_post_directional_guide"
        jam_layout = compute_ohj_layout(task.geometry_config, "jam_right")
        guide_position = np.asarray(p.getBasePositionAndOrientation(
            task.directional_guide_body_id)[0], dtype=np.float64)
        np.testing.assert_allclose(
            guide_position, jam_layout["directional_guide_center"])
        guide_contacts = sum(
            len(p.getClosestPoints(
                task.directional_guide_body_id, bead, distance=0.0))
            for bead in task.cable_bead_IDs)
        assert guide_contacts == 0
    finally:
        p.disconnect()
