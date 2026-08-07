import types

import numpy as np
import pybullet as p
import pytest

from ravens.tasks.ccda_ohj_cable import OHJCablePhase0
from ravens.tasks.ccda_ohj_geometry import compute_ohj_layout


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
