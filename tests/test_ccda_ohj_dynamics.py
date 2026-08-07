import types

import numpy as np
import pybullet as p

from ravens.tasks.ccda_ohj_cable import OHJCablePhase0
from ravens.tasks.ccda_ohj_geometry import compute_ohj_layout


def test_constraints_use_midpoint_anchors_and_stabilizer_releases():
    p.connect(p.DIRECT)
    task = OHJCablePhase0()
    task.object_points, task._IDs = {}, {}
    task.cable_bead_IDs, task.cable_constraint_ids = [], []
    layout = compute_ohj_layout(task.geometry_config, "free")
    task._create_cable(types.SimpleNamespace(objects=[]), layout)
    try:
        info = p.getConstraintInfo(task.cable_constraint_ids[0])
        parent = p.multiplyTransforms(
            *p.getBasePositionAndOrientation(info[0]), info[6],
            [0, 0, 0, 1])[0]
        midpoint = 0.5 * (layout["bead_positions"][0]
                          + layout["bead_positions"][1])
        assert np.allclose(parent, midpoint, atol=1e-7)
        assert task.active_endpoint_stabilizer_id is not None
        task.release_active_endpoint_stabilizer()
        assert task.active_endpoint_stabilizer_id is None
    finally:
        p.disconnect()
