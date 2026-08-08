import types

import pybullet as p

from ravens.tasks.ccda_dhr_cable import (
    DHRCableControlSmoke,
)
from ravens.tasks.ccda_dhr_geometry import (
    DHRGeometryConfig,
    compute_dhr_layout,
)


def test_dhr_jam_layout_has_no_initial_pocket_penetration():
    p.connect(p.DIRECT)

    try:
        task = DHRCableControlSmoke()

        task.geometry_config = (
            DHRGeometryConfig())

        task.object_points = {}
        task._IDs = {}
        task.cable_bead_IDs = []
        task.cable_constraint_ids = []

        layout = compute_dhr_layout(
            task.geometry_config,
            "jam_right")

        task._create_cable(
            types.SimpleNamespace(
                objects=[]),
            layout)

        task._create_hidden_pocket(
            layout)

        perform_collision_detection = getattr(
            p,
            "performCollisionDetection",
            None)

        if perform_collision_detection is not None:
            perform_collision_detection()

        for hidden_body in (
                task.pocket_bottom_body_id,
                task.pocket_left_body_id,
                task.pocket_right_body_id):
            penetrating = sum(
                len(
                    p.getClosestPoints(
                        hidden_body,
                        bead,
                        distance=0.0))
                for bead
                in task.cable_bead_IDs)

            assert penetrating == 0

    finally:
        p.disconnect()
