import json
import os

import numpy as np

from ravens.tasks import (
    ccda_hidden_routing_gate_cable
    as routing_task,
)
from ravens.tasks.ccda_hidden_routing_gate_cable import (
    FROZEN_BRANCH_ARM_LAYOUT_MODE,
    FROZEN_PUBLIC_LAYOUT_ENV,
    PUBLIC_LAYOUT_MODE_ENV,
    CCDAHiddenRoutingGateCable,
)


def public_layout():
    return {
        "layout_version": (
            "ccda_hidden_routing_gate_"
            "public_v1"
        ),
        "topology_id": (
            "hidden_routing_gate_v1r4"
        ),
        "probe_index": 7,
        "endpoint_index": 0,
        "leading_segment_indices": [
            0,
            1,
            2,
            3,
        ],
        "normal_xy": [0.0, 1.0],
        "tangent_xy": [1.0, 0.0],
        "stage1_target_xy": [
            0.40,
            -0.32,
        ],
        "final_target_xy": [
            0.40,
            -0.28,
        ],
        "target_plane_point_xy": [
            0.40,
            -0.315,
        ],
        "target_corridor_half_width": (
            0.035
        ),
        "target_zone_center_xy": [
            0.40,
            -0.3025,
        ],
        "target_zone_depth": 0.025,
        "target_zone_half_extents_xy": [
            0.0125,
            0.035,
        ],
        "target_zone_yaw": (
            np.pi / 2
        ),
    }


def make_task():
    task = object.__new__(
        CCDAHiddenRoutingGateCable
    )
    task._hook_layout = {
        "public_task": public_layout(),
    }
    task._frozen_public_routing_layout = (
        None
    )
    task._target_zone_body_id = None
    return task


def test_freeze_exports_exact_public_layout(
    monkeypatch,
):
    monkeypatch.setenv(
        PUBLIC_LAYOUT_MODE_ENV,
        FROZEN_BRANCH_ARM_LAYOUT_MODE,
    )
    task = make_task()
    task._freeze_public_routing_layout()

    expected = public_layout()
    assert (
        task._frozen_public_routing_layout
        == expected
    )
    assert json.loads(
        os.environ[
            FROZEN_PUBLIC_LAYOUT_ENV
        ]
    ) == expected
    assert (
        task._current_public_routing_layout()
        == expected
    )


def test_current_public_layout_prefers_frozen():
    task = make_task()
    frozen = public_layout()
    frozen["stage1_target_xy"] = [
        0.41,
        -0.31,
    ]
    task._frozen_public_routing_layout = (
        frozen
    )
    assert (
        task._current_public_routing_layout()
        ["stage1_target_xy"]
        == [0.41, -0.31]
    )


def test_target_zone_reposition_uses_frozen_pose(
    monkeypatch,
):
    task = make_task()
    task._target_zone_body_id = 42
    calls = []

    monkeypatch.setattr(
        routing_task.p,
        "getQuaternionFromEuler",
        lambda value: (
            "quaternion",
            tuple(value),
        ),
    )
    monkeypatch.setattr(
        routing_task.p,
        "resetBasePositionAndOrientation",
        lambda body, position, orientation: (
            calls.append(
                (
                    body,
                    position,
                    orientation,
                )
            )
        ),
    )

    task._position_visible_target_zone(
        public_layout()
    )

    assert len(calls) == 1
    body, position, orientation = calls[0]
    assert body == 42
    assert position == (
        0.40,
        -0.3025,
        0.001,
    )
    assert orientation == (
        "quaternion",
        (
            0,
            0,
            np.pi / 2,
        ),
    )
