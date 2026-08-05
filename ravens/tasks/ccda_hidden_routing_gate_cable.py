"""Outcome-aligned hidden routing-gate cable task."""
from __future__ import annotations

import copy
import os

import numpy as np
import pybullet as p

from ravens.tasks.ccda_hidden_hook_cable import (
    CCDAHiddenHookCable,
)
from ravens.tasks.ccda_hidden_routing_gate_geometry import (
    HiddenRoutingGateGeometryConfig,
    compute_hidden_routing_gate_layout,
    public_routing_layout,
)


class CCDAHiddenRoutingGateCable(
    CCDAHiddenHookCable
):
    """Hidden probe roof plus endpoint-routing barrier.

    The internal binary condition name remains ``hidden_hook`` only
    for compatibility with the existing exact-counterfactual tooling.
    Its physical meaning in this task is ``hidden_routing_gate``.
    """

    ENVIRONMENT_VERSION = (
        "ccda_hidden_routing_gate_cable_v1"
    )

    def __init__(self):
        super().__init__()
        self._name = (
            "ccda-hidden-routing-gate-cable"
        )
        self._target_zone_body_id = None
        self._target_zone_visual_shape_index = (
            None
        )

    def _routing_config(self):
        return HiddenRoutingGateGeometryConfig(
            center_ratio=self._env_float(
                "CCDA_ROUTING_CENTER_RATIO",
                "0.45",
            ),
            probe_roof_clearance=self._env_float(
                "CCDA_ROUTING_PROBE_ROOF_CLEARANCE",
                "0.002",
            ),
            probe_roof_depth=self._env_float(
                "CCDA_ROUTING_PROBE_ROOF_DEPTH",
                "0.022",
            ),
            probe_roof_width=self._env_float(
                "CCDA_ROUTING_PROBE_ROOF_WIDTH",
                "0.022",
            ),
            probe_roof_thickness=self._env_float(
                "CCDA_ROUTING_PROBE_ROOF_THICKNESS",
                "0.002",
            ),
            barrier_offset=self._env_float(
                "CCDA_ROUTING_BARRIER_OFFSET",
                "0.065",
            ),
            barrier_thickness=self._env_float(
                "CCDA_ROUTING_BARRIER_THICKNESS",
                "0.004",
            ),
            barrier_width=self._env_float(
                "CCDA_ROUTING_BARRIER_WIDTH",
                "0.340",
            ),
            barrier_height=self._env_float(
                "CCDA_ROUTING_BARRIER_HEIGHT",
                "0.030",
            ),
            stage1_pull_distance=self._env_float(
                "CCDA_ROUTING_STAGE1_DISTANCE",
                "0.080",
            ),
            final_pull_distance=self._env_float(
                "CCDA_ROUTING_FINAL_DISTANCE",
                "0.120",
            ),
            target_plane_offset=self._env_float(
                "CCDA_ROUTING_TARGET_PLANE_OFFSET",
                "0.085",
            ),
            target_zone_depth=self._env_float(
                "CCDA_ROUTING_TARGET_ZONE_DEPTH",
                "0.025",
            ),
            target_corridor_half_width=self._env_float(
                "CCDA_ROUTING_CORRIDOR_HALF_WIDTH",
                "0.035",
            ),
            leading_segment_size=self._env_int(
                "CCDA_ROUTING_LEADING_SEGMENT_SIZE",
                "4",
            ),
            workspace_x=(
                self._env_float(
                    "CCDA_ROUTING_WORKSPACE_X_MIN",
                    "0.25",
                ),
                self._env_float(
                    "CCDA_ROUTING_WORKSPACE_X_MAX",
                    "0.75",
                ),
            ),
            workspace_y=(
                self._env_float(
                    "CCDA_ROUTING_WORKSPACE_Y_MIN",
                    "-0.45",
                ),
                self._env_float(
                    "CCDA_ROUTING_WORKSPACE_Y_MAX",
                    "0.45",
                ),
            ),
            topology_id=os.environ.get(
                "CCDA_ROUTING_TOPOLOGY_ID",
                "hidden_routing_gate_v1",
            ),
        )

    def _compute_hidden_layout(self, positions):
        config = self._routing_config()
        layout = (
            compute_hidden_routing_gate_layout(
                positions,
                config,
            )
        )

        probe_index = int(
            layout["probe_index"]
        )
        if (
            probe_index
            >= len(self.cable_bead_IDs)
        ):
            raise RuntimeError(
                "routing probe index exceeds "
                "cable bead IDs"
            )

        orientation = (
            p.getBasePositionAndOrientation(
                int(
                    self.cable_bead_IDs[
                        probe_index
                    ]
                )
            )[1]
        )
        rotation = np.asarray(
            p.getMatrixFromQuaternion(
                orientation
            ),
            dtype=np.float64,
        ).reshape(3, 3)
        vertical_support = float(
            config.bead_radius
            * np.sum(
                np.abs(rotation[2, :])
            )
        )
        construction_epsilon = 5e-5
        roof_bottom = float(
            positions[probe_index, 2]
            + vertical_support
            + config.probe_roof_clearance
            + construction_epsilon
        )

        for box in layout["boxes"]:
            if box["name"] == "probe_roof":
                box["center_z"] = float(
                    roof_bottom
                    + config.probe_roof_thickness / 2
                )

        layout["roof_bottom_z"] = roof_bottom
        layout[
            "probe_anchor_vertical_collision_support"
        ] = vertical_support
        layout[
            "expected_surface_clearance"
        ] = float(min(
            layout["expected_surface_clearance"],
            config.probe_roof_clearance
            + construction_epsilon,
        ))
        return layout

    def _create_visible_target_zone(self):
        public = public_routing_layout(
            self._hook_layout
        )
        center_xy = np.asarray(
            public["target_zone_center_xy"],
            dtype=np.float64,
        )
        half_xy = np.asarray(
            public[
                "target_zone_half_extents_xy"
            ],
            dtype=np.float64,
        )
        yaw = float(
            public["target_zone_yaw"]
        )

        visual = int(
            p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[
                    float(half_xy[0]),
                    float(half_xy[1]),
                    0.0005,
                ],
                rgbaColor=[
                    0.20,
                    0.80,
                    0.20,
                    0.35,
                ],
            )
        )
        body_id = int(
            p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=visual,
                basePosition=(
                    float(center_xy[0]),
                    float(center_xy[1]),
                    0.001,
                ),
                baseOrientation=(
                    p.getQuaternionFromEuler(
                        (0, 0, yaw)
                    )
                ),
            )
        )
        self._target_zone_body_id = body_id
        self._target_zone_visual_shape_index = (
            visual
        )
        self._IDs[body_id] = (
            "ccda_public_routing_target_zone"
        )

    def reset(self, env, last_info=None):
        self._target_zone_body_id = None
        self._target_zone_visual_shape_index = (
            None
        )
        super().reset(
            env,
            last_info=last_info,
        )
        self._minimum_initial_clearance = (
            self._env_float(
                "CCDA_ROUTING_MIN_INITIAL_CLEARANCE",
                "0.002",
            )
        )
        if self._minimum_initial_clearance <= 0:
            raise ValueError(
                "minimum routing-gate clearance "
                "must be positive"
            )
        self._create_visible_target_zone()

    def ccda_public_task_state(self):
        return {
            "environment_version": (
                self.ENVIRONMENT_VERSION
            ),
            "task_family": (
                "hidden_routing_gate"
            ),
            "public_task": copy.deepcopy(
                public_routing_layout(
                    self._hook_layout
                )
            ),
            "target_zone_body_id": (
                self._target_zone_body_id
            ),
            "target_zone_has_collision": False,
        }

    def ccda_privileged_state(self):
        state = super().ccda_privileged_state()
        layout = state.pop("hook_layout")
        result = {
            **state,
            "environment_version": (
                self.ENVIRONMENT_VERSION
            ),
            "hidden_factor": (
                "invisible_routing_gate_fixture"
            ),
            "topology": str(
                layout["topology"]
            ),
            "public_task": copy.deepcopy(
                public_routing_layout(layout)
            ),
            "routing_gate_layout_privileged": (
                layout
            ),
            "routing_gate_body_ids": state.pop(
                "hook_body_ids"
            ),
            "routing_gate_collision_enabled": (
                state.pop(
                    "hook_collision_enabled"
                )
            ),
            "routing_gate_initial_clearance": (
                state.pop(
                    "hook_initial_clearance"
                )
            ),
            "routing_gate_visual_shape_indices": (
                state.pop(
                    "hook_visual_shape_indices"
                )
            ),
            "routing_gate_visual_shape_disabled": (
                state.pop(
                    "hook_visual_shape_disabled"
                )
            ),
            "target_zone_body_id": (
                self._target_zone_body_id
            ),
            "target_zone_visual_shape_index": (
                self._target_zone_visual_shape_index
            ),
            "target_zone_has_collision": False,
        }
        return result

