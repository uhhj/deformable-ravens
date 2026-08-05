"""Fixed invisible delayed Z-latch cable task for Phase 0I smoke."""
from __future__ import annotations

import numpy as np
import pybullet as p

from ravens.tasks.ccda_hidden_hook_cable import CCDAHiddenHookCable
from ravens.tasks.ccda_hidden_latch_geometry import (
    HiddenLatchGeometryConfig,
    compute_hidden_latch_layout,
)


class CCDAHiddenLatchCable(CCDAHiddenHookCable):
    CONDITIONS = ("free", "hidden_hook")
    ENVIRONMENT_VERSION = "ccda_hidden_latch_cable_v1"

    def __init__(self):
        super().__init__()
        self._name = "ccda-hidden-latch-cable"

    def _latch_config(self):
        return HiddenLatchGeometryConfig(
            center_ratio=self._env_float("CCDA_LATCH_CENTER_RATIO", "0.45"),
            stop_clearance=self._env_float("CCDA_LATCH_STOP_CLEARANCE", "0.002"),
            roof_clearance=self._env_float("CCDA_LATCH_ROOF_CLEARANCE", "0.002"),
            wall_thickness=self._env_float("CCDA_LATCH_WALL_THICKNESS", "0.002"),
            wall_width=self._env_float("CCDA_LATCH_WALL_WIDTH", "0.032"),
            wall_height=self._env_float("CCDA_LATCH_WALL_HEIGHT", "0.020"),
            roof_depth=self._env_float("CCDA_LATCH_ROOF_DEPTH", "0.020"),
            roof_width=self._env_float("CCDA_LATCH_ROOF_WIDTH", "0.036"),
            roof_thickness=self._env_float("CCDA_LATCH_ROOF_THICKNESS", "0.002"),
            main_pull_distance=self._env_float(
                "CCDA_LATCH_MAIN_PULL_DISTANCE", "0.08"
            ),
            workspace_x=(
                self._env_float("CCDA_LATCH_WORKSPACE_X_MIN", "0.25"),
                self._env_float("CCDA_LATCH_WORKSPACE_X_MAX", "0.75"),
            ),
            workspace_y=(
                self._env_float("CCDA_LATCH_WORKSPACE_Y_MIN", "-0.45"),
                self._env_float("CCDA_LATCH_WORKSPACE_Y_MAX", "0.45"),
            ),
        )

    def _compute_hidden_layout(self, positions):
        config = self._latch_config()
        layout = compute_hidden_latch_layout(positions, config)
        probe_index = int(layout["probe_index"])
        if probe_index >= len(self.cable_bead_IDs):
            raise RuntimeError("latch probe index exceeds cable bead IDs")
        orientation = p.getBasePositionAndOrientation(
            int(self.cable_bead_IDs[probe_index])
        )[1]
        rotation = np.asarray(
            p.getMatrixFromQuaternion(orientation), dtype=np.float64
        ).reshape(3, 3)
        vertical_support = float(
            config.bead_radius * np.sum(np.abs(rotation[2, :]))
        )
        construction_epsilon = 5e-5
        roof_bottom = float(
            positions[probe_index, 2] + vertical_support
            + config.roof_clearance + construction_epsilon
        )
        for box in layout["boxes"]:
            if box["name"] == "roof":
                box["center_z"] = float(
                    roof_bottom + config.roof_thickness / 2
                )
        layout["roof_bottom_z"] = roof_bottom
        layout["anchor_vertical_collision_support"] = vertical_support
        layout["expected_surface_clearance"] = float(min(
            layout["expected_surface_clearance"],
            config.roof_clearance + construction_epsilon,
        ))
        return layout

    def _create_invisible_hook(self):
        super()._create_invisible_hook()
        body_ids = [int(p.getBodyUniqueId(i)) for i in range(p.getNumBodies())]
        for latch_id in self._hook_body_ids:
            for body_id in body_ids:
                if body_id == latch_id:
                    continue
                for link_index in range(-1, int(p.getNumJoints(body_id))):
                    p.setCollisionFilterPair(
                        int(latch_id), body_id, -1, link_index, 0
                    )

    def reset(self, env, last_info=None):
        super().reset(env, last_info=last_info)
        self._minimum_initial_clearance = self._env_float(
            "CCDA_LATCH_MIN_INITIAL_CLEARANCE", "0.002"
        )

    def ccda_privileged_state(self):
        state = super().ccda_privileged_state()
        state.update({
            "environment_version": self.ENVIRONMENT_VERSION,
            "hidden_factor": "invisible_z_latch",
            "topology": "delayed_z_latch_v1",
            "latch_layout": state.pop("hook_layout"),
            "latch_body_ids": state.pop("hook_body_ids"),
            "latch_collision_enabled": state.pop("hook_collision_enabled"),
            "latch_initial_clearance": state.pop("hook_initial_clearance"),
            "latch_visual_shape_indices": state.pop("hook_visual_shape_indices"),
            "latch_visual_shape_disabled": state.pop("hook_visual_shape_disabled"),
        })
        return state
