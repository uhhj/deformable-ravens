"""Hidden-Hook Cable Routing task for Stage M0 auditing."""
from __future__ import annotations

import copy
from typing import Any, Dict, List

import numpy as np
import pybullet as p

from ravens.tasks.ccda_hidden_friction_cable import CCDAHiddenFrictionCable
from ravens.tasks.ccda_hidden_hook_geometry import (
    HiddenHookGeometryConfig,
    compute_hidden_hook_layout,
)


class CCDAHiddenHookCable(CCDAHiddenFrictionCable):
    CONDITIONS = ("free", "hidden_hook")
    ENVIRONMENT_VERSION = "ccda_hidden_hook_cable_v1"

    def __init__(self):
        super().__init__()
        self._name = "ccda-hidden-hook-cable"
        self._hook_body_ids: List[int] = []
        self._hook_visual_shape_indices: List[int] = []
        self._hook_layout: Dict[str, Any] = {}
        self._hook_collision_enabled = False
        self._minimum_initial_clearance = 0.002
        self._initial_hook_clearance = None

    def _hook_config(self):
        return HiddenHookGeometryConfig(
            center_ratio=self._env_float("CCDA_HOOK_CENTER_RATIO", "0.45"),
            mouth_offset=self._env_float("CCDA_HOOK_MOUTH_OFFSET", "0.008"),
            width=self._env_float("CCDA_HOOK_WIDTH", "0.028"),
            depth=self._env_float("CCDA_HOOK_DEPTH", "0.020"),
            thickness=self._env_float("CCDA_HOOK_THICKNESS", "0.003"),
            height=self._env_float("CCDA_HOOK_HEIGHT", "0.025"),
            probe_distance=self._env_float("CCDA_HOOK_PROBE_DISTANCE", "0.012"),
            main_pull_distance=self._env_float("CCDA_HOOK_MAIN_PULL_DISTANCE", "0.08"),
            workspace_x=(
                self._env_float("CCDA_HOOK_WORKSPACE_X_MIN", "0.25"),
                self._env_float("CCDA_HOOK_WORKSPACE_X_MAX", "0.75"),
            ),
            workspace_y=(
                self._env_float("CCDA_HOOK_WORKSPACE_Y_MIN", "-0.45"),
                self._env_float("CCDA_HOOK_WORKSPACE_Y_MAX", "0.45"),
            ),
        )

    def reset(self, env, last_info=None):
        super().reset(env, last_info=last_info)
        self._hook_body_ids = []
        self._hook_visual_shape_indices = []
        self._hook_layout = compute_hidden_hook_layout(
            self._ordered_bead_positions(), self._hook_config()
        )
        self._hook_collision_enabled = False
        self._initial_hook_clearance = None
        self._minimum_initial_clearance = self._env_float(
            "CCDA_HOOK_MIN_INITIAL_CLEARANCE", "0.002"
        )
        if self._minimum_initial_clearance <= 0:
            raise ValueError("minimum hook clearance must be positive")
        self._create_invisible_hook()

    def _create_invisible_hook(self):
        for box in self._hook_layout["boxes"]:
            half = [float(value) for value in box["half_extents"]]
            collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
            center = box["center_xy"]
            body_id = int(p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=-1,
                basePosition=(float(center[0]), float(center[1]), half[2]),
                baseOrientation=p.getQuaternionFromEuler((0, 0, float(box["yaw"]))),
            ))
            self._hook_body_ids.append(body_id)
            self._hook_visual_shape_indices.append(-1)
            self._IDs[body_id] = "ccda_hidden_hook_" + box["name"]

        body_ids = [int(p.getBodyUniqueId(i)) for i in range(p.getNumBodies())]
        for hook_id in self._hook_body_ids:
            for body_id in body_ids:
                if body_id != hook_id:
                    p.setCollisionFilterPair(hook_id, body_id, -1, -1, 0)

    def _position_hook_bodies(self):
        if len(self._hook_body_ids) != len(self._hook_layout["boxes"]):
            raise RuntimeError("hidden hook body/layout count mismatch")
        for hook_id, box in zip(self._hook_body_ids, self._hook_layout["boxes"]):
            half = [float(value) for value in box["half_extents"]]
            center = box["center_xy"]
            p.resetBasePositionAndOrientation(
                int(hook_id),
                (float(center[0]), float(center[1]), half[2]),
                p.getQuaternionFromEuler((0, 0, float(box["yaw"]))),
            )

    def _minimum_hook_cable_distance(self):
        minimum = float("inf")
        for hook_id in self._hook_body_ids:
            for bead_id in self.cable_bead_IDs:
                for point in p.getClosestPoints(
                    bodyA=int(hook_id), bodyB=int(bead_id), distance=0.05
                ):
                    minimum = min(minimum, float(point[8]))
        return minimum

    def arm_ccda_hidden_factor_after_settle(self):
        if self._hidden_friction_armed:
            return {"already_armed": True}
        before = self._ordered_bead_positions()
        self._hook_layout = compute_hidden_hook_layout(
            before, self._hook_config()
        )
        self._position_hook_bodies()
        clearance = self._minimum_hook_cable_distance()
        if clearance < self._minimum_initial_clearance:
            raise RuntimeError(
                f"hidden hook too close to initial cable: {clearance}"
            )
        enabled = self.hidden_condition == "hidden_hook"
        for hook_id in self._hook_body_ids:
            for bead_id in self.cable_bead_IDs:
                p.setCollisionFilterPair(
                    int(hook_id), int(bead_id), -1, -1, int(enabled)
                )
        after = self._ordered_bead_positions()
        jump = np.abs(after - before)
        self._arm_max_abs_jump = float(np.max(jump)) if jump.size else 0.0
        self._arm_mae_jump = float(np.mean(jump)) if jump.size else 0.0
        self._hidden_friction_pending = False
        self._hidden_friction_armed = True
        self._hook_collision_enabled = bool(enabled)
        self._initial_hook_clearance = float(clearance)
        self._last_contact = self._zero_contact()
        return {
            "already_armed": False,
            "hidden_factor": "hook_collision",
            "collision_enabled": bool(enabled),
            "initial_clearance": float(clearance),
            "arm_max_abs_jump": self._arm_max_abs_jump,
            "arm_mae_jump": self._arm_mae_jump,
        }

    def arm_hidden_friction_after_settle(self):
        return self.arm_ccda_hidden_factor_after_settle()

    def reset_ccda_branch(self, condition):
        super().reset_ccda_branch(condition)
        self._hook_collision_enabled = False
        self._initial_hook_clearance = None
        self._last_contact = self._zero_contact()

    def physics_pre_step_hook(self):
        self._last_contact = self._zero_contact()

    def _hook_contact_observation(self):
        sum_force = np.zeros(3, dtype=np.float64)
        force_norms = []
        active_beads = set()
        speeds = []
        for hook_id in self._hook_body_ids:
            for bead_id in self.cable_bead_IDs:
                bead_norm = 0.0
                for point in p.getContactPoints(bodyA=int(hook_id), bodyB=int(bead_id)):
                    force = float(point[9]) * np.asarray(point[7], dtype=np.float64)
                    if len(point) >= 14:
                        force += (
                            float(point[10]) * np.asarray(point[11], dtype=np.float64)
                            + float(point[12]) * np.asarray(point[13], dtype=np.float64)
                        )
                    sum_force += force
                    bead_norm += float(np.linalg.norm(force))
                if bead_norm > 0:
                    active_beads.add(int(bead_id))
                    force_norms.append(bead_norm)
        for bead_id in active_beads:
            velocity = p.getBaseVelocity(int(bead_id))[0]
            speeds.append(float(np.linalg.norm(np.asarray(velocity[:2]))))
        return {
            "force_xyz": sum_force.astype(float).tolist(),
            "force_norm": float(np.sum(force_norms)) if force_norms else 0.0,
            "max_force_norm": float(np.max(force_norms)) if force_norms else 0.0,
            "active_beads": len(active_beads),
            "mean_speed": float(np.mean(speeds)) if speeds else 0.0,
        }

    def physics_step_hook(self):
        self._physics_step_count += 1
        if self._hidden_friction_armed:
            self._last_contact = self._hook_contact_observation()
        if (
            not self._hidden_friction_armed
            or self._physics_step_count % self._trace_stride != 0
        ):
            return

        positions = self._ordered_bead_positions()
        velocities = np.asarray(
            [p.getBaseVelocity(int(bead))[0] for bead in self.cable_bead_IDs],
            dtype=np.float64,
        )
        robot = self._robot_state()
        oracle = self.ccda_contact_observation()
        sensor = self.ccda_sensor_observation()
        self._trace.append({
            "physics_step": int(self._physics_step_count),
            "phase": self._phase,
            "bead_positions": positions.tolist(),
            "bead_velocities": velocities.tolist(),
            "joint_positions": robot["joint_positions"],
            "joint_velocities": robot["joint_velocities"],
            "ee_position": robot["ee_position"],
            "ee_orientation": robot["ee_orientation"],
            "contact_force_xyz": oracle["force_xyz"],
            "contact_force_norm": oracle["force_norm"],
            "contact_max_force_norm": oracle["max_force_norm"],
            "contact_active_beads": oracle["active_beads"],
            "contact_mean_speed": oracle["mean_speed"],
            "sensor_joint_motor_torque": sensor["joint_motor_torque"],
            "sensor_joint_motor_torque_norm": sensor["joint_motor_torque_norm"],
            "sensor_joint_reaction_force_torque": sensor[
                "joint_reaction_force_torque"
            ],
            "sensor_joint_reaction_force_torque_norm": sensor[
                "joint_reaction_force_torque_norm"
            ],
            "sensor_suction_force_xyz": sensor["suction_force_xyz"],
            "sensor_suction_force_norm": sensor["suction_force_norm"],
            "sensor_suction_torque_xyz": sensor["suction_torque_xyz"],
            "sensor_suction_torque_norm": sensor["suction_torque_norm"],
            "sensor_grasp_active": sensor["grasp_active"],
            "sensor_constraint_available": sensor["constraint_available"],
        })

    def ccda_privileged_state(self):
        return {
            "environment_version": self.ENVIRONMENT_VERSION,
            "hidden_condition": self.hidden_condition,
            "hidden_factor": "invisible_u_hook",
            "hook_layout": copy.deepcopy(self._hook_layout),
            "hook_body_ids": list(self._hook_body_ids),
            "hook_collision_enabled": bool(self._hook_collision_enabled),
            "hook_initial_clearance": self._initial_hook_clearance,
            "hook_visual_shape_indices": list(
                self._hook_visual_shape_indices
            ),
            "hook_visual_shape_disabled": bool(
                self._hook_visual_shape_indices
                and all(index == -1 for index in self._hook_visual_shape_indices)
            ),
            "arm_max_abs_jump": float(self._arm_max_abs_jump),
            "arm_mae_jump": float(self._arm_mae_jump),
            "visible_seed": self.ccda_visible_seed,
            "pair_group": self.ccda_pair_group,
        }
