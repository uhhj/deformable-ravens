#!/usr/bin/env python3
"""Formal v2-only CCDA slack cable task."""
from __future__ import annotations

import os
import collections
from typing import Any, Dict, List

import numpy as np
import pybullet as p

from ravens.tasks.ccda_slack_breakaway import (
    SlackBreakawayConfig,
    UnilateralSlackBreakaway,
)
from ravens.tasks.defs_cables import CableLineNoTarget


class CCDASlackCableV2(CableLineNoTarget):
    """Cable line task with a versioned latent unilateral tether."""

    CONDITIONS = ("free", "hidden_slack_breakaway_pin_v2")
    ENVIRONMENT_VERSION = "ccda_hidden_slack_breakaway_v2"

    def __init__(self):
        super().__init__()
        self._name = "ccda-slack-cable-v2"
        self.metric = "cable-target"
        self.primitive = "pick_place"
        self.ee = "suction"
        self._ccda_env = None
        self.hidden_condition = os.environ.get("CCDA_HIDDEN_CONDITION", "free")
        self.ccda_visible_seed = os.environ.get("CCDA_VISIBLE_SEED", "")
        self.ccda_pair_group = os.environ.get("CCDA_PAIR_GROUP", "")
        self._reset_ccda_state()

    def _reset_ccda_state(self) -> None:
        self.hidden_contact_applied = False
        self.hidden_body_ids: List[int] = []
        self.hidden_constraint_ids: List[int] = []
        self.hidden_contact_meta: Dict[str, Any] = {}
        self._ccda_step_count = 0
        self._ccda_physics_step_count = 0
        self._hidden_contact_pending = False
        self._hidden_contact_armed = False
        self._hidden_contact_arm_mode = "reset"
        self._hidden_contact_arm_step = None
        self._hidden_contact_arm_physics_step = None
        self._hidden_contact_arm_max_abs_jump = 0.0
        self._hidden_contact_arm_mae_jump = 0.0
        self._slack_model = None
        self._slack_bead_id = None
        self._slack_bead_local_index = None
        self._slack_last_output = None
        self._slack_last_applied_force = [0.0, 0.0, 0.0]

    def reset(self, env, last_info=None):
        self._ccda_env = env
        self.hidden_condition = os.environ.get("CCDA_HIDDEN_CONDITION", "free")
        self.ccda_visible_seed = os.environ.get("CCDA_VISIBLE_SEED", "")
        self.ccda_pair_group = os.environ.get("CCDA_PAIR_GROUP", "")
        if self.hidden_condition not in self.CONDITIONS:
            raise ValueError(
                f"unknown formal condition {self.hidden_condition!r}; "
                f"expected one of {self.CONDITIONS}"
            )
        self._reset_ccda_state()
        self.hidden_contact_meta = {
            "environment_semantics_version": self.ENVIRONMENT_VERSION,
            "condition": self.hidden_condition,
            "applied": False,
            "hidden_body_ids": [],
            "hidden_constraint_ids": [],
        }
        # The upstream cable builder keys target-place construction on its
        # legacy task name. Reuse that geometry path only while resetting,
        # then restore the formal v2 identity immediately.
        formal_name = self._name
        self._name = "cable-line-notarget"
        try:
            super().reset(env, last_info=last_info)
        finally:
            self._name = formal_name
        defer = os.environ.get("CCDA_DEFER_HIDDEN_CONTACT_ARMING", "1") == "1"
        if defer:
            self._hidden_contact_pending = True
            self._hidden_contact_armed = False
            self._hidden_contact_arm_mode = "deferred_after_visible_settle"
        else:
            self.arm_hidden_contact_after_settle(env)
        self._update_metadata()

    def _ordered_bead_xy(self) -> np.ndarray:
        return np.asarray(
            [p.getBasePositionAndOrientation(int(bead))[0][:2] for bead in self.cable_bead_IDs],
            dtype=np.float32,
        )

    def arm_hidden_contact_after_settle(self, env=None) -> Dict[str, Any]:
        if self._hidden_contact_armed:
            self._update_metadata()
            return {"already_armed": True, "condition": self.hidden_condition}
        before = self._ordered_bead_xy()
        if self.hidden_condition == "hidden_slack_breakaway_pin_v2":
            self._apply_slack_v2()
        else:
            self.hidden_contact_applied = True
            self.hidden_contact_meta.update({"applied": True, "force_model": "none"})
        after = self._ordered_bead_xy()
        difference = np.abs(after - before)
        self._hidden_contact_pending = False
        self._hidden_contact_armed = True
        self._hidden_contact_arm_mode = "free_noop" if self.hidden_condition == "free" else "deferred_zero_offset"
        self._hidden_contact_arm_step = int(self._ccda_step_count)
        self._hidden_contact_arm_physics_step = int(self._ccda_physics_step_count)
        self._hidden_contact_arm_max_abs_jump = float(np.max(difference)) if difference.size else 0.0
        self._hidden_contact_arm_mae_jump = float(np.mean(difference)) if difference.size else 0.0
        self._update_metadata()
        return {
            "already_armed": False,
            "condition": self.hidden_condition,
            "max_abs_jump": self._hidden_contact_arm_max_abs_jump,
            "mae_jump": self._hidden_contact_arm_mae_jump,
            "hidden_body_ids": [],
            "hidden_constraint_ids": [],
        }

    def physics_pre_step_hook(self) -> None:
        if self.hidden_condition != "hidden_slack_breakaway_pin_v2" or not self._hidden_contact_armed:
            return
        if self._slack_model is None or self._slack_bead_id is None:
            raise RuntimeError("armed v2 task has no slack model")
        bead = int(self._slack_bead_id)
        position = p.getBasePositionAndOrientation(bead)[0]
        velocity = p.getBaseVelocity(bead)[0]
        output = self._slack_model.evaluate(
            position,
            velocity,
            physics_step=int(self._ccda_physics_step_count) + 1,
        )
        force = np.asarray(output.force_xyz, dtype=np.float64)
        if force.shape != (3,) or not np.all(np.isfinite(force)):
            raise RuntimeError("non-finite formal slack force")
        self._slack_last_output = output
        self._slack_last_applied_force = force.astype(float).tolist()
        if np.any(force != 0.0):
            p.applyExternalForce(
                bead,
                -1,
                forceObj=self._slack_last_applied_force,
                posObj=[float(value) for value in position],
                flags=p.WORLD_FRAME,
            )
        self._update_metadata()

    def physics_step_hook(self) -> None:
        self._ccda_physics_step_count += 1
        self._update_metadata()

    def reward(self):
        self._ccda_step_count += 1
        reward, extras = super().reward()
        extras.update(self._ccda_extras())
        return reward, extras

    def oracle(self, env):
        """Deterministic geometry oracle used only to create the free sequence."""
        OracleAgent = collections.namedtuple("OracleAgent", ["act"])

        def act(obs, info):
            action = {"primitive": None}
            if self.done():
                return action
            bead_positions = np.asarray(
                [p.getBasePositionAndOrientation(int(bead))[0] for bead in self.cable_bead_IDs],
                dtype=np.float64,
            )
            targets = np.asarray(
                [self.goal["places"][int(bead)][0] for bead in self.cable_bead_IDs],
                dtype=np.float64,
            )
            index = int(np.argmax(np.linalg.norm(bead_positions[:, :2] - targets[:, :2], axis=1)))
            quaternion = (0.0, 0.0, 0.0, 1.0)
            action.update(
                {
                    "primitive": "pick_place",
                    "params": {
                        "pose0": ((float(bead_positions[index, 0]), float(bead_positions[index, 1]), 0.001), quaternion),
                        "pose1": ((float(targets[index, 0]), float(targets[index, 1]), 0.001), quaternion),
                    },
                }
            )
            return action

        return OracleAgent(act)

    def _robot_pose_proxy(self) -> Dict[str, Any]:
        env = self._ccda_env
        body_id = None
        if env is not None:
            for attribute in ("ur5", "ur5_id", "robot_id", "robot"):
                value = getattr(env, attribute, None)
                if isinstance(value, (int, np.integer)):
                    body_id = int(value)
                    break
        if body_id is not None:
            try:
                count = int(p.getNumJoints(body_id))
                states = [p.getJointState(body_id, index) for index in range(count)]
                position = [0.0, 0.0, 0.0]
                orientation = [0.0, 0.0, 0.0, 1.0]
                if count:
                    link = p.getLinkState(body_id, count - 1)
                    position = [float(value) for value in link[0]]
                    orientation = [float(value) for value in link[1]]
                return {
                    "source": "pybullet_robot_body",
                    "body_id": body_id,
                    "joint_positions": [float(value[0]) for value in states],
                    "joint_velocities": [float(value[1]) for value in states],
                    "ee_position": position,
                    "ee_orientation": orientation,
                }
            except Exception:
                pass
        return {
            "source": "missing_zero_proxy",
            "body_id": None,
            "joint_positions": [],
            "joint_velocities": [],
            "ee_position": [0.0, 0.0, 0.0],
            "ee_orientation": [0.0, 0.0, 0.0, 1.0],
        }

    def _ordered_bead_states(self) -> List[Dict[str, Any]]:
        states = []
        for index, bead in enumerate(self.cable_bead_IDs):
            position, orientation = p.getBasePositionAndOrientation(int(bead))
            states.append(
                {
                    "local_index": int(index),
                    "id": int(bead),
                    "position": [float(value) for value in position],
                    "orientation": [float(value) for value in orientation],
                }
            )
        return states

    def _ccda_extras(self) -> Dict[str, Any]:
        self._update_metadata()
        beads = self._ordered_bead_states()
        try:
            visible_seed: Any = int(self.ccda_visible_seed)
        except (TypeError, ValueError):
            visible_seed = self.ccda_visible_seed
        return {
            "ccda_task": self._name,
            "hidden_condition": self.hidden_condition,
            "ccda_visible_seed": visible_seed,
            "ccda_pair_group": self.ccda_pair_group,
            "hidden_contact_meta": dict(self.hidden_contact_meta),
            "robot_pose_proxy": self._robot_pose_proxy(),
            "bead_ids": [item["id"] for item in beads],
            "bead_positions": [item["position"] for item in beads],
            "bead_orientations": [item["orientation"] for item in beads],
        }

    def _slack_breakaway_config(self) -> SlackBreakawayConfig:
        return SlackBreakawayConfig(
            slack_distance=float(os.environ.get("CCDA_SLACK_V2_DISTANCE", "0.015")),
            spring_stiffness=float(os.environ.get("CCDA_SLACK_V2_STIFFNESS", "50.0")),
            radial_damping=float(os.environ.get("CCDA_SLACK_V2_DAMPING", "0.1")),
            max_tension=float(os.environ.get("CCDA_SLACK_V2_MAX_TENSION", "4.0")),
            breakaway_extension=float(os.environ.get("CCDA_SLACK_V2_BREAKAWAY_EXTENSION", "0.030")),
            breakaway_force=float(os.environ.get("CCDA_SLACK_V2_BREAKAWAY_FORCE", "3.0")),
        )

    def _apply_slack_v2(self) -> None:
        if self.hidden_body_ids or self.hidden_constraint_ids:
            raise RuntimeError("formal v2 task must not own hidden geometry")
        ratio = float(os.environ.get("CCDA_SLACK_V2_BEAD_RATIO", "0.45"))
        ratio = min(max(ratio, 0.05), 0.95)
        index = int(np.clip(round(ratio * (len(self.cable_bead_IDs) - 1)), 0, len(self.cable_bead_IDs) - 1))
        bead = int(self.cable_bead_IDs[index])
        position = p.getBasePositionAndOrientation(bead)[0]
        self._slack_bead_id = bead
        self._slack_bead_local_index = index
        self._slack_model = UnilateralSlackBreakaway(
            config=self._slack_breakaway_config(),
            anchor_position=position,
        )
        self.hidden_contact_applied = True
        self.hidden_contact_meta.update(
            {
                "applied": True,
                "force_model": "unilateral_deadband_spring",
                "uses_world_constraint": False,
                "recoverability_params": self._slack_model.snapshot()["config"],
            }
        )

    def _update_metadata(self) -> None:
        snapshot = self._slack_model.snapshot() if self._slack_model is not None else None
        self.hidden_contact_meta.update(
            {
                "environment_semantics_version": self.ENVIRONMENT_VERSION,
                "condition": self.hidden_condition,
                "hidden_contact_pending": bool(self._hidden_contact_pending),
                "hidden_contact_armed": bool(self._hidden_contact_armed),
                "hidden_contact_arm_mode": self._hidden_contact_arm_mode,
                "hidden_contact_arm_step": self._hidden_contact_arm_step,
                "hidden_contact_arm_physics_step": self._hidden_contact_arm_physics_step,
                "hidden_contact_arm_max_abs_jump": float(self._hidden_contact_arm_max_abs_jump),
                "hidden_contact_arm_mae_jump": float(self._hidden_contact_arm_mae_jump),
                "hidden_body_ids": [],
                "hidden_constraint_ids": [],
                "slack_state": None if snapshot is None else snapshot["state"],
                "slack_engagement_physics_step": None if snapshot is None else snapshot["engagement_physics_step"],
                "slack_release_physics_step": None if snapshot is None else snapshot["release_physics_step"],
                "slack_release_reason": None if snapshot is None else snapshot["release_reason"],
                "slack_last_tension": 0.0 if snapshot is None else snapshot["last_tension"],
                "slack_last_force": list(self._slack_last_applied_force),
            }
        )

    def ccda_snapshot_state(self) -> Dict[str, Any]:
        return {
            "snapshot_version": "ccda_task_snapshot_v2",
            "hidden_condition": self.hidden_condition,
            "ccda_step_count": int(self._ccda_step_count),
            "ccda_physics_step_count": int(self._ccda_physics_step_count),
            "hidden_contact_pending": bool(self._hidden_contact_pending),
            "hidden_contact_armed": bool(self._hidden_contact_armed),
            "slack_bead_id": self._slack_bead_id,
            "slack_bead_local_index": self._slack_bead_local_index,
            "slack_last_applied_force": list(self._slack_last_applied_force),
            "slack_model": None if self._slack_model is None else self._slack_model.snapshot(),
        }

    def ccda_restore_state(self, snapshot: Dict[str, Any]) -> None:
        if snapshot.get("snapshot_version") != "ccda_task_snapshot_v2":
            raise ValueError("unsupported task snapshot")
        if snapshot.get("hidden_condition") != self.hidden_condition:
            raise ValueError("snapshot condition mismatch")
        self._ccda_step_count = int(snapshot["ccda_step_count"])
        self._ccda_physics_step_count = int(snapshot["ccda_physics_step_count"])
        self._hidden_contact_pending = bool(snapshot["hidden_contact_pending"])
        self._hidden_contact_armed = bool(snapshot["hidden_contact_armed"])
        self._slack_bead_id = snapshot.get("slack_bead_id")
        self._slack_bead_local_index = snapshot.get("slack_bead_local_index")
        self._slack_last_applied_force = [float(value) for value in snapshot.get("slack_last_applied_force", [0.0, 0.0, 0.0])]
        model = snapshot.get("slack_model")
        self._slack_model = None if model is None else UnilateralSlackBreakaway.from_snapshot(model)
        self._slack_last_output = None
        self._update_metadata()
