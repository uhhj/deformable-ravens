#!/usr/bin/env python
"""CCDA hidden-contact cable tasks.

This task is a Phase1 fork of cable-line-notarget. It keeps the visible
goal and the original pick-place oracle structure, but injects hidden
contact conditions after reset. Hidden contact metadata and ordered bead
states are added to info['extras'] through reward().
"""

# PHASE3_12D_R24_SLACK_BREAKAWAY_V2
import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pybullet as p

from ravens.tasks.defs_cables import CableLineNoTarget
from ravens.tasks.ccda_slack_breakaway import (
    SlackBreakawayConfig,
    UnilateralSlackBreakaway,
)


class HiddenContactCableLine(CableLineNoTarget):
    """Cable-line-notarget with controlled hidden contact conditions.

    Supported hidden conditions:
      - free: no hidden contact.
      - hidden_pin: hard/impossible diagnostic pin.
      - hidden_high_friction: weak high-friction control.
      - hidden_partial_pin: limited-span recoverable pin candidate.
      - hidden_soft_pin: low-force recoverable pin candidate.
      - hidden_breakaway_pin: soft breakaway-style recoverable pin candidate.
      - hidden_friction_patch: localized recoverable friction patch candidate.

    The hidden geometry is not added to env.objects or env.fixed_objects,
    so it is not returned by Environment.info. Collision-only bodies have
    no visual shape and should not appear in RGB renderings.
    """

    CONDITIONS = (
        "free",
        "hidden_pin",
        "hidden_high_friction",
        "hidden_partial_pin",
        "hidden_soft_pin",
        "hidden_breakaway_pin",
        "hidden_slack_breakaway_pin_v2",
        "hidden_friction_patch",
    )

    def __init__(self):
        super().__init__()
        self._name = "hidden-contact-cable-line"

        # Keep the original cable-line target metric and primitive.
        self.metric = "cable-target"
        self.primitive = "pick_place"
        self.ee = "suction"

        # Phase1 metadata.
        self.hidden_condition = os.environ.get("CCDA_HIDDEN_CONDITION", "free")
        self.ccda_visible_seed = os.environ.get("CCDA_VISIBLE_SEED", "")
        self.ccda_pair_group = os.environ.get("CCDA_PAIR_GROUP", "")

        self.hidden_contact_applied = False
        self.hidden_constraint_ids: List[int] = []
        self.hidden_body_ids: List[int] = []
        self.hidden_contact_meta: Dict[str, Any] = {}
        self._ccda_env = None

        self._ccda_step_count = 0
        self._ccda_physics_step_count = 0
        self._breakaway_released = False
        self._breakaway_release_step = None
        self._breakaway_release_physics_step = None
        self._breakaway_anchor_pos = None
        self._breakaway_bead_id = None
        self._breakaway_constraint_id = None
        self._breakaway_max_disp_seen = 0.0
        self._reset_slack_breakaway_v2_state()

        # PHASE3_12D_R2_DEFERRED_ARMING: hidden conditions are created only after the common
        # visible cable settling phase.
        self._hidden_contact_pending = False
        self._hidden_contact_armed = False
        self._hidden_contact_arm_mode = "uninitialized"
        self._hidden_contact_arm_step = None
        self._hidden_contact_arm_physics_step = None
        self._hidden_contact_arm_pre_xy = None
        self._hidden_contact_arm_post_xy = None
        self._hidden_contact_arm_max_abs_jump = 0.0
        self._hidden_contact_arm_mae_jump = 0.0

    def _reset_slack_breakaway_v2_state(self) -> None:
        self._slack_model = None
        self._slack_bead_id = None
        self._slack_bead_local_index = None
        self._slack_last_output = None
        self._slack_last_applied_force = [0.0, 0.0, 0.0]
        self._slack_environment_semantics_version = (
            "ccda_hidden_slack_breakaway_v2"
        )

    def reset(self, env, last_info=None):
        """Reset the base cable-line task and inject hidden contact.

        The hidden contact is applied after the original cable has been
        created and settled. This preserves the visible initial state as
        much as possible while changing only the hidden physical condition.
        """
        self._ccda_env = env

        condition = os.environ.get("CCDA_HIDDEN_CONDITION", self.hidden_condition)
        self.hidden_condition = condition
        self.ccda_visible_seed = os.environ.get("CCDA_VISIBLE_SEED", self.ccda_visible_seed)
        self.ccda_pair_group = os.environ.get("CCDA_PAIR_GROUP", self.ccda_pair_group)

        if self.hidden_condition not in self.CONDITIONS:
            raise ValueError(
                "Unknown hidden condition: {}. Valid: {}".format(
                    self.hidden_condition, self.CONDITIONS
                )
            )

        self.hidden_contact_applied = False
        self.hidden_constraint_ids = []
        self.hidden_body_ids = []
        self.hidden_contact_meta = {
            "condition": self.hidden_condition,
            "applied": False,
            "hidden_body_ids": [],
            "hidden_constraint_ids": [],
        }
        self._ccda_step_count = 0
        self._ccda_physics_step_count = 0
        self._breakaway_released = False
        self._breakaway_release_step = None
        self._breakaway_release_physics_step = None
        self._breakaway_anchor_pos = None
        self._breakaway_bead_id = None
        self._breakaway_constraint_id = None
        self._breakaway_max_disp_seen = 0.0
        self._reset_slack_breakaway_v2_state()
        self._hidden_contact_pending = False
        self._hidden_contact_armed = False
        self._hidden_contact_arm_mode = "reset"
        self._hidden_contact_arm_step = None
        self._hidden_contact_arm_physics_step = None
        self._hidden_contact_arm_pre_xy = None
        self._hidden_contact_arm_post_xy = None
        self._hidden_contact_arm_max_abs_jump = 0.0
        self._hidden_contact_arm_mae_jump = 0.0

        super().reset(env, last_info=last_info)

        # Phase3.12d-r2: all non-free conditions may defer hidden-contact
        # creation until a shared, condition-independent visible settling
        # phase has completed.
        defer_hidden = (
            os.environ.get("CCDA_DEFER_HIDDEN_CONTACT_ARMING", "0") == "1"
        )
        if self.hidden_condition == "free" or not defer_hidden:
            self._apply_hidden_contact(env)
            self._hidden_contact_pending = False
            self._hidden_contact_armed = True
            self._hidden_contact_arm_mode = "reset_immediate"
            self._hidden_contact_arm_step = int(self._ccda_step_count)
            self._hidden_contact_arm_physics_step = int(
                self._ccda_physics_step_count
            )
        else:
            self._hidden_contact_pending = True
            self._hidden_contact_armed = False
            self._hidden_contact_arm_mode = "deferred_after_visible_settle"
            self.hidden_contact_meta.update(
                {
                    "applied": False,
                    "hidden_contact_pending": True,
                    "hidden_contact_armed": False,
                    "hidden_contact_arm_mode": self._hidden_contact_arm_mode,
                }
            )

        # Long settling after hidden-contact creation reintroduces visible
        # condition leakage. This is legacy opt-in only and defaults to zero.
        post_arm_settle_seconds = float(
            os.environ.get("CCDA_POST_ARM_SETTLE_SECONDS", "0")
        )
        if (
            self._hidden_contact_armed
            and self.hidden_condition != "free"
            and post_arm_settle_seconds > 0
        ):
            env.start()
            time.sleep(post_arm_settle_seconds)
            env.pause()


    def _ordered_bead_xy_array(self) -> np.ndarray:
        """Return ordered bead XY for hidden-contact arming diagnostics."""
        if not self.cable_bead_IDs:
            return np.zeros((0, 2), dtype=np.float32)
        return np.asarray(
            [
                p.getBasePositionAndOrientation(int(bead_id))[0][:2]
                for bead_id in self.cable_bead_IDs
            ],
            dtype=np.float32,
        )

    def _update_hidden_arm_meta_fields(self) -> None:
        self.hidden_contact_meta["hidden_contact_pending"] = bool(
            self._hidden_contact_pending
        )
        self.hidden_contact_meta["hidden_contact_armed"] = bool(
            self._hidden_contact_armed
        )
        self.hidden_contact_meta["hidden_contact_arm_mode"] = str(
            self._hidden_contact_arm_mode
        )
        self.hidden_contact_meta["hidden_contact_arm_step"] = (
            self._hidden_contact_arm_step
        )
        self.hidden_contact_meta["hidden_contact_arm_physics_step"] = (
            self._hidden_contact_arm_physics_step
        )
        self.hidden_contact_meta["hidden_contact_arm_max_abs_jump"] = float(
            self._hidden_contact_arm_max_abs_jump
        )
        self.hidden_contact_meta["hidden_contact_arm_mae_jump"] = float(
            self._hidden_contact_arm_mae_jump
        )

    def arm_hidden_contact_after_settle(self, env=None) -> Dict[str, Any]:
        """Install latent contact at the current settled pose, exactly once.

        This method performs no physics step. Point constraints therefore use
        the current bead pose as a zero-offset world anchor, and the caller can
        separately audit immediate installation jump and short-horizon drift.
        """
        if env is None:
            env = self._ccda_env
        if env is None:
            raise RuntimeError("No environment available for hidden-contact arming")

        if self._hidden_contact_armed:
            self._update_hidden_arm_meta_fields()
            return {
                "already_armed": True,
                "condition": self.hidden_condition,
                "arm_mode": self._hidden_contact_arm_mode,
                "max_abs_jump": float(self._hidden_contact_arm_max_abs_jump),
                "mae_jump": float(self._hidden_contact_arm_mae_jump),
            }

        if self.hidden_condition == "free":
            if not self.hidden_contact_applied:
                self._apply_hidden_contact(env)
            self._hidden_contact_pending = False
            self._hidden_contact_armed = True
            self._hidden_contact_arm_mode = "free_noop"
            self._hidden_contact_arm_step = int(self._ccda_step_count)
            self._hidden_contact_arm_physics_step = int(
                self._ccda_physics_step_count
            )
            self._update_hidden_arm_meta_fields()
            return {
                "already_armed": False,
                "condition": self.hidden_condition,
                "arm_mode": self._hidden_contact_arm_mode,
                "max_abs_jump": 0.0,
                "mae_jump": 0.0,
            }

        before = self._ordered_bead_xy_array()
        self._apply_hidden_contact(env)
        after = self._ordered_bead_xy_array()

        if before.shape != after.shape:
            raise RuntimeError(
                "Bead shape changed while arming hidden contact: "
                f"{before.shape} -> {after.shape}"
            )

        diff = np.abs(after - before)
        self._hidden_contact_arm_pre_xy = before.copy()
        self._hidden_contact_arm_post_xy = after.copy()
        self._hidden_contact_arm_max_abs_jump = (
            float(np.max(diff)) if diff.size else 0.0
        )
        self._hidden_contact_arm_mae_jump = (
            float(np.mean(diff)) if diff.size else 0.0
        )
        self._hidden_contact_pending = False
        self._hidden_contact_armed = True
        self._hidden_contact_arm_mode = "deferred_zero_offset"
        self._hidden_contact_arm_step = int(self._ccda_step_count)
        self._hidden_contact_arm_physics_step = int(
            self._ccda_physics_step_count
        )
        self._update_hidden_arm_meta_fields()

        return {
            "already_armed": False,
            "condition": self.hidden_condition,
            "arm_mode": self._hidden_contact_arm_mode,
            "max_abs_jump": float(self._hidden_contact_arm_max_abs_jump),
            "mae_jump": float(self._hidden_contact_arm_mae_jump),
            "hidden_body_ids": [int(x) for x in self.hidden_body_ids],
            "hidden_constraint_ids": [int(x) for x in self.hidden_constraint_ids],
        }

    def physics_pre_step_hook(self):
        # Apply v2 unilateral tether force before the simulation step.
        if self.hidden_condition != "hidden_slack_breakaway_pin_v2":
            return
        if not self._hidden_contact_armed:
            self._update_slack_breakaway_v2_meta_fields()
            return
        if self._slack_model is None or self._slack_bead_id is None:
            raise RuntimeError(
                "slack-breakaway v2 is armed without a model or bead"
            )

        bead_id = int(self._slack_bead_id)
        position = p.getBasePositionAndOrientation(bead_id)[0]
        linear_velocity = p.getBaseVelocity(bead_id)[0]
        output = self._slack_model.evaluate(
            position,
            linear_velocity,
            physics_step=int(self._ccda_physics_step_count) + 1,
        )
        force = np.asarray(output.force_xyz, dtype=np.float64)
        if force.shape != (3,) or not np.all(np.isfinite(force)):
            raise RuntimeError("non-finite slack tether force")

        self._slack_last_output = output
        self._slack_last_applied_force = force.astype(float).tolist()

        if np.any(force != 0.0):
            p.applyExternalForce(
                bead_id,
                -1,
                forceObj=force.astype(float).tolist(),
                posObj=[float(value) for value in position],
                flags=p.WORLD_FRAME,
            )

        self._update_slack_breakaway_v2_meta_fields()

    def physics_step_hook(self):
        """Update recoverable hidden contact at every simulated physics step.

        The old implementation checked displacement only from reward(), after a
        full pick-place primitive and settling. A bead could exceed the release
        threshold transiently during the pull and return before reward(), which
        incorrectly kept the pin attached.
        """
        self._ccda_physics_step_count += 1
        self._maybe_update_breakaway(check_source="physics")
        self._update_slack_breakaway_v2_meta_fields()

    def reward(self):
        """Call original reward, then add CCDA logging fields."""
        self._ccda_step_count += 1
        # Fallback checks keep direct/manual stepping and legacy callers safe.
        self._maybe_update_breakaway(check_source="reward_pre")
        reward, extras = super().reward()
        self._maybe_update_breakaway(check_source="reward_post")
        extras.update(self._ccda_extras())
        return reward, extras

    def _robot_pose_proxy(self):
        """Return a fixed-format robot proprio/proxy dict.

        DeformableRavens uses high-level pick-place primitives, so this is not
        full robot proprioception. We log the strongest available proxy:
          1. PyBullet joint positions/velocities for env.ur5 if available.
          2. End-effector/tool link pose if link state is accessible.
          3. Otherwise a fixed zero vector with source='missing_zero_proxy'.

        This must never include hidden_contact metadata.
        """
        env = getattr(self, "_ccda_env", None)
        candidate_body_attrs = ["ur5", "ur5_id", "robot_id", "robot"]
        body_id = None
        if env is not None:
            for attr in candidate_body_attrs:
                value = getattr(env, attr, None)
                if isinstance(value, (int, np.integer)):
                    body_id = int(value)
                    break

        if body_id is not None:
            try:
                n_joints = int(p.getNumJoints(body_id))
                joint_positions = []
                joint_velocities = []
                for j in range(n_joints):
                    js = p.getJointState(body_id, j)
                    joint_positions.append(float(js[0]))
                    joint_velocities.append(float(js[1]))

                ee_position = [0.0, 0.0, 0.0]
                ee_orientation = [0.0, 0.0, 0.0, 1.0]
                if n_joints > 0:
                    try:
                        link_state = p.getLinkState(body_id, n_joints - 1)
                        ee_position = [float(v) for v in link_state[0]]
                        ee_orientation = [float(v) for v in link_state[1]]
                    except Exception:
                        pass

                return {
                    "source": "pybullet_robot_body",
                    "body_id": int(body_id),
                    "joint_positions": joint_positions,
                    "joint_velocities": joint_velocities,
                    "ee_position": ee_position,
                    "ee_orientation": ee_orientation,
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

    def _ccda_extras(self) -> Dict[str, Any]:
        self._update_hidden_arm_meta_fields()
        self._update_breakaway_meta_fields()
        self._update_slack_breakaway_v2_meta_fields()
        bead_states = self._ordered_bead_states()
        return {
            "ccda_task": self._name,
            "hidden_condition": self.hidden_condition,
            "ccda_visible_seed": self._safe_int_or_str(self.ccda_visible_seed),
            "ccda_pair_group": self.ccda_pair_group,
            "hidden_contact_applied": self.hidden_contact_applied,
            "hidden_contact_meta": self.hidden_contact_meta,
            "robot_pose_proxy": self._robot_pose_proxy(),
            "bead_ids": [int(x["id"]) for x in bead_states],
            "bead_positions": [x["position"] for x in bead_states],
            "bead_orientations": [x["orientation"] for x in bead_states],
            "bead_velocities": [x["linear_velocity"] for x in bead_states],
        }

    @staticmethod
    def _safe_int_or_str(value):
        try:
            return int(value)
        except Exception:
            return value

    def _ordered_bead_states(self) -> List[Dict[str, Any]]:
        states = []
        for local_idx, bead_id in enumerate(self.cable_bead_IDs):
            pos, orn = p.getBasePositionAndOrientation(bead_id)
            lin_vel, ang_vel = p.getBaseVelocity(bead_id)
            states.append(
                {
                    "local_index": int(local_idx),
                    "id": int(bead_id),
                    "position": [float(v) for v in pos],
                    "orientation": [float(v) for v in orn],
                    "linear_velocity": [float(v) for v in lin_vel],
                    "angular_velocity": [float(v) for v in ang_vel],
                }
            )
        return states

    def _recoverability_class_candidate(self, condition: str) -> str:
        if condition == "free":
            return "free"
        if condition == "hidden_pin":
            return "hard_impossible_candidate"
        if condition == "hidden_high_friction":
            return "weak_candidate"
        return "recoverable_candidate"

    def _base_recoverability_params(self, condition: str) -> Dict[str, Any]:
        return {
            "condition": condition,
            "class_candidate": self._recoverability_class_candidate(condition),
        }

    def _apply_hidden_contact(self, env) -> None:
        if len(self.cable_bead_IDs) == 0:
            raise RuntimeError("No cable beads found. Did add_cable() run?")

        self._apply_hidden_contact_condition(env, self.hidden_condition)

        self.hidden_contact_meta["hidden_body_ids"] = [int(x) for x in self.hidden_body_ids]
        self.hidden_contact_meta["hidden_constraint_ids"] = [
            int(x) for x in self.hidden_constraint_ids
        ]
        self.hidden_contact_meta["recoverability_class_candidate"] = self._recoverability_class_candidate(self.hidden_condition)
        self.hidden_contact_meta.setdefault("recoverability_params", self._base_recoverability_params(self.hidden_condition))
        self.hidden_contact_meta["applied"] = True
        self.hidden_contact_applied = True

    def _apply_hidden_contact_condition(self, env, condition: str) -> None:
        if condition == "free":
            return self._apply_free(env)
        if condition == "hidden_pin":
            return self._apply_hidden_pin(mode="hard")
        if condition == "hidden_high_friction":
            return self._apply_hidden_high_friction(scale=8.0, span=5)
        if condition == "hidden_partial_pin":
            return self._apply_hidden_pin(mode="partial")
        if condition == "hidden_soft_pin":
            return self._apply_hidden_pin(mode="soft")
        if condition == "hidden_breakaway_pin":
            return self._apply_hidden_breakaway_pin()
        if condition == "hidden_slack_breakaway_pin_v2":
            return self._apply_hidden_slack_breakaway_pin_v2()
        if condition == "hidden_friction_patch":
            return self._apply_hidden_friction_patch()
        raise ValueError("unknown hidden contact condition: {}".format(condition))

    def _apply_free(self, env) -> None:
        self.hidden_contact_meta.update(
            {
                "applied": True,
                "condition": "free",
                "note": "No hidden contact injected.",
                "recoverability_class_candidate": "free",
                "recoverability_params": self._base_recoverability_params("free"),
            }
        )

    def _middle_bead_index(self) -> int:
        return int(len(self.cable_bead_IDs) // 2)

    def _bead_position(self, bead_id: int) -> Tuple[float, float, float]:
        return p.getBasePositionAndOrientation(bead_id)[0]

    def _make_invisible_anchor(self, position, radius=0.002) -> int:
        collision = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
        body = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=-1,
            basePosition=position,
            baseOrientation=(0, 0, 0, 1),
        )
        self.hidden_body_ids.append(body)
        return body

    def _pin_indices_for_mode(self, mode: str) -> List[int]:
        n = len(self.cable_bead_IDs)
        if mode == "partial":
            ratio = float(os.environ.get("CCDA_PARTIAL_PIN_BEAD_RATIO", "0.45"))
            center = int(np.clip(round((n - 1) * ratio), 0, n - 1))
            span = max(1, int(os.environ.get("CCDA_PARTIAL_PIN_SPAN", "1")))
        else:
            center = self._middle_bead_index()
            span = 1
        lo = max(0, center - span // 2)
        hi = min(n, lo + span)
        lo = max(0, hi - span)
        return list(range(lo, hi))

    def _pin_force_for_mode(self, mode: str) -> float:
        if mode == "hard":
            return 80.0
        if mode == "partial":
            return float(os.environ.get("CCDA_PARTIAL_PIN_FORCE", "2.0"))
        if mode == "soft":
            return float(os.environ.get("CCDA_SOFT_PIN_FORCE", "0.8"))
        if mode == "breakaway":
            return float(os.environ.get("CCDA_BREAKAWAY_FORCE", "1.2"))
        raise ValueError(mode)

    def _update_breakaway_meta_fields(self) -> None:
        if self.hidden_condition != "hidden_breakaway_pin":
            return
        self.hidden_contact_meta["breakaway_released"] = bool(
            self._breakaway_released
        )
        self.hidden_contact_meta["breakaway_release_step"] = (
            self._breakaway_release_step
        )
        self.hidden_contact_meta["breakaway_release_physics_step"] = (
            self._breakaway_release_physics_step
        )
        self.hidden_contact_meta["breakaway_max_disp_seen"] = float(
            self._breakaway_max_disp_seen
        )
        self.hidden_contact_meta["ccda_physics_step_count"] = int(
            self._ccda_physics_step_count
        )

    def _maybe_update_breakaway(self, check_source="reward") -> None:
        if self.hidden_condition != "hidden_breakaway_pin":
            return
        if not self._hidden_contact_armed:
            self._update_breakaway_meta_fields()
            return
        if self._breakaway_released:
            self._update_breakaway_meta_fields()
            return
        if self._breakaway_bead_id is None or self._breakaway_anchor_pos is None:
            self._update_breakaway_meta_fields()
            return

        try:
            bead_pos = np.asarray(
                p.getBasePositionAndOrientation(
                    int(self._breakaway_bead_id)
                )[0],
                dtype=np.float32,
            )
            anchor = np.asarray(
                self._breakaway_anchor_pos, dtype=np.float32
            )
            disp = float(np.linalg.norm((bead_pos - anchor)[:2]))
        except Exception:
            self._update_breakaway_meta_fields()
            return

        self._breakaway_max_disp_seen = max(
            float(self._breakaway_max_disp_seen), disp
        )
        threshold = float(
            os.environ.get("CCDA_BREAKAWAY_DISP", "0.035")
        )
        min_physics_steps = int(
            os.environ.get("CCDA_BREAKAWAY_MIN_PHYSICS_STEPS", "1")
        )

        # reward() may run before the background thread has advanced. Keep the
        # old action-step fallback only when explicitly requested; normal task
        # execution releases according to actual physics-step displacement.
        physics_ready = (
            self._ccda_physics_step_count >= min_physics_steps
        )
        reward_fallback = (
            str(check_source).startswith("reward")
            and self._ccda_step_count
            >= int(os.environ.get("CCDA_BREAKAWAY_MIN_STEP", "1"))
        )
        if disp < threshold or not (physics_ready or reward_fallback):
            self._update_breakaway_meta_fields()
            return

        released_constraint_id = self._breakaway_constraint_id
        try:
            if released_constraint_id is not None:
                p.removeConstraint(int(released_constraint_id))
        except Exception:
            # If it was already absent, treat the latent contact as released.
            pass

        self.hidden_constraint_ids = [
            int(cid)
            for cid in self.hidden_constraint_ids
            if int(cid) != int(released_constraint_id)
        ] if released_constraint_id is not None else list(
            self.hidden_constraint_ids
        )
        self._breakaway_constraint_id = None
        self._breakaway_released = True
        self._breakaway_release_step = int(self._ccda_step_count)
        self._breakaway_release_physics_step = int(
            self._ccda_physics_step_count
        )
        self.hidden_contact_meta["breakaway_release_source"] = str(
            check_source
        )
        self._update_breakaway_meta_fields()

    def _apply_hidden_breakaway_pin(self) -> None:
        bead_ratio = float(os.environ.get("CCDA_BREAKAWAY_BEAD_RATIO", "0.45"))
        bead_ratio = min(max(bead_ratio, 0.05), 0.95)
        idx = int(round(bead_ratio * (len(self.cable_bead_IDs) - 1)))
        idx = max(0, min(len(self.cable_bead_IDs) - 1, idx))

        bead_id = int(self.cable_bead_IDs[idx])
        bead_pos = self._bead_position(bead_id)
        max_force = float(os.environ.get("CCDA_BREAKAWAY_FORCE", "1.2"))
        disp = float(os.environ.get("CCDA_BREAKAWAY_DISP", "0.035"))
        damping = float(os.environ.get("CCDA_BREAKAWAY_DAMPING", "0.0"))

        cid = p.createConstraint(
            parentBodyUniqueId=bead_id,
            parentLinkIndex=-1,
            childBodyUniqueId=-1,
            childLinkIndex=-1,
            jointType=p.JOINT_POINT2POINT,
            jointAxis=(0, 0, 0),
            parentFramePosition=(0, 0, 0),
            childFramePosition=bead_pos,
        )
        p.changeConstraint(cid, maxForce=max_force)
        if damping > 0:
            p.changeDynamics(bead_id, -1, linearDamping=damping, angularDamping=damping)

        self.hidden_constraint_ids.append(int(cid))
        self._breakaway_bead_id = int(bead_id)
        self._breakaway_constraint_id = int(cid)
        self._breakaway_anchor_pos = [float(x) for x in bead_pos]
        self._breakaway_released = False
        self._breakaway_release_step = None
        self._breakaway_release_physics_step = None
        self._breakaway_max_disp_seen = 0.0

        params = self._base_recoverability_params("hidden_breakaway_pin")
        params.update(
            {
                "breakaway_force": float(max_force),
                "breakaway_disp": float(disp),
                "breakaway_bead_ratio": float(bead_ratio),
                "breakaway_damping": float(damping),
            }
        )
        self.hidden_contact_meta.update(
            {
                "condition": "hidden_breakaway_pin",
                "pin_mode": "breakaway",
                "pin_bead_local_index": int(idx),
                "pin_bead_id": int(bead_id),
                "pin_anchor_position": [float(x) for x in bead_pos],
                "pin_position": [float(x) for x in bead_pos],
                "constraint_id": int(cid),
                "max_force": float(max_force),
                "breakaway_force": float(max_force),
                "breakaway_disp": float(disp),
                "breakaway_released": False,
                "breakaway_release_step": None,
                "breakaway_release_physics_step": None,
                "ccda_physics_step_count": int(
                    self._ccda_physics_step_count
                ),
                "breakaway_max_disp_seen": 0.0,
                "recoverability_class_candidate": "recoverable_candidate",
                "recoverability_params": params,
            }
        )

    def _slack_breakaway_v2_config(self) -> SlackBreakawayConfig:
        return SlackBreakawayConfig(
            slack_distance=float(
                os.environ.get("CCDA_SLACK_V2_DISTANCE", "0.010")
            ),
            spring_stiffness=float(
                os.environ.get("CCDA_SLACK_V2_STIFFNESS", "100.0")
            ),
            radial_damping=float(
                os.environ.get("CCDA_SLACK_V2_DAMPING", "0.20")
            ),
            max_tension=float(
                os.environ.get("CCDA_SLACK_V2_MAX_TENSION", "4.0")
            ),
            breakaway_extension=float(
                os.environ.get(
                    "CCDA_SLACK_V2_BREAKAWAY_EXTENSION",
                    "0.030",
                )
            ),
            breakaway_force=float(
                os.environ.get(
                    "CCDA_SLACK_V2_BREAKAWAY_FORCE",
                    "3.0",
                )
            ),
        )

    def _update_slack_breakaway_v2_meta_fields(self) -> None:
        if self.hidden_condition != "hidden_slack_breakaway_pin_v2":
            return

        model = self._slack_model
        snapshot = model.snapshot() if model is not None else None
        self.hidden_contact_meta.update(
            {
                "environment_semantics_version":
                    self._slack_environment_semantics_version,
                "condition": "hidden_slack_breakaway_pin_v2",
                "force_model": "unilateral_deadband_spring",
                "uses_world_constraint": False,
                "contains_no_action_force_below_deadband": True,
                "slack_bead_id": self._slack_bead_id,
                "slack_bead_local_index": self._slack_bead_local_index,
                "slack_state": (
                    snapshot["state"] if snapshot is not None else None
                ),
                "slack_anchor_xy": (
                    snapshot["anchor_xy"] if snapshot is not None else None
                ),
                "slack_engagement_physics_step": (
                    snapshot["engagement_physics_step"]
                    if snapshot is not None
                    else None
                ),
                "slack_release_physics_step": (
                    snapshot["release_physics_step"]
                    if snapshot is not None
                    else None
                ),
                "slack_release_reason": (
                    snapshot["release_reason"]
                    if snapshot is not None
                    else None
                ),
                "slack_max_radial_distance": (
                    snapshot["max_radial_distance"]
                    if snapshot is not None
                    else 0.0
                ),
                "slack_max_extension": (
                    snapshot["max_extension"]
                    if snapshot is not None
                    else 0.0
                ),
                "slack_max_tension": (
                    snapshot["max_tension"]
                    if snapshot is not None
                    else 0.0
                ),
                "slack_last_tension": (
                    snapshot["last_tension"]
                    if snapshot is not None
                    else 0.0
                ),
                "slack_last_force": list(
                    self._slack_last_applied_force
                ),
                "hidden_body_ids": [
                    int(value) for value in self.hidden_body_ids
                ],
                "hidden_constraint_ids": [
                    int(value) for value in self.hidden_constraint_ids
                ],
            }
        )

    def _apply_hidden_slack_breakaway_pin_v2(self) -> None:
        if self.hidden_body_ids or self.hidden_constraint_ids:
            raise RuntimeError(
                "slack-breakaway v2 must not create hidden bodies or "
                "PyBullet constraints"
            )

        bead_ratio = float(
            os.environ.get("CCDA_SLACK_V2_BEAD_RATIO", "0.45")
        )
        bead_ratio = min(max(bead_ratio, 0.05), 0.95)
        index = int(
            round(bead_ratio * (len(self.cable_bead_IDs) - 1))
        )
        index = max(0, min(len(self.cable_bead_IDs) - 1, index))
        bead_id = int(self.cable_bead_IDs[index])
        bead_position = self._bead_position(bead_id)

        self._slack_bead_id = bead_id
        self._slack_bead_local_index = index
        self._slack_model = UnilateralSlackBreakaway(
            config=self._slack_breakaway_v2_config(),
            anchor_position=bead_position,
        )
        self._slack_last_output = None
        self._slack_last_applied_force = [0.0, 0.0, 0.0]

        parameters = self._base_recoverability_params(
            "hidden_slack_breakaway_pin_v2"
        )
        parameters.update(self._slack_model.snapshot()["config"])

        self.hidden_contact_meta.update(
            {
                "condition": "hidden_slack_breakaway_pin_v2",
                "recoverability_class_candidate":
                    "recoverable_candidate",
                "recoverability_params": parameters,
                "environment_semantics_version":
                    self._slack_environment_semantics_version,
                "force_model": "unilateral_deadband_spring",
                "uses_world_constraint": False,
                "contains_no_action_force_below_deadband": True,
            }
        )
        self._update_slack_breakaway_v2_meta_fields()

    def ccda_snapshot_state(self) -> Dict[str, Any]:
        # Capture Python-side state not included in p.saveState().
        return {
            "snapshot_version": "ccda_task_snapshot_v2",
            "hidden_condition": str(self.hidden_condition),
            "ccda_step_count": int(self._ccda_step_count),
            "ccda_physics_step_count": int(
                self._ccda_physics_step_count
            ),
            "hidden_contact_pending": bool(
                self._hidden_contact_pending
            ),
            "hidden_contact_armed": bool(
                self._hidden_contact_armed
            ),
            "slack_bead_id": self._slack_bead_id,
            "slack_bead_local_index": self._slack_bead_local_index,
            "slack_last_applied_force": list(
                self._slack_last_applied_force
            ),
            "slack_model": (
                self._slack_model.snapshot()
                if self._slack_model is not None
                else None
            ),
        }

    def ccda_restore_state(self, snapshot: Dict[str, Any]) -> None:
        # Restore Python-side state after p.restoreState().
        if snapshot.get("snapshot_version") != "ccda_task_snapshot_v2":
            raise ValueError("unsupported CCDA task snapshot")
        if str(snapshot["hidden_condition"]) != str(
            self.hidden_condition
        ):
            raise ValueError("snapshot hidden condition mismatch")

        self._ccda_step_count = int(snapshot["ccda_step_count"])
        self._ccda_physics_step_count = int(
            snapshot["ccda_physics_step_count"]
        )
        self._hidden_contact_pending = bool(
            snapshot["hidden_contact_pending"]
        )
        self._hidden_contact_armed = bool(
            snapshot["hidden_contact_armed"]
        )
        self._slack_bead_id = snapshot.get("slack_bead_id")
        self._slack_bead_local_index = snapshot.get(
            "slack_bead_local_index"
        )
        self._slack_last_applied_force = [
            float(value)
            for value in snapshot.get(
                "slack_last_applied_force",
                [0.0, 0.0, 0.0],
            )
        ]

        model_snapshot = snapshot.get("slack_model")
        self._slack_model = (
            UnilateralSlackBreakaway.from_snapshot(model_snapshot)
            if model_snapshot is not None
            else None
        )
        self._slack_last_output = None
        self._update_hidden_arm_meta_fields()
        self._update_slack_breakaway_v2_meta_fields()

    def _apply_hidden_pin(self, mode: str = "hard") -> None:
        indices = self._pin_indices_for_mode(mode)
        max_force = self._pin_force_for_mode(mode)
        damping = float(os.environ.get("CCDA_SOFT_PIN_DAMPING", "0.2")) if mode in ("soft", "breakaway") else 0.0
        pin_records = []

        for idx in indices:
            bead_id = self.cable_bead_IDs[idx]
            bead_pos = self._bead_position(bead_id)
            try:
                cid = p.createConstraint(
                    parentBodyUniqueId=bead_id,
                    parentLinkIndex=-1,
                    childBodyUniqueId=-1,
                    childLinkIndex=-1,
                    jointType=p.JOINT_POINT2POINT,
                    jointAxis=(0, 0, 0),
                    parentFramePosition=(0, 0, 0),
                    childFramePosition=bead_pos,
                )
                p.changeConstraint(cid, maxForce=max_force)
            except Exception:
                anchor = self._make_invisible_anchor(bead_pos)
                cid = p.createConstraint(
                    parentBodyUniqueId=bead_id,
                    parentLinkIndex=-1,
                    childBodyUniqueId=anchor,
                    childLinkIndex=-1,
                    jointType=p.JOINT_POINT2POINT,
                    jointAxis=(0, 0, 0),
                    parentFramePosition=(0, 0, 0),
                    childFramePosition=(0, 0, 0),
                )
                p.changeConstraint(cid, maxForce=max_force)
            if damping > 0:
                p.changeDynamics(bead_id, -1, linearDamping=damping, angularDamping=damping)
            self.hidden_constraint_ids.append(cid)
            pin_records.append(
                {
                    "pin_bead_local_index": int(idx),
                    "pin_bead_id": int(bead_id),
                    "pin_position": [float(v) for v in bead_pos],
                    "constraint_id": int(cid),
                }
            )

        condition = "hidden_pin" if mode == "hard" else "hidden_{}_pin".format(mode)
        if mode == "breakaway":
            condition = "hidden_breakaway_pin"
        if mode == "partial":
            condition = "hidden_partial_pin"
        if mode == "soft":
            condition = "hidden_soft_pin"

        params = self._base_recoverability_params(condition)
        params.update(
            {
                "pin_mode": mode,
                "max_force": float(max_force),
                "damping": float(damping),
                "pin_span": int(len(indices)),
                "breakaway_force": float(os.environ.get("CCDA_BREAKAWAY_FORCE", "1.2")),
                "breakaway_disp": float(os.environ.get("CCDA_BREAKAWAY_DISP", "0.03")),
                "soft_breakaway_approximation": bool(mode == "breakaway"),
            }
        )
        self.hidden_contact_meta.update(
            {
                "condition": condition,
                "pin_mode": mode,
                "pins": pin_records,
                "pin_bead_local_index": int(pin_records[0]["pin_bead_local_index"]),
                "pin_bead_id": int(pin_records[0]["pin_bead_id"]),
                "pin_position": pin_records[0]["pin_position"],
                "max_force": float(max_force),
                "recoverability_class_candidate": self._recoverability_class_candidate(condition),
                "recoverability_params": params,
                "breakaway_released": False,
                "breakaway_step": None,
            }
        )

    def _apply_hidden_high_friction(self, scale: float = 8.0, span: int = 5) -> None:
        mid = self._middle_bead_index()
        span = max(1, int(span))
        lo = max(0, mid - span // 2)
        hi = min(len(self.cable_bead_IDs), lo + span)
        lo = max(0, hi - span)
        selected = self.cable_bead_IDs[lo:hi]

        for bead_id in selected:
            p.changeDynamics(
                bead_id,
                -1,
                lateralFriction=float(scale),
                spinningFriction=max(0.2, float(scale) * 0.125),
                rollingFriction=max(0.1, float(scale) * 0.0625),
                linearDamping=min(1.0, max(0.1, float(scale) * 0.1)),
                angularDamping=min(1.0, max(0.1, float(scale) * 0.1)),
            )

        params = self._base_recoverability_params("hidden_high_friction")
        params.update({"friction_scale": float(scale), "friction_span": int(span)})
        self.hidden_contact_meta.update(
            {
                "condition": "hidden_high_friction",
                "friction_bead_local_indices": list(range(lo, hi)),
                "friction_bead_ids": [int(x) for x in selected],
                "lateralFriction": float(scale),
                "spinningFriction": max(0.2, float(scale) * 0.125),
                "rollingFriction": max(0.1, float(scale) * 0.0625),
                "recoverability_class_candidate": "weak_candidate",
                "recoverability_params": params,
            }
        )

    def _apply_hidden_friction_patch(self) -> None:
        scale = float(os.environ.get("CCDA_FRICTION_PATCH_SCALE", "2.5"))
        span = int(os.environ.get("CCDA_FRICTION_PATCH_SPAN", "3"))
        mid = self._middle_bead_index()
        span = max(1, int(span))
        lo = max(0, mid - span // 2)
        hi = min(len(self.cable_bead_IDs), lo + span)
        lo = max(0, hi - span)
        selected = self.cable_bead_IDs[lo:hi]
        for bead_id in selected:
            p.changeDynamics(
                bead_id,
                -1,
                lateralFriction=float(scale),
                spinningFriction=max(0.1, float(scale) * 0.08),
                rollingFriction=max(0.05, float(scale) * 0.04),
                linearDamping=min(0.8, max(0.05, float(scale) * 0.08)),
                angularDamping=min(0.8, max(0.05, float(scale) * 0.08)),
            )
        params = self._base_recoverability_params("hidden_friction_patch")
        params.update({"friction_patch_scale": float(scale), "friction_patch_span": int(span)})
        self.hidden_contact_meta.update(
            {
                "condition": "hidden_friction_patch",
                "friction_patch_bead_local_indices": list(range(lo, hi)),
                "friction_patch_bead_ids": [int(x) for x in selected],
                "lateralFriction": float(scale),
                "recoverability_class_candidate": "recoverable_candidate",
                "recoverability_params": params,
            }
        )
