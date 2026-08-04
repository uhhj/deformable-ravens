"""Hidden-friction cable task for Phase 0 CCDA physical auditing."""
from __future__ import annotations

import copy
import os
from typing import Any, Dict, List

import numpy as np
import pybullet as p

from ravens.tasks.ccda_hidden_friction import (
    HiddenFrictionConfig,
    HiddenPlanarFrictionPatch,
)
from ravens.tasks.defs_cables import CableLineNoTarget


class CCDAHiddenFrictionCable(CableLineNoTarget):
    CONDITIONS = ("free", "hidden_high_friction")
    ENVIRONMENT_VERSION = "ccda_hidden_friction_cable_v1"
    VALID_PHASES = ("no_action", "preload", "main_pull", "post_main")

    def __init__(self):
        super().__init__()
        self._name = "ccda-hidden-friction-cable"
        self.metric = "cable-target"
        self.primitive = "pick_place"
        self.ee = "suction"
        self._env = None
        self._reset_ccda_state()

    @staticmethod
    def _env_float(name: str, default: str) -> float:
        value = float(os.environ.get(name, default))
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value

    @staticmethod
    def _env_int(name: str, default: str) -> int:
        text = os.environ.get(name, default)
        value = int(text)
        if str(value) != text.strip() and text.strip() not in {f"+{value}", f"-{abs(value)}"}:
            raise ValueError(f"{name} must be an integer")
        return value

    def _reset_ccda_state(self) -> None:
        self.hidden_condition = "free"
        self.ccda_visible_seed = ""
        self.ccda_pair_group = ""
        self._physics_step_count = 0
        self._phase = "no_action"
        self._trace: List[Dict[str, Any]] = []
        self._trace_stride = 4
        self._hidden_friction_pending = True
        self._hidden_friction_armed = False
        self._friction_model = None
        self._patch_center_xy = np.zeros(2, dtype=np.float64)
        self._selected_local_indices: List[int] = []
        self._selected_bead_ids: List[int] = []
        self._arm_max_abs_jump = 0.0
        self._arm_mae_jump = 0.0
        self._last_contact = self._zero_contact()

    @staticmethod
    def _zero_contact() -> Dict[str, Any]:
        return {
            "force_xyz": [0.0, 0.0, 0.0],
            "force_norm": 0.0,
            "max_force_norm": 0.0,
            "active_beads": 0,
            "mean_speed": 0.0,
        }

    def reset(self, env, last_info=None):
        self._env = env
        self._reset_ccda_state()
        self.hidden_condition = os.environ.get("CCDA_HIDDEN_CONDITION", "free")
        self.ccda_visible_seed = os.environ.get("CCDA_VISIBLE_SEED", "")
        self.ccda_pair_group = os.environ.get("CCDA_PAIR_GROUP", "")
        if self.hidden_condition not in self.CONDITIONS:
            raise ValueError(
                f"unknown hidden condition {self.hidden_condition!r}; "
                f"expected one of {self.CONDITIONS}"
            )
        self._trace_stride = self._env_int("CCDA_TRACE_STRIDE", "4")
        if self._trace_stride <= 0:
            raise ValueError("CCDA_TRACE_STRIDE must be positive")

        formal_name = self._name
        self._name = "cable-line-notarget"
        try:
            super().reset(env, last_info=last_info)
        finally:
            self._name = formal_name
        # Environment.reset() calls arm_hidden_friction_after_settle() only
        # after its visible, force-free settling step has completed.
        self._hidden_friction_pending = True

    def _ordered_bead_positions(self) -> np.ndarray:
        return np.asarray(
            [p.getBasePositionAndOrientation(int(bead))[0] for bead in self.cable_bead_IDs],
            dtype=np.float64,
        )

    def _friction_config(self) -> HiddenFrictionConfig:
        return HiddenFrictionConfig(
            patch_radius=self._env_float("CCDA_FRICTION_PATCH_RADIUS", "0.045"),
            viscous_gain=self._env_float("CCDA_FRICTION_VISCOUS_GAIN", "1.5"),
            coulomb_force=self._env_float("CCDA_FRICTION_COULOMB_FORCE", "0.12"),
            max_force=self._env_float("CCDA_FRICTION_MAX_FORCE", "0.75"),
            speed_epsilon=self._env_float("CCDA_FRICTION_SPEED_EPSILON", "0.0001"),
            contact_height=self._env_float("CCDA_FRICTION_CONTACT_HEIGHT", "0.03"),
        )

    def arm_hidden_friction_after_settle(self) -> Dict[str, Any]:
        if self._hidden_friction_armed:
            return {"already_armed": True}
        before = self._ordered_bead_positions()
        if before.shape[0] == 0:
            raise RuntimeError("cannot arm hidden friction without cable beads")
        ratio = self._env_float("CCDA_FRICTION_CENTER_RATIO", "0.45")
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("CCDA_FRICTION_CENTER_RATIO must lie in [0, 1]")
        selected_count = self._env_int("CCDA_FRICTION_SELECTED_COUNT", "5")
        if selected_count <= 0 or selected_count % 2 != 1:
            raise ValueError("CCDA_FRICTION_SELECTED_COUNT must be a positive odd integer")
        if selected_count > before.shape[0]:
            raise ValueError("CCDA_FRICTION_SELECTED_COUNT exceeds bead count")

        center_index = int(np.clip(round(ratio * (before.shape[0] - 1)), 0, before.shape[0] - 1))
        start = center_index - selected_count // 2
        start = int(np.clip(start, 0, before.shape[0] - selected_count))
        self._selected_local_indices = list(range(start, start + selected_count))
        self._selected_bead_ids = [int(self.cable_bead_IDs[index]) for index in self._selected_local_indices]
        self._patch_center_xy = np.mean(before[self._selected_local_indices, :2], axis=0)
        self._friction_model = HiddenPlanarFrictionPatch(
            config=self._friction_config(),
            center_xy=self._patch_center_xy,
            enabled=self.hidden_condition == "hidden_high_friction",
        )
        after = self._ordered_bead_positions()
        jump = np.abs(after - before)
        self._arm_max_abs_jump = float(np.max(jump)) if jump.size else 0.0
        self._arm_mae_jump = float(np.mean(jump)) if jump.size else 0.0
        self._hidden_friction_pending = False
        self._hidden_friction_armed = True
        self._last_contact = self._zero_contact()
        return {
            "already_armed": False,
            "arm_max_abs_jump": self._arm_max_abs_jump,
            "arm_mae_jump": self._arm_mae_jump,
        }

    def physics_pre_step_hook(self) -> None:
        self._last_contact = self._zero_contact()
        if not self._hidden_friction_armed:
            return
        if self._friction_model is None:
            raise RuntimeError("armed hidden-friction task has no friction model")

        sum_force = np.zeros(3, dtype=np.float64)
        force_norms = []
        speeds = []
        active_count = 0
        for bead_id in self._selected_bead_ids:
            position = p.getBasePositionAndOrientation(bead_id)[0]
            velocity = p.getBaseVelocity(bead_id)[0]
            output = self._friction_model.evaluate(position, velocity)
            force = np.asarray(output.force_xyz, dtype=np.float64)
            sum_force += force
            force_norms.append(float(output.force_norm))
            speeds.append(float(output.planar_speed))
            active_count += int(output.active)
            if np.any(force != 0.0):
                p.applyExternalForce(
                    bead_id,
                    -1,
                    forceObj=force.tolist(),
                    posObj=[float(value) for value in position],
                    flags=p.WORLD_FRAME,
                )

        self._last_contact = {
            "force_xyz": [float(value) for value in sum_force],
            "force_norm": float(np.sum(force_norms)) if force_norms else 0.0,
            "max_force_norm": float(np.max(force_norms)) if force_norms else 0.0,
            "active_beads": int(active_count),
            "mean_speed": float(np.mean(speeds)) if speeds else 0.0,
        }
        if self.hidden_condition == "free" and any(
            value != 0.0 for value in self._last_contact["force_xyz"]
        ):
            raise RuntimeError("free condition produced non-zero hidden force")

    def _robot_state(self) -> Dict[str, Any]:
        if self._env is None:
            return {
                "joint_positions": [],
                "joint_velocities": [],
                "ee_position": [0.0, 0.0, 0.0],
                "ee_orientation": [0.0, 0.0, 0.0, 1.0],
            }
        states = [p.getJointState(self._env.ur5, int(joint)) for joint in self._env.joints]
        ee_state = p.getLinkState(self._env.ur5, self._env.ee_tip_link)
        return {
            "joint_positions": [float(state[0]) for state in states],
            "joint_velocities": [float(state[1]) for state in states],
            "ee_position": [float(value) for value in ee_state[0]],
            "ee_orientation": [float(value) for value in ee_state[1]],
        }

    def physics_step_hook(self) -> None:
        self._physics_step_count += 1
        if not self._hidden_friction_armed or self._physics_step_count % self._trace_stride != 0:
            return
        positions = self._ordered_bead_positions()
        velocities = np.asarray(
            [p.getBaseVelocity(int(bead))[0] for bead in self.cable_bead_IDs],
            dtype=np.float64,
        )
        robot = self._robot_state()
        contact = self.ccda_contact_observation()
        self._trace.append(
            {
                "physics_step": int(self._physics_step_count),
                "phase": self._phase,
                "bead_positions": positions.tolist(),
                "bead_velocities": velocities.tolist(),
                "joint_positions": robot["joint_positions"],
                "joint_velocities": robot["joint_velocities"],
                "ee_position": robot["ee_position"],
                "ee_orientation": robot["ee_orientation"],
                "contact_force_xyz": contact["force_xyz"],
                "contact_force_norm": contact["force_norm"],
                "contact_max_force_norm": contact["max_force_norm"],
                "contact_active_beads": contact["active_beads"],
                "contact_mean_speed": contact["mean_speed"],
            }
        )

    def set_ccda_phase(self, phase: str) -> None:
        if phase not in self.VALID_PHASES:
            raise ValueError(f"invalid CCDA phase {phase!r}; expected one of {self.VALID_PHASES}")
        self._phase = phase

    def ccda_trace(self) -> List[Dict[str, Any]]:
        return copy.deepcopy(self._trace)

    def physics_step_count(self) -> int:
        return int(self._physics_step_count)

    def ccda_contact_observation(self) -> Dict[str, Any]:
        return copy.deepcopy(self._last_contact)

    def ccda_privileged_state(self) -> Dict[str, Any]:
        if self._friction_model is None:
            raise RuntimeError("hidden friction has not been armed")
        return {
            "environment_version": self.ENVIRONMENT_VERSION,
            "hidden_condition": self.hidden_condition,
            "patch_center_xy": [float(value) for value in self._patch_center_xy],
            "selected_local_indices": list(self._selected_local_indices),
            "selected_bead_ids": list(self._selected_bead_ids),
            "friction_model": self._friction_model.snapshot(),
            "arm_max_abs_jump": float(self._arm_max_abs_jump),
            "arm_mae_jump": float(self._arm_mae_jump),
            "visible_seed": self.ccda_visible_seed,
            "pair_group": self.ccda_pair_group,
        }

    def reward(self):
        reward, extras = super().reward()
        extras["ccda_environment_version"] = self.ENVIRONMENT_VERSION
        extras["ccda_phase"] = self._phase
        extras["ccda_physics_step"] = self._physics_step_count
        extras["ccda_contact_observation"] = self.ccda_contact_observation()
        return reward, extras
