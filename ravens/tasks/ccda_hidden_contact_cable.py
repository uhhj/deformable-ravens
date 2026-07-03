#!/usr/bin/env python
"""CCDA hidden-contact cable tasks.

This task is a Phase1 fork of cable-line-notarget. It keeps the visible
goal and the original pick-place oracle structure, but injects hidden
contact conditions after reset. Hidden contact metadata and ordered bead
states are added to info['extras'] through reward().
"""

import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pybullet as p

from ravens.tasks.defs_cables import CableLineNoTarget


class HiddenContactCableLine(CableLineNoTarget):
    """Cable-line-notarget with controlled hidden contact conditions.

    Supported hidden conditions:
      - free: no hidden contact.
      - hidden_pin: one bead is constrained to world / invisible anchor.
      - hidden_high_friction: several beads get high friction/damping.

    The hidden geometry is not added to env.objects or env.fixed_objects,
    so it is not returned by Environment.info. Collision-only bodies have
    no visual shape and should not appear in RGB renderings.
    """

    CONDITIONS = (
        "free",
        "hidden_pin",
        "hidden_high_friction",
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

        super().reset(env, last_info=last_info)

        # Apply hidden contact after cable creation.
        self._apply_hidden_contact(env)

        # Let contacts settle briefly. Keep this short for smoke tests.
        env.start()
        time.sleep(0.25)
        env.pause()

    def reward(self):
        """Call original reward, then add CCDA logging fields."""
        reward, extras = super().reward()
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

    def _apply_hidden_contact(self, env) -> None:
        if len(self.cable_bead_IDs) == 0:
            raise RuntimeError("No cable beads found. Did add_cable() run?")

        if self.hidden_condition == "free":
            self.hidden_contact_meta.update(
                {
                    "applied": True,
                    "condition": "free",
                    "note": "No hidden contact injected.",
                }
            )
            self.hidden_contact_applied = True
            return

        if self.hidden_condition == "hidden_pin":
            self._apply_hidden_pin()
        elif self.hidden_condition == "hidden_high_friction":
            self._apply_hidden_high_friction()
        else:
            raise ValueError(self.hidden_condition)

        self.hidden_contact_meta["hidden_body_ids"] = [int(x) for x in self.hidden_body_ids]
        self.hidden_contact_meta["hidden_constraint_ids"] = [
            int(x) for x in self.hidden_constraint_ids
        ]
        self.hidden_contact_meta["applied"] = True
        self.hidden_contact_applied = True

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

    def _apply_hidden_pin(self) -> None:
        idx = self._middle_bead_index()
        bead_id = self.cable_bead_IDs[idx]
        bead_pos = self._bead_position(bead_id)

        # First try to constrain the bead directly to world.
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
            p.changeConstraint(cid, maxForce=80)
        except Exception:
            # Fallback: create an invisible static anchor and constrain to it.
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
            p.changeConstraint(cid, maxForce=80)

        self.hidden_constraint_ids.append(cid)
        self.hidden_contact_meta.update(
            {
                "condition": "hidden_pin",
                "pin_bead_local_index": int(idx),
                "pin_bead_id": int(bead_id),
                "pin_position": [float(v) for v in bead_pos],
                "max_force": 80,
            }
        )

    def _apply_hidden_high_friction(self) -> None:
        mid = self._middle_bead_index()
        lo = max(0, mid - 2)
        hi = min(len(self.cable_bead_IDs), mid + 3)
        selected = self.cable_bead_IDs[lo:hi]

        for bead_id in selected:
            p.changeDynamics(
                bead_id,
                -1,
                lateralFriction=8.0,
                spinningFriction=1.0,
                rollingFriction=0.5,
                linearDamping=0.8,
                angularDamping=0.8,
            )

        self.hidden_contact_meta.update(
            {
                "condition": "hidden_high_friction",
                "friction_bead_local_indices": list(range(lo, hi)),
                "friction_bead_ids": [int(x) for x in selected],
                "lateralFriction": 8.0,
                "spinningFriction": 1.0,
                "rollingFriction": 0.5,
                "linearDamping": 0.8,
                "angularDamping": 0.8,
            }
        )
