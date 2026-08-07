"""Minimal occluded hidden-jam cable task for CCDA Phase 0."""
from __future__ import annotations

import numpy as np
import pybullet as p

from ravens import utils
from ravens.tasks.defs_cables import CableEnv
from ravens.tasks.ccda_ohj_geometry import (
    CONDITIONS, OHJGeometryConfig, compute_ohj_layout)


def _world_point_to_local(body_position, body_orientation, world_point):
    inverse = p.invertTransform(body_position, body_orientation)
    local, _ = p.multiplyTransforms(
        inverse[0], inverse[1], world_point, [0, 0, 0, 1])
    return local


def _surface_contact_force_world(point):
    force = (
        np.asarray(point[7], dtype=np.float64)
        * float(point[9]))
    if len(point) > 11:
        force = force + (
            np.asarray(point[11], dtype=np.float64)
            * float(point[10]))
    if len(point) > 13:
        force = force + (
            np.asarray(point[13], dtype=np.float64)
            * float(point[12]))
    return force


def _tactile_patch_index(local_position):
    x = float(local_position[0])
    y = float(local_position[1])
    if x >= 0.0 and y >= 0.0:
        return 0
    if x < 0.0 and y >= 0.0:
        return 1
    if x < 0.0 and y < 0.0:
        return 2
    return 3


class OHJCablePhase0(CableEnv):
    def __init__(self):
        super().__init__()
        self._name = "ccda-ohj-cable-phase0"
        self.hidden_condition = "free"
        self.pair_id = ""
        self.visible_seed = 0
        self.geometry_config = OHJGeometryConfig()
        self.initial_settle_seconds = 1.0
        self.cable_constraint_ids = []
        self.latch_body_id = None
        self.occluder_body_id = None
        self.active_endpoint_stabilizer_id = None
        self._physics_step_count = 0
        self._phase = "no_action"
        self._trace = []
        self._layout = {}
        self._environment = None
        self._ee_target_position = np.zeros(3, dtype=np.float64)
        self._arm_metadata = {}
        self.reference_passive_xyz = None
        self.ee = "suction"
        self.primitive = "pick_place"
        self.metric = "cable-target"
        self.max_steps = 4
        self.def_threshold = 0.030
        self.def_nb_anchors = 1
        self.def_IDs = []

    def configure_phase0(
            self, *, pair_id, seed, settle_seconds=1.0,
            geometry_config=None):
        self.pair_id = str(pair_id)
        self.visible_seed = int(seed)
        self.initial_settle_seconds = float(settle_seconds)
        if geometry_config is not None:
            self.geometry_config = geometry_config

    def reset(self, env, last_info=None):
        del last_info
        self._environment = env
        self.total_rewards = 0
        self.goal = {"places": {}, "steps": [{}]}
        self.object_points = {}
        self._IDs = {}
        self.cable_bead_IDs = []
        self.cable_constraint_ids = []
        self.latch_body_id = None
        self.occluder_body_id = None
        self.active_endpoint_stabilizer_id = None
        self._layout = compute_ohj_layout(self.geometry_config, "free")
        self._create_cable(env, self._layout)
        self._create_hidden_latch(self._layout)
        self._create_occluder(self._layout)
        self.reset_branch_runtime("free")
        env.settle_for_seconds(self.initial_settle_seconds)
        for joint_id in env.joints:
            p.enableJointForceTorqueSensor(
                env.ur5, int(joint_id), enableSensor=True)
        self.reset_branch_runtime("free")
        active_id = self.cable_bead_IDs[self._layout["active_endpoint_index"]]
        self._ee_target_position = np.asarray(
            p.getBasePositionAndOrientation(active_id)[0], dtype=np.float64)

    def _create_cable(self, env, layout):
        cfg = self.geometry_config
        positions = layout["bead_positions"]
        orientations = [p.getQuaternionFromEuler([0.0, 0.0, float(yaw)])
                        for yaw in layout["bead_yaw"]]
        collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.45 * cfg.spacing_m,
                         cfg.cable_radius_m, cfg.cable_radius_m])
        visual = p.createVisualShape(
            p.GEOM_SPHERE, radius=1.5 * cfg.cable_radius_m,
            rgbaColor=utils.COLORS["blue"] + [1])
        endpoint_visual = p.createVisualShape(
            p.GEOM_SPHERE, radius=1.5 * cfg.cable_radius_m,
            rgbaColor=utils.COLORS["yellow"] + [1])
        for index, position in enumerate(positions):
            part_id = p.createMultiBody(
                baseMass=cfg.bead_mass_kg,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=(endpoint_visual if index == 31 else visual),
                basePosition=position.tolist(),
                baseOrientation=orientations[index])
            p.changeDynamics(
                part_id, -1, lateralFriction=cfg.bead_lateral_friction,
                linearDamping=cfg.linear_damping,
                angularDamping=cfg.angular_damping)
            if index:
                midpoint = 0.5 * (positions[index - 1] + positions[index])
                parent_frame = _world_point_to_local(
                    positions[index - 1], orientations[index - 1], midpoint)
                child_frame = _world_point_to_local(
                    positions[index], orientations[index], midpoint)
                constraint = p.createConstraint(
                    parentBodyUniqueId=self.cable_bead_IDs[-1],
                    parentLinkIndex=-1, childBodyUniqueId=part_id,
                    childLinkIndex=-1, jointType=p.JOINT_POINT2POINT,
                    jointAxis=(0, 0, 0), parentFramePosition=parent_frame,
                    childFramePosition=child_frame)
                p.changeConstraint(
                    constraint, maxForce=cfg.constraint_max_force_n)
                p.setCollisionFilterPair(
                    self.cable_bead_IDs[-1], part_id, -1, -1,
                    enableCollision=0)
                self.cable_constraint_ids.append(int(constraint))
            self.cable_bead_IDs.append(int(part_id))
            env.objects.append(int(part_id))
            self.object_points[int(part_id)] = np.zeros((3, 1), dtype=np.float32)
            self._IDs[int(part_id)] = "ohj_cable_bead_{:02d}".format(index)
        active = self.cable_bead_IDs[layout["active_endpoint_index"]]
        self.active_endpoint_stabilizer_id = int(p.createConstraint(
            parentBodyUniqueId=active, parentLinkIndex=-1,
            childBodyUniqueId=-1, childLinkIndex=-1,
            jointType=p.JOINT_FIXED, jointAxis=(0, 0, 0),
            parentFramePosition=(0, 0, 0),
            parentFrameOrientation=(0, 0, 0, 1),
            childFramePosition=positions[layout["active_endpoint_index"]].tolist(),
            childFrameOrientation=orientations[layout["active_endpoint_index"]]))

    def release_active_endpoint_stabilizer(self):
        if self.active_endpoint_stabilizer_id is not None:
            p.removeConstraint(self.active_endpoint_stabilizer_id)
            self.active_endpoint_stabilizer_id = None

    def _create_hidden_latch(self, layout):
        collision = p.createCollisionShape(
            p.GEOM_CYLINDER, radius=layout["latch_radius"],
            height=layout["latch_height"])
        self.latch_body_id = int(p.createMultiBody(
            baseMass=0.0, baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=-1,
            basePosition=layout["latch_center"].tolist()))
        p.changeVisualShape(
            self.latch_body_id, -1, rgbaColor=[0.0, 0.0, 0.0, 0.0])
        p.changeDynamics(
            self.latch_body_id, -1,
            lateralFriction=self.geometry_config.bead_lateral_friction)
        self._IDs[self.latch_body_id] = "ohj_hidden_latch"

    def _create_occluder(self, layout):
        visual = p.createVisualShape(
            p.GEOM_BOX, halfExtents=layout["occluder_half_extents"].tolist(),
            rgbaColor=[0.18, 0.18, 0.18, 1.0])
        self.occluder_body_id = int(p.createMultiBody(
            baseMass=0.0, baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual,
            basePosition=layout["occluder_center"].tolist()))
        self._IDs[self.occluder_body_id] = "ohj_visual_occluder"

    def arm_condition(self, condition):
        if condition not in CONDITIONS:
            raise ValueError("unknown OHJ condition: {}".format(condition))
        before = self._bead_positions()
        layout = compute_ohj_layout(self.geometry_config, condition)
        p.resetBasePositionAndOrientation(
            self.latch_body_id, layout["latch_center"].tolist(), [0, 0, 0, 1])
        after = self._bead_positions()
        self.hidden_condition = condition
        self._layout = layout
        self._arm_metadata = {
            "condition": condition,
            "latch_center": layout["latch_center"].astype(float).tolist(),
            "arm_max_bead_jump": float(np.max(np.linalg.norm(
                after - before, axis=1))),
        }
        return dict(self._arm_metadata)

    def reset_branch_runtime(self, condition):
        self.hidden_condition = condition
        self._phase = "no_action"
        self._physics_step_count = 0
        self._trace = []

    def set_ccda_phase(self, phase):
        self._phase = str(phase)

    def set_ee_target_position(self, position):
        self._ee_target_position = np.asarray(position, dtype=np.float64).copy()

    def physics_step_count(self):
        return int(self._physics_step_count)

    def physics_pre_step_hook(self):
        pass

    def physics_step_hook(self):
        self._physics_step_count += 1
        self._trace.append(self._trace_sample())

    def _bead_positions(self):
        return np.asarray([p.getBasePositionAndOrientation(body)[0]
                           for body in self.cable_bead_IDs], dtype=np.float64)

    def visible_keypoints(self):
        return self._bead_positions()[self._layout["visible_keypoint_indices"]]

    def statediff_state(self):
        ee = np.asarray(p.getLinkState(
            self._environment.ur5, self._environment.ee_tip_link,
            computeForwardKinematics=True)[0], dtype=np.float64)
        state = np.concatenate([self.visible_keypoints().reshape(-1), ee])
        assert state.shape == (51,)
        return state

    def formal_contact_sensor(self):
        obs = self._environment.ccda_sensor_observation()
        sensor = np.concatenate([
            np.asarray(obs["suction_force_xyz"]),
            np.asarray(obs["suction_torque_xyz"]),
        ]).astype(np.float64)
        assert sensor.shape == (6,)
        return sensor

    def _gripper_surface_contacts(self):
        ee = self._environment.ee
        active_id = self.cable_bead_IDs[
            self._layout["active_endpoint_index"]]
        return p.getContactPoints(
            bodyA=ee.body,
            bodyB=active_id,
            linkIndexA=0,
        )

    def gripper_surface_tactile_force(self):
        contacts = self._gripper_surface_contacts()
        force_world = np.zeros(3, dtype=np.float64)
        for point in contacts:
            force_world += _surface_contact_force_world(point)
        ee = self._environment.ee
        link_state = p.getLinkState(
            ee.body, 0, computeForwardKinematics=True)
        rotation = np.asarray(
            p.getMatrixFromQuaternion(link_state[1]),
            dtype=np.float64,
        ).reshape(3, 3)
        force_local = rotation.T @ force_world
        assert force_local.shape == (3,)
        return force_local

    def gripper_surface_tactile_patches(self):
        contacts = self._gripper_surface_contacts()
        ee = self._environment.ee
        link_state = p.getLinkState(
            ee.body, 0, computeForwardKinematics=True)
        tip_position = np.asarray(link_state[0], dtype=np.float64)
        rotation = np.asarray(
            p.getMatrixFromQuaternion(link_state[1]),
            dtype=np.float64,
        ).reshape(3, 3)
        patch_force = np.zeros((4, 3), dtype=np.float64)
        patch_count = np.zeros(4, dtype=np.int64)
        for point in contacts:
            position_world = np.asarray(point[5], dtype=np.float64)
            position_local = rotation.T @ (position_world - tip_position)
            patch = _tactile_patch_index(position_local)
            force_world = _surface_contact_force_world(point)
            force_local = rotation.T @ force_world
            patch_force[patch] += force_local
            patch_count[patch] += 1
        return patch_force, patch_count

    def combined_contact_sensor(self):
        sensor = np.concatenate([
            self.formal_contact_sensor(),
            self.gripper_surface_tactile_force(),
        ]).astype(np.float64)
        assert sensor.shape == (9,)
        return sensor

    def spatial_contact_sensor(self):
        patch_force, _ = self.gripper_surface_tactile_patches()
        sensor = np.concatenate([
            self.formal_contact_sensor(),
            patch_force.reshape(-1),
        ]).astype(np.float64)
        assert sensor.shape == (18,)
        return sensor

    def extraction_progress_m(self, reference_passive_xyz):
        current = self._bead_positions()[:4].mean(axis=0)
        direction = np.asarray(self._layout["pull_direction"], dtype=np.float64)
        return float(np.dot(
            current - np.asarray(reference_passive_xyz), direction))

    def _oracle_latch_contact(self):
        total_force = 0.0
        indices = []
        for index, bead in enumerate(self.cable_bead_IDs):
            contacts = p.getContactPoints(self.latch_body_id, bead)
            if contacts:
                indices.append(index)
            for contact in contacts:
                total_force += float(contact[9]) if len(contact) > 9 else 0.0
        return float(total_force), indices

    def oracle_internal_cable_constraint_force_xyz(self):
        rows = []
        for constraint in self.cable_constraint_ids:
            state = np.asarray(
                p.getConstraintState(int(constraint)),
                dtype=np.float64,
            ).reshape(-1)
            if state.size < 3:
                raise RuntimeError(
                    "OHJ P2P constraint diagnostic must expose "
                    "3 translational force components")
            rows.append(state[:3].copy())
        force = np.asarray(rows, dtype=np.float64)
        assert force.shape == (len(self.cable_constraint_ids), 3)
        return force

    def _trace_sample(self):
        all_beads = self._bead_positions()
        ee = np.asarray(p.getLinkState(
            self._environment.ur5, self._environment.ee_tip_link,
            computeForwardKinematics=True)[0], dtype=np.float64)
        observation = self._environment.ccda_sensor_observation()
        latch_force, latch_beads = self._oracle_latch_contact()
        latch_bead_mask = np.zeros(
            len(self.cable_bead_IDs), dtype=np.int8)
        if latch_beads:
            latch_bead_mask[np.asarray(latch_beads, dtype=np.int64)] = 1
        internal_constraint_force = (
            self.oracle_internal_cable_constraint_force_xyz())
        reference = (all_beads[:4].mean(axis=0)
                     if self.reference_passive_xyz is None
                     else np.asarray(self.reference_passive_xyz))
        wrist_wrench = self.formal_contact_sensor()
        patch_force, patch_count = self.gripper_surface_tactile_patches()
        surface_tactile = np.sum(patch_force, axis=0)
        surface_contact_count = int(np.sum(patch_count))
        formal_sensor = np.concatenate([
            wrist_wrench,
            surface_tactile,
        ]).astype(np.float64)
        formal_sensor_spatial = np.concatenate([
            wrist_wrench,
            patch_force.reshape(-1),
        ]).astype(np.float64)
        return {
            "physics_step": self.physics_step_count(),
            "phase": self._phase,
            "statediff_state": self.statediff_state().astype(float).tolist(),
            "visible_keypoints": self.visible_keypoints().astype(float).tolist(),
            "all_bead_positions": all_beads.astype(float).tolist(),
            "ee_position": ee.astype(float).tolist(),
            "formal_wrench": wrist_wrench.astype(float).tolist(),
            "gripper_surface_tactile_force": surface_tactile.astype(
                float).tolist(),
            "gripper_surface_contact_count": surface_contact_count,
            "formal_sensor": formal_sensor.astype(float).tolist(),
            "gripper_surface_tactile_patch_force": patch_force.astype(
                float).tolist(),
            "gripper_surface_tactile_patch_contact_count": patch_count.astype(
                int).tolist(),
            "formal_sensor_spatial": formal_sensor_spatial.astype(
                float).tolist(),
            "joint_motor_torque": observation["joint_motor_torque"],
            "joint_reaction_wrench": observation[
                "joint_reaction_force_torque"],
            "extraction_progress_m": self.extraction_progress_m(reference),
            "oracle_latch_contact_force": latch_force,
            "oracle_latch_contact_count": len(latch_beads),
            "oracle_latch_contact_bead_mask": latch_bead_mask.astype(
                int).tolist(),
            "oracle_internal_cable_constraint_force_xyz": (
                internal_constraint_force.astype(float).tolist()),
        }

    def ccda_trace(self):
        return [dict(row) for row in self._trace]

    def reward(self):
        return 0.0, {}

    def done(self):
        return False
