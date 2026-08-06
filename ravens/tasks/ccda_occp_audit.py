"""Deterministic single-pin OCCP task for paired audit rollouts."""
from __future__ import annotations

import numpy as np
import pybullet as p

from ravens import utils
from ravens.tasks.defs_cables import CableEnv
from ravens.tasks.ccda_occp_geometry import (
    CONDITIONS,
    OCCPGeometryConfig,
    compute_occp_layout,
)


class OCCPAuditCable(CableEnv):
    def __init__(self):
        super().__init__()
        self._name = 'ccda-occp-audit'
        self.hidden_condition = 'free'
        self.pair_id = ''
        self.visible_seed = 0
        self.geometry_config = OCCPGeometryConfig()
        self.initial_settle_seconds = 1.0
        self.cable_constraint_ids = []
        self.passive_endpoint_constraint_id = None
        self.pin_body_id = None
        self.occluder_body_id = None
        self._physics_step_count = 0
        self._phase = 'no_action'
        self._trace = []
        self._trace_stride = 2
        self._layout = {}
        self._environment = None
        self._ee_target_position = np.zeros(3, dtype=np.float64)
        self._arm_metadata = {}
        self._oracle_peak_pin_force = 0.0
        self._oracle_contact_beads = set()
        self.ee = 'suction'
        self.primitive = 'pick_place'
        self.metric = 'cable-target'
        self.max_steps = 4
        self.def_threshold = 0.030
        self.def_nb_anchors = 1
        self.def_IDs = []

    def configure_audit(
            self, *, pair_id, seed, trace_stride=2,
            settle_seconds=1.0, geometry_config=None):
        self.pair_id = str(pair_id)
        self.visible_seed = int(seed)
        self._trace_stride = int(trace_stride)
        self.initial_settle_seconds = float(settle_seconds)
        if self._trace_stride <= 0:
            raise ValueError('trace_stride must be positive')
        if self.initial_settle_seconds < 0:
            raise ValueError('settle_seconds must be non-negative')
        if geometry_config is not None:
            self.geometry_config = geometry_config

    def reset(self, env, last_info=None):
        del last_info
        self._environment = env
        self.total_rewards = 0
        self.goal = {'places': {}, 'steps': [{}]}
        self.object_points = {}
        self._IDs = {}
        self.cable_bead_IDs = []
        self.cable_constraint_ids = []
        self.passive_endpoint_constraint_id = None
        self.pin_body_id = None
        self.occluder_body_id = None
        self.reset_branch_runtime('free')
        self._layout = compute_occp_layout(self.geometry_config, 'free')
        self._create_cable(env, self._layout)
        self._create_hidden_pin(self._layout)
        self._create_occluder(self._layout)
        env.settle_for_seconds(self.initial_settle_seconds)
        for joint_id in env.joints:
            p.enableJointForceTorqueSensor(
                env.ur5, int(joint_id), enableSensor=True)
        self.reset_branch_runtime('free')
        active_id = self.cable_bead_IDs[
            self._layout['active_endpoint_index']]
        self._ee_target_position = np.asarray(
            p.getBasePositionAndOrientation(active_id)[0], dtype=np.float64)

    def _create_cable(self, env, layout):
        radius = self.geometry_config.cable_radius
        spacing = layout['spacing']
        collision = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[radius] * 3)
        visual = p.createVisualShape(
            p.GEOM_SPHERE, radius=radius * 1.5,
            rgbaColor=utils.COLORS['blue'] + [1])
        endpoint_visual = p.createVisualShape(
            p.GEOM_SPHERE, radius=radius * 1.5,
            rgbaColor=utils.COLORS['yellow'] + [1])
        for index, position in enumerate(layout['bead_positions']):
            mass = 0.0 if index == 0 else self.geometry_config.bead_mass
            part_id = p.createMultiBody(
                baseMass=mass,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=(
                    endpoint_visual if index == len(layout['bead_positions']) - 1
                    else visual),
                basePosition=position.tolist())
            if index:
                constraint_id = p.createConstraint(
                    parentBodyUniqueId=self.cable_bead_IDs[-1],
                    parentLinkIndex=-1,
                    childBodyUniqueId=part_id,
                    childLinkIndex=-1,
                    jointType=p.JOINT_POINT2POINT,
                    jointAxis=(0, 0, 0),
                    parentFramePosition=(spacing, 0, 0),
                    childFramePosition=(0, 0, 0))
                p.changeConstraint(constraint_id, maxForce=100)
                self.cable_constraint_ids.append(int(constraint_id))
            self.cable_bead_IDs.append(int(part_id))
            env.objects.append(int(part_id))
            self.object_points[int(part_id)] = np.zeros((3, 1), dtype=np.float32)
            self._IDs[int(part_id)] = 'occp_cable_bead_{:02d}'.format(index)

    def _create_hidden_pin(self, layout):
        collision = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=layout['pin_radius'],
            height=layout['pin_height'])
        self.pin_body_id = int(p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=-1,
            basePosition=layout['pin_center'].tolist()))
        # PyBullet 3.0.4 reports a collision-derived visual entry even when
        # baseVisualShapeIndex=-1. Keep that compatibility entry transparent.
        p.changeVisualShape(
            self.pin_body_id, -1, rgbaColor=[0.0, 0.0, 0.0, 0.0])
        self._IDs[self.pin_body_id] = 'occp_hidden_pin'

    def _create_occluder(self, layout):
        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=layout['occluder_half_extents'].tolist(),
            rgbaColor=[0.18, 0.18, 0.18, 1.0])
        self.occluder_body_id = int(p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual,
            basePosition=layout['occluder_center'].tolist()))
        self._IDs[self.occluder_body_id] = 'occp_visual_occluder'

    def arm_condition(self, condition):
        if condition not in CONDITIONS:
            raise ValueError('unknown OCCP condition: {}'.format(condition))
        before = self._bead_positions()
        layout = compute_occp_layout(self.geometry_config, condition)
        p.resetBasePositionAndOrientation(
            self.pin_body_id, layout['pin_center'].tolist(), [0, 0, 0, 1])
        after = self._bead_positions()
        jumps = np.linalg.norm(after - before, axis=1)
        self.hidden_condition = condition
        self._layout = layout
        self._arm_metadata = {
            'condition': condition,
            'pin_center': layout['pin_center'].astype(float).tolist(),
            'arm_max_bead_jump': float(np.max(jumps)),
            'arm_mean_bead_jump': float(np.mean(jumps)),
        }
        return dict(self._arm_metadata)

    def reset_branch_runtime(self, condition):
        if condition not in CONDITIONS:
            raise ValueError('unknown OCCP condition: {}'.format(condition))
        self.hidden_condition = condition
        self._phase = 'no_action'
        self._physics_step_count = 0
        self._trace = []
        self._oracle_peak_pin_force = 0.0
        self._oracle_contact_beads = set()

    def set_ccda_phase(self, phase):
        if phase not in ('no_action', 'probe', 'test_pull', 'post_test'):
            raise ValueError('unknown OCCP phase: {}'.format(phase))
        self._phase = str(phase)

    def set_ee_target_position(self, position):
        self._ee_target_position = np.asarray(position, dtype=np.float64).copy()

    def physics_step_count(self):
        return int(self._physics_step_count)

    def physics_pre_step_hook(self):
        pass

    def physics_step_hook(self):
        self._physics_step_count += 1
        if self._physics_step_count % self._trace_stride == 0:
            self._trace.append(self._trace_sample())

    def _bead_positions(self):
        return np.asarray([
            p.getBasePositionAndOrientation(body_id)[0]
            for body_id in self.cable_bead_IDs], dtype=np.float64)

    def _oracle_pin_contact(self):
        total_force = 0.0
        indices = []
        for index, bead_id in enumerate(self.cable_bead_IDs):
            contacts = p.getContactPoints(self.pin_body_id, bead_id)
            if contacts:
                indices.append(index)
            for contact in contacts:
                normal = float(contact[9]) if len(contact) > 9 else 0.0
                lateral_1 = float(contact[10]) if len(contact) > 10 else 0.0
                lateral_2 = float(contact[12]) if len(contact) > 12 else 0.0
                total_force += float(np.linalg.norm(
                    [normal, lateral_1, lateral_2]))
        self._oracle_peak_pin_force = max(
            self._oracle_peak_pin_force, total_force)
        self._oracle_contact_beads.update(indices)
        return {
            'force': float(total_force),
            'count': int(len(indices)),
            'bead_indices': [int(value) for value in indices],
        }

    def _trace_sample(self):
        bead_positions = self._bead_positions()
        bead_velocities = np.asarray([
            p.getBaseVelocity(body_id)[0] for body_id in self.cable_bead_IDs])
        joint_states = [
            p.getJointState(self._environment.ur5, int(joint_id))
            for joint_id in self._environment.joints]
        ee_state = p.getLinkState(
            self._environment.ur5,
            self._environment.ee_tip_link,
            computeLinkVelocity=1,
            computeForwardKinematics=True)
        sensor = self._environment.ccda_sensor_observation()
        oracle = self._oracle_pin_contact()
        ee_position = np.asarray(ee_state[0], dtype=np.float64)
        ee_target_position = np.asarray(
            self._ee_target_position, dtype=np.float64)
        return {
            'physics_step': self.physics_step_count(),
            'phase': self._phase,
            'bead_positions': bead_positions.astype(float).tolist(),
            'bead_velocities': bead_velocities.astype(float).tolist(),
            'joint_positions': [float(state[0]) for state in joint_states],
            'joint_velocities': [float(state[1]) for state in joint_states],
            'ee_position': ee_position.astype(float).tolist(),
            'ee_orientation': [float(value) for value in ee_state[1]],
            'ee_linear_velocity': [float(value) for value in ee_state[6]],
            'ee_angular_velocity': [float(value) for value in ee_state[7]],
            'ee_target_position': ee_target_position.astype(float).tolist(),
            'ee_tracking_error': float(np.linalg.norm(
                ee_position - ee_target_position)),
            'sensor_joint_motor_torque': sensor['joint_motor_torque'],
            'sensor_joint_reaction_force_torque': (
                sensor['joint_reaction_force_torque']),
            'sensor_suction_force_xyz': sensor['suction_force_xyz'],
            'sensor_suction_torque_xyz': sensor['suction_torque_xyz'],
            'sensor_grasp_active': sensor['grasp_active'],
            'sensor_constraint_available': sensor['constraint_available'],
            'oracle_pin_contact_force': oracle['force'],
            'oracle_pin_contact_count': oracle['count'],
            'oracle_pin_contact_bead_indices': oracle['bead_indices'],
        }

    def ccda_trace(self):
        return [dict(row) for row in self._trace]

    def ccda_privileged_state(self):
        return {
            'condition': self.hidden_condition,
            'pin_center': self._layout['pin_center'].astype(float).tolist(),
            'arm': dict(self._arm_metadata),
            'oracle_peak_pin_force': float(self._oracle_peak_pin_force),
            'oracle_contact_bead_indices': sorted(self._oracle_contact_beads),
        }

    def reward(self):
        return 0.0, {}

    def done(self):
        return False
