"""Directional Hidden-Release cable benchmark smoke."""
from __future__ import annotations

import numpy as np
import pybullet as p

from ravens.tasks.ccda_dhr_geometry import (
    CONDITIONS,
    DHRGeometryConfig,
    compute_dhr_layout,
)
from ravens.tasks.ccda_ohj_cable import (
    OHJCablePhase0,
)


class DHRCableControlSmoke(
        OHJCablePhase0):
    def __init__(self):
        super().__init__()

        self._name = (
            "ccda-dhr-cable-smoke")

        self.geometry_config = (
            DHRGeometryConfig())

        self.smoke_condition = "free"

        self.pocket_bottom_body_id = None
        self.pocket_left_body_id = None
        self.pocket_right_body_id = None

    def configure_control_smoke(
            self,
            *,
            condition,
            run_id,
            seed,
            settle_seconds=1.0,
            geometry_config=None):
        if condition not in CONDITIONS:
            raise ValueError(
                "unknown DHR condition: {}"
                .format(condition))

        self.smoke_condition = str(
            condition)

        self.pair_id = str(
            run_id)

        self.visible_seed = int(
            seed)

        self.initial_settle_seconds = float(
            settle_seconds)

        if geometry_config is not None:
            self.geometry_config = (
                geometry_config)

    def reset(
            self,
            env,
            last_info=None):
        del last_info

        self._environment = env
        self.total_rewards = 0
        self.goal = {
            "places": {},
            "steps": [{}],
        }

        self.object_points = {}
        self._IDs = {}
        self.cable_bead_IDs = []
        self.cable_constraint_ids = []

        self.latch_body_id = None
        self.directional_guide_body_id = None
        self.hook_side_body_id = None
        self.hook_stop_body_id = None

        self.pocket_bottom_body_id = None
        self.pocket_left_body_id = None
        self.pocket_right_body_id = None

        self.occluder_body_id = None
        self.active_endpoint_stabilizer_id = None

        self._layout = compute_dhr_layout(
            self.geometry_config,
            self.smoke_condition)

        self._create_cable(
            env,
            self._layout)

        self._create_hidden_pocket(
            self._layout)

        self._create_occluder(
            self._layout)

        self.reset_branch_runtime(
            self.smoke_condition)

        env.settle_for_seconds(
            self.initial_settle_seconds)

        for joint_id in env.joints:
            p.enableJointForceTorqueSensor(
                env.ur5,
                int(joint_id),
                enableSensor=True)

        self.reset_branch_runtime(
            self.smoke_condition)

        active_id = self.cable_bead_IDs[
            self._layout[
                "active_endpoint_index"]]

        self._ee_target_position = np.asarray(
            p.getBasePositionAndOrientation(
                active_id)[0],
            dtype=np.float64)

    def _create_hidden_pocket(
            self,
            layout):
        bottom_collision = (
            p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=layout[
                    "pocket_bottom_half_extents"
                ].tolist()))

        wall_collision = (
            p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=layout[
                    "pocket_wall_half_extents"
                ].tolist()))

        self.pocket_bottom_body_id = int(
            p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=
                    bottom_collision,
                baseVisualShapeIndex=-1,
                basePosition=layout[
                    "pocket_bottom_center"
                ].tolist()))

        self.pocket_left_body_id = int(
            p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=
                    wall_collision,
                baseVisualShapeIndex=-1,
                basePosition=layout[
                    "pocket_left_center"
                ].tolist()))

        self.pocket_right_body_id = int(
            p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=
                    wall_collision,
                baseVisualShapeIndex=-1,
                basePosition=layout[
                    "pocket_right_center"
                ].tolist()))

        for body in (
                self.pocket_bottom_body_id,
                self.pocket_left_body_id,
                self.pocket_right_body_id):
            p.changeVisualShape(
                body,
                -1,
                rgbaColor=[
                    0.0, 0.0, 0.0, 0.0])

            p.changeDynamics(
                body,
                -1,
                lateralFriction=
                    self.geometry_config
                    .bead_lateral_friction)

        self._IDs[
            self.pocket_bottom_body_id
        ] = "dhr_hidden_pocket_bottom"

        self._IDs[
            self.pocket_left_body_id
        ] = "dhr_hidden_pocket_left"

        self._IDs[
            self.pocket_right_body_id
        ] = "dhr_hidden_pocket_right"

    def _oracle_latch_contact(self):
        total_force = 0.0
        indices = set()

        hidden_bodies = (
            self.pocket_bottom_body_id,
            self.pocket_left_body_id,
            self.pocket_right_body_id,
        )

        for hidden_body in hidden_bodies:
            for index, bead in enumerate(
                    self.cable_bead_IDs):
                contacts = p.getContactPoints(
                    hidden_body,
                    bead)

                if contacts:
                    indices.add(
                        int(index))

                for contact in contacts:
                    total_force += (
                        float(contact[9])
                        if len(contact) > 9
                        else 0.0)

        return (
            float(total_force),
            sorted(indices))
