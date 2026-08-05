import numpy as np

from ravens import environment
from ravens.environment import (
    Environment,
)


class FakeTask:
    def __init__(self):
        self.step = 0
        self.def_IDs = [100]
        self.cable_bead_IDs = [
            40,
            41,
            42,
        ]
        self.task_stage = 1
        self.primitive_params = {
            1: {
                "speed": 0.001,
                "delta_z": -0.0005,
            }
        }

    def physics_step_count(self):
        return self.step


class FakeEE:
    def __init__(
        self,
        grasp_success=True,
    ):
        self.grasp_success = bool(
            grasp_success
        )
        self.contact_constraint = None
        self.detect_calls = 0
        self.active = False
        self.target_object_id = None

    def detect_contact(self, ids):
        self.detect_calls += 1
        return self.detect_calls >= 2

    def detect_target_contact(
        self,
        target_object_id,
    ):
        self.target_object_id = int(
            target_object_id
        )
        self.detect_calls += 1
        return self.detect_calls >= 2

    def activate(
        self,
        objects,
        ids,
        target_object_id=None,
    ):
        self.target_object_id = (
            None
            if target_object_id is None
            else int(target_object_id)
        )
        self.active = (
            self.grasp_success
        )
        self.contact_constraint = (
            17
            if self.grasp_success
            else None
        )

    def check_grasp(self):
        return bool(self.active)

    def release(self):
        self.active = False
        self.contact_constraint = None


def make_env(
    monkeypatch,
    grasp_success=True,
):
    env = object.__new__(
        Environment
    )
    env.deterministic = True
    env.task = FakeTask()
    env.ee = FakeEE(
        grasp_success=grasp_success
    )
    env.objects = []
    env._ccda_motion_events = []
    env.ur5 = 1
    env.ee_tip_link = 12

    def movep(*args, **kwargs):
        return True

    def step_physics(steps):
        env.task.step += int(steps)

    def movep_precise(
        target,
        speed=0.001,
        joint_tolerance=1e-4,
        cartesian_tolerance=2e-4,
        max_corrections=3,
        label="precise_move",
        primitive=(
            "pick_precise_probe_return"
        ),
        record_event=True,
    ):
        if record_event:
            env._ccda_motion_events.append({
                "primitive": primitive,
                "stage": label,
                "label": label,
                "success": True,
                "achieved_fraction": 1.0,
                "physics_step_start": (
                    env.task.step
                ),
                "physics_step_end": (
                    env.task.step
                ),
                "physics_step_count": 0,
                "joint_timeout_count": 0,
                "joint_motion_success": True,
                "timeout_reason": None,
            })
        return True

    env.movep = movep
    env.movep_precise = (
        movep_precise
    )
    env.step_physics = step_physics

    monkeypatch.setattr(
        environment.p,
        "getLinkState",
        lambda *args, **kwargs: (
            (
                0.40,
                -0.30,
                0.006,
            ),
            (
                0.0,
                0.0,
                0.0,
                1.0,
            ),
        ),
    )
    monkeypatch.setattr(
        environment.p,
        "getConstraintInfo",
        lambda constraint_id: (
            0,
            0,
            env.ee.target_object_id,
            -1,
        ),
    )
    return env


def pose0():
    return (
        (
            0.40,
            -0.30,
            0.001,
        ),
        (
            0.0,
            0.0,
            0.0,
            1.0,
        ),
    )


def test_precise_probe_acquisition_reaches_formal_events(
    monkeypatch,
):
    env = make_env(
        monkeypatch,
        grasp_success=True,
    )

    success = (
        env.pick_precise_latch_probe(
            pose0(),
            acquisition_motion_mode=(
                "precise_endpoint_recovery"
            ),
            target_bead_index=1,
        )
    )

    assert success
    stages = [
        event["stage"]
        for event
        in env._ccda_motion_events
    ]
    assert stages.count(
        "routing_probe_acquisition"
    ) == 1

    formal = {
        stage
        for stage in stages
        if stage.startswith(
            "latch_probe_"
        )
    }
    assert formal == {
        "latch_probe_lift",
        "latch_probe_hold",
        "latch_probe_lower_release",
        "latch_probe_return_hold",
        "latch_probe_post_release",
    }
    assert (
        "routing_probe_lower_step"
        not in stages
    )
    assert (
        "routing_probe_approach"
        not in stages
    )

    acquisition = next(
        event
        for event
        in env._ccda_motion_events
        if event["stage"]
        == "routing_probe_acquisition"
    )
    assert acquisition["success"]
    assert (
        acquisition["failure_reason"]
        is None
    )
    assert acquisition[
        "contact_detected"
    ]
    assert acquisition[
        "grasp_active_after"
    ]
    assert acquisition[
        "constraint_available_after"
    ]
    assert acquisition[
        "target_bead_index"
    ] == 1
    assert acquisition[
        "target_body_id"
    ] == 41
    assert acquisition[
        "target_contact_detected"
    ]
    assert acquisition[
        "target_constraint_created"
    ]
    assert env.ee.target_object_id == 41


def test_probe_grasp_failure_is_explicit(
    monkeypatch,
):
    env = make_env(
        monkeypatch,
        grasp_success=False,
    )

    success = (
        env.pick_precise_latch_probe(
            pose0(),
            acquisition_motion_mode=(
                "precise_endpoint_recovery"
            ),
            target_bead_index=1,
        )
    )

    assert not success
    stages = [
        event["stage"]
        for event
        in env._ccda_motion_events
    ]
    assert stages == [
        "routing_probe_acquisition"
    ]

    acquisition = (
        env._ccda_motion_events[0]
    )
    assert not acquisition["success"]
    assert acquisition[
        "failure_reason"
    ] == "target_constraint_not_created"
    assert acquisition[
        "contact_detected"
    ]
    assert not acquisition[
        "grasp_active_after"
    ]
    assert not acquisition[
        "target_constraint_created"
    ]


def test_wrong_target_constraint_is_rejected(
    monkeypatch,
):
    env = make_env(
        monkeypatch,
        grasp_success=True,
    )
    monkeypatch.setattr(
        environment.p,
        "getConstraintInfo",
        lambda constraint_id: (
            0,
            0,
            40,
            -1,
        ),
    )

    success = (
        env.pick_precise_latch_probe(
            pose0(),
            acquisition_motion_mode=(
                "precise_endpoint_recovery"
            ),
            target_bead_index=1,
        )
    )

    assert not success
    acquisition = (
        env._ccda_motion_events[0]
    )
    assert acquisition[
        "failure_reason"
    ] == "target_constraint_not_created"
    assert acquisition[
        "constraint_available_after"
    ]
    assert not acquisition[
        "target_constraint_created"
    ]


def test_legacy_probe_mode_has_no_new_event(
    monkeypatch,
):
    env = make_env(
        monkeypatch,
        grasp_success=True,
    )

    success = (
        env.pick_precise_latch_probe(
            pose0(),
        )
    )

    assert success
    assert all(
        event["stage"]
        != "routing_probe_acquisition"
        for event
        in env._ccda_motion_events
    )
