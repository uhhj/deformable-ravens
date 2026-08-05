import numpy as np

from ravens import environment
from ravens.environment import Environment


class FakeTask:
    def __init__(self):
        self.step = 0
        self.def_IDs = [100]
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

    def detect_contact(self, ids):
        self.detect_calls += 1
        return self.detect_calls >= 2

    def activate(self, objects, ids):
        self.active = self.grasp_success
        self.contact_constraint = (
            11
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
    env = object.__new__(Environment)
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
                "grasp_active_after": True,
                "constraint_available_after": True,
                "physics_step_start": 0,
                "physics_step_end": 0,
                "physics_step_count": 0,
                "joint_timeout_count": 0,
                "timeout_reason": None,
            })
        return True

    env.movep = movep
    env.movep_precise = movep_precise

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
    return env


def poses():
    rotation = (
        0.0,
        0.0,
        0.0,
        1.0,
    )
    return (
        (
            (
                0.40,
                -0.30,
                0.001,
            ),
            rotation,
        ),
        (
            (
                0.40,
                -0.22,
                0.001,
            ),
            rotation,
        ),
        (
            (
                0.40,
                -0.18,
                0.001,
            ),
            rotation,
        ),
    )


def test_precise_acquisition_reaches_formal_motion(
    monkeypatch,
):
    env = make_env(
        monkeypatch,
        grasp_success=True,
    )
    pose0, stage1, final = poses()

    success = (
        env.pick_precise_tension_extension(
            pose0,
            stage1,
            final,
            acquisition_motion_mode=(
                "precise_endpoint_recovery"
            ),
        )
    )

    assert success
    stages = [
        event["stage"]
        for event
        in env._ccda_motion_events
    ]
    assert stages.count(
        "tension_pull_acquisition"
    ) == 1
    assert "tension_pull_approach" in stages
    assert "tension_pull_lift" in stages
    assert "tension_pull_stage1" in stages
    assert "tension_pull_stage2" in stages
    assert (
        "tension_pull_lower_release"
        in stages
    )

    acquisition = next(
        event
        for event
        in env._ccda_motion_events
        if event["stage"]
        == "tension_pull_acquisition"
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


def test_grasp_failure_is_recorded_before_lift(
    monkeypatch,
):
    env = make_env(
        monkeypatch,
        grasp_success=False,
    )
    pose0, stage1, final = poses()

    success = (
        env.pick_precise_tension_extension(
            pose0,
            stage1,
            final,
            acquisition_motion_mode=(
                "precise_endpoint_recovery"
            ),
        )
    )

    assert not success
    stages = [
        event["stage"]
        for event
        in env._ccda_motion_events
    ]
    assert stages.count(
        "tension_pull_acquisition"
    ) == 1
    assert "tension_pull_lift" not in stages

    acquisition = next(
        event
        for event
        in env._ccda_motion_events
        if event["stage"]
        == "tension_pull_acquisition"
    )
    assert not acquisition["success"]
    assert acquisition[
        "failure_reason"
    ] == "grasp_failed"
    assert acquisition[
        "contact_detected"
    ]
    assert not acquisition[
        "grasp_active_after"
    ]


def test_lower_steps_do_not_create_event_rows(
    monkeypatch,
):
    env = make_env(
        monkeypatch,
        grasp_success=True,
    )
    pose0, stage1, final = poses()

    env.pick_precise_tension_extension(
        pose0,
        stage1,
        final,
        acquisition_motion_mode=(
            "precise_endpoint_recovery"
        ),
    )

    assert all(
        event["stage"]
        != "tension_pull_lower_step"
        for event
        in env._ccda_motion_events
    )
