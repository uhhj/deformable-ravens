import inspect
from types import SimpleNamespace

import pytest

from ravens import environment as environment_module
from ravens.environment import Environment


POSE0 = ((0.40, 0.00, 0.001), (0.0, 0.0, 0.0, 1.0))
STAGE1 = ((0.40, 0.08, 0.001), (0.0, 0.0, 0.0, 1.0))
FINAL = ((0.40, 0.12, 0.001), (0.0, 0.0, 0.0, 1.0))


class FakeEE:
    def __init__(self, checks=None):
        self.checks = list(checks or [True] * 8)
        self.activate_count = 0
        self.release_count = 0
        self.contact_constraint = object()

    def detect_contact(self, _):
        return True

    def activate(self, *_):
        self.activate_count += 1

    def check_grasp(self):
        return self.checks.pop(0) if self.checks else True

    def release(self):
        self.release_count += 1
        self.contact_constraint = None


def make_env(monkeypatch, deterministic=True, checks=None):
    env = Environment.__new__(Environment)
    env.deterministic = deterministic
    env.task = SimpleNamespace(def_IDs=[])
    env.ee = FakeEE(checks)
    env.objects = []
    env.ur5 = 1
    env.ee_tip_link = 2
    env._ccda_motion_events = []
    env.movep = lambda *args, **kwargs: True

    def precise(_target, label, primitive, **_kwargs):
        env._ccda_motion_events.append({
            "stage": label,
            "primitive": primitive,
            "success": True,
            "achieved_fraction": 1.0,
            "physics_step_end": len(env._ccda_motion_events) + 1,
        })
        return True

    env.movep_precise = precise
    monkeypatch.setattr(
        environment_module.p,
        "getLinkState",
        lambda *_args, **_kwargs: ((0.40, 0.00, 0.01), (0, 0, 0, 1)),
    )
    return env


def run(env, stage1=STAGE1, final=FINAL):
    return env.pick_precise_tension_extension(
        POSE0, stage1, final,
        lift_height=0.004,
        approach_height=0.02,
        retreat_z=0.3,
        joint_tolerance=1e-4,
        cartesian_tolerance=2e-4,
        min_achieved_fraction=0.8,
    )


def test_primitive_is_registered():
    source = inspect.getsource(Environment.__init__)
    assert "'pick_precise_tension_extension': self.pick_precise_tension_extension" in source


def test_primitive_requires_deterministic_mode(monkeypatch):
    env = make_env(monkeypatch, deterministic=False)
    with pytest.raises(RuntimeError, match="fixed-step"):
        run(env)


def test_primitive_rejects_non_collinear_path(monkeypatch):
    env = make_env(monkeypatch)
    with pytest.raises(ValueError, match="collinear"):
        run(env, final=((0.44, 0.12, 0.001), FINAL[1]))


def test_primitive_rejects_shorter_final_target(monkeypatch):
    env = make_env(monkeypatch)
    with pytest.raises(ValueError, match="exceed stage 1"):
        run(env, final=((0.40, 0.06, 0.001), FINAL[1]))


def test_primitive_activates_once_and_releases_only_after_both_stages(monkeypatch):
    env = make_env(monkeypatch)
    assert run(env)
    assert env.ee.activate_count == 1
    assert env.ee.release_count == 1
    stages = [row["stage"] for row in env._ccda_motion_events]
    assert stages == [
        "tension_pull_lift",
        "tension_pull_stage1",
        "tension_pull_stage2",
        "tension_pull_lower_release",
    ]


def test_primitive_records_grasp_and_constraint_audit(monkeypatch):
    env = make_env(monkeypatch)
    assert run(env)
    for row in env._ccda_motion_events[:3]:
        assert row["grasp_active_after"] is True
        assert row["constraint_available_after"] is True
    assert env._ccda_motion_events[-1]["grasp_active_before_release"] is True


def test_primitive_fails_when_grasp_is_lost(monkeypatch):
    env = make_env(monkeypatch, checks=[True, True, False])
    assert not run(env)
    assert env.ee.activate_count == 1
    assert env.ee.release_count == 1
    assert [row["stage"] for row in env._ccda_motion_events] == [
        "tension_pull_lift", "tension_pull_stage1"
    ]


def test_primitive_does_not_read_hidden_condition_or_oracle():
    source = inspect.getsource(Environment.pick_precise_tension_extension)
    for forbidden in ("hidden_condition", "topology", "oracle"):
        assert forbidden not in source.lower()
