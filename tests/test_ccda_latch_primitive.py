import inspect
import types

import numpy as np

from ravens.environment import Environment


class FakeEE:
    def detect_contact(self, _):
        return True

    def activate(self, *_):
        return None

    def check_grasp(self):
        return True

    def release(self):
        return None


def fake_environment(monkeypatch, precise_success=True):
    env = Environment.__new__(Environment)
    env.deterministic = True
    env.ee = FakeEE()
    env.objects = []
    env.ur5 = 1
    env.ee_tip_link = 12
    env._ccda_motion_events = []
    state = {"step": 100}
    env.task = types.SimpleNamespace(
        def_IDs=[], physics_step_count=lambda: state["step"]
    )
    env.movep = types.MethodType(lambda self, *args, **kwargs: True, env)

    def precise(self, pose, **kwargs):
        self._ccda_motion_events.append({
            "stage": kwargs["label"],
            "label": kwargs["label"],
            "achieved_fraction": 1.0,
            "success": precise_success,
        })
        calls.append((np.asarray(pose), kwargs))
        return precise_success

    calls = []
    env.movep_precise = types.MethodType(precise, env)
    env.step_physics = lambda steps=1: state.__setitem__("step", state["step"] + steps)
    monkeypatch.setattr(
        "ravens.environment.p.getLinkState",
        lambda *args, **kwargs: ((0.5, 0.1, 0.02), (0, 0, 0, 1)),
    )
    return env, calls


def test_latch_primitive_stage_order_exact_holds_tolerances_and_xy(monkeypatch):
    env, calls = fake_environment(monkeypatch)
    assert env.pick_precise_latch_probe(
        ((0.5, 0.1, 0.001), (0, 0, 0, 1)),
        hold_steps=7, return_hold_steps=11, post_release_steps=13,
        joint_tolerance=1e-4, cartesian_tolerance=2e-4,
    )
    assert [event["stage"] for event in env._ccda_motion_events] == [
        "latch_probe_lift", "latch_probe_hold",
        "latch_probe_lower_release", "latch_probe_return_hold",
        "latch_probe_post_release",
    ]
    holds = [event for event in env._ccda_motion_events if event.get("event_kind") == "hold"]
    assert [event["physics_step_count"] for event in holds] == [7, 11, 13]
    assert np.array_equal(calls[0][0][:2], calls[1][0][:2])
    assert all(call[1]["joint_tolerance"] == 1e-4 for call in calls)
    assert all(call[1]["cartesian_tolerance"] == 2e-4 for call in calls)


def test_latch_endpoint_miss_returns_false_and_has_no_wall_clock_sleep(monkeypatch):
    env, _ = fake_environment(monkeypatch, precise_success=False)
    assert not env.pick_precise_latch_probe(
        ((0.5, 0.1, 0.001), (0, 0, 0, 1))
    )
    source = inspect.getsource(Environment.pick_precise_latch_probe)
    assert "time.sleep" not in source
    assert "joint_tolerance=joint_tolerance" in source
    assert "cartesian_tolerance=cartesian_tolerance" in source
