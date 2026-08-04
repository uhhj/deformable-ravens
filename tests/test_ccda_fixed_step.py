import threading

import pytest

from ravens.environment import Environment


class _DummyTask:
    def __init__(self):
        self.events = []

    def physics_pre_step_hook(self):
        self.events.append("pre")

    def physics_step_hook(self):
        self.events.append("post")


class _DummyEE:
    def __init__(self, events):
        self.events = events

    def step(self):
        self.events.append("ee")


def _bare_environment():
    env = Environment.__new__(Environment)
    env.deterministic = True
    env.running = False
    env.hz = 480
    env.control_substeps = 1
    env.post_action_settle_steps = 240
    env._ccda_step_lock = threading.RLock()
    env._ccda_physics_hook_error = None
    env._ccda_video_recorder = None
    env._ccda_video_label = ""
    return env


def test_step_physics_calls_hooks_in_order(monkeypatch):
    env = _bare_environment()
    task = _DummyTask()
    env.task = task
    env.ee = _DummyEE(task.events)
    monkeypatch.setattr(
        "ravens.environment.p.stepSimulation",
        lambda: task.events.append("physics"),
    )

    env.step_physics(2)
    assert task.events == [
        "pre", "physics", "ee", "post",
        "pre", "physics", "ee", "post",
    ]


def test_step_physics_rejects_negative_steps():
    env = _bare_environment()
    env.task = None
    env.ee = None
    with pytest.raises(ValueError):
        env.step_physics(-1)


def test_threaded_running_environment_rejects_manual_step():
    env = _bare_environment()
    env.deterministic = False
    env.running = True
    env.task = None
    env.ee = None
    with pytest.raises(RuntimeError):
        env.step_physics(1)


def test_settle_for_seconds_rounds_to_exact_steps(monkeypatch):
    env = _bare_environment()
    observed = []
    monkeypatch.setattr(env, "step_physics", lambda steps: observed.append(steps))
    env.settle_for_seconds(0.5)
    assert observed == [240]
