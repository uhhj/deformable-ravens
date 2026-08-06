import threading

import pytest

from ravens.environment import Environment


def bare_environment():
    env = Environment.__new__(Environment)
    env.deterministic = True
    env.running = False
    env.hz = 480
    env.control_substeps = 1
    env._ccda_step_lock = threading.RLock()
    env._ccda_physics_hook_error = None
    env._ccda_video_recorder = None
    env._ccda_video_label = ''
    return env


def test_step_physics_calls_hooks_in_order(monkeypatch):
    events = []
    task = type('Task', (), {
        'physics_pre_step_hook': lambda self: events.append('pre'),
        'physics_step_hook': lambda self: events.append('post')})()
    ee = type('EE', (), {'step': lambda self: events.append('ee')})()
    env = bare_environment()
    env.task, env.ee = task, ee
    monkeypatch.setattr(
        'ravens.environment.p.stepSimulation',
        lambda: events.append('physics'))
    env.step_physics(1)
    assert events == ['pre', 'physics', 'ee', 'post']


def test_step_physics_rejects_negative_steps():
    env = bare_environment()
    env.task = env.ee = None
    with pytest.raises(ValueError):
        env.step_physics(-1)


def test_threaded_running_environment_rejects_manual_step():
    env = bare_environment()
    env.deterministic = False
    env.running = True
    env.task = env.ee = None
    with pytest.raises(RuntimeError):
        env.step_physics(1)


def test_settle_for_seconds_rounds_to_exact_steps(monkeypatch):
    env = bare_environment()
    observed = []
    monkeypatch.setattr(env, 'step_physics', observed.append)
    env.settle_for_seconds(0.5)
    assert observed == [240]
