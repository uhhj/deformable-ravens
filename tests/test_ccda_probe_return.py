import types

import numpy as np
import pytest

from ravens.environment import Environment


class DummyEE:
    def __init__(self):
        self.activated = False
        self.released = False

    def detect_contact(self, _):
        return True

    def activate(self, *_):
        self.activated = True

    def check_grasp(self):
        return self.activated

    def release(self):
        self.activated = False
        self.released = True


def bare_env():
    env = Environment.__new__(Environment)
    env.deterministic = True
    env.task = types.SimpleNamespace(
        task_stage=1,
        primitive_params={
            1: {
                'speed': 0.001,
                'delta_z': -0.001,
                'postpick_z': 0.04,
                'preplace_z': 0.04,
                'pause_place': 0.0,
            }
        },
    )
    env.ee = DummyEE()
    env.objects = []
    env.movements = []
    env.labels = []
    env.stepped = []
    env.movep = lambda pose, speed=0.01: (
        env.movements.append((np.asarray(pose), float(speed))) or True
    )
    env._ccda_record_frame = lambda label='': env.labels.append(label)
    env.step_physics = lambda steps=1: env.stepped.append(int(steps))
    return env


def pose(x, y, z=0.001):
    return ((x, y, z), (0, 0, 0, 1))


def test_probe_return_keeps_one_grasp_and_steps_hold():
    env = bare_env()
    result = env.pick_probe_return(
        pose0=pose(0.5, 0.0),
        pose_probe=pose(0.51, 0.0),
        pose_return=pose(0.5, 0.0),
        hold_steps=60,
    )
    assert result is True
    assert env.ee.released is True
    assert env.stepped == [60]
    assert 'probe_out' in env.labels
    assert 'probe_return' in env.labels
    assert 'probe_release' in env.labels


def test_probe_return_requires_fixed_step_mode():
    env = bare_env()
    env.deterministic = False
    with pytest.raises(RuntimeError):
        env.pick_probe_return(
            pose0=pose(0.5, 0.0),
            pose_probe=pose(0.51, 0.0),
            pose_return=pose(0.5, 0.0),
        )


def test_probe_return_rejects_negative_hold():
    env = bare_env()
    with pytest.raises(ValueError):
        env.pick_probe_return(
            pose0=pose(0.5, 0.0),
            pose_probe=pose(0.51, 0.0),
            pose_return=pose(0.5, 0.0),
            hold_steps=-1,
        )
