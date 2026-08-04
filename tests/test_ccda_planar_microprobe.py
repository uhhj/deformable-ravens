import types

import numpy as np
import pytest

from ravens.environment import Environment


class DummyEE:
    def __init__(self):
        self.activated = False
        self.activate_count = 0
        self.release_count = 0

    def detect_contact(self, _):
        return True

    def activate(self, *_):
        self.activated = True
        self.activate_count += 1

    def check_grasp(self):
        return self.activated

    def release(self):
        self.activated = False
        self.release_count += 1


def bare_env(monkeypatch):
    env = Environment.__new__(Environment)
    env.deterministic = True
    env.ur5 = 2
    env.ee_tip_link = 12
    env.task = types.SimpleNamespace(
        task_stage=1,
        def_IDs=[31],
        primitive_params={1: {"speed": 0.001, "delta_z": -0.0005}},
    )
    env.ee = DummyEE()
    env.objects = []
    env.movements = []
    env.labels = []
    env.stepped = []
    env.movep = lambda pose, speed=0.01: (
        env.movements.append((np.asarray(pose).copy(), float(speed))) or True
    )
    env._ccda_record_frame = lambda label="": env.labels.append(label)
    env.step_physics = lambda steps=1: env.stepped.append(int(steps))
    monkeypatch.setattr(
        "ravens.environment.p.getLinkState",
        lambda *args, **kwargs: ((0.5, 0.0, 0.001), (0, 0, 0, 1)),
    )
    return env


def pose(x, y, z=0.001):
    return ((x, y, z), (0, 0, 0, 1))


def test_microprobe_success_path_is_low_and_continuous(monkeypatch):
    env = bare_env(monkeypatch)
    result = env.pick_planar_microprobe(
        pose0=pose(0.5, 0.0),
        pose_probe=pose(0.5005, 0.0),
        pose_return=pose(0.5, 0.0),
        lift_height=0.0015,
        hold_steps=60,
        return_hold_steps=120,
        post_release_steps=180,
    )
    assert result is True
    assert env.ee.activate_count == 1
    assert env.ee.release_count == 1
    assert env.stepped == [60, 120, 180]
    assert {"micro_probe_out", "micro_probe_return", "micro_release"}.issubset(env.labels)
    waypoint_z = [
        pose_value[2]
        for pose_value, _ in env.movements
        if abs(pose_value[0] - 0.5) <= 0.001
        and pose_value[2] < 0.01
    ]
    assert waypoint_z
    assert max(waypoint_z) <= 0.001 + 0.0015


def test_microprobe_requires_deterministic_mode(monkeypatch):
    env = bare_env(monkeypatch)
    env.deterministic = False
    with pytest.raises(RuntimeError):
        env.pick_planar_microprobe(
            pose0=pose(0.5, 0.0),
            pose_probe=pose(0.5005, 0.0),
            pose_return=pose(0.5, 0.0),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lift_height": 0.0},
        {"hold_steps": -1},
        {"return_hold_steps": -1},
        {"post_release_steps": -1},
    ],
)
def test_microprobe_rejects_invalid_values(monkeypatch, kwargs):
    env = bare_env(monkeypatch)
    with pytest.raises(ValueError):
        env.pick_planar_microprobe(
            pose0=pose(0.5, 0.0),
            pose_probe=pose(0.5005, 0.0),
            pose_return=pose(0.5, 0.0),
            **kwargs,
        )
