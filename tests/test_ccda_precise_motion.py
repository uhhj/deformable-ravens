import types

import numpy as np
import pytest

from ravens.environment import Environment


def test_movep_precise_records_half_mm_motion(monkeypatch):
    env = Environment.__new__(Environment)
    env.deterministic = True
    env.ur5 = 2
    env.ee_tip_link = 12
    env._ccda_motion_events = []
    positions = iter([(0.5, 0.0, 0.3), (0.5005, 0.0, 0.3)])
    monkeypatch.setattr(
        "ravens.environment.p.getLinkState",
        lambda *args, **kwargs: (next(positions), (0, 0, 0, 1)),
    )
    env.movep = types.MethodType(
        lambda self, pose, speed=0.01, joint_tolerance=None: True,
        env,
    )
    target = np.asarray([0.5005, 0.0, 0.3, 0, 0, 0, 1])
    assert env.movep_precise(target, label="half_mm")
    event = env.ccda_motion_events()[0]
    assert event["achieved_fraction"] == pytest.approx(1.0)
    assert event["requested_distance"] == pytest.approx(0.0005)


def test_movep_precise_rejects_async_mode():
    env = Environment.__new__(Environment)
    env.deterministic = False
    with pytest.raises(RuntimeError):
        env.movep_precise([0, 0, 0, 0, 0, 0, 1])


def test_explicit_tolerance_executes_sub_legacy_threshold(monkeypatch):
    env = Environment.__new__(Environment)
    env.hz = 480
    env.control_substeps = 1
    env.ur5 = 2
    env.joints = [0]
    env._ccda_record_frame = lambda label="": None
    env.step_physics = lambda steps=1: None
    state = {"position": 0.0, "commands": 0}
    monkeypatch.setattr(
        "ravens.environment.p.getJointState",
        lambda *args, **kwargs: (state["position"], 0, 0, 0),
    )

    def control(**kwargs):
        state["commands"] += 1
        state["position"] = float(kwargs["targetPositions"][0])

    monkeypatch.setattr(
        "ravens.environment.p.setJointMotorControlArray", control
    )
    assert env._movej_fixed(
        np.asarray([0.0005]),
        speed=0.00025,
        t_lim=1,
        joint_tolerance=1e-4,
    )
    assert state["commands"] == 2
