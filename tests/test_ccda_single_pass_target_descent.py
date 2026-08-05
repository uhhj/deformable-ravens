import numpy as np

from ravens.environment import (
    Environment,
)


class FakeEE:
    def __init__(
        self,
        contacts,
    ):
        self.contacts = list(
            contacts
        )
        self.queries = []

    def detect_target_contact(
        self,
        target_body_id,
    ):
        self.queries.append(
            int(target_body_id)
        )
        if not self.contacts:
            return False
        return bool(
            self.contacts.pop(0)
        )


def make_env(
    contacts,
    motion_results,
):
    env = object.__new__(
        Environment
    )
    env.ee = FakeEE(contacts)
    env.motion_results = list(
        motion_results
    )
    env.motion_targets = []

    def movep_precise(
        target,
        **kwargs,
    ):
        env.motion_targets.append(
            np.asarray(
                target,
                dtype=np.float64,
            ).copy()
        )
        return bool(
            env.motion_results.pop(0)
        )

    env.movep_precise = (
        movep_precise
    )
    return env


def approach():
    return np.asarray(
        [
            0.4,
            -0.3,
            0.021,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        dtype=np.float64,
    )


def run(env):
    return (
        env._ccda_precise_target_descent(
            approach=approach(),
            pick_z=0.001,
            floor_limit=0.0,
            target_body_id=11,
            speed=0.001,
            joint_tolerance=1e-4,
            cartesian_tolerance=2e-4,
            primitive=(
                "pick_precise_latch_probe"
            ),
            label_prefix=(
                "routing_probe"
            ),
        )
    )


def test_contact_at_pick_z_uses_one_command():
    env = make_env(
        contacts=[True],
        motion_results=[True],
    )

    result = run(env)

    assert result == (
        True,
        True,
        1,
    )
    assert len(
        env.motion_targets
    ) == 1
    assert np.isclose(
        env.motion_targets[0][2],
        0.001,
    )
    assert env.ee.queries == [11]


def test_contact_at_floor_uses_two_commands():
    env = make_env(
        contacts=[
            False,
            True,
        ],
        motion_results=[
            True,
            True,
        ],
    )

    result = run(env)

    assert result == (
        True,
        True,
        2,
    )
    assert [
        float(target[2])
        for target
        in env.motion_targets
    ] == [
        0.001,
        0.0,
    ]
    assert env.ee.queries == [
        11,
        11,
    ]


def test_no_contact_does_not_add_commands():
    env = make_env(
        contacts=[
            False,
            False,
        ],
        motion_results=[
            True,
            True,
        ],
    )

    result = run(env)

    assert result == (
        True,
        False,
        2,
    )
    assert len(
        env.motion_targets
    ) == 2


def test_motion_failure_stops_descent():
    env = make_env(
        contacts=[],
        motion_results=[False],
    )

    result = run(env)

    assert result == (
        False,
        False,
        1,
    )
    assert env.ee.queries == []
