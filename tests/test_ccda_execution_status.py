from ravens.environment import (
    Environment,
)


class FakeTask:
    def __init__(
        self,
        *,
        task_success=False,
        exit_gracefully=False,
    ):
        self._task_success = bool(
            task_success
        )
        self.exit_gracefully = bool(
            exit_gracefully
        )
        self.def_IDs = []

    def reward(self):
        return 0.0, {}

    def done(self):
        return self._task_success


class StatusEnvironment(
    Environment
):
    @property
    def info(self):
        return {}


def make_env(
    *,
    primitive_success,
    task_success=False,
    exit_gracefully=False,
):
    env = object.__new__(
        StatusEnvironment
    )
    env.task = FakeTask(
        task_success=task_success,
        exit_gracefully=(
            exit_gracefully
        ),
    )
    env.primitives = {
        "fake": lambda: bool(
            primitive_success
        )
    }
    env.deterministic = True
    env.post_action_settle_steps = 0
    return env


def action():
    return {
        "primitive": "fake",
        "params": {},
    }


def test_primitive_failure_is_not_task_success():
    env = make_env(
        primitive_success=False,
    )

    _, _, legacy_done, info = (
        env.step(action())
    )
    status = info[
        "ccda_execution_status"
    ]

    assert legacy_done
    assert status[
        "action_completed"
    ]
    assert not status[
        "primitive_succeeded"
    ]
    assert not status[
        "task_success"
    ]
    assert status[
        "episode_terminated"
    ]
    assert status[
        "termination_reason"
    ] == "primitive_failed"


def test_successful_action_can_leave_task_running():
    env = make_env(
        primitive_success=True,
        task_success=False,
    )

    _, _, legacy_done, info = (
        env.step(action())
    )
    status = info[
        "ccda_execution_status"
    ]

    assert not legacy_done
    assert status[
        "action_completed"
    ]
    assert status[
        "primitive_succeeded"
    ]
    assert not status[
        "task_success"
    ]
    assert not status[
        "episode_terminated"
    ]
    assert status[
        "termination_reason"
    ] is None


def test_task_success_is_separate_from_action_success():
    env = make_env(
        primitive_success=True,
        task_success=True,
    )

    _, _, legacy_done, info = (
        env.step(action())
    )
    status = info[
        "ccda_execution_status"
    ]

    assert legacy_done
    assert status[
        "primitive_succeeded"
    ]
    assert status[
        "task_success"
    ]
    assert status[
        "episode_terminated"
    ]
    assert status[
        "termination_reason"
    ] == "task_success"
