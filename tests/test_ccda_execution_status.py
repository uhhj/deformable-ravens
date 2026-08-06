from ravens.environment import Environment


class FakeTask:
    def __init__(self, task_success=False):
        self.task_success = task_success
        self.exit_gracefully = False
        self.def_IDs = []

    def reward(self):
        return 0., {}

    def done(self):
        return self.task_success


class StatusEnvironment(Environment):
    @property
    def info(self):
        return {}


def make_env(primitive_success, task_success=False):
    env = object.__new__(StatusEnvironment)
    env.task = FakeTask(task_success)
    env.primitives = {'fake': lambda: primitive_success}
    env.deterministic = True
    env.post_action_settle_steps = 0
    return env


def action():
    return {'primitive': 'fake', 'params': {}}


def test_primitive_failure_is_not_task_success():
    _, _, done, info = make_env(False).step(action())
    status = info['ccda_execution_status']
    assert done and not status['primitive_succeeded']
    assert not status['task_success']
    assert status['termination_reason'] == 'primitive_failed'


def test_successful_action_can_leave_task_running():
    _, _, done, info = make_env(True).step(action())
    status = info['ccda_execution_status']
    assert not done and status['primitive_succeeded']
    assert not status['task_success']


def test_task_success_is_separate_from_action_success():
    _, _, done, info = make_env(True, True).step(action())
    status = info['ccda_execution_status']
    assert done and status['primitive_succeeded'] and status['task_success']
    assert status['termination_reason'] == 'task_success'
