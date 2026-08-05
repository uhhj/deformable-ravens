import pytest

from ravens import tasks
from ravens.tasks.ccda_hidden_hook_cable import CCDAHiddenHookCable


def test_hidden_hook_task_is_registered():
    assert tasks.names["ccda-hidden-hook-cable"] is CCDAHiddenHookCable
    task = CCDAHiddenHookCable()
    assert task._name == "ccda-hidden-hook-cable"
    assert task.CONDITIONS == ("free", "hidden_hook")


@pytest.mark.parametrize("condition", ["free", "hidden_hook"])
def test_hidden_hook_branch_reset(condition):
    task = CCDAHiddenHookCable()
    task._hook_collision_enabled = True
    task.reset_ccda_branch(condition)
    assert task.hidden_condition == condition
    assert task._hook_collision_enabled is False
    assert task.ccda_trace() == []


def test_hidden_hook_branch_reset_rejects_unknown_condition():
    task = CCDAHiddenHookCable()
    with pytest.raises(ValueError):
        task.reset_ccda_branch("unknown")
