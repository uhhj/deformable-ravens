import pytest

from ravens.tasks.ccda_hidden_friction_cable import CCDAHiddenFrictionCable


def test_reset_ccda_branch_preserves_geometry_metadata():
    task = CCDAHiddenFrictionCable()
    sentinel_env = object()
    task._env = sentinel_env
    task.cable_bead_IDs = [4, 5, 6]
    task.ccda_visible_seed = "70001"
    task.ccda_pair_group = "hf_070001"
    task._trace_stride = 4

    task._physics_step_count = 99
    task._phase = "main_pull"
    task._trace = [{"physics_step": 4}]
    task._hidden_friction_armed = True
    task._selected_local_indices = [1]
    task._selected_bead_ids = [5]
    task.total_rewards = 0.7
    task.exit_gracefully = True
    task.t = 2

    task.reset_ccda_branch("hidden_high_friction")

    assert task.hidden_condition == "hidden_high_friction"
    assert task._env is sentinel_env
    assert task.cable_bead_IDs == [4, 5, 6]
    assert task.ccda_visible_seed == "70001"
    assert task.ccda_pair_group == "hf_070001"
    assert task._trace_stride == 4
    assert task.physics_step_count() == 0
    assert task.ccda_phase() == "no_action"
    assert task.ccda_trace() == []
    assert not task.ccda_is_armed()
    assert task.total_rewards == 0
    assert task.exit_gracefully is False
    assert task.t == 0


def test_reset_ccda_branch_rejects_unknown_condition():
    task = CCDAHiddenFrictionCable()
    with pytest.raises(ValueError):
        task.reset_ccda_branch("unknown")
