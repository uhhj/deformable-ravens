import inspect

import pytest

from ravens import tasks
from ravens.environment import Environment
from ravens.tasks.ccda_hidden_hook_cable import CCDAHiddenHookCable
from ravens.tasks.ccda_hidden_latch_cable import CCDAHiddenLatchCable


def test_latch_task_is_registered_without_replacing_prior_tasks():
    assert tasks.names["ccda-hidden-latch-cable"] is CCDAHiddenLatchCable
    assert tasks.names["ccda-hidden-hook-cable"] is CCDAHiddenHookCable
    task = CCDAHiddenLatchCable()
    assert task.CONDITIONS == ("free", "hidden_hook")
    assert task._name == "ccda-hidden-latch-cable"


def test_latch_reuses_invisible_equal_body_and_delayed_arming_policy():
    create_source = inspect.getsource(CCDAHiddenHookCable._create_invisible_hook)
    arm_source = inspect.getsource(CCDAHiddenHookCable.arm_ccda_hidden_factor_after_settle)
    assert "baseVisualShapeIndex=-1" in create_source
    assert "self._hook_layout[\"boxes\"]" in create_source
    assert 'enabled = self.hidden_condition == "hidden_hook"' in arm_source
    assert "int(enabled)" in arm_source
    assert "before = self._ordered_bead_positions()" in arm_source
    assert "after = self._ordered_bead_positions()" in arm_source


def test_privileged_latch_state_is_separate_from_formal_sensor_and_trace_has_reaction():
    sensor_source = inspect.getsource(Environment.ccda_sensor_observation)
    for forbidden in ("latch_layout", "latch_body_ids", "hidden_condition"):
        assert forbidden not in sensor_source
    trace_source = inspect.getsource(CCDAHiddenHookCable.physics_step_hook)
    assert "sensor_joint_reaction_force_torque" in trace_source


def test_topology_environment_is_dynamic_and_phase0i_default_is_unchanged(monkeypatch):
    monkeypatch.delenv("CCDA_LATCH_TOPOLOGY_ID", raising=False)
    task = CCDAHiddenLatchCable()
    assert task._latch_config().topology_id == "delayed_z_latch_v1"
    monkeypatch.setenv("CCDA_LATCH_TOPOLOGY_ID", "wide_stop_z_latch_v2")
    assert task._latch_config().topology_id == "wide_stop_z_latch_v2"


def test_privileged_topology_comes_from_layout_and_formal_sensor_ignores_it(monkeypatch):
    monkeypatch.setattr(
        CCDAHiddenHookCable,
        "ccda_privileged_state",
        lambda self: {
            "hook_layout": {"topology": "wide_stop_z_latch_v2"},
            "hook_body_ids": [],
            "hook_collision_enabled": False,
            "hook_initial_clearance": 0.002,
            "hook_visual_shape_indices": [],
            "hook_visual_shape_disabled": True,
        },
    )
    state = CCDAHiddenLatchCable().ccda_privileged_state()
    assert state["topology"] == "wide_stop_z_latch_v2"
    assert state["latch_layout"]["topology"] == "wide_stop_z_latch_v2"
    sensor_source = inspect.getsource(Environment.ccda_sensor_observation)
    assert "topology" not in sensor_source
