import inspect

from ravens import tasks
from ravens.environment import Environment
from ravens.tasks.ccda_hidden_hook_cable import CCDAHiddenHookCable
from ravens.tasks.ccda_hidden_routing_gate_cable import CCDAHiddenRoutingGateCable


def test_task_is_registered():
    assert tasks.names["ccda-hidden-routing-gate-cable"] is CCDAHiddenRoutingGateCable
    assert CCDAHiddenRoutingGateCable().CONDITIONS == ("free", "hidden_hook")


def test_hidden_fixture_has_no_visual_shapes():
    source = inspect.getsource(CCDAHiddenHookCable._create_invisible_hook)
    assert "baseVisualShapeIndex=-1" in source


def test_target_zone_has_visual_shape_and_no_collision():
    source = inspect.getsource(CCDAHiddenRoutingGateCable._create_visible_target_zone)
    assert "createVisualShape" in source
    assert "baseCollisionShapeIndex=-1" in source
    assert '"ccda_public_routing_target_zone"' in source


def test_public_task_state_has_no_hidden_geometry():
    task = CCDAHiddenRoutingGateCable()
    task._hook_layout = {"public_task": {"topology_id": "hidden_routing_gate_v1"}}
    state = task.ccda_public_task_state()
    assert state["target_zone_has_collision"] is False
    assert "routing_gate_layout_privileged" not in state
    assert "boxes" not in str(state)


def test_privileged_state_separates_public_and_hidden_layout(monkeypatch):
    monkeypatch.setattr(
        CCDAHiddenHookCable,
        "ccda_privileged_state",
        lambda self: {
            "hook_layout": {
                "topology": "hidden_routing_gate_v1",
                "public_task": {"topology_id": "hidden_routing_gate_v1"},
                "boxes": [{"name": "routing_barrier"}],
            },
            "hook_body_ids": [2, 3],
            "hook_collision_enabled": True,
            "hook_initial_clearance": 0.002,
            "hook_visual_shape_indices": [-1, -1],
            "hook_visual_shape_disabled": True,
        },
    )
    state = CCDAHiddenRoutingGateCable().ccda_privileged_state()
    assert "boxes" not in str(state["public_task"])
    assert state["routing_gate_layout_privileged"]["boxes"]
    assert state["routing_gate_visual_shape_disabled"] is True


def test_free_and_hidden_conditions_control_fixture_collision():
    source = inspect.getsource(CCDAHiddenHookCable.arm_ccda_hidden_factor_after_settle)
    assert 'enabled = self.hidden_condition == "hidden_hook"' in source
    assert "int(enabled)" in source


def test_formal_sensor_observation_has_no_topology_or_condition():
    source = inspect.getsource(Environment.ccda_sensor_observation)
    for forbidden in ("topology", "hidden_condition", "routing_gate"):
        assert forbidden not in source
