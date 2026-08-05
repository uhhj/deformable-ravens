import json

import numpy as np

from ravens.tasks.ccda_hidden_routing_gate_geometry import (
    HiddenRoutingGateGeometryConfig,
    compute_hidden_routing_gate_layout,
    public_routing_layout,
)


def config():
    return HiddenRoutingGateGeometryConfig(
        center_ratio=0.45,
        probe_roof_clearance=0.002,
        probe_roof_depth=0.022,
        probe_roof_width=0.022,
        probe_roof_thickness=0.002,
        barrier_offset=0.065,
        barrier_thickness=0.004,
        barrier_width=0.340,
        barrier_height=0.030,
        stage1_pull_distance=0.080,
        final_pull_distance=0.120,
        target_plane_offset=0.085,
        target_zone_depth=0.025,
        target_corridor_half_width=0.035,
        leading_segment_size=4,
        workspace_x=(0.25, 0.75),
        workspace_y=(-0.45, 0.45),
    )


def beads():
    return np.column_stack((
        np.linspace(0.35, 0.65, 25),
        np.zeros(25),
        np.full(25, 0.005),
    ))


def layout():
    return compute_hidden_routing_gate_layout(beads(), config())


def test_public_layout_contains_no_hidden_geometry():
    public = public_routing_layout(layout())
    serialized = json.dumps(public, sort_keys=True).lower()
    for forbidden in ("boxes", "routing_barrier", "probe_roof"):
        assert forbidden not in serialized


def test_barrier_covers_entire_initial_cable_tangent_span():
    result = layout()
    barrier = next(box for box in result["boxes"] if box["name"] == "routing_barrier")
    assert np.isclose(barrier["half_extents"][0], 0.170)
    assert result["barrier_tangent_coverage_margin"] > 0


def test_target_plane_and_final_target_are_beyond_barrier():
    public = public_routing_layout(layout())
    start = beads()[public["endpoint_index"], :2]
    normal = np.asarray(public["normal_xy"])
    plane = np.dot(np.asarray(public["target_plane_point_xy"]) - start, normal)
    final = np.dot(np.asarray(public["final_target_xy"]) - start, normal)
    assert 0.065 + 0.004 / 2 + 0.005 < plane
    assert plane < final


def test_stage1_and_final_targets_are_collinear():
    public = public_routing_layout(layout())
    start = beads()[public["endpoint_index"], :2]
    stage1 = np.asarray(public["stage1_target_xy"])
    final = np.asarray(public["final_target_xy"])
    assert np.isclose(np.linalg.norm(stage1 - start), 0.080)
    assert np.isclose(np.linalg.norm(final - start), 0.120)
    assert np.allclose((stage1 - start) / 0.080, (final - start) / 0.120)


def test_leading_segment_follows_selected_endpoint():
    public = public_routing_layout(layout())
    expected = list(range(4)) if public["endpoint_index"] == 0 else list(range(21, 25))
    assert public["leading_segment_indices"] == expected


def test_all_geometry_and_targets_fit_workspace():
    result = layout()
    assert result["workspace_margin"] > 0
    assert result["expected_surface_clearance"] >= 0.002 - 1e-9


def test_layout_is_deterministic():
    first = layout()
    second = layout()
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
