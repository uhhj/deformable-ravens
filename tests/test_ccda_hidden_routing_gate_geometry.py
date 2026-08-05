import json
from dataclasses import replace

import numpy as np
import pytest

from ravens.tasks.ccda_hidden_routing_gate_geometry import (
    HiddenRoutingGateGeometryError,
    HiddenRoutingGateGeometryConfig,
    compute_hidden_routing_gate_layout,
    evaluate_hidden_routing_gate_candidates,
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
        barrier_width=0.350,
        barrier_height=0.030,
        stage1_pull_distance=0.080,
        final_pull_distance=0.120,
        target_plane_offset=0.085,
        target_zone_depth=0.025,
        target_corridor_half_width=0.035,
        leading_segment_size=4,
        workspace_x=(0.25, 0.75),
        workspace_y=(-0.45, 0.45),
        topology_id="hidden_routing_gate_v1r1",
    )


def beads():
    radius = 0.005
    count = 24
    distance = 2.0 * radius * np.sqrt(2.0)
    x = 0.34 + np.arange(count) * distance
    return np.column_stack((
        x,
        np.zeros(count),
        np.full(count, radius),
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
    assert np.isclose(barrier["half_extents"][0], 0.175)
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
    expected = list(range(4)) if public["endpoint_index"] == 0 else list(range(20, 24))
    assert public["leading_segment_indices"] == expected


def test_all_geometry_and_targets_fit_workspace():
    result = layout()
    assert result["workspace_margin"] > 0
    assert result["expected_surface_clearance"] >= 0.002 - 1e-9


def test_layout_is_deterministic():
    first = layout()
    second = layout()
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_original_0340_width_is_analytically_too_short():
    original = replace(
        config(), barrier_width=0.340, topology_id="hidden_routing_gate_v1"
    )
    audit = evaluate_hidden_routing_gate_candidates(beads(), original)
    assert audit["accepted_candidate_count"] == 0
    assert audit["candidate_count"] == 4
    assert audit["rejection_counts"] == {"barrier_coverage": 4}
    required = {
        round(row["required_barrier_width"], 12)
        for row in audit["candidates"]
    }
    assert len(required) == 1
    assert np.isclose(next(iter(required)), 0.3425896274215007)
    assert all(row["coverage_margin"] < 0 for row in audit["candidates"])


def test_original_failure_contains_all_candidate_provenance():
    original = replace(
        config(), barrier_width=0.340, topology_id="hidden_routing_gate_v1"
    )
    with pytest.raises(HiddenRoutingGateGeometryError) as captured:
        compute_hidden_routing_gate_layout(beads(), original)
    diagnostics = captured.value.to_dict()["diagnostics"]
    assert diagnostics["candidate_count"] == 4
    assert diagnostics["accepted_candidate_count"] == 0
    assert len(diagnostics["bead_positions_xyz"]) == 24
    assert all(
        row["rejection_reasons"] == ["barrier_coverage"]
        for row in diagnostics["candidates"]
    )


def test_resume1_0350_width_has_positive_coverage():
    result = layout()
    assert result["barrier_tangent_coverage_margin"] > 0.003
    assert result["barrier_width_shortfall"] == 0.0
    assert result["geometry_audit"]["accepted_candidate_count"] >= 1


def test_geometry_audit_records_component_workspace_margins():
    rows = layout()["geometry_audit"]["candidates"]
    expected = {
        "stage1_target", "final_target", "target_plane", "routing_barrier",
        "probe_roof", "target_zone",
    }
    assert len(rows) == 4
    assert all(set(row["component_workspace_margin"]) == expected for row in rows)


def test_public_layout_does_not_expose_geometry_audit():
    serialized = json.dumps(public_routing_layout(layout()), sort_keys=True).lower()
    assert "geometry_audit" not in serialized
    assert "required_barrier_width" not in serialized
    assert "rejection" not in serialized
