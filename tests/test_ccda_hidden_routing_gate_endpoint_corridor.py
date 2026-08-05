import json
from pathlib import Path

import numpy as np
import pytest

from ravens.tasks.ccda_hidden_routing_gate_geometry import (
    ENDPOINT_CORRIDOR_BARRIER_MODE,
    WHOLE_CABLE_BARRIER_MODE,
    HiddenRoutingGateGeometryConfig,
    compute_hidden_routing_gate_layout,
    evaluate_hidden_routing_gate_candidates,
    public_routing_layout,
)


HERE = Path(__file__).resolve().parent
FIXTURE = (
    HERE
    / "data"
    / (
        "ccda_hidden_routing_gate_"
        "seed71001_settled_beads.json"
    )
)


def beads():
    return np.asarray(
        json.loads(
            FIXTURE.read_text(
                encoding="utf-8"
            )
        ),
        dtype=np.float64,
    )


def endpoint_config():
    return HiddenRoutingGateGeometryConfig(
        center_ratio=0.45,
        probe_roof_clearance=0.002,
        probe_roof_depth=0.022,
        probe_roof_width=0.022,
        probe_roof_thickness=0.002,
        barrier_offset=0.065,
        barrier_thickness=0.004,
        barrier_width=0.0,
        barrier_height=0.030,
        stage1_pull_distance=0.080,
        final_pull_distance=0.120,
        target_plane_offset=0.085,
        target_zone_depth=0.025,
        target_corridor_half_width=0.035,
        leading_segment_size=4,
        workspace_x=(0.25, 0.75),
        workspace_y=(-0.45, 0.45),
        barrier_mode=(
            ENDPOINT_CORRIDOR_BARRIER_MODE
        ),
        barrier_safety_margin=0.005,
        topology_id=(
            "hidden_routing_gate_v1r2"
        ),
    )


def test_endpoint_mode_derives_fixed_width():
    config = endpoint_config()
    expected = (
        2.0 * (
            0.035
            + np.sqrt(3.0) * 0.005
            + 0.005
        )
    )
    assert np.isclose(
        config.resolved_barrier_width,
        expected,
        atol=1e-15,
    )
    assert np.isclose(
        config.resolved_barrier_width,
        0.09732050807568877,
        atol=1e-15,
    )
    assert np.isclose(
        config.corridor_required_width,
        0.08732050807568878,
        atol=1e-15,
    )


def test_endpoint_mode_rejects_manual_width():
    config = endpoint_config()
    values = dict(
        config.__dict__
    )
    values["barrier_width"] = 0.1
    with pytest.raises(
        ValueError,
        match="derives width",
    ):
        HiddenRoutingGateGeometryConfig(
            **values
        )


def test_endpoint_mode_requires_safety_margin():
    config = endpoint_config()
    values = dict(
        config.__dict__
    )
    values[
        "barrier_safety_margin"
    ] = 0.0
    with pytest.raises(
        ValueError,
        match="positive fixed",
    ):
        HiddenRoutingGateGeometryConfig(
            **values
        )


def test_real_seed_has_two_legal_candidates():
    audit = (
        evaluate_hidden_routing_gate_candidates(
            beads(),
            endpoint_config(),
        )
    )
    assert audit[
        "candidate_count"
    ] == 4
    assert audit[
        "accepted_candidate_count"
    ] == 2
    assert audit[
        "rejection_counts"
    ] == {
        "workspace": 2,
    }


def test_accepted_candidates_have_fixed_corridor_margin():
    audit = (
        evaluate_hidden_routing_gate_candidates(
            beads(),
            endpoint_config(),
        )
    )
    accepted = [
        row
        for row in audit["candidates"]
        if row["accepted"]
    ]
    assert len(accepted) == 2
    for row in accepted:
        assert row["barrier_mode"] == (
            ENDPOINT_CORRIDOR_BARRIER_MODE
        )
        assert row[
            "coverage_reference"
        ] == (
            "pulled_endpoint_target_corridor"
        )
        assert np.isclose(
            row["coverage_margin"],
            0.005,
            atol=1e-12,
        )
        assert np.isclose(
            row[
                "barrier_tangent_center_error"
            ],
            0.0,
            atol=1e-12,
        )
        assert (
            row[
                "component_workspace_margin"
            ]["routing_barrier"]
            > 0
        )
        assert (
            row["minimum_clearance"]
            >= row[
                "expected_minimum_clearance"
            ] - 1e-9
        )


def test_real_seed_inward_candidate_margins():
    audit = (
        evaluate_hidden_routing_gate_candidates(
            beads(),
            endpoint_config(),
        )
    )
    accepted = {
        int(row["endpoint_index"]): row
        for row in audit["candidates"]
        if row["accepted"]
    }
    assert set(accepted) == {0, 23}

    first = accepted[0]
    assert np.isclose(
        first[
            "component_workspace_margin"
        ]["routing_barrier"],
        0.06549782746555344,
        atol=1e-12,
    )
    assert np.isclose(
        first[
            "component_workspace_margin"
        ]["final_target"],
        0.07693698452907227,
        atol=1e-12,
    )

    last = accepted[23]
    assert np.isclose(
        last[
            "component_workspace_margin"
        ]["routing_barrier"],
        0.03939921547581432,
        atol=1e-12,
    )
    assert np.isclose(
        last[
            "component_workspace_margin"
        ]["final_target"],
        0.047303782632164226,
        atol=1e-12,
    )


def test_full_cable_width_is_diagnostic_only():
    audit = (
        evaluate_hidden_routing_gate_candidates(
            beads(),
            endpoint_config(),
        )
    )
    accepted = [
        row
        for row in audit["candidates"]
        if row["accepted"]
    ]
    assert accepted
    for row in accepted:
        assert (
            row[
                "full_cable_required_width_diagnostic"
            ]
            > row[
                "required_barrier_width"
            ]
        )
        assert (
            row[
                "barrier_width_shortfall"
            ]
            == 0.0
        )


def test_public_layout_does_not_leak_constructor():
    layout = (
        compute_hidden_routing_gate_layout(
            beads(),
            endpoint_config(),
        )
    )
    public = public_routing_layout(
        layout
    )
    text = json.dumps(
        public,
        sort_keys=True,
    ).lower()
    for forbidden in (
        "barrier",
        "safety",
        "coverage",
        "geometry_audit",
        "boxes",
    ):
        assert forbidden not in text


def test_whole_cable_mode_remains_available():
    config = HiddenRoutingGateGeometryConfig(
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
        barrier_mode=(
            WHOLE_CABLE_BARRIER_MODE
        ),
        barrier_safety_margin=0.0,
    )
    assert (
        config.resolved_barrier_width
        == 0.350
    )
