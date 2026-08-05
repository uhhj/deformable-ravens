import json
from pathlib import Path

import numpy as np

from ravens.tasks.ccda_hidden_routing_gate_geometry import (
    ALL_BEAD_CLEARANCE_PROBE_SELECTOR,
    ENDPOINT_CORRIDOR_BARRIER_MODE,
    FIXED_CENTER_RATIO_PROBE_SELECTOR,
    HiddenRoutingGateGeometryConfig,
    compute_hidden_routing_gate_layout,
    evaluate_hidden_routing_gate_candidates,
    public_routing_layout,
    select_probe_roof,
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


def config(
    selector,
    override=-1,
):
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
        probe_selector_mode=selector,
        probe_index_override=override,
        barrier_mode=(
            ENDPOINT_CORRIDOR_BARRIER_MODE
        ),
        barrier_safety_margin=0.005,
        topology_id=(
            "hidden_routing_gate_v1r3"
        ),
    )


def test_fixed_center_probe_matches_resume2_failure():
    selected = select_probe_roof(
        beads(),
        config(
            FIXED_CENTER_RATIO_PROBE_SELECTOR
        ),
    )
    assert selected[
        "preferred_probe_index"
    ] == 10
    assert selected[
        "probe_index"
    ] == 10
    assert selected[
        "nearest_bead_index"
    ] == 15
    assert np.isclose(
        selected[
            "all_bead_clearance"
        ],
        -0.005153851761092307,
        atol=1e-12,
    )


def test_all_bead_selector_chooses_index_7():
    selected = select_probe_roof(
        beads(),
        config(
            ALL_BEAD_CLEARANCE_PROBE_SELECTOR
        ),
    )
    assert selected[
        "preferred_probe_index"
    ] == 10
    assert selected[
        "probe_index"
    ] == 7
    assert (
        selected[
            "all_bead_clearance"
        ]
        >= 0.002
    )
    assert 7 in selected[
        "legal_probe_indices"
    ]


def test_override_reuses_selected_probe_index():
    selected = select_probe_roof(
        beads(),
        config(
            ALL_BEAD_CLEARANCE_PROBE_SELECTOR,
            override=7,
        ),
    )
    assert selected[
        "probe_index"
    ] == 7


def test_resume3_geometry_has_legal_candidate():
    audit = (
        evaluate_hidden_routing_gate_candidates(
            beads(),
            config(
                ALL_BEAD_CLEARANCE_PROBE_SELECTOR
            ),
        )
    )
    assert audit[
        "accepted_candidate_count"
    ] >= 1
    assert audit[
        "probe_selection"
    ][
        "selected_probe_index"
    ] == 7
    assert audit[
        "probe_selection"
    ][
        "selected_all_bead_clearance"
    ] >= 0.002


def test_resume3_layout_keeps_endpoint_corridor():
    layout = (
        compute_hidden_routing_gate_layout(
            beads(),
            config(
                ALL_BEAD_CLEARANCE_PROBE_SELECTOR
            ),
        )
    )
    assert layout[
        "barrier_mode"
    ] == (
        ENDPOINT_CORRIDOR_BARRIER_MODE
    )
    assert layout[
        "selected_probe_index"
    ] == 7
    assert np.isclose(
        layout[
            "resolved_barrier_width"
        ],
        0.09732050807568877,
    )


def test_public_layout_does_not_leak_selector_audit():
    layout = (
        compute_hidden_routing_gate_layout(
            beads(),
            config(
                ALL_BEAD_CLEARANCE_PROBE_SELECTOR
            ),
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
        "legal_probe",
        "nearest_bead",
        "all_bead_clearance",
        "geometry_audit",
        "barrier",
    ):
        assert forbidden not in text
