import json

import numpy as np
import pytest

from ravens.tasks.ccda_hidden_latch_geometry import (
    HiddenLatchGeometryConfig,
    compute_hidden_latch_layout,
)


def config():
    return HiddenLatchGeometryConfig(
        center_ratio=0.45, stop_clearance=0.002, roof_clearance=0.002,
        wall_thickness=0.002, wall_width=0.032, wall_height=0.020,
        roof_depth=0.020, roof_width=0.036, roof_thickness=0.002,
        main_pull_distance=0.08, workspace_x=(0.25, 0.75),
        workspace_y=(-0.45, 0.45),
    )


def beads():
    return np.column_stack((
        np.linspace(0.35, 0.65, 25), np.zeros(25), np.full(25, 0.005)
    ))


def test_layout_is_deterministic_legal_and_clear():
    first = compute_hidden_latch_layout(beads(), config())
    second = compute_hidden_latch_layout(beads(), config())
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["topology"] == "delayed_z_latch_v1"
    assert 0 < first["probe_index"] < 24
    assert first["endpoint_index"] in (0, 24)
    assert first["workspace_margin"] > 0
    assert first["roof_bottom_z"] > first["anchor_bead_center_z"] + 0.005
    assert first["expected_surface_clearance"] >= 0.002 - 1e-9
    assert len(first["boxes"]) == 4
    assert all(np.isfinite(box["center_z"]) for box in first["boxes"])


def test_invalid_latch_dimensions_fail():
    values = config().__dict__.copy()
    values["roof_thickness"] = 0
    with pytest.raises(ValueError):
        HiddenLatchGeometryConfig(**values)
    with pytest.raises(ValueError):
        compute_hidden_latch_layout(np.zeros((2, 3)), config())
