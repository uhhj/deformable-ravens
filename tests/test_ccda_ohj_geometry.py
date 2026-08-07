import numpy as np

from ravens.tasks.ccda_ohj_geometry import (
    CONDITIONS, VISIBLE_KEYPOINT_INDICES, OHJGeometryConfig,
    build_common_bead_polyline, compute_ohj_layout)


def test_ohj_geometry_contract_and_common_cable():
    cfg = OHJGeometryConfig()
    free = compute_ohj_layout(cfg, "free")
    jam = compute_ohj_layout(cfg, "jam_right")
    assert CONDITIONS == ("free", "jam_right")
    assert free["bead_positions"].shape == (32, 3)
    assert VISIBLE_KEYPOINT_INDICES == tuple(
        list(range(8)) + list(range(24, 32)))
    assert np.array_equal(free["bead_positions"], jam["bead_positions"])
    assert not np.array_equal(free["latch_center"], jam["latch_center"])
    assert free["active_endpoint_index"] == 31
    assert free["passive_endpoint_index"] == 0


def test_polyline_has_exact_spacing_and_condition_independent_layout():
    cfg = OHJGeometryConfig()
    positions, yaw = build_common_bead_polyline(cfg)
    assert np.allclose(np.linalg.norm(np.diff(positions, axis=0), axis=1),
                       cfg.spacing_m)
    assert yaw.shape == (32,)
    free = compute_ohj_layout(cfg, "free")
    jam = compute_ohj_layout(cfg, "jam_right")
    for key in ("bead_yaw", "visible_keypoint_indices",
                "hidden_bead_indices", "occluder_center",
                "occluder_half_extents", "pull_direction"):
        assert np.array_equal(free[key], jam[key])
