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


def test_dual_post_guide_is_dimension_derived_and_keeps_primary_post():
    base_cfg = OHJGeometryConfig(jam_surface_clearance_m=0.00025)
    dual_cfg = OHJGeometryConfig(
        jam_surface_clearance_m=0.00025,
        latch_topology="dual_post_directional_guide")
    base_jam = compute_ohj_layout(base_cfg, "jam_right")
    dual_jam = compute_ohj_layout(dual_cfg, "jam_right")
    dual_free = compute_ohj_layout(dual_cfg, "free")
    np.testing.assert_allclose(
        dual_jam["jam_latch_center"], base_jam["jam_latch_center"])
    latch_index = (
        dual_cfg.hidden_start_index + dual_cfg.hidden_end_index) // 2
    expected_offset = int(np.ceil(
        (2.0 * dual_cfg.latch_radius_m) / dual_cfg.spacing_m))
    expected_index = latch_index + expected_offset
    assert dual_jam["directional_guide_index"] == expected_index == 17
    positions = dual_jam["bead_positions"]
    yaw = dual_jam["bead_yaw"][expected_index]
    expected_half_y = (
        abs(np.sin(yaw)) * (0.45 * dual_cfg.spacing_m)
        + abs(np.cos(yaw)) * dual_cfg.cable_radius_m)
    assert np.isclose(
        dual_jam["directional_guide_y_half_extent"], expected_half_y)
    guide = dual_jam["jam_directional_guide_center"]
    actual_clearance = (
        positions[expected_index, 1] - expected_half_y
        - (guide[1] + dual_cfg.latch_radius_m))
    assert np.isclose(actual_clearance, dual_cfg.jam_surface_clearance_m)
    np.testing.assert_allclose(
        dual_free["directional_guide_center"]
        - dual_jam["directional_guide_center"],
        [0.0, dual_cfg.free_lateral_offset_m, 0.0])
    assert np.array_equal(
        dual_free["bead_positions"], dual_jam["bead_positions"])
