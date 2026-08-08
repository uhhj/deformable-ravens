import numpy as np

from ravens.tasks.ccda_dhr_geometry import (
    DHRGeometryConfig,
    build_condition_polyline,
    compute_dhr_layout,
)


def test_dhr_routes_keep_visible_geometry_and_edge_lengths():
    cfg = DHRGeometryConfig()

    free = compute_dhr_layout(
        cfg, "free")

    jam = compute_dhr_layout(
        cfg, "jam_right")

    visible = free[
        "visible_keypoint_indices"]

    np.testing.assert_allclose(
        free["bead_positions"][visible],
        jam["bead_positions"][visible],
        atol=1e-12,
        rtol=0.0)

    for condition in (
            "free",
            "jam_right"):
        positions, _, _ = (
            build_condition_polyline(
                cfg, condition))

        lengths = np.linalg.norm(
            np.diff(
                positions,
                axis=0),
            axis=1)

        np.testing.assert_allclose(
            lengths,
            cfg.spacing_m,
            atol=1e-12,
            rtol=0.0)


def test_dhr_jam_is_lower_hidden_route_with_same_boundary_directions():
    cfg = DHRGeometryConfig()

    free_positions, _, free_angles = (
        build_condition_polyline(
            cfg, "free"))

    jam_positions, _, jam_angles = (
        build_condition_polyline(
            cfg, "jam_right"))

    first_edge = (
        cfg.hidden_start_index - 1)

    last_edge = (
        cfg.hidden_end_index)

    assert np.isclose(
        free_angles[first_edge],
        jam_angles[first_edge])

    assert np.isclose(
        free_angles[last_edge],
        jam_angles[last_edge])

    hidden = slice(
        cfg.hidden_start_index,
        cfg.hidden_end_index + 1)

    assert (
        np.max(
            free_positions[
                hidden, 1])
        > 0.03)

    assert (
        np.min(
            jam_positions[
                hidden, 1])
        < -0.03)


def test_dhr_static_pocket_is_identical_across_conditions():
    cfg = DHRGeometryConfig()

    free = compute_dhr_layout(
        cfg, "free")

    jam = compute_dhr_layout(
        cfg, "jam_right")

    for key in (
        "pocket_bottom_center",
        "pocket_bottom_half_extents",
        "pocket_left_center",
        "pocket_right_center",
        "pocket_wall_half_extents",
    ):
        np.testing.assert_allclose(
            free[key],
            jam[key],
            atol=0.0,
            rtol=0.0)

    assert (
        free[
            "pocket_left_index"]
        == 12)

    assert (
        free[
            "pocket_right_index"]
        == 19)
