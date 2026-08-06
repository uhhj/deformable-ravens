import numpy as np
import pytest

from ravens.tasks.ccda_occp_geometry import (
    CONDITIONS, OCCPGeometryConfig, build_common_bead_polyline,
    compute_occp_layout, hidden_edge_angles, interpolate_polyline_y_at_x)


def test_occp_conditions_are_exact():
    assert CONDITIONS == ('free', 'right_hidden_jam')
    with pytest.raises(ValueError):
        compute_occp_layout(OCCPGeometryConfig(), 'hidden_friction')


def test_segment_lengths_are_exact():
    config = OCCPGeometryConfig()
    positions, _ = build_common_bead_polyline(config)
    lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    assert np.allclose(lengths, config.cable_radius * config.spacing_scale,
                       atol=1e-12, rtol=0)


def test_hidden_bend_returns_to_centerline():
    config = OCCPGeometryConfig()
    positions, _ = build_common_bead_polyline(config)
    exit_index = config.hidden_start_index + config.hidden_edge_count
    assert np.max(positions[:, 1]) > config.cable_center_y
    assert positions[exit_index, 1] == pytest.approx(
        config.cable_center_y, abs=1e-12)
    assert np.any(hidden_edge_angles(config) > 0)
    assert np.any(hidden_edge_angles(config) < 0)


def test_polyline_interpolation_and_bounds():
    positions, _ = build_common_bead_polyline(OCCPGeometryConfig())
    x = 0.5 * (positions[10, 0] + positions[11, 0])
    assert interpolate_polyline_y_at_x(positions, x) == pytest.approx(
        0.5 * (positions[10, 1] + positions[11, 1]))
    with pytest.raises(ValueError):
        interpolate_polyline_y_at_x(positions, positions[0, 0] - 1e-3)


def test_initial_slack_is_between_two_and_four_spacings():
    layout = compute_occp_layout(OCCPGeometryConfig(), 'free')
    assert 2 * layout['spacing'] <= layout['initial_slack'] <= 4 * layout['spacing']


def test_pin_dimensions_location_and_clearances():
    config = OCCPGeometryConfig()
    free = compute_occp_layout(config, 'free')
    jam = compute_occp_layout(config, 'right_hidden_jam')
    spacing_gap = (free['active_exit_x'] - free['pin_center'][0]) / free['spacing']
    assert 4 <= spacing_gap <= 6
    assert free['pin_radius'] == pytest.approx(
        config.pin_radius_diameter_scale * 2 * config.cable_radius)
    assert jam['jam_initial_surface_clearance'] == pytest.approx(
        config.jam_initial_clearance_diameter_scale * 2 * config.cable_radius)
    assert free['free_initial_surface_clearance'] == pytest.approx(
        config.free_initial_clearance_diameter_scale * 2 * config.cable_radius)
    assert jam['jam_predicted_straight_penetration'] > 0
    assert free['free_predicted_straight_clearance'] > 0


def test_visible_readout_is_active_side_outside_occluder():
    layout = compute_occp_layout(OCCPGeometryConfig(), 'free')
    assert layout['visible_readout_indices'] == [22, 23, 24, 25, 26, 27]
    assert all(layout['bead_positions'][i, 0] > layout['occluder_x_max']
               for i in layout['visible_readout_indices'])
    assert layout['bead_positions'][layout['active_endpoint_index'], 0] > layout['occluder_x_max']


def test_occluder_covers_hidden_path_and_both_pin_positions():
    layout = compute_occp_layout(OCCPGeometryConfig(), 'free')
    center, half = layout['occluder_center'], layout['occluder_half_extents']
    for pin in (layout['free_pin_center'], layout['jam_pin_center']):
        assert center[0] - half[0] <= pin[0] <= center[0] + half[0]
        assert center[1] - half[1] <= pin[1] - layout['pin_radius']
        assert center[1] + half[1] >= pin[1] + layout['pin_radius']


def test_free_and_jam_layouts_differ_only_in_pin_position():
    free = compute_occp_layout(OCCPGeometryConfig(), 'free')
    jam = compute_occp_layout(OCCPGeometryConfig(), 'right_hidden_jam')
    for key in ('bead_positions', 'bead_yaw', 'visible_mask',
                'occluder_center', 'occluder_half_extents', 'probe_delta',
                'test_delta'):
        assert np.array_equal(free[key], jam[key])
    assert free['visible_readout_indices'] == jam['visible_readout_indices']
    assert not np.array_equal(free['pin_center'], jam['pin_center'])
