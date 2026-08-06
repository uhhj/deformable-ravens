import numpy as np
import pytest

from ravens.tasks.ccda_occp_geometry import (
    CONDITIONS, OCCPGeometryConfig, compute_occp_layout)


def test_occp_conditions_are_exact():
    assert CONDITIONS == ('free', 'right_hidden_jam')
    with pytest.raises(ValueError):
        compute_occp_layout(OCCPGeometryConfig(), 'hidden_friction')


def test_layout_differs_only_in_pin_lateral_position():
    config = OCCPGeometryConfig()
    free = compute_occp_layout(config, 'free')
    jam = compute_occp_layout(config, 'right_hidden_jam')
    assert np.array_equal(free['bead_positions'], jam['bead_positions'])
    assert free['pin_center'][0] == jam['pin_center'][0]
    assert free['pin_center'][2] == jam['pin_center'][2]
    assert free['pin_center'][1] > jam['pin_center'][1]
    assert np.array_equal(free['occluder_center'], jam['occluder_center'])


def test_ordered_bead_count_spacing_and_actions():
    config = OCCPGeometryConfig()
    layout = compute_occp_layout(config, 'free')
    beads = layout['bead_positions']
    assert beads.shape == (24, 3)
    assert np.allclose(np.diff(beads[:, 0]), layout['spacing'])
    assert layout['passive_endpoint_index'] == 0
    assert layout['active_endpoint_index'] == 23
    assert np.allclose(
        layout['probe_delta'], [0.75 * layout['spacing'], 0., 0.])
    assert np.linalg.norm(layout['test_delta']) == pytest.approx(
        6.0 * layout['spacing'])


def test_visible_readout_excludes_occluded_region():
    layout = compute_occp_layout(OCCPGeometryConfig(), 'free')
    indices = layout['visible_readout_indices']
    assert indices == [0, 1, 2, 3, 4, 5]
    visible_x = layout['bead_positions'][indices, 0]
    occluder_min_x = (
        layout['occluder_center'][0] - layout['occluder_half_extents'][0])
    assert np.max(visible_x) < occluder_min_x
