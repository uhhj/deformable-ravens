import numpy as np

from ravens.tasks.ccda_occp_geometry import OCCPGeometryConfig, compute_occp_layout


def test_default_hidden_bend_is_common_to_both_conditions():
    config = OCCPGeometryConfig()
    free = compute_occp_layout(config, 'free')
    jam = compute_occp_layout(config, 'right_hidden_jam')
    assert np.array_equal(free['bead_positions'], jam['bead_positions'])
    assert free['initial_slack'] > 0


def test_predicted_probe_sweep_selects_only_jam_pin():
    config = OCCPGeometryConfig()
    free = compute_occp_layout(config, 'free')
    jam = compute_occp_layout(config, 'right_hidden_jam')
    assert jam['initial_path_y_at_pin'] > jam['jam_pin_center'][1]
    assert jam['jam_predicted_straight_penetration'] > 0
    assert free['free_predicted_straight_clearance'] > 0


def test_visible_mask_marks_occluder_and_active_readout():
    layout = compute_occp_layout(OCCPGeometryConfig(), 'free')
    assert all(layout['visible_mask'][i] == 0 for i in layout['occluded_bead_indices'])
    assert all(layout['visible_mask'][i] == 1 for i in layout['visible_readout_indices'])
