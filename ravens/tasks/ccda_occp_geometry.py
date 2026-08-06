"""Pure geometry for the clean single-pin OCCP audit task."""
from dataclasses import dataclass

import numpy as np


CONDITIONS = ('free', 'right_hidden_jam')


@dataclass(frozen=True)
class OCCPGeometryConfig:
    cable_radius: float = 0.005
    num_beads: int = 24
    bead_mass: float = 0.05
    spacing_scale: float = 2.15
    passive_endpoint_x: float = 0.34
    cable_center_y: float = 0.0
    cable_z: float = 0.012
    pin_radius_scale: float = 2.0
    pin_height: float = 0.06
    pin_from_active_exit_spacing: float = 5.0
    jam_clearance_scale: float = 0.10
    free_clearance_scale: float = 1.00
    occluder_length_x: float = 0.16
    occluder_width_y: float = 0.14
    occluder_bottom_z: float = 0.025
    occluder_height: float = 0.08
    visible_readout_count: int = 6
    probe_spacing_scale: float = 0.75
    test_spacing_scale: float = 6.0
    test_angle_deg: float = 18.0


def validate_occp_config(config):
    if config.cable_radius <= 0 or config.bead_mass <= 0:
        raise ValueError('cable radius and bead mass must be positive')
    if config.spacing_scale <= 2.0:
        raise ValueError('spacing_scale must exceed two radii')
    if config.num_beads < 8:
        raise ValueError('num_beads must be at least 8')
    if not 1 <= config.visible_readout_count < config.num_beads:
        raise ValueError('visible_readout_count is invalid')
    if config.pin_radius_scale <= 0 or config.pin_height <= 0:
        raise ValueError('pin dimensions must be positive')
    if config.occluder_length_x <= 0 or config.occluder_width_y <= 0:
        raise ValueError('occluder dimensions must be positive')
    if config.occluder_height <= 0 or config.occluder_bottom_z <= 0:
        raise ValueError('occluder vertical dimensions must be positive')
    if not 0 < config.probe_spacing_scale <= 2:
        raise ValueError('probe_spacing_scale is outside the audit range')
    if not 2 <= config.test_spacing_scale <= 10:
        raise ValueError('test_spacing_scale is outside the audit range')
    if not 0 < config.test_angle_deg < 45:
        raise ValueError('test_angle_deg is outside the audit range')

    spacing = config.cable_radius * config.spacing_scale
    active_x = config.passive_endpoint_x + spacing * (config.num_beads - 1)
    pin_x = active_x - config.pin_from_active_exit_spacing * spacing
    if config.passive_endpoint_x < 0.25 or active_x > 0.75:
        raise ValueError('cable leaves the Ravens workspace')
    if not config.passive_endpoint_x < pin_x < active_x:
        raise ValueError('pin must lie between cable endpoints')
    if abs(config.cable_center_y) + config.occluder_width_y / 2 > 0.5:
        raise ValueError('occluder leaves the Ravens workspace')


def compute_occp_layout(config, condition):
    validate_occp_config(config)
    if condition not in CONDITIONS:
        raise ValueError('condition must be one of {}'.format(CONDITIONS))

    cable_diameter = 2 * config.cable_radius
    spacing = config.cable_radius * config.spacing_scale
    bead_positions = np.zeros((config.num_beads, 3), dtype=np.float64)
    bead_positions[:, 0] = (
        config.passive_endpoint_x
        + np.arange(config.num_beads, dtype=np.float64) * spacing)
    bead_positions[:, 1] = config.cable_center_y
    bead_positions[:, 2] = config.cable_z
    active_index = config.num_beads - 1
    pin_radius = config.pin_radius_scale * config.cable_radius
    clearance_scale = (
        config.jam_clearance_scale
        if condition == 'right_hidden_jam'
        else config.free_clearance_scale)
    pin_y = (
        config.cable_center_y + pin_radius + config.cable_radius
        + clearance_scale * config.cable_radius)
    pin_x = (
        bead_positions[active_index, 0]
        - config.pin_from_active_exit_spacing * spacing)
    pin_center = np.array(
        [pin_x, pin_y, config.pin_height / 2], dtype=np.float64)
    occluder_center = np.array([
        pin_x,
        config.cable_center_y,
        config.occluder_bottom_z + config.occluder_height / 2,
    ], dtype=np.float64)
    occluder_half_extents = np.array([
        config.occluder_length_x / 2,
        config.occluder_width_y / 2,
        config.occluder_height / 2,
    ], dtype=np.float64)
    test_distance = config.test_spacing_scale * spacing
    angle = np.deg2rad(config.test_angle_deg)
    return {
        'bead_positions': bead_positions,
        'spacing': float(spacing),
        'cable_diameter': float(cable_diameter),
        'active_endpoint_index': int(active_index),
        'passive_endpoint_index': 0,
        'visible_readout_indices': list(range(config.visible_readout_count)),
        'pin_center': pin_center,
        'pin_radius': float(pin_radius),
        'pin_height': float(config.pin_height),
        'occluder_center': occluder_center,
        'occluder_half_extents': occluder_half_extents,
        'probe_delta': np.array(
            [config.probe_spacing_scale * spacing, 0., 0.]),
        'test_delta': np.array([
            test_distance * np.cos(angle),
            test_distance * np.sin(angle),
            0.,
        ]),
    }
