"""Pure geometry for the probe-triggered single-pin OCCP audit task."""
from dataclasses import dataclass

import numpy as np


CONDITIONS = ('free', 'right_hidden_jam')


@dataclass(frozen=True)
class OCCPGeometryConfig:
    cable_radius: float = 0.005
    num_beads: int = 28
    bead_mass: float = 0.05
    spacing_scale: float = 2.15
    passive_endpoint_x: float = 0.28
    cable_center_y: float = 0.0
    cable_z: float = 0.012

    hidden_start_index: int = 8
    hidden_edge_count: int = 13
    hidden_peak_tangent_deg: float = 55.0
    visible_readout_count: int = 6

    pin_radius_diameter_scale: float = 1.75
    pin_height: float = 0.06
    pin_from_active_exit_spacing: float = 4.5
    jam_initial_clearance_diameter_scale: float = 0.25
    free_initial_clearance_diameter_scale: float = 0.90

    collision_margin_m: float = 0.0005
    bead_lateral_friction: float = 0.50
    pin_lateral_friction: float = 0.90
    linear_damping: float = 0.04
    angular_damping: float = 0.90
    constraint_max_force: float = 100.0

    occluder_padding_x_spacing: float = 0.75
    occluder_padding_y_m: float = 0.025
    occluder_bottom_z: float = 0.025
    occluder_height: float = 0.08

    probe_spacing_scale: float = 1.0
    test_spacing_scale: float = 6.0
    test_angle_deg: float = 18.0


def validate_occp_config(config):
    dc = 2.0 * config.cable_radius
    exit_index = config.hidden_start_index + config.hidden_edge_count
    if config.cable_radius <= 0 or config.bead_mass <= 0:
        raise ValueError('cable radius and bead mass must be positive')
    if config.num_beads < 16 or config.spacing_scale <= 2.0:
        raise ValueError('invalid bead count or spacing scale')
    if config.hidden_start_index < 2 or config.hidden_edge_count < 6:
        raise ValueError('hidden bend is too short')
    if exit_index >= config.num_beads - 2:
        raise ValueError('active exit must precede the final two nodes')
    if config.num_beads - exit_index - 1 != config.visible_readout_count:
        raise ValueError(
            'visible_readout_count must exactly cover nodes outside active exit')
    if not 1.5 <= config.pin_radius_diameter_scale <= 2.5:
        raise ValueError('pin radius scale must be in cable diameters')
    if not 4.0 <= config.pin_from_active_exit_spacing <= 6.0:
        raise ValueError('pin must be four to six spacings from active exit')
    if not 0.0 <= config.jam_initial_clearance_diameter_scale <= 0.50:
        raise ValueError('jam clearance is outside the audit range')
    if config.free_initial_clearance_diameter_scale < 0.75:
        raise ValueError('free clearance is too small')
    if not 0.5 <= config.probe_spacing_scale <= 1.0:
        raise ValueError('probe distance is outside the audit range')
    if not 4.0 <= config.test_spacing_scale <= 8.0:
        raise ValueError('test distance is outside the audit range')
    if not 10.0 <= config.test_angle_deg <= 25.0:
        raise ValueError('test angle is outside the audit range')
    if not 0.0 <= config.collision_margin_m <= 0.1 * dc:
        raise ValueError('collision margin is outside the audit range')
    if min(config.pin_height, config.bead_lateral_friction,
           config.pin_lateral_friction, config.constraint_max_force) <= 0:
        raise ValueError('physics parameters must be positive')
    if min(config.occluder_padding_x_spacing,
           config.occluder_padding_y_m, config.occluder_bottom_z,
           config.occluder_height) <= 0:
        raise ValueError('occluder parameters must be positive')


def hidden_edge_angles(config):
    """Return N-1 tangent angles in radians."""
    validate_occp_config(config)
    angles = np.zeros(config.num_beads - 1, dtype=np.float64)
    peak = np.deg2rad(config.hidden_peak_tangent_deg)
    for local_index in range(config.hidden_edge_count):
        phase = 2.0 * np.pi * (local_index + 0.5) / config.hidden_edge_count
        edge_index = config.hidden_start_index + local_index
        angles[edge_index] = peak * np.sin(phase)
    return angles


def build_common_bead_polyline(config):
    """Return equal-segment bead positions [N,3] and bead yaw [N]."""
    spacing = config.cable_radius * config.spacing_scale
    angles = hidden_edge_angles(config)
    positions = np.zeros((config.num_beads, 3), dtype=np.float64)
    positions[0] = [
        config.passive_endpoint_x, config.cable_center_y, config.cable_z]
    for edge_index, theta in enumerate(angles):
        positions[edge_index + 1] = positions[edge_index] + spacing * np.array(
            [np.cos(theta), np.sin(theta), 0.0], dtype=np.float64)
    tangents = np.zeros((config.num_beads, 2), dtype=np.float64)
    tangents[0] = positions[1, :2] - positions[0, :2]
    tangents[-1] = positions[-1, :2] - positions[-2, :2]
    tangents[1:-1] = positions[2:, :2] - positions[:-2, :2]
    yaw = np.arctan2(tangents[:, 1], tangents[:, 0])
    return positions, yaw


def interpolate_polyline_y_at_x(positions, x):
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] < 2:
        raise ValueError('positions must have shape [N, >=2]')
    if np.any(np.diff(positions[:, 0]) <= 0):
        raise ValueError('polyline x must be strictly increasing')
    if x < positions[0, 0] or x > positions[-1, 0]:
        raise ValueError('x is outside the polyline')
    index = min(int(np.searchsorted(positions[:, 0], x, side='right')) - 1,
                len(positions) - 2)
    index = max(index, 0)
    left, right = positions[index], positions[index + 1]
    alpha = (x - left[0]) / (right[0] - left[0])
    return float(left[1] + alpha * (right[1] - left[1]))


def compute_occp_layout(config, condition):
    validate_occp_config(config)
    if condition not in CONDITIONS:
        raise ValueError('condition must be one of {}'.format(CONDITIONS))
    positions, yaw = build_common_bead_polyline(config)
    dc = 2.0 * config.cable_radius
    spacing = config.cable_radius * config.spacing_scale
    passive_entry_index = config.hidden_start_index
    active_exit_index = passive_entry_index + config.hidden_edge_count
    active_exit_x = positions[active_exit_index, 0]
    pin_radius = config.pin_radius_diameter_scale * dc
    pin_x = active_exit_x - config.pin_from_active_exit_spacing * spacing
    path_y = interpolate_polyline_y_at_x(positions, pin_x)
    combined_radius = pin_radius + config.cable_radius
    jam_clearance = config.jam_initial_clearance_diameter_scale * dc
    free_clearance = config.free_initial_clearance_diameter_scale * dc
    jam_pin_y = path_y - combined_radius - jam_clearance
    free_pin_y = path_y + combined_radius + free_clearance
    pin_z = config.pin_height / 2.0
    jam_pin_center = np.array([pin_x, jam_pin_y, pin_z], dtype=np.float64)
    free_pin_center = np.array([pin_x, free_pin_y, pin_z], dtype=np.float64)
    selected_pin = jam_pin_center if condition == 'right_hidden_jam' else free_pin_center

    occluder_x_min = (positions[passive_entry_index, 0]
                      - config.occluder_padding_x_spacing * spacing)
    occluder_x_max = (active_exit_x
                      + config.occluder_padding_x_spacing * spacing)
    visible_readout_indices = list(range(active_exit_index + 1, config.num_beads))
    if not all(positions[index, 0] > occluder_x_max
               for index in visible_readout_indices):
        raise ValueError('visible readout overlaps occluder')
    visible_mask = np.logical_or(
        positions[:, 0] < occluder_x_min,
        positions[:, 0] > occluder_x_max).astype(np.int64)
    occluded_indices = np.flatnonzero(visible_mask == 0).astype(int).tolist()
    hidden_indices = np.arange(passive_entry_index, active_exit_index + 1)
    y_min = min(float(positions[hidden_indices, 1].min()),
                jam_pin_y - pin_radius, free_pin_y - pin_radius)
    y_max = max(float(positions[hidden_indices, 1].max()),
                jam_pin_y + pin_radius, free_pin_y + pin_radius)
    y_min -= config.occluder_padding_y_m
    y_max += config.occluder_padding_y_m
    occluder_center = np.array([
        0.5 * (occluder_x_min + occluder_x_max),
        0.5 * (y_min + y_max),
        config.occluder_bottom_z + config.occluder_height / 2.0,
    ], dtype=np.float64)
    occluder_half_extents = np.array([
        0.5 * (occluder_x_max - occluder_x_min),
        0.5 * (y_max - y_min),
        config.occluder_height / 2.0,
    ], dtype=np.float64)
    polyline_length = float(np.sum(np.linalg.norm(
        np.diff(positions, axis=0), axis=1)))
    endpoint_distance = float(np.linalg.norm(positions[-1] - positions[0]))
    test_distance = config.test_spacing_scale * spacing
    angle = np.deg2rad(config.test_angle_deg)
    return {
        'bead_positions': positions,
        'bead_yaw': yaw,
        'spacing': float(spacing),
        'cable_diameter': float(dc),
        'initial_slack': float(polyline_length - endpoint_distance),
        'active_endpoint_index': int(config.num_beads - 1),
        'passive_endpoint_index': 0,
        'passive_entry_index': int(passive_entry_index),
        'active_exit_index': int(active_exit_index),
        'visible_readout_indices': visible_readout_indices,
        'visible_mask': visible_mask,
        'occluded_bead_indices': occluded_indices,
        'initial_path_y_at_pin': float(path_y),
        'pin_center': selected_pin.copy(),
        'free_pin_center': free_pin_center,
        'jam_pin_center': jam_pin_center,
        'parked_pin_center': free_pin_center.copy(),
        'pin_radius': float(pin_radius),
        'pin_height': float(config.pin_height),
        'jam_initial_surface_clearance': float(jam_clearance),
        'free_initial_surface_clearance': float(free_clearance),
        'jam_predicted_straight_penetration': float(
            combined_radius - abs(jam_pin_y - config.cable_center_y)),
        'free_predicted_straight_clearance': float(
            abs(free_pin_y - config.cable_center_y) - combined_radius),
        'occluder_x_min': float(occluder_x_min),
        'occluder_x_max': float(occluder_x_max),
        'active_exit_x': float(active_exit_x),
        'occluder_center': occluder_center,
        'occluder_half_extents': occluder_half_extents,
        'probe_delta': np.array([
            config.probe_spacing_scale * spacing, 0.0, 0.0]),
        'test_delta': np.array([
            test_distance * np.cos(angle), test_distance * np.sin(angle), 0.0]),
    }
