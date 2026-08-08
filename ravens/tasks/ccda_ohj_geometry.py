"""Pure geometry for the occluded hidden-jam cable task."""
from dataclasses import dataclass

import numpy as np


CONDITIONS = ("free", "jam_right")
VISIBLE_KEYPOINT_INDICES = tuple(list(range(0, 8)) + list(range(24, 32)))


@dataclass(frozen=True)
class OHJGeometryConfig:
    num_beads: int = 32
    cable_radius_m: float = 0.005
    bead_mass_kg: float = 0.05
    spacing_m: float = 0.011
    cable_z_m: float = 0.012
    entry_x_m: float = 0.255
    center_y_m: float = 0.0
    hidden_start_index: int = 8
    hidden_end_index: int = 23
    hidden_peak_tangent_deg: float = 35.0
    constraint_max_force_n: float = 100.0
    bead_lateral_friction: float = 0.45
    linear_damping: float = 0.04
    angular_damping: float = 0.90
    latch_radius_m: float = 0.009
    latch_height_m: float = 0.06
    jam_surface_clearance_m: float = 0.0005
    free_lateral_offset_m: float = 0.040
    latch_topology: str = "single_post"
    occluder_padding_x_m: float = 0.012
    occluder_padding_y_m: float = 0.030
    occluder_bottom_z_m: float = 0.025
    occluder_height_m: float = 0.080


def _validate(cfg):
    if cfg.num_beads != 32:
        raise ValueError("OHJ Phase 0 requires 32 beads")
    if not (0 <= cfg.hidden_start_index <= cfg.hidden_end_index < cfg.num_beads):
        raise ValueError("invalid hidden bead interval")
    if min(cfg.cable_radius_m, cfg.bead_mass_kg, cfg.spacing_m,
           cfg.constraint_max_force_n, cfg.latch_radius_m) <= 0:
        raise ValueError("OHJ dimensions and masses must be positive")
    if cfg.latch_topology not in (
            "single_post", "dual_post_directional_guide",
            "continuous_l_slot_hook"):
        raise ValueError(
            "unknown OHJ latch topology: {}".format(cfg.latch_topology))


def build_common_bead_polyline(cfg):
    """Return the condition-independent 32-bead cable centerline and yaw."""
    _validate(cfg)
    edge_angles = np.zeros(cfg.num_beads - 1, dtype=np.float64)
    first_edge = cfg.hidden_start_index - 1
    last_edge = cfg.hidden_end_index
    count = last_edge - first_edge + 1
    peak = np.deg2rad(cfg.hidden_peak_tangent_deg)
    for local in range(count):
        phase = 2.0 * np.pi * (local + 0.5) / count
        edge_angles[first_edge + local] = peak * np.sin(phase)
    positions = np.zeros((cfg.num_beads, 3), dtype=np.float64)
    positions[0] = [cfg.entry_x_m, cfg.center_y_m, cfg.cable_z_m]
    for index, angle in enumerate(edge_angles):
        positions[index + 1] = positions[index] + cfg.spacing_m * np.array(
            [np.cos(angle), np.sin(angle), 0.0], dtype=np.float64)
    tangent = np.empty((cfg.num_beads, 2), dtype=np.float64)
    tangent[0] = positions[1, :2] - positions[0, :2]
    tangent[-1] = positions[-1, :2] - positions[-2, :2]
    tangent[1:-1] = positions[2:, :2] - positions[:-2, :2]
    yaw = np.arctan2(tangent[:, 1], tangent[:, 0])
    return positions, yaw


def _box_world_xy_half_extents(cfg, yaw):
    half_length = 0.45 * cfg.spacing_m
    cosine = abs(np.cos(float(yaw)))
    sine = abs(np.sin(float(yaw)))
    return np.array([
        cosine * half_length + sine * cfg.cable_radius_m,
        sine * half_length + cosine * cfg.cable_radius_m,
    ], dtype=np.float64)


def _box_world_y_half_extent(cfg, yaw):
    return float(_box_world_xy_half_extents(cfg, yaw)[1])


def _directional_guide_index(cfg, latch_index):
    offset = int(np.ceil((2.0 * cfg.latch_radius_m) / cfg.spacing_m))
    return int(min(
        cfg.hidden_end_index, latch_index + max(1, offset)))


def _continuous_l_hook_layout(
        cfg, positions, yaw, latch_index, jam_latch_center):
    downstream_index = _directional_guide_index(cfg, latch_index)
    support_indices = np.arange(
        latch_index, downstream_index + 1, dtype=np.int64)
    lower_y = []
    for index in support_indices:
        half_xy = _box_world_xy_half_extents(cfg, yaw[index])
        lower_y.append(float(positions[index, 1] - half_xy[1]))
    stop_top_y = min(lower_y) - cfg.jam_surface_clearance_m
    wall_half = 0.5 * cfg.cable_radius_m
    rail_top_y = stop_top_y - cfg.cable_radius_m
    rail_bottom_y = rail_top_y - 2.0 * wall_half
    stop_center_x = float(0.5 * (
        positions[downstream_index - 1, 0]
        + positions[downstream_index, 0]))
    rail_start_x = float(jam_latch_center[0])
    rail_center_x = float(0.5 * (rail_start_x + stop_center_x))
    rail_half_x = float(
        0.5 * (stop_center_x - rail_start_x) + wall_half)
    rail_center_y = float(rail_top_y - wall_half)
    stop_bottom_y = rail_bottom_y
    stop_center_y = float(0.5 * (stop_bottom_y + stop_top_y))
    stop_half_y = float(0.5 * (stop_top_y - stop_bottom_y))
    half_z = float(cfg.latch_height_m / 2.0)
    jam_side_center = np.array([
        rail_center_x, rail_center_y, half_z], dtype=np.float64)
    side_half_extents = np.array([
        rail_half_x, wall_half, half_z], dtype=np.float64)
    jam_stop_center = np.array([
        stop_center_x, stop_center_y, half_z], dtype=np.float64)
    stop_half_extents = np.array([
        wall_half, stop_half_y, half_z], dtype=np.float64)
    free_shift = np.array(
        [0.0, cfg.free_lateral_offset_m, 0.0], dtype=np.float64)
    return {
        "hook_downstream_index": int(downstream_index),
        "hook_support_indices": support_indices,
        "hook_stop_top_y": float(stop_top_y),
        "jam_hook_side_center": jam_side_center,
        "free_hook_side_center": jam_side_center + free_shift,
        "hook_side_half_extents": side_half_extents,
        "jam_hook_stop_center": jam_stop_center,
        "free_hook_stop_center": jam_stop_center + free_shift,
        "hook_stop_half_extents": stop_half_extents,
    }


def compute_ohj_layout(cfg, condition):
    """Return common cable geometry and the selected hidden latch pose."""
    _validate(cfg)
    if condition not in CONDITIONS:
        raise ValueError("unknown OHJ condition: {}".format(condition))
    positions, yaw = build_common_bead_polyline(cfg)
    latch_index = (cfg.hidden_start_index + cfg.hidden_end_index) // 2
    path = positions[latch_index]
    jam_latch_center = np.array([
        path[0],
        path[1] - cfg.latch_radius_m - cfg.cable_radius_m
        - cfg.jam_surface_clearance_m,
        cfg.latch_height_m / 2.0,
    ], dtype=np.float64)
    free_latch_center = jam_latch_center + np.array(
        [0.0, cfg.free_lateral_offset_m, 0.0], dtype=np.float64)
    latch_center = (jam_latch_center if condition == "jam_right"
                    else free_latch_center).copy()
    guide_index = _directional_guide_index(cfg, latch_index)
    guide_path = positions[guide_index]
    guide_y_half_extent = _box_world_y_half_extent(
        cfg, yaw[guide_index])
    jam_directional_guide_center = np.array([
        guide_path[0],
        guide_path[1] - guide_y_half_extent - cfg.latch_radius_m
        - cfg.jam_surface_clearance_m,
        cfg.latch_height_m / 2.0,
    ], dtype=np.float64)
    free_directional_guide_center = (
        jam_directional_guide_center
        + np.array(
            [0.0, cfg.free_lateral_offset_m, 0.0], dtype=np.float64))
    directional_guide_center = (
        jam_directional_guide_center
        if condition == "jam_right"
        else free_directional_guide_center).copy()
    hook = _continuous_l_hook_layout(
        cfg, positions, yaw, latch_index, jam_latch_center)
    hook_side_center = (
        hook["jam_hook_side_center"]
        if condition == "jam_right"
        else hook["free_hook_side_center"]).copy()
    hook_stop_center = (
        hook["jam_hook_stop_center"]
        if condition == "jam_right"
        else hook["free_hook_stop_center"]).copy()
    hidden = np.arange(
        cfg.hidden_start_index, cfg.hidden_end_index + 1, dtype=np.int64)
    hidden_x_min = float(positions[hidden, 0].min())
    hidden_x_max = float(positions[hidden, 0].max())
    hidden_y_min = float(positions[hidden, 1].min())
    hidden_y_max = float(positions[hidden, 1].max())
    obstacle_centers = [jam_latch_center, free_latch_center]
    if cfg.latch_topology == "dual_post_directional_guide":
        obstacle_centers.extend([
            jam_directional_guide_center,
            free_directional_guide_center])
    obstacle_centers = np.asarray(obstacle_centers, dtype=np.float64)
    obstacle_x_min = float(
        np.min(obstacle_centers[:, 0]) - cfg.latch_radius_m)
    obstacle_x_max = float(
        np.max(obstacle_centers[:, 0]) + cfg.latch_radius_m)
    obstacle_y_min = float(
        np.min(obstacle_centers[:, 1]) - cfg.latch_radius_m)
    obstacle_y_max = float(
        np.max(obstacle_centers[:, 1]) + cfg.latch_radius_m)
    if cfg.latch_topology == "continuous_l_slot_hook":
        hook_bounds = []
        for center, half in (
                (hook["jam_hook_side_center"],
                 hook["hook_side_half_extents"]),
                (hook["free_hook_side_center"],
                 hook["hook_side_half_extents"]),
                (hook["jam_hook_stop_center"],
                 hook["hook_stop_half_extents"]),
                (hook["free_hook_stop_center"],
                 hook["hook_stop_half_extents"])):
            hook_bounds.append((
                float(center[0] - half[0]),
                float(center[0] + half[0]),
                float(center[1] - half[1]),
                float(center[1] + half[1])))
        obstacle_x_min = min(
            obstacle_x_min, min(row[0] for row in hook_bounds))
        obstacle_x_max = max(
            obstacle_x_max, max(row[1] for row in hook_bounds))
        obstacle_y_min = min(
            obstacle_y_min, min(row[2] for row in hook_bounds))
        obstacle_y_max = max(
            obstacle_y_max, max(row[3] for row in hook_bounds))
    x_min = min(hidden_x_min, obstacle_x_min) - cfg.occluder_padding_x_m
    x_max = max(hidden_x_max, obstacle_x_max) + cfg.occluder_padding_x_m
    y_min = min(hidden_y_min, obstacle_y_min) - cfg.occluder_padding_y_m
    y_max = max(hidden_y_max, obstacle_y_max) + cfg.occluder_padding_y_m
    occluder_center = np.array([
        0.5 * (x_min + x_max), 0.5 * (y_min + y_max),
        cfg.occluder_bottom_z_m + cfg.occluder_height_m / 2.0,
    ], dtype=np.float64)
    occluder_half_extents = np.array([
        0.5 * (x_max - x_min), 0.5 * (y_max - y_min),
        cfg.occluder_height_m / 2.0,
    ], dtype=np.float64)
    return {
        "bead_positions": positions,
        "bead_yaw": yaw,
        "visible_keypoint_indices": np.asarray(
            VISIBLE_KEYPOINT_INDICES, dtype=np.int64),
        "hidden_bead_indices": hidden,
        "active_endpoint_index": 31,
        "passive_endpoint_index": 0,
        "latch_center": latch_center,
        "jam_latch_center": jam_latch_center,
        "free_latch_center": free_latch_center,
        "latch_topology": str(cfg.latch_topology),
        "directional_guide_index": int(guide_index),
        "directional_guide_y_half_extent": float(guide_y_half_extent),
        "directional_guide_center": directional_guide_center,
        "jam_directional_guide_center": jam_directional_guide_center,
        "free_directional_guide_center": free_directional_guide_center,
        "hook_downstream_index": int(hook["hook_downstream_index"]),
        "hook_support_indices": hook["hook_support_indices"],
        "hook_side_center": hook_side_center,
        "jam_hook_side_center": hook["jam_hook_side_center"],
        "free_hook_side_center": hook["free_hook_side_center"],
        "hook_side_half_extents": hook["hook_side_half_extents"],
        "hook_stop_center": hook_stop_center,
        "jam_hook_stop_center": hook["jam_hook_stop_center"],
        "free_hook_stop_center": hook["free_hook_stop_center"],
        "hook_stop_half_extents": hook["hook_stop_half_extents"],
        "hook_stop_top_y": float(hook["hook_stop_top_y"]),
        "latch_radius": float(cfg.latch_radius_m),
        "latch_height": float(cfg.latch_height_m),
        "occluder_center": occluder_center,
        "occluder_half_extents": occluder_half_extents,
        "pull_direction": np.array([1.0, 0.0, 0.0], dtype=np.float64),
    }
