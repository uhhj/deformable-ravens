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
            "single_post", "dual_post_directional_guide"):
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


def _box_world_y_half_extent(cfg, yaw):
    half_length = 0.45 * cfg.spacing_m
    return float(
        abs(np.sin(float(yaw))) * half_length
        + abs(np.cos(float(yaw))) * cfg.cable_radius_m)


def _directional_guide_index(cfg, latch_index):
    offset = int(np.ceil((2.0 * cfg.latch_radius_m) / cfg.spacing_m))
    return int(min(
        cfg.hidden_end_index, latch_index + max(1, offset)))


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
        "latch_radius": float(cfg.latch_radius_m),
        "latch_height": float(cfg.latch_height_m),
        "occluder_center": occluder_center,
        "occluder_half_extents": occluder_half_extents,
        "pull_direction": np.array([1.0, 0.0, 0.0], dtype=np.float64),
    }
