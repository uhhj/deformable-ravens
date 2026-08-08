"""Pure geometry for the Directional Hidden-Release cable smoke."""
from dataclasses import dataclass

import numpy as np


CONDITIONS = ("free", "jam_right")
VISIBLE_KEYPOINT_INDICES = tuple(
    list(range(0, 8)) + list(range(24, 32)))


@dataclass(frozen=True)
class DHRGeometryConfig:
    num_beads: int = 32
    cable_radius_m: float = 0.005
    bead_mass_kg: float = 0.05
    spacing_m: float = 0.011
    cable_z_m: float = 0.012
    entry_x_m: float = 0.255
    center_y_m: float = 0.0
    hidden_start_index: int = 8
    hidden_end_index: int = 23
    route_peak_tangent_deg: float = 35.0
    constraint_max_force_n: float = 100.0
    bead_lateral_friction: float = 0.45
    linear_damping: float = 0.04
    angular_damping: float = 0.90
    pocket_clearance_m: float = 0.00025
    pocket_height_m: float = 0.060
    occluder_padding_x_m: float = 0.012
    occluder_padding_y_m: float = 0.030
    occluder_bottom_z_m: float = 0.025
    occluder_height_m: float = 0.080


def _validate(cfg):
    if cfg.num_beads != 32:
        raise ValueError(
            "DHR smoke requires 32 beads")
    if not (
            0 <= cfg.hidden_start_index
            <= cfg.hidden_end_index
            < cfg.num_beads):
        raise ValueError(
            "invalid hidden bead interval")
    if min(
            cfg.cable_radius_m,
            cfg.bead_mass_kg,
            cfg.spacing_m,
            cfg.constraint_max_force_n,
            cfg.pocket_height_m) <= 0:
        raise ValueError(
            "DHR dimensions and masses "
            "must be positive")


def _hidden_angle_multiset(cfg):
    first_edge = (
        cfg.hidden_start_index - 1)
    last_edge = cfg.hidden_end_index
    count = (
        last_edge - first_edge + 1)
    peak = np.deg2rad(
        cfg.route_peak_tangent_deg)
    return np.asarray([
        peak * np.sin(
            2.0 * np.pi
            * (local + 0.5) / count)
        for local in range(count)
    ], dtype=np.float64)


def _condition_hidden_angles(
        cfg,
        condition):
    canonical = _hidden_angle_multiset(
        cfg)

    if condition == "free":
        return canonical

    if condition != "jam_right":
        raise ValueError(
            "unknown DHR condition: {}"
            .format(condition))

    interior = np.sort(
        canonical[1:-1])

    return np.concatenate([
        canonical[:1],
        interior,
        canonical[-1:],
    ])


def build_condition_polyline(
        cfg,
        condition):
    _validate(cfg)

    hidden_angles = (
        _condition_hidden_angles(
            cfg, condition))

    edge_angles = np.zeros(
        cfg.num_beads - 1,
        dtype=np.float64)

    first_edge = (
        cfg.hidden_start_index - 1)
    last_edge = cfg.hidden_end_index

    edge_angles[
        first_edge:last_edge + 1
    ] = hidden_angles

    positions = np.zeros(
        (cfg.num_beads, 3),
        dtype=np.float64)

    positions[0] = [
        cfg.entry_x_m,
        cfg.center_y_m,
        cfg.cable_z_m,
    ]

    for index, angle in enumerate(
            edge_angles):
        positions[index + 1] = (
            positions[index]
            + cfg.spacing_m
            * np.array([
                np.cos(angle),
                np.sin(angle),
                0.0,
            ], dtype=np.float64))

    tangent = np.empty(
        (cfg.num_beads, 2),
        dtype=np.float64)

    tangent[0] = (
        positions[1, :2]
        - positions[0, :2])

    tangent[-1] = (
        positions[-1, :2]
        - positions[-2, :2])

    tangent[1:-1] = (
        positions[2:, :2]
        - positions[:-2, :2])

    yaw = np.arctan2(
        tangent[:, 1],
        tangent[:, 0])

    return positions, yaw, edge_angles


def _box_world_xy_half_extents(
        cfg,
        yaw):
    half_length = (
        0.45 * cfg.spacing_m)
    cosine = abs(
        np.cos(float(yaw)))
    sine = abs(
        np.sin(float(yaw)))
    return np.array([
        cosine * half_length
        + sine * cfg.cable_radius_m,
        sine * half_length
        + cosine * cfg.cable_radius_m,
    ], dtype=np.float64)


def _lower_surface_y(
        cfg,
        positions,
        yaw,
        index):
    half_xy = (
        _box_world_xy_half_extents(
            cfg, yaw[index]))
    return float(
        positions[index, 1]
        - half_xy[1])


def _pocket_layout(
        cfg,
        jam_positions,
        jam_yaw):
    hidden_count = (
        cfg.hidden_end_index
        - cfg.hidden_start_index
        + 1)

    quarter = max(
        1,
        hidden_count // 4)

    left_index = (
        cfg.hidden_start_index
        + quarter)

    right_index = (
        cfg.hidden_end_index
        - quarter)

    wall_top_y = (
        min(
            _lower_surface_y(
                cfg,
                jam_positions,
                jam_yaw,
                left_index),
            _lower_surface_y(
                cfg,
                jam_positions,
                jam_yaw,
                right_index))
        - cfg.pocket_clearance_m)

    hidden_indices = np.arange(
        cfg.hidden_start_index,
        cfg.hidden_end_index + 1,
        dtype=np.int64)

    rail_top_y = (
        min(
            _lower_surface_y(
                cfg,
                jam_positions,
                jam_yaw,
                int(index))
            for index
            in hidden_indices)
        - cfg.pocket_clearance_m)

    wall_half = (
        0.5 * cfg.cable_radius_m)

    rail_center_y = (
        rail_top_y - wall_half)

    rail_bottom_y = (
        rail_top_y
        - 2.0 * wall_half)

    left_x = float(
        jam_positions[
            left_index, 0])

    right_x = float(
        jam_positions[
            right_index, 0])

    rail_center_x = float(
        0.5 * (left_x + right_x))

    rail_half_x = float(
        0.5 * (right_x - left_x)
        + wall_half)

    wall_center_y = float(
        0.5 * (
            rail_bottom_y
            + wall_top_y))

    wall_half_y = float(
        0.5 * (
            wall_top_y
            - rail_bottom_y))

    half_z = float(
        cfg.pocket_height_m / 2.0)

    return {
        "pocket_left_index":
            int(left_index),

        "pocket_right_index":
            int(right_index),

        "pocket_wall_top_y":
            float(wall_top_y),

        "pocket_rail_top_y":
            float(rail_top_y),

        "pocket_bottom_center":
            np.array([
                rail_center_x,
                rail_center_y,
                half_z,
            ], dtype=np.float64),

        "pocket_bottom_half_extents":
            np.array([
                rail_half_x,
                wall_half,
                half_z,
            ], dtype=np.float64),

        "pocket_left_center":
            np.array([
                left_x,
                wall_center_y,
                half_z,
            ], dtype=np.float64),

        "pocket_right_center":
            np.array([
                right_x,
                wall_center_y,
                half_z,
            ], dtype=np.float64),

        "pocket_wall_half_extents":
            np.array([
                wall_half,
                wall_half_y,
                half_z,
            ], dtype=np.float64),
    }


def compute_dhr_layout(
        cfg,
        condition):
    _validate(cfg)

    if condition not in CONDITIONS:
        raise ValueError(
            "unknown DHR condition: {}"
            .format(condition))

    free_positions, free_yaw, _ = (
        build_condition_polyline(
            cfg, "free"))

    jam_positions, jam_yaw, _ = (
        build_condition_polyline(
            cfg, "jam_right"))

    if condition == "free":
        positions = free_positions.copy()
        yaw = free_yaw.copy()
    else:
        positions = jam_positions.copy()
        yaw = jam_yaw.copy()

    pocket = _pocket_layout(
        cfg,
        jam_positions,
        jam_yaw)

    hidden = np.arange(
        cfg.hidden_start_index,
        cfg.hidden_end_index + 1,
        dtype=np.int64)

    all_route_positions = np.vstack([
        free_positions[hidden],
        jam_positions[hidden],
    ])

    pocket_x_min = float(
        pocket["pocket_left_center"][0]
        - pocket[
            "pocket_wall_half_extents"
        ][0])

    pocket_x_max = float(
        pocket["pocket_right_center"][0]
        + pocket[
            "pocket_wall_half_extents"
        ][0])

    pocket_y_min = float(
        pocket["pocket_bottom_center"][1]
        - pocket[
            "pocket_bottom_half_extents"
        ][1])

    x_min = min(
        float(
            all_route_positions[:, 0].min()),
        pocket_x_min,
    ) - cfg.occluder_padding_x_m

    x_max = max(
        float(
            all_route_positions[:, 0].max()),
        pocket_x_max,
    ) + cfg.occluder_padding_x_m

    y_min = min(
        float(
            all_route_positions[:, 1].min()),
        pocket_y_min,
    ) - cfg.occluder_padding_y_m

    y_max = (
        float(
            all_route_positions[:, 1].max())
        + cfg.occluder_padding_y_m)

    occluder_center = np.array([
        0.5 * (x_min + x_max),
        0.5 * (y_min + y_max),
        cfg.occluder_bottom_z_m
        + cfg.occluder_height_m / 2.0,
    ], dtype=np.float64)

    occluder_half_extents = np.array([
        0.5 * (x_max - x_min),
        0.5 * (y_max - y_min),
        cfg.occluder_height_m / 2.0,
    ], dtype=np.float64)

    return {
        "condition":
            condition,

        "bead_positions":
            positions,

        "bead_yaw":
            yaw,

        "free_bead_positions":
            free_positions,

        "jam_bead_positions":
            jam_positions,

        "visible_keypoint_indices":
            np.asarray(
                VISIBLE_KEYPOINT_INDICES,
                dtype=np.int64),

        "hidden_bead_indices":
            hidden,

        "active_endpoint_index":
            31,

        "passive_endpoint_index":
            0,

        "pull_direction":
            np.array(
                [1.0, 0.0, 0.0],
                dtype=np.float64),

        "occluder_center":
            occluder_center,

        "occluder_half_extents":
            occluder_half_extents,

        **pocket,
    }
