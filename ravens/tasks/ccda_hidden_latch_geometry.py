"""Pure deterministic geometry for the fixed delayed hidden Z-latch."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import numpy as np


@dataclass(frozen=True)
class HiddenLatchGeometryConfig:
    center_ratio: float
    stop_clearance: float
    roof_clearance: float
    wall_thickness: float
    wall_width: float
    wall_height: float
    roof_depth: float
    roof_width: float
    roof_thickness: float
    main_pull_distance: float
    workspace_x: Sequence[float]
    workspace_y: Sequence[float]
    bead_radius: float = 0.005

    def __post_init__(self):
        if not 0.0 <= float(self.center_ratio) <= 1.0:
            raise ValueError("center_ratio must lie in [0,1]")
        for name in (
            "stop_clearance", "roof_clearance", "wall_thickness",
            "wall_width", "wall_height", "roof_depth", "roof_width",
            "roof_thickness", "main_pull_distance", "bead_radius",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name, bounds in (
            ("workspace_x", self.workspace_x), ("workspace_y", self.workspace_y)
        ):
            values = np.asarray(bounds, dtype=np.float64).reshape(-1)
            if values.size != 2 or not np.all(np.isfinite(values)) or values[0] >= values[1]:
                raise ValueError(f"invalid {name}")


def _normalize(value):
    value = np.asarray(value, dtype=np.float64).reshape(2)
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("near-zero latch geometry vector")
    return value / norm


def _point_box_distance(point, box):
    point = np.asarray(point, dtype=np.float64)
    center_xy = np.asarray(box["center_xy"], dtype=np.float64)
    yaw = float(box["yaw"])
    cosine, sine = np.cos(yaw), np.sin(yaw)
    delta = point[:2] - center_xy
    local_xy = np.asarray([
        cosine * delta[0] + sine * delta[1],
        -sine * delta[0] + cosine * delta[1],
    ])
    half = np.asarray(box["half_extents"], dtype=np.float64)
    local = np.asarray([
        local_xy[0], local_xy[1], point[2] - float(box["center_z"])
    ])
    outside = np.abs(local) - half
    return float(
        np.linalg.norm(np.maximum(outside, 0.0))
        + min(float(np.max(outside)), 0.0)
    )


def _axis_aligned_bead_box_clearance(point, box, bead_half_extent):
    """Face-axis clearance between an axis-aligned bead box and yawed box."""
    point = np.asarray(point, dtype=np.float64)
    center_xy = np.asarray(box["center_xy"], dtype=np.float64)
    yaw = float(box["yaw"])
    cosine, sine = np.cos(yaw), np.sin(yaw)
    delta = point[:2] - center_xy
    local = np.asarray([
        cosine * delta[0] + sine * delta[1],
        -sine * delta[0] + cosine * delta[1],
        point[2] - float(box["center_z"]),
    ])
    support_xy = bead_half_extent * (abs(cosine) + abs(sine))
    expanded_half = np.asarray(box["half_extents"], dtype=np.float64) + np.asarray([
        support_xy, support_xy, bead_half_extent
    ])
    outside = np.abs(local) - expanded_half
    return float(
        np.linalg.norm(np.maximum(outside, 0.0))
        + min(float(np.max(outside)), 0.0)
    )


def _workspace_margin(points, x_bounds, y_bounds):
    return float(min(
        min(
            point[0] - x_bounds[0], x_bounds[1] - point[0],
            point[1] - y_bounds[0], y_bounds[1] - point[1],
        )
        for point in points
    ))


def _box_corners(box):
    yaw = float(box["yaw"])
    axis_x = np.asarray([np.cos(yaw), np.sin(yaw)])
    axis_y = np.asarray([-np.sin(yaw), np.cos(yaw)])
    center = np.asarray(box["center_xy"], dtype=np.float64)
    hx, hy = np.asarray(box["half_extents"], dtype=np.float64)[:2]
    return [
        center + sx * hx * axis_x + sy * hy * axis_y
        for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)
    ]


def compute_hidden_latch_layout(
    bead_positions: np.ndarray,
    config: HiddenLatchGeometryConfig,
) -> Dict[str, Any]:
    beads = np.asarray(bead_positions, dtype=np.float64)
    if beads.ndim != 2 or beads.shape[0] < 5 or beads.shape[1] != 3:
        raise ValueError(f"expected ordered beads [N,3], got {beads.shape}")
    if not np.all(np.isfinite(beads)):
        raise ValueError("beads contain NaN or Inf")

    preferred = int(np.clip(
        round(float(config.center_ratio) * (beads.shape[0] - 1)),
        1, beads.shape[0] - 2,
    ))
    x_bounds = np.asarray(config.workspace_x, dtype=np.float64)
    y_bounds = np.asarray(config.workspace_y, dtype=np.float64)
    candidates = []
    # Keep the requested 2 mm surface clearance on the safe side of
    # PyBullet's sub-micrometre collision-distance rounding.
    construction_epsilon = 5e-5
    for probe_index in range(1, beads.shape[0] - 1):
        try:
            tangent = _normalize(
                beads[probe_index + 1, :2] - beads[probe_index - 1, :2]
            )
        except ValueError:
            continue
        base_normal = np.asarray([-tangent[1], tangent[0]])
        anchor = beads[probe_index]
        for sign in (-1.0, 1.0):
            normal = sign * base_normal
            yaw_tangent = float(np.arctan2(tangent[1], tangent[0]))
            yaw_normal = float(np.arctan2(normal[1], normal[0]))
            # Cable collision beads are cubes with ``bead_radius`` as their
            # half extent and may have arbitrary settled orientation. Use the
            # half diagonal as an orientation-independent support bound.
            collision_support_radius = float(
                np.sqrt(3.0) * config.bead_radius
            )
            normal_support = collision_support_radius
            stop_near = (
                normal_support + config.stop_clearance + construction_epsilon
            )
            roof_bottom = (
                float(anchor[2]) + config.bead_radius
                + config.roof_clearance + construction_epsilon
            )
            stop_center = anchor[:2] + normal * (
                stop_near + config.wall_thickness / 2
            )
            # The overhang starts at the anchor plane and extends toward the
            # stop wall. The bead's finite XY support still overlaps the roof
            # edge during a vertical probe, while cable behind the anchor is
            # not unnecessarily covered.
            roof_center = anchor[:2] + normal * (config.roof_depth / 2)
            side_center_normal = stop_near + config.roof_depth / 2
            side_tangent = config.roof_width / 2 + config.wall_thickness / 2
            boxes = [
                {
                    "name": "stop_wall",
                    "center_xy": stop_center.astype(float).tolist(),
                    "center_z": float(config.wall_height / 2),
                    "half_extents": [
                        config.wall_width / 2,
                        config.wall_thickness / 2,
                        config.wall_height / 2,
                    ],
                    "yaw": yaw_tangent,
                },
                {
                    "name": "roof",
                    "center_xy": roof_center.astype(float).tolist(),
                    "center_z": float(roof_bottom + config.roof_thickness / 2),
                    "half_extents": [
                        config.roof_depth / 2,
                        config.roof_width / 2,
                        config.roof_thickness / 2,
                    ],
                    "yaw": yaw_normal,
                },
                {
                    "name": "side_a",
                    "center_xy": (
                        anchor[:2] + normal * side_center_normal
                        + tangent * side_tangent
                    ).astype(float).tolist(),
                    "center_z": float(config.wall_height / 2),
                    "half_extents": [
                        config.roof_depth / 2,
                        config.wall_thickness / 2,
                        config.wall_height / 2,
                    ],
                    "yaw": yaw_normal,
                },
                {
                    "name": "side_b",
                    "center_xy": (
                        anchor[:2] + normal * side_center_normal
                        - tangent * side_tangent
                    ).astype(float).tolist(),
                    "center_z": float(config.wall_height / 2),
                    "half_extents": [
                        config.roof_depth / 2,
                        config.wall_thickness / 2,
                        config.wall_height / 2,
                    ],
                    "yaw": yaw_normal,
                },
            ]
            center_clearance = min(
                _point_box_distance(bead, box)
                for bead in beads for box in boxes
            )
            surface_clearance = min(
                _point_box_distance(bead, box) - (
                    config.bead_radius
                    if box["name"] == "roof" and bead_index == probe_index
                    else collision_support_radius
                )
                for bead_index, bead in enumerate(beads) for box in boxes
            )
            nominal_surface_clearance = min(
                _point_box_distance(bead, box) - config.bead_radius
                for bead in beads for box in boxes
            )
            for endpoint_index in (0, beads.shape[0] - 1):
                main_target = (
                    beads[endpoint_index, :2]
                    + normal * config.main_pull_distance
                )
                points = [main_target]
                for box in boxes:
                    points.extend(_box_corners(box))
                margin = _workspace_margin(points, x_bounds, y_bounds)
                if margin > 0:
                    candidates.append({
                        "probe_index": probe_index,
                        "endpoint_index": endpoint_index,
                        "normal_sign": sign,
                        "normal": normal,
                        "tangent": tangent,
                        "anchor": anchor.copy(),
                        "boxes": boxes,
                        "main_target": main_target,
                        "workspace_margin": margin,
                        "center_clearance": center_clearance,
                        "surface_clearance": surface_clearance,
                        "nominal_surface_clearance": nominal_surface_clearance,
                        "roof_bottom": roof_bottom,
                        "stop_near": stop_near,
                    })
    if not candidates:
        raise RuntimeError("no legal fixed hidden-latch orientation and endpoint")
    minimum_expected = min(config.stop_clearance, config.roof_clearance)
    valid = [
        item for item in candidates
        if item["surface_clearance"] + 1e-9 >= minimum_expected
    ]
    if not valid:
        valid = [
            item for item in candidates
            if item["nominal_surface_clearance"] + 1e-9 >= minimum_expected
        ]
        if not valid:
            best = max(item["nominal_surface_clearance"] for item in candidates)
            raise RuntimeError(
                "fixed hidden latch intersects/approaches initial cable: "
                f"best nominal clearance {best} < {minimum_expected}"
            )
    selected = max(valid, key=lambda item: (
        -abs(int(item["probe_index"]) - preferred),
        round(float(item["workspace_margin"]), 12),
        round(float(item["surface_clearance"]), 12),
        -int(item["endpoint_index"]),
        float(item["normal_sign"]),
    ))
    anchor = selected["anchor"]
    return {
        "snapshot_version": "ccda_hidden_latch_layout_v1",
        "topology": "delayed_z_latch_v1",
        "probe_index": int(selected["probe_index"]),
        "endpoint_index": int(selected["endpoint_index"]),
        "normal_sign": float(selected["normal_sign"]),
        "normal_xy": selected["normal"].astype(float).tolist(),
        "tangent_xy": selected["tangent"].astype(float).tolist(),
        "cable_anchor_xyz": anchor.astype(float).tolist(),
        "anchor_bead_center_z": float(anchor[2]),
        "bead_collision_half_extent": float(config.bead_radius),
        "bead_collision_bounding_radius": float(
            np.sqrt(3.0) * config.bead_radius
        ),
        "main_target_xy": selected["main_target"].astype(float).tolist(),
        "roof_bottom_z": float(selected["roof_bottom"]),
        "stop_near_face_distance": float(selected["stop_near"]),
        "geometric_center_clearance": float(selected["center_clearance"]),
        "expected_surface_clearance": float(max(
            selected["surface_clearance"],
            selected["nominal_surface_clearance"],
        )),
        "conservative_surface_clearance": float(
            selected["surface_clearance"]
        ),
        "workspace_margin": float(selected["workspace_margin"]),
        "boxes": selected["boxes"],
    }
