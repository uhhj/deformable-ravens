"""Pure geometry for the fixed hidden routing-gate cable task."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np


@dataclass(frozen=True)
class HiddenRoutingGateGeometryConfig:
    center_ratio: float
    probe_roof_clearance: float
    probe_roof_depth: float
    probe_roof_width: float
    probe_roof_thickness: float
    barrier_offset: float
    barrier_thickness: float
    barrier_width: float
    barrier_height: float
    stage1_pull_distance: float
    final_pull_distance: float
    target_plane_offset: float
    target_zone_depth: float
    target_corridor_half_width: float
    leading_segment_size: int
    workspace_x: Sequence[float]
    workspace_y: Sequence[float]
    bead_radius: float = 0.005
    topology_id: str = "hidden_routing_gate_v1"

    def __post_init__(self):
        if not str(self.topology_id).strip():
            raise ValueError(
                "topology_id must be non-empty"
            )
        if not 0.0 <= float(self.center_ratio) <= 1.0:
            raise ValueError(
                "center_ratio must lie in [0,1]"
            )

        positive = (
            "probe_roof_clearance",
            "probe_roof_depth",
            "probe_roof_width",
            "probe_roof_thickness",
            "barrier_offset",
            "barrier_thickness",
            "barrier_width",
            "barrier_height",
            "stage1_pull_distance",
            "final_pull_distance",
            "target_plane_offset",
            "target_zone_depth",
            "target_corridor_half_width",
            "bead_radius",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(
                    f"{name} must be finite and positive"
                )

        if int(self.leading_segment_size) <= 0:
            raise ValueError(
                "leading_segment_size must be positive"
            )
        if (
            float(self.final_pull_distance)
            <= float(self.stage1_pull_distance)
        ):
            raise ValueError(
                "final pull must exceed stage-1 pull"
            )

        barrier_far_face = (
            float(self.barrier_offset)
            + float(self.barrier_thickness) / 2
        )
        if (
            float(self.target_plane_offset)
            <= barrier_far_face
            + float(self.bead_radius)
        ):
            raise ValueError(
                "target plane must lie beyond the "
                "barrier and one bead radius"
            )
        if (
            float(self.final_pull_distance)
            <= float(self.target_plane_offset)
        ):
            raise ValueError(
                "final target must lie beyond target plane"
            )

        for name, bounds in (
            ("workspace_x", self.workspace_x),
            ("workspace_y", self.workspace_y),
        ):
            values = np.asarray(
                bounds,
                dtype=np.float64,
            ).reshape(-1)
            if (
                values.size != 2
                or not np.all(np.isfinite(values))
                or values[0] >= values[1]
            ):
                raise ValueError(f"invalid {name}")


def _normalize(value) -> np.ndarray:
    vector = np.asarray(
        value,
        dtype=np.float64,
    ).reshape(2)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError(
            "near-zero routing geometry vector"
        )
    return vector / norm


def _box_corners(box: Dict[str, Any]) -> List[np.ndarray]:
    yaw = float(box["yaw"])
    axis_x = np.asarray(
        [np.cos(yaw), np.sin(yaw)],
        dtype=np.float64,
    )
    axis_y = np.asarray(
        [-np.sin(yaw), np.cos(yaw)],
        dtype=np.float64,
    )
    center = np.asarray(
        box["center_xy"],
        dtype=np.float64,
    )
    half = np.asarray(
        box["half_extents"],
        dtype=np.float64,
    )
    return [
        center
        + sx * half[0] * axis_x
        + sy * half[1] * axis_y
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
    ]


def _rectangle_corners(
    center_xy,
    normal,
    tangent,
    half_normal,
    half_tangent,
) -> List[np.ndarray]:
    center = np.asarray(
        center_xy,
        dtype=np.float64,
    )
    normal = np.asarray(
        normal,
        dtype=np.float64,
    )
    tangent = np.asarray(
        tangent,
        dtype=np.float64,
    )
    return [
        center
        + sn * float(half_normal) * normal
        + st * float(half_tangent) * tangent
        for sn in (-1.0, 1.0)
        for st in (-1.0, 1.0)
    ]


def _workspace_margin(
    points,
    x_bounds,
    y_bounds,
) -> float:
    return float(min(
        min(
            point[0] - x_bounds[0],
            x_bounds[1] - point[0],
            point[1] - y_bounds[0],
            y_bounds[1] - point[1],
        )
        for point in points
    ))


def _point_box_distance(point, box) -> float:
    point = np.asarray(
        point,
        dtype=np.float64,
    )
    center_xy = np.asarray(
        box["center_xy"],
        dtype=np.float64,
    )
    yaw = float(box["yaw"])
    cosine = float(np.cos(yaw))
    sine = float(np.sin(yaw))
    delta = point[:2] - center_xy
    local = np.asarray([
        cosine * delta[0] + sine * delta[1],
        -sine * delta[0] + cosine * delta[1],
        point[2] - float(box["center_z"]),
    ])
    half = np.asarray(
        box["half_extents"],
        dtype=np.float64,
    )
    outside = np.abs(local) - half
    return float(
        np.linalg.norm(
            np.maximum(outside, 0.0)
        )
        + min(float(np.max(outside)), 0.0)
    )


def _leading_indices(
    bead_count: int,
    endpoint_index: int,
    size: int,
) -> List[int]:
    size = min(
        int(size),
        int(bead_count),
    )
    if endpoint_index == 0:
        return list(range(size))
    if endpoint_index == bead_count - 1:
        return list(range(
            bead_count - size,
            bead_count,
        ))
    raise ValueError(
        "endpoint_index must be an ordered endpoint"
    )


def public_routing_layout(
    layout: Dict[str, Any],
) -> Dict[str, Any]:
    """Strip all hidden fixture geometry from a layout."""
    public = dict(layout["public_task"])
    forbidden = {
        key
        for key in public
        if (
            "barrier" in str(key).lower()
            or "roof" in str(key).lower()
            or "hidden" in str(key).lower()
            or "box" in str(key).lower()
        )
    }
    if forbidden:
        raise RuntimeError(
            "public routing layout leaks hidden keys: "
            f"{sorted(forbidden)}"
        )
    return public


def compute_hidden_routing_gate_layout(
    bead_positions: np.ndarray,
    config: HiddenRoutingGateGeometryConfig,
) -> Dict[str, Any]:
    beads = np.asarray(
        bead_positions,
        dtype=np.float64,
    )
    if (
        beads.ndim != 2
        or beads.shape[0] < 5
        or beads.shape[1] != 3
    ):
        raise ValueError(
            f"expected ordered beads [N,3], "
            f"got {beads.shape}"
        )
    if not np.all(np.isfinite(beads)):
        raise ValueError(
            "beads contain NaN or Inf"
        )
    if (
        int(config.leading_segment_size)
        >= beads.shape[0]
    ):
        raise ValueError(
            "leading segment must be smaller "
            "than the cable"
        )

    endpoint_axis = _normalize(
        beads[-1, :2] - beads[0, :2]
    )
    base_normal = np.asarray(
        [-endpoint_axis[1], endpoint_axis[0]],
        dtype=np.float64,
    )
    preferred_probe = int(np.clip(
        round(
            float(config.center_ratio)
            * (beads.shape[0] - 1)
        ),
        1,
        beads.shape[0] - 2,
    ))
    centroid = np.mean(
        beads[:, :2],
        axis=0,
    )
    x_bounds = np.asarray(
        config.workspace_x,
        dtype=np.float64,
    )
    y_bounds = np.asarray(
        config.workspace_y,
        dtype=np.float64,
    )
    bounding_radius = float(
        np.sqrt(3.0) * config.bead_radius
    )
    construction_epsilon = 5e-5

    candidates: List[Dict[str, Any]] = []

    for endpoint_index in (
        0,
        beads.shape[0] - 1,
    ):
        endpoint = beads[endpoint_index]
        tangent = (
            endpoint_axis
            if endpoint_index == 0
            else -endpoint_axis
        )
        for normal_sign in (-1.0, 1.0):
            normal = (
                normal_sign * base_normal
            )
            yaw_tangent = float(
                np.arctan2(
                    tangent[1],
                    tangent[0],
                )
            )
            yaw_normal = float(
                np.arctan2(
                    normal[1],
                    normal[0],
                )
            )

            # Preserve the endpoint's normal coordinate,
            # but center the barrier along the cable tangent.
            endpoint_from_centroid = (
                endpoint[:2] - centroid
            )
            barrier_center = (
                centroid
                + normal * (
                    float(np.dot(
                        endpoint_from_centroid,
                        normal,
                    ))
                    + config.barrier_offset
                )
            )
            barrier = {
                "name": "routing_barrier",
                "center_xy": (
                    barrier_center
                    .astype(float)
                    .tolist()
                ),
                "center_z": float(
                    config.barrier_height / 2
                ),
                "half_extents": [
                    float(config.barrier_width / 2),
                    float(
                        config.barrier_thickness / 2
                    ),
                    float(config.barrier_height / 2),
                ],
                "yaw": yaw_tangent,
            }

            tangent_coordinates = (
                beads[:, :2]
                - barrier_center.reshape(1, 2)
            ) @ tangent
            coverage_margin = float(
                config.barrier_width / 2
                - np.max(
                    np.abs(tangent_coordinates)
                )
                - bounding_radius
            )
            if coverage_margin <= 0:
                continue

            probe_index = preferred_probe
            probe = beads[probe_index]
            roof_bottom = float(
                probe[2]
                + config.bead_radius
                + config.probe_roof_clearance
                + construction_epsilon
            )
            probe_roof = {
                "name": "probe_roof",
                "center_xy": (
                    probe[:2]
                    .astype(float)
                    .tolist()
                ),
                "center_z": float(
                    roof_bottom
                    + config.probe_roof_thickness / 2
                ),
                "half_extents": [
                    float(
                        config.probe_roof_depth / 2
                    ),
                    float(
                        config.probe_roof_width / 2
                    ),
                    float(
                        config.probe_roof_thickness / 2
                    ),
                ],
                "yaw": yaw_tangent,
            }

            stage1_target = (
                endpoint[:2]
                + normal
                * config.stage1_pull_distance
            )
            final_target = (
                endpoint[:2]
                + normal
                * config.final_pull_distance
            )
            target_plane_point = (
                endpoint[:2]
                + normal
                * config.target_plane_offset
            )
            target_zone_center = (
                target_plane_point
                + normal
                * (config.target_zone_depth / 2)
            )

            target_zone_corners = _rectangle_corners(
                target_zone_center,
                normal,
                tangent,
                config.target_zone_depth / 2,
                config.target_corridor_half_width,
            )
            points = [
                stage1_target,
                final_target,
                target_plane_point,
            ]
            points.extend(_box_corners(barrier))
            points.extend(_box_corners(probe_roof))
            points.extend(target_zone_corners)
            workspace_margin = _workspace_margin(
                points,
                x_bounds,
                y_bounds,
            )
            if workspace_margin <= 0:
                continue

            barrier_surface_clearance = min(
                _point_box_distance(
                    bead,
                    barrier,
                ) - bounding_radius
                for bead in beads
            )
            roof_surface_clearance = (
                _point_box_distance(
                    probe,
                    probe_roof,
                )
                - config.bead_radius
            )
            minimum_clearance = min(
                barrier_surface_clearance,
                roof_surface_clearance,
            )
            expected_minimum = min(
                config.probe_roof_clearance,
                (
                    config.barrier_offset
                    - config.barrier_thickness / 2
                    - bounding_radius
                ),
            )
            if (
                minimum_clearance + 1e-9
                < expected_minimum
            ):
                continue

            leading = _leading_indices(
                beads.shape[0],
                endpoint_index,
                int(config.leading_segment_size),
            )
            public_task = {
                "layout_version": (
                    "ccda_hidden_routing_gate_"
                    "public_v1"
                ),
                "topology_id": str(
                    config.topology_id
                ),
                "probe_index": int(probe_index),
                "endpoint_index": int(
                    endpoint_index
                ),
                "leading_segment_indices": [
                    int(value)
                    for value in leading
                ],
                "normal_xy": (
                    normal.astype(float).tolist()
                ),
                "tangent_xy": (
                    tangent.astype(float).tolist()
                ),
                "stage1_target_xy": (
                    stage1_target
                    .astype(float)
                    .tolist()
                ),
                "final_target_xy": (
                    final_target
                    .astype(float)
                    .tolist()
                ),
                "target_plane_point_xy": (
                    target_plane_point
                    .astype(float)
                    .tolist()
                ),
                "target_corridor_half_width": float(
                    config.target_corridor_half_width
                ),
                "target_zone_center_xy": (
                    target_zone_center
                    .astype(float)
                    .tolist()
                ),
                "target_zone_depth": float(
                    config.target_zone_depth
                ),
                "target_zone_half_extents_xy": [
                    float(
                        config.target_zone_depth / 2
                    ),
                    float(
                        config.target_corridor_half_width
                    ),
                ],
                "target_zone_yaw": yaw_normal,
            }
            candidates.append({
                "endpoint_index": endpoint_index,
                "normal_sign": normal_sign,
                "probe_index": probe_index,
                "public_task": public_task,
                "boxes": [
                    probe_roof,
                    barrier,
                ],
                "workspace_margin": (
                    workspace_margin
                ),
                "coverage_margin": coverage_margin,
                "minimum_clearance": (
                    minimum_clearance
                ),
                "barrier_surface_clearance": (
                    barrier_surface_clearance
                ),
                "roof_surface_clearance": (
                    roof_surface_clearance
                ),
                "roof_bottom_z": roof_bottom,
            })

    if not candidates:
        raise RuntimeError(
            "no legal fixed hidden routing-gate "
            "orientation and endpoint"
        )

    selected = max(
        candidates,
        key=lambda item: (
            round(
                float(item["workspace_margin"]),
                12,
            ),
            round(
                float(item["coverage_margin"]),
                12,
            ),
            -int(item["endpoint_index"]),
            float(item["normal_sign"]),
        ),
    )
    return {
        "snapshot_version": (
            "ccda_hidden_routing_gate_layout_v1"
        ),
        "topology": str(config.topology_id),
        "public_task": selected["public_task"],
        "probe_index": int(
            selected["probe_index"]
        ),
        "endpoint_index": int(
            selected["endpoint_index"]
        ),
        "normal_sign": float(
            selected["normal_sign"]
        ),
        "roof_bottom_z": float(
            selected["roof_bottom_z"]
        ),
        "bead_collision_half_extent": float(
            config.bead_radius
        ),
        "bead_collision_bounding_radius": float(
            bounding_radius
        ),
        "barrier_tangent_coverage_margin": float(
            selected["coverage_margin"]
        ),
        "barrier_surface_clearance": float(
            selected["barrier_surface_clearance"]
        ),
        "probe_roof_surface_clearance": float(
            selected["roof_surface_clearance"]
        ),
        "expected_surface_clearance": float(
            selected["minimum_clearance"]
        ),
        "workspace_margin": float(
            selected["workspace_margin"]
        ),
        "boxes": selected["boxes"],
    }

