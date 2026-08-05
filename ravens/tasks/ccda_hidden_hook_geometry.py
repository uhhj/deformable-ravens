"""Pure deterministic geometry for Hidden-Hook Cable Routing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import numpy as np


@dataclass(frozen=True)
class HiddenHookGeometryConfig:
    center_ratio: float
    mouth_offset: float
    width: float
    depth: float
    thickness: float
    height: float
    probe_distance: float
    main_pull_distance: float
    workspace_x: Sequence[float]
    workspace_y: Sequence[float]

    def __post_init__(self) -> None:
        if not 0 <= float(self.center_ratio) <= 1:
            raise ValueError("center_ratio must lie in [0,1]")
        for name in (
            "mouth_offset", "width", "depth", "thickness", "height",
            "probe_distance", "main_pull_distance",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.thickness >= self.width or self.thickness >= self.depth:
            raise ValueError("hook thickness must be smaller than width and depth")
        for name, bounds in (("workspace_x", self.workspace_x), ("workspace_y", self.workspace_y)):
            value = np.asarray(bounds, dtype=np.float64).reshape(-1)
            if value.size != 2 or not np.all(np.isfinite(value)) or value[0] >= value[1]:
                raise ValueError(f"invalid {name}")


def _normalize(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64).reshape(2)
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("near-zero geometry vector")
    return value / norm


def _margin(xy, x_bounds, y_bounds):
    return float(min(
        xy[0] - x_bounds[0], x_bounds[1] - xy[0],
        xy[1] - y_bounds[0], y_bounds[1] - xy[1],
    ))


def _rectangle_clearance(point, center, half_extents, yaw):
    delta = np.asarray(point, dtype=np.float64) - np.asarray(
        center, dtype=np.float64
    )
    cosine, sine = np.cos(yaw), np.sin(yaw)
    local = np.asarray([
        cosine * delta[0] + sine * delta[1],
        -sine * delta[0] + cosine * delta[1],
    ])
    outside = np.abs(local) - np.asarray(half_extents, dtype=np.float64)
    return float(
        np.linalg.norm(np.maximum(outside, 0.0))
        + min(float(np.max(outside)), 0.0)
    )


def compute_hidden_hook_layout(
    bead_positions: np.ndarray,
    config: HiddenHookGeometryConfig,
) -> Dict[str, Any]:
    beads = np.asarray(bead_positions, dtype=np.float64)
    if beads.ndim != 2 or beads.shape[0] < 5 or beads.shape[1] != 3:
        raise ValueError(f"expected ordered beads [N,3], got {beads.shape}")
    if not np.all(np.isfinite(beads)):
        raise ValueError("beads contain NaN or Inf")

    count = beads.shape[0]
    preferred_probe_index = int(np.clip(
        round(float(config.center_ratio) * (count - 1)), 1, count - 2
    ))
    x_bounds = np.asarray(config.workspace_x, dtype=np.float64)
    y_bounds = np.asarray(config.workspace_y, dtype=np.float64)

    candidates = []
    for probe_index in range(1, count - 1):
        try:
            tangent = _normalize(
                beads[probe_index + 1, :2] - beads[probe_index - 1, :2]
            )
        except ValueError:
            continue
        base_normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
        cable_xy = beads[probe_index, :2]
        for sign in (-1.0, 1.0):
            normal = base_normal * sign
            for endpoint_index in (0, count - 1):
                probe_target = cable_xy + normal * config.probe_distance
                main_target = beads[endpoint_index, :2] + normal * config.main_pull_distance
                hook_far = cable_xy + normal * (
                    config.mouth_offset + config.depth + config.thickness
                )
                points = (
                    probe_target,
                    main_target,
                    hook_far + tangent * config.width / 2,
                    hook_far - tangent * config.width / 2,
                )
                workspace_score = min(
                    _margin(point, x_bounds, y_bounds) for point in points
                )
                mouth = cable_xy + normal * config.mouth_offset
                back = mouth + normal * config.depth
                side_center = mouth + normal * config.depth / 2
                side_offset = config.width / 2 - config.thickness / 2
                yaw_tangent = float(np.arctan2(tangent[1], tangent[0]))
                yaw_normal = float(np.arctan2(normal[1], normal[0]))
                rectangles = (
                    (back, (config.width / 2, config.thickness / 2), yaw_tangent),
                    (
                        side_center + tangent * side_offset,
                        (config.depth / 2, config.thickness / 2),
                        yaw_normal,
                    ),
                    (
                        side_center - tangent * side_offset,
                        (config.depth / 2, config.thickness / 2),
                        yaw_normal,
                    ),
                )
                cable_clearance = min(
                    _rectangle_clearance(bead[:2], center, half, yaw)
                    for bead in beads
                    for center, half, yaw in rectangles
                )
                if workspace_score > 0:
                    candidates.append({
                        "cable_clearance": cable_clearance,
                        "workspace_score": workspace_score,
                        "probe_index": probe_index,
                        "endpoint_index": endpoint_index,
                        "normal_sign": sign,
                        "normal": normal,
                        "tangent": tangent,
                        "cable_xy": cable_xy,
                    })
    if not candidates:
        raise RuntimeError("no legal hidden-hook orientation and endpoint")

    selected = max(
        candidates,
        key=lambda item: (
            round(float(item["cable_clearance"]), 12),
            -abs(int(item["probe_index"]) - preferred_probe_index),
            float(item["workspace_score"]),
            -int(item["endpoint_index"]),
            float(item["normal_sign"]),
        ),
    )
    score = selected["workspace_score"]
    probe_index = selected["probe_index"]
    endpoint_index = selected["endpoint_index"]
    normal_sign = selected["normal_sign"]
    normal = selected["normal"]
    tangent = selected["tangent"]
    cable_xy = selected["cable_xy"]
    mouth = cable_xy + normal * config.mouth_offset
    back = mouth + normal * config.depth
    side_center = mouth + normal * config.depth / 2
    side_offset = config.width / 2 - config.thickness / 2
    yaw_tangent = float(np.arctan2(tangent[1], tangent[0]))
    yaw_normal = float(np.arctan2(normal[1], normal[0]))

    boxes = [
        {
            "name": "back",
            "center_xy": back.astype(float).tolist(),
            "half_extents": [config.width / 2, config.thickness / 2, config.height / 2],
            "yaw": yaw_tangent,
        },
        {
            "name": "side_a",
            "center_xy": (side_center + tangent * side_offset).astype(float).tolist(),
            "half_extents": [config.depth / 2, config.thickness / 2, config.height / 2],
            "yaw": yaw_normal,
        },
        {
            "name": "side_b",
            "center_xy": (side_center - tangent * side_offset).astype(float).tolist(),
            "half_extents": [config.depth / 2, config.thickness / 2, config.height / 2],
            "yaw": yaw_normal,
        },
    ]
    return {
        "snapshot_version": "ccda_hidden_hook_layout_v1",
        "probe_index": probe_index,
        "endpoint_index": int(endpoint_index),
        "normal_sign": float(normal_sign),
        "tangent_xy": tangent.astype(float).tolist(),
        "normal_xy": normal.astype(float).tolist(),
        "cable_anchor_xy": cable_xy.astype(float).tolist(),
        "probe_target_xy": (cable_xy + normal * config.probe_distance).astype(float).tolist(),
        "main_target_xy": (
            beads[endpoint_index, :2] + normal * config.main_pull_distance
        ).astype(float).tolist(),
        "workspace_margin": float(score),
        "geometric_cable_clearance": float(selected["cable_clearance"]),
        "boxes": boxes,
    }
