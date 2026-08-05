"""Pure geometry for the fixed hidden routing-gate cable task."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence

import numpy as np


WHOLE_CABLE_BARRIER_MODE = (
    "whole_cable_span"
)
ENDPOINT_CORRIDOR_BARRIER_MODE = (
    "endpoint_corridor"
)
BARRIER_MODES = (
    WHOLE_CABLE_BARRIER_MODE,
    ENDPOINT_CORRIDOR_BARRIER_MODE,
)
FIXED_CENTER_RATIO_PROBE_SELECTOR = (
    "fixed_center_ratio"
)
ALL_BEAD_CLEARANCE_PROBE_SELECTOR = (
    "all_bead_clearance_nearest_center"
)
PROBE_SELECTOR_MODES = (
    FIXED_CENTER_RATIO_PROBE_SELECTOR,
    ALL_BEAD_CLEARANCE_PROBE_SELECTOR,
)



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
    probe_selector_mode: str = (
        FIXED_CENTER_RATIO_PROBE_SELECTOR
    )
    probe_index_override: int = -1
    barrier_mode: str = (
        WHOLE_CABLE_BARRIER_MODE
    )
    barrier_safety_margin: float = 0.0
    bead_radius: float = 0.005
    topology_id: str = (
        "hidden_routing_gate_v1"
    )

    def __post_init__(self):
        if not str(self.topology_id).strip():
            raise ValueError(
                "topology_id must be non-empty"
            )
        if not (
            0.0
            <= float(self.center_ratio)
            <= 1.0
        ):
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
            "barrier_height",
            "stage1_pull_distance",
            "final_pull_distance",
            "target_plane_offset",
            "target_zone_depth",
            "target_corridor_half_width",
            "bead_radius",
        )
        for name in positive:
            value = float(
                getattr(self, name)
            )
            if (
                not np.isfinite(value)
                or value <= 0
            ):
                raise ValueError(
                    f"{name} must be finite "
                    "and positive"
                )

        mode = str(
            self.barrier_mode
        ).strip()
        if mode not in BARRIER_MODES:
            raise ValueError(
                "unsupported barrier_mode "
                f"{mode!r}"
            )

        selector = str(
            self.probe_selector_mode
        )
        if selector not in PROBE_SELECTOR_MODES:
            raise ValueError(
                "unsupported probe_selector_mode "
                f"{selector!r}"
            )

        override = int(
            self.probe_index_override
        )
        if override < -1:
            raise ValueError(
                "probe_index_override must be "
                "-1 or an interior bead index"
            )

        width = float(
            self.barrier_width
        )
        safety = float(
            self.barrier_safety_margin
        )
        if not np.isfinite(width):
            raise ValueError(
                "barrier_width must be finite"
            )
        if not np.isfinite(safety):
            raise ValueError(
                "barrier_safety_margin "
                "must be finite"
            )

        if (
            mode
            == WHOLE_CABLE_BARRIER_MODE
        ):
            if width <= 0:
                raise ValueError(
                    "whole-cable barrier_width "
                    "must be positive"
                )
            if abs(safety) > 1e-12:
                raise ValueError(
                    "whole-cable mode must not "
                    "use barrier_safety_margin"
                )
        else:
            if abs(width) > 1e-12:
                raise ValueError(
                    "endpoint-corridor mode "
                    "derives width and requires "
                    "barrier_width=0"
                )
            if safety <= 0:
                raise ValueError(
                    "endpoint-corridor mode "
                    "requires a positive fixed "
                    "barrier_safety_margin"
                )

        if int(
            self.leading_segment_size
        ) <= 0:
            raise ValueError(
                "leading_segment_size "
                "must be positive"
            )
        if (
            float(
                self.final_pull_distance
            )
            <= float(
                self.stage1_pull_distance
            )
        ):
            raise ValueError(
                "final pull must exceed "
                "stage-1 pull"
            )

        barrier_far_face = (
            float(self.barrier_offset)
            + float(
                self.barrier_thickness
            ) / 2
        )
        if (
            float(
                self.target_plane_offset
            )
            <= barrier_far_face
            + float(self.bead_radius)
        ):
            raise ValueError(
                "target plane must lie beyond "
                "the barrier and one bead radius"
            )
        if (
            float(
                self.final_pull_distance
            )
            <= float(
                self.target_plane_offset
            )
        ):
            raise ValueError(
                "final target must lie beyond "
                "target plane"
            )

        for name, bounds in (
            (
                "workspace_x",
                self.workspace_x,
            ),
            (
                "workspace_y",
                self.workspace_y,
            ),
        ):
            values = np.asarray(
                bounds,
                dtype=np.float64,
            ).reshape(-1)
            if (
                values.size != 2
                or not np.all(
                    np.isfinite(values)
                )
                or values[0] >= values[1]
            ):
                raise ValueError(
                    f"invalid {name}"
                )

    @property
    def bead_collision_support(
        self,
    ) -> float:
        return float(
            np.sqrt(3.0)
            * float(self.bead_radius)
        )

    @property
    def corridor_required_width(
        self,
    ) -> float:
        return float(
            2.0 * (
                float(
                    self
                    .target_corridor_half_width
                )
                + self.bead_collision_support
            )
        )

    @property
    def resolved_barrier_width(
        self,
    ) -> float:
        if (
            str(self.barrier_mode)
            == WHOLE_CABLE_BARRIER_MODE
        ):
            return float(
                self.barrier_width
            )
        return float(
            self.corridor_required_width
            + 2.0
            * float(
                self.barrier_safety_margin
            )
        )


class HiddenRoutingGateGeometryError(RuntimeError):
    """No legal fixed routing geometry, with complete provenance."""

    def __init__(self, message, diagnostics):
        super().__init__(str(message))
        self.diagnostics = diagnostics

    def to_dict(self):
        return {
            "exception_type": type(self).__name__,
            "message": str(self),
            "diagnostics": self.diagnostics,
        }


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


def _margin_for_points(
    points,
    x_bounds,
    y_bounds,
):
    values = list(points)
    if not values:
        raise ValueError(
            "workspace margin point set is empty"
        )
    return _workspace_margin(
        values,
        x_bounds,
        y_bounds,
    )


def _json_float(value):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(
            "geometry diagnostic is non-finite"
        )
    return value


def _probe_roof_candidate(
    beads,
    config,
    probe_index,
    yaw,
):
    probe_index = int(
        probe_index
    )
    probe = np.asarray(
        beads[probe_index],
        dtype=np.float64,
    )
    construction_epsilon = 5e-5
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
        "yaw": float(yaw),
    }

    clearances = [
        float(
            _point_box_distance(
                bead,
                probe_roof,
            )
            - config.bead_radius
        )
        for bead in beads
    ]
    nearest_index = int(
        np.argmin(clearances)
    )
    return {
        "probe_index": probe_index,
        "probe_roof": probe_roof,
        "roof_bottom_z": roof_bottom,
        "all_bead_clearance": float(
            clearances[nearest_index]
        ),
        "nearest_bead_index": (
            nearest_index
        ),
    }


def select_probe_roof(
    bead_positions,
    config,
):
    beads = np.asarray(
        bead_positions,
        dtype=np.float64,
    )
    bead_count = int(
        beads.shape[0]
    )
    preferred_index = int(np.clip(
        round(
            float(config.center_ratio)
            * (bead_count - 1)
        ),
        1,
        bead_count - 2,
    ))

    endpoint_axis = _normalize(
        beads[-1, :2]
        - beads[0, :2]
    )
    yaw = float(
        np.arctan2(
            endpoint_axis[1],
            endpoint_axis[0],
        )
    )

    override = int(
        config.probe_index_override
    )
    selector = str(
        config.probe_selector_mode
    )

    if override >= 0:
        indices = [override]
    elif (
        selector
        == FIXED_CENTER_RATIO_PROBE_SELECTOR
    ):
        indices = [preferred_index]
    else:
        indices = list(
            range(
                1,
                bead_count - 1,
            )
        )

    rows = [
        _probe_roof_candidate(
            beads,
            config,
            index,
            yaw,
        )
        for index in indices
    ]

    if (
        selector
        == ALL_BEAD_CLEARANCE_PROBE_SELECTOR
        and override < 0
    ):
        legal = [
            row
            for row in rows
            if (
                row["all_bead_clearance"]
                + 1e-12
                >= config.probe_roof_clearance
            )
        ]
        if not legal:
            best = max(
                rows,
                key=lambda row: (
                    row["all_bead_clearance"],
                    -abs(
                        row["probe_index"]
                        - preferred_index
                    ),
                    -row["probe_index"],
                ),
            )
            raise HiddenRoutingGateGeometryError(
                (
                    "no interior probe bead "
                    "satisfies all-bead roof "
                    "clearance"
                ),
                {
                    "failure": (
                        "probe_roof_all_bead_"
                        "clearance"
                    ),
                    "preferred_probe_index": (
                        preferred_index
                    ),
                    "best_probe_index": int(
                        best["probe_index"]
                    ),
                    "best_clearance": float(
                        best[
                            "all_bead_clearance"
                        ]
                    ),
                    "required_clearance": float(
                        config
                        .probe_roof_clearance
                    ),
                },
            )
        selected = min(
            legal,
            key=lambda row: (
                abs(
                    row["probe_index"]
                    - preferred_index
                ),
                -row[
                    "all_bead_clearance"
                ],
                row["probe_index"],
            ),
        )
        legal_indices = [
            int(row["probe_index"])
            for row in legal
        ]
    else:
        selected = rows[0]
        legal_indices = (
            [int(selected["probe_index"])]
            if (
                selected[
                    "all_bead_clearance"
                ]
                + 1e-12
                >= config
                .probe_roof_clearance
            )
            else []
        )

    return {
        **selected,
        "selector_mode": selector,
        "preferred_probe_index": (
            preferred_index
        ),
        "legal_probe_indices": (
            legal_indices
        ),
    }


def _barrier_candidate_geometry(
    beads,
    endpoint,
    tangent,
    normal,
    centroid,
    bounding_radius,
    config,
):
    mode = str(
        config.barrier_mode
    )
    resolved_width = float(
        config.resolved_barrier_width
    )

    tangent_projection = (
        beads[:, :2] @ tangent
    )
    tangent_min = float(
        np.min(tangent_projection)
    )
    tangent_max = float(
        np.max(tangent_projection)
    )
    tangent_midpoint = 0.5 * (
        tangent_min + tangent_max
    )

    if (
        mode
        == WHOLE_CABLE_BARRIER_MODE
    ):
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
        tangent_reference = (
            tangent_midpoint
        )
        coverage_reference = (
            "whole_cable_tangent_span"
        )
    elif (
        mode
        == ENDPOINT_CORRIDOR_BARRIER_MODE
    ):
        barrier_center = (
            endpoint[:2]
            + normal
            * config.barrier_offset
        )
        tangent_reference = float(
            np.dot(
                endpoint[:2],
                tangent,
            )
        )
        coverage_reference = (
            "pulled_endpoint_target_corridor"
        )
    else:
        raise RuntimeError(
            "validated barrier mode "
            "became unsupported"
        )

    centered_tangent_coordinate = float(
        np.dot(
            barrier_center,
            tangent,
        )
    )
    tangent_center_error = float(
        centered_tangent_coordinate
        - tangent_reference
    )
    tangent_coordinates = (
        beads[:, :2]
        - barrier_center.reshape(1, 2)
    ) @ tangent
    max_abs_tangent_coordinate = float(
        np.max(
            np.abs(
                tangent_coordinates
            )
        )
    )
    full_cable_required_width = float(
        2.0 * (
            max_abs_tangent_coordinate
            + bounding_radius
        )
    )

    if (
        mode
        == WHOLE_CABLE_BARRIER_MODE
    ):
        required_width = (
            full_cable_required_width
        )
        coverage_margin = float(
            resolved_width / 2
            - max_abs_tangent_coordinate
            - bounding_radius
        )
    else:
        required_width = float(
            config.corridor_required_width
        )
        coverage_margin = float(
            resolved_width / 2
            - float(
                config
                .target_corridor_half_width
            )
            - bounding_radius
        )

    yaw_tangent = float(
        np.arctan2(
            tangent[1],
            tangent[0],
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
            float(
                resolved_width / 2
            ),
            float(
                config
                .barrier_thickness / 2
            ),
            float(
                config.barrier_height / 2
            ),
        ],
        "yaw": yaw_tangent,
    }
    return {
        "barrier_mode": mode,
        "coverage_reference": (
            coverage_reference
        ),
        "barrier": barrier,
        "barrier_center_xy": (
            barrier_center
            .astype(float)
            .tolist()
        ),
        "barrier_center_tangent_coordinate": (
            centered_tangent_coordinate
        ),
        "barrier_tangent_reference": (
            tangent_reference
        ),
        "barrier_tangent_center_error": (
            tangent_center_error
        ),
        "tangent_projection_min": (
            tangent_min
        ),
        "tangent_projection_max": (
            tangent_max
        ),
        "tangent_projection_midpoint": (
            tangent_midpoint
        ),
        "max_abs_tangent_coordinate": (
            max_abs_tangent_coordinate
        ),
        "configured_barrier_width": (
            float(config.barrier_width)
        ),
        "resolved_barrier_width": (
            resolved_width
        ),
        "required_barrier_width": (
            required_width
        ),
        "full_cable_required_width_diagnostic": (
            full_cable_required_width
        ),
        "barrier_width_shortfall": (
            float(max(
                0.0,
                required_width
                - resolved_width,
            ))
        ),
        "coverage_margin": (
            coverage_margin
        ),
        "barrier_safety_margin": (
            float(
                config
                .barrier_safety_margin
            )
        ),
    }
def _candidate_rejection_reasons(row):
    reasons = []
    if row["coverage_margin"] <= 0:
        reasons.append("barrier_coverage")
    if row["workspace_margin"] <= 0:
        reasons.append("workspace")
    if (
        row["minimum_clearance"] + 1e-9
        < row["expected_minimum_clearance"]
    ):
        reasons.append("initial_clearance")
    return reasons


def evaluate_hidden_routing_gate_candidates(
    bead_positions: np.ndarray,
    config: HiddenRoutingGateGeometryConfig,
) -> Dict[str, Any]:
    """Evaluate all four fixed endpoint/sign candidates.

    This function never discards a candidate without recording why.
    It performs no search over task parameters.
    """
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
    probe_selection = (
        select_probe_roof(
            beads,
            config,
        )
    )
    probe_index = int(
        probe_selection["probe_index"]
    )
    probe_roof = (
        probe_selection["probe_roof"]
    )
    roof_bottom = float(
        probe_selection["roof_bottom_z"]
    )
    roof_all_bead_clearance = float(
        probe_selection[
            "all_bead_clearance"
        ]
    )
    if (
        str(config.probe_selector_mode)
        == ALL_BEAD_CLEARANCE_PROBE_SELECTOR
    ):
        roof_surface_clearance = (
            roof_all_bead_clearance
        )
    else:
        roof_surface_clearance = float(
            _point_box_distance(
                beads[probe_index],
                probe_roof,
            )
            - config.bead_radius
        )
    roof_nearest_bead_index = int(
        probe_selection[
            "nearest_bead_index"
        ]
    )
    base_normal = np.asarray(
        [-endpoint_axis[1], endpoint_axis[0]],
        dtype=np.float64,
    )
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
    rows: List[Dict[str, Any]] = []

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
            barrier_geometry = (
                _barrier_candidate_geometry(
                    beads=beads,
                    endpoint=endpoint,
                    tangent=tangent,
                    normal=normal,
                    centroid=centroid,
                    bounding_radius=(
                        bounding_radius
                    ),
                    config=config,
                )
            )
            barrier = (
                barrier_geometry["barrier"]
            )
            yaw_tangent = float(
                barrier["yaw"]
            )
            yaw_normal = float(
                np.arctan2(
                    normal[1],
                    normal[0],
                )
            )

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
                * (
                    config.target_zone_depth / 2
                )
            )
            target_zone_corners = (
                _rectangle_corners(
                    target_zone_center,
                    normal,
                    tangent,
                    (
                        config.target_zone_depth
                        / 2
                    ),
                    (
                        config
                        .target_corridor_half_width
                    ),
                )
            )

            component_workspace_margin = {
                "stage1_target": _margin_for_points(
                    [stage1_target],
                    x_bounds,
                    y_bounds,
                ),
                "final_target": _margin_for_points(
                    [final_target],
                    x_bounds,
                    y_bounds,
                ),
                "target_plane": _margin_for_points(
                    [target_plane_point],
                    x_bounds,
                    y_bounds,
                ),
                "routing_barrier": _margin_for_points(
                    _box_corners(barrier),
                    x_bounds,
                    y_bounds,
                ),
                "probe_roof": _margin_for_points(
                    _box_corners(probe_roof),
                    x_bounds,
                    y_bounds,
                ),
                "target_zone": _margin_for_points(
                    target_zone_corners,
                    x_bounds,
                    y_bounds,
                ),
            }
            workspace_margin = float(min(
                component_workspace_margin.values()
            ))

            barrier_surface_clearance = min(
                _point_box_distance(
                    bead,
                    barrier,
                ) - bounding_radius
                for bead in beads
            )
            minimum_clearance = float(min(
                barrier_surface_clearance,
                roof_surface_clearance,
            ))
            expected_minimum = float(min(
                config.probe_roof_clearance,
                (
                    config.barrier_offset
                    - config.barrier_thickness / 2
                    - bounding_radius
                ),
            ))

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
                "probe_index": int(
                    probe_index
                ),
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
                "target_corridor_half_width": (
                    float(
                        config
                        .target_corridor_half_width
                    )
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
                        config.target_zone_depth
                        / 2
                    ),
                    float(
                        config
                        .target_corridor_half_width
                    ),
                ],
                "target_zone_yaw": yaw_normal,
            }

            row = {
                "endpoint_index": int(
                    endpoint_index
                ),
                "normal_sign": float(
                    normal_sign
                ),
                "probe_index": int(
                    probe_index
                ),
                "probe_selector_mode": str(
                    probe_selection[
                        "selector_mode"
                    ]
                ),
                "preferred_probe_index": int(
                    probe_selection[
                        "preferred_probe_index"
                    ]
                ),
                "probe_roof_nearest_bead_index": (
                    roof_nearest_bead_index
                ),
                "legal_probe_indices": [
                    int(value)
                    for value in probe_selection[
                        "legal_probe_indices"
                    ]
                ],
                "normal_xy": (
                    normal.astype(float).tolist()
                ),
                "tangent_xy": (
                    tangent.astype(float).tolist()
                ),
                **{
                    key: (
                        _json_float(value)
                        if isinstance(
                            value,
                            (
                                float,
                                int,
                                np.floating,
                                np.integer,
                            ),
                        )
                        else value
                    )
                    for key, value
                    in barrier_geometry.items()
                    if key != "barrier"
                },
                "workspace_margin": (
                    _json_float(
                        workspace_margin
                    )
                ),
                "component_workspace_margin": {
                    key: _json_float(value)
                    for key, value
                    in component_workspace_margin.items()
                },
                "barrier_surface_clearance": (
                    _json_float(
                        barrier_surface_clearance
                    )
                ),
                "probe_roof_surface_clearance": (
                    _json_float(
                        roof_surface_clearance
                    )
                ),
                "minimum_clearance": (
                    _json_float(
                        minimum_clearance
                    )
                ),
                "expected_minimum_clearance": (
                    _json_float(
                        expected_minimum
                    )
                ),
                "public_task": public_task,
                "boxes": [
                    probe_roof,
                    barrier,
                ],
                "roof_bottom_z": (
                    _json_float(
                        roof_bottom
                    )
                ),
            }
            reasons = (
                _candidate_rejection_reasons(
                    row
                )
            )
            row["rejection_reasons"] = reasons
            row["accepted"] = not reasons
            rows.append(row)

    rejection_counts = Counter(
        reason
        for row in rows
        for reason in row[
            "rejection_reasons"
        ]
    )
    accepted = [
        row
        for row in rows
        if row["accepted"]
    ]
    return {
        "audit_version": (
            "hidden_routing_gate_"
            "geometry_audit_v1"
        ),
        "barrier_mode": str(
            config.barrier_mode
        ),
        "configured_barrier_width": float(
            config.barrier_width
        ),
        "resolved_barrier_width": float(
            config.resolved_barrier_width
        ),
        "corridor_required_width": float(
            config.corridor_required_width
        ),
        "barrier_safety_margin": float(
            config.barrier_safety_margin
        ),
        "probe_selection": {
            "selector_mode": str(
                probe_selection[
                    "selector_mode"
                ]
            ),
            "preferred_probe_index": int(
                probe_selection[
                    "preferred_probe_index"
                ]
            ),
            "selected_probe_index": int(
                probe_index
            ),
            "selected_all_bead_clearance": (
                float(
                    roof_all_bead_clearance
                )
            ),
            "nearest_bead_index": int(
                roof_nearest_bead_index
            ),
            "legal_probe_indices": [
                int(value)
                for value in probe_selection[
                    "legal_probe_indices"
                ]
            ],
        },
        "topology_id": str(
            config.topology_id
        ),
        "config": asdict(config),
        "bead_count": int(
            beads.shape[0]
        ),
        "bead_positions_xyz": (
            beads.astype(float).tolist()
        ),
        "ordered_endpoint_center_span": float(
            np.linalg.norm(
                beads[-1, :2]
                - beads[0, :2]
            )
        ),
        "bead_collision_bounding_radius": (
            bounding_radius
        ),
        "candidate_count": len(rows),
        "accepted_candidate_count": len(
            accepted
        ),
        "rejection_counts": {
            str(key): int(value)
            for key, value
            in sorted(
                rejection_counts.items()
            )
        },
        "candidates": rows,
    }


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
    audit = (
        evaluate_hidden_routing_gate_candidates(
            bead_positions,
            config,
        )
    )
    candidates = [
        row
        for row in audit["candidates"]
        if row["accepted"]
    ]
    if not candidates:
        raise HiddenRoutingGateGeometryError(
            (
                "no legal fixed hidden "
                "routing-gate orientation "
                "and endpoint"
            ),
            audit,
        )

    selected = max(
        candidates,
        key=lambda item: (
            round(
                float(
                    item["workspace_margin"]
                ),
                12,
            ),
            round(
                float(
                    item["coverage_margin"]
                ),
                12,
            ),
            -int(
                item["endpoint_index"]
            ),
            float(
                item["normal_sign"]
            ),
        ),
    )
    return {
        "snapshot_version": (
            "ccda_hidden_routing_gate_"
            "layout_v1r3"
            if (
                str(config.barrier_mode)
                == ENDPOINT_CORRIDOR_BARRIER_MODE
                and str(
                    config.probe_selector_mode
                )
                == (
                    ALL_BEAD_CLEARANCE_PROBE_SELECTOR
                )
            )
            else
            "ccda_hidden_routing_gate_"
            "layout_v1r2"
            if (
                str(config.barrier_mode)
                == ENDPOINT_CORRIDOR_BARRIER_MODE
            )
            else
            "ccda_hidden_routing_gate_"
            "layout_v1r1"
        ),
        "probe_selector_mode": str(
            config.probe_selector_mode
        ),
        "preferred_probe_index": int(
            audit["probe_selection"][
                "preferred_probe_index"
            ]
        ),
        "selected_probe_index": int(
            audit["probe_selection"][
                "selected_probe_index"
            ]
        ),
        "probe_roof_nearest_bead_index": (
            int(
                audit["probe_selection"][
                    "nearest_bead_index"
                ]
            )
        ),
        "probe_roof_all_bead_clearance": (
            float(
                audit["probe_selection"][
                    "selected_all_bead_clearance"
                ]
            )
        ),
        "barrier_mode": str(
            selected["barrier_mode"]
        ),
        "coverage_reference": str(
            selected["coverage_reference"]
        ),
        "resolved_barrier_width": float(
            selected["resolved_barrier_width"]
        ),
        "corridor_required_width": float(
            config.corridor_required_width
        ),
        "barrier_safety_margin": float(
            selected["barrier_safety_margin"]
        ),
        "full_cable_required_width_diagnostic": float(
            selected["full_cable_required_width_diagnostic"]
        ),
        "topology": str(
            config.topology_id
        ),
        "public_task": (
            selected["public_task"]
        ),
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
        "bead_collision_half_extent": (
            float(config.bead_radius)
        ),
        "bead_collision_bounding_radius": (
            float(
                audit[
                    "bead_collision_bounding_radius"
                ]
            )
        ),
        "barrier_tangent_coverage_margin": (
            float(
                selected["coverage_margin"]
            )
        ),
        "barrier_required_width": float(
            selected[
                "required_barrier_width"
            ]
        ),
        "barrier_width_shortfall": float(
            selected[
                "barrier_width_shortfall"
            ]
        ),
        "barrier_tangent_center_error": float(
            selected[
                "barrier_tangent_center_error"
            ]
        ),
        "barrier_surface_clearance": float(
            selected[
                "barrier_surface_clearance"
            ]
        ),
        "probe_roof_surface_clearance": (
            float(
                selected[
                    "probe_roof_surface_clearance"
                ]
            )
        ),
        "expected_surface_clearance": (
            float(
                selected[
                    "minimum_clearance"
                ]
            )
        ),
        "workspace_margin": float(
            selected["workspace_margin"]
        ),
        "component_workspace_margin": (
            selected[
                "component_workspace_margin"
            ]
        ),
        "boxes": selected["boxes"],
        # Privileged engineering provenance only.
        # public_routing_layout() never exposes it.
        "geometry_audit": audit,
    }
