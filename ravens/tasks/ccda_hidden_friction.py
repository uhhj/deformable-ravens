"""Pure hidden planar-friction model for CCDA cable experiments."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Sequence

import numpy as np


@dataclass(frozen=True)
class HiddenFrictionConfig:
    patch_radius: float
    viscous_gain: float
    coulomb_force: float
    max_force: float
    speed_epsilon: float
    contact_height: float

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not np.isfinite(float(value)):
                raise ValueError(f"{name} must be finite, got {value!r}")
            if float(value) <= 0:
                raise ValueError(f"{name} must be positive, got {value!r}")
        if self.coulomb_force > self.max_force:
            raise ValueError("coulomb_force must not exceed max_force")


@dataclass(frozen=True)
class NativeSegmentFrictionConfig:
    base_lateral_friction: float
    hidden_lateral_friction: float
    spinning_friction: float
    rolling_friction: float
    restitution: float

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.hidden_lateral_friction <= self.base_lateral_friction:
            raise ValueError(
                "hidden_lateral_friction must exceed base_lateral_friction"
            )

    def snapshot(self) -> Dict[str, Any]:
        return {
            "snapshot_version": "ccda_native_segment_friction_v1",
            "config": {
                name: float(value)
                for name, value in asdict(self).items()
            },
        }


@dataclass(frozen=True)
class HiddenFrictionOutput:
    force_xyz: np.ndarray
    active: bool
    inside_patch: bool
    planar_speed: float
    force_norm: float
    radial_distance: float

    def __post_init__(self) -> None:
        force = np.asarray(self.force_xyz, dtype=np.float64)
        if force.shape != (3,):
            raise ValueError(f"force_xyz must have shape (3,), got {force.shape}")
        if not np.all(np.isfinite(force)):
            raise ValueError("force_xyz contains NaN or Inf")


class HiddenPlanarFrictionPatch:
    """Velocity-opposing force inside a latent circular XY patch."""

    SNAPSHOT_VERSION = "ccda_hidden_planar_friction_v1"

    def __init__(
        self,
        config: HiddenFrictionConfig,
        center_xy: Sequence[float],
        enabled: bool,
    ) -> None:
        center = np.asarray(center_xy, dtype=np.float64).reshape(-1)
        if center.size != 2:
            raise ValueError(f"center_xy must contain exactly two values, got {center.size}")
        if not np.all(np.isfinite(center)):
            raise ValueError("center_xy contains NaN or Inf")
        self.config = config
        self.center_xy = center.copy()
        self.enabled = bool(enabled)

    @staticmethod
    def _vector(value: Sequence[float], *, name: str) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
        if vector.size < 3:
            raise ValueError(f"{name} must contain XYZ")
        if not np.all(np.isfinite(vector[:3])):
            raise ValueError(f"{name} contains NaN or Inf")
        return vector[:3].copy()

    def evaluate(
        self,
        position: Sequence[float],
        linear_velocity: Sequence[float],
    ) -> HiddenFrictionOutput:
        position_xyz = self._vector(position, name="position")
        velocity_xyz = self._vector(linear_velocity, name="linear_velocity")
        delta_xy = position_xyz[:2] - self.center_xy
        radial_distance = float(np.linalg.norm(delta_xy))
        planar_speed = float(np.linalg.norm(velocity_xyz[:2]))
        inside_patch = bool(
            radial_distance <= self.config.patch_radius
            and position_xyz[2] <= self.config.contact_height
        )

        force_xyz = np.zeros(3, dtype=np.float64)
        active = bool(
            self.enabled
            and inside_patch
            and planar_speed > self.config.speed_epsilon
        )
        if active:
            magnitude = min(
                self.config.max_force,
                self.config.coulomb_force + self.config.viscous_gain * planar_speed,
            )
            force_xy = -magnitude * velocity_xyz[:2] / planar_speed
            force_xyz = np.array([force_xy[0], force_xy[1], 0.0], dtype=np.float64)

        force_norm = float(np.linalg.norm(force_xyz))
        if force_norm > self.config.max_force + 1e-12:
            raise RuntimeError("computed hidden-friction force exceeds max_force")
        if float(np.dot(force_xyz[:2], velocity_xyz[:2])) > 1e-12:
            raise RuntimeError("hidden-friction force is not velocity opposing")
        return HiddenFrictionOutput(
            force_xyz=force_xyz,
            active=active,
            inside_patch=inside_patch,
            planar_speed=planar_speed,
            force_norm=force_norm,
            radial_distance=radial_distance,
        )

    def snapshot(self) -> Dict[str, Any]:
        return {
            "snapshot_version": self.SNAPSHOT_VERSION,
            "config": {name: float(value) for name, value in asdict(self.config).items()},
            "center_xy": [float(value) for value in self.center_xy],
            "enabled": bool(self.enabled),
        }
