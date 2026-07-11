from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np


DORMANT = "dormant"
ENGAGED = "engaged"
RELEASED = "released"
_VALID_STATES = {DORMANT, ENGAGED, RELEASED}


@dataclass(frozen=True)
class SlackBreakawayConfig:
    """Configuration for a unilateral deadband tether in the XY plane."""

    slack_distance: float
    spring_stiffness: float
    radial_damping: float
    max_tension: float
    breakaway_extension: float
    breakaway_force: float

    def __post_init__(self) -> None:
        values = asdict(self)
        for name, value in values.items():
            if not np.isfinite(float(value)):
                raise ValueError(f"{name} must be finite, got {value!r}")

        if self.slack_distance <= 0:
            raise ValueError("slack_distance must be positive")
        if self.spring_stiffness <= 0:
            raise ValueError("spring_stiffness must be positive")
        if self.radial_damping < 0:
            raise ValueError("radial_damping must be non-negative")
        if self.max_tension <= 0:
            raise ValueError("max_tension must be positive")
        if self.breakaway_extension <= 0:
            raise ValueError("breakaway_extension must be positive")
        if self.breakaway_force <= 0:
            raise ValueError("breakaway_force must be positive")
        if self.breakaway_force > self.max_tension:
            raise ValueError(
                "breakaway_force must not exceed max_tension, otherwise "
                "the force release criterion is unreachable"
            )


@dataclass(frozen=True)
class SlackBreakawayOutput:
    """One pre-physics-step force evaluation."""

    state: str
    force_xyz: np.ndarray
    radial_distance: float
    radial_speed: float
    extension: float
    tension: float
    engaged_now: bool
    released_now: bool
    release_reason: Optional[str]

    def __post_init__(self) -> None:
        force = np.asarray(self.force_xyz, dtype=np.float64)
        if force.shape != (3,):
            raise ValueError(f"force_xyz must be [3], got {force.shape}")
        if not np.all(np.isfinite(force)):
            raise ValueError("force_xyz contains NaN or Inf")
        if self.state not in _VALID_STATES:
            raise ValueError(f"invalid state: {self.state}")


class UnilateralSlackBreakaway:
    """State machine for a hidden slack tether.

    The tether is physically inactive while the selected bead remains within
    ``slack_distance`` of the latent anchor. When the radial distance exceeds
    the deadband, a unilateral spring-damper pulls the bead toward the anchor.
    It never pushes the bead away from the anchor.

    The class is independent from PyBullet. The task integration is
    responsible for reading body state and applying ``force_xyz`` before the
    simulation step.
    """

    SNAPSHOT_VERSION = "ccda_slack_breakaway_v2_state_v1"

    def __init__(
        self,
        config: SlackBreakawayConfig,
        anchor_position: Sequence[float],
    ) -> None:
        self.config = config
        anchor = np.asarray(anchor_position, dtype=np.float64).reshape(-1)
        if anchor.size not in (2, 3):
            raise ValueError(
                f"anchor_position must have 2 or 3 elements, got {anchor.size}"
            )
        if not np.all(np.isfinite(anchor)):
            raise ValueError("anchor_position contains NaN or Inf")

        self.anchor_xy = anchor[:2].copy()
        self.anchor_z = float(anchor[2]) if anchor.size == 3 else 0.0

        self.state = DORMANT
        self.engagement_physics_step: Optional[int] = None
        self.release_physics_step: Optional[int] = None
        self.release_reason: Optional[str] = None

        self.max_radial_distance = 0.0
        self.max_extension = 0.0
        self.max_tension = 0.0

        self.last_radial_distance = 0.0
        self.last_radial_speed = 0.0
        self.last_extension = 0.0
        self.last_tension = 0.0
        self.last_force_xyz = np.zeros(3, dtype=np.float64)

    @staticmethod
    def _xy(value: Sequence[float], *, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
        if array.size < 2:
            raise ValueError(f"{name} must contain XY")
        if not np.all(np.isfinite(array[:2])):
            raise ValueError(f"{name} contains NaN or Inf")
        return array[:2].copy()

    def _zero_output(
        self,
        *,
        radial_distance: float,
        radial_speed: float,
        extension: float,
        engaged_now: bool = False,
        released_now: bool = False,
        release_reason: Optional[str] = None,
    ) -> SlackBreakawayOutput:
        self.last_radial_distance = float(radial_distance)
        self.last_radial_speed = float(radial_speed)
        self.last_extension = float(extension)
        self.last_tension = 0.0
        self.last_force_xyz = np.zeros(3, dtype=np.float64)
        return SlackBreakawayOutput(
            state=self.state,
            force_xyz=self.last_force_xyz.copy(),
            radial_distance=float(radial_distance),
            radial_speed=float(radial_speed),
            extension=float(extension),
            tension=0.0,
            engaged_now=bool(engaged_now),
            released_now=bool(released_now),
            release_reason=release_reason,
        )

    def evaluate(
        self,
        position: Sequence[float],
        linear_velocity: Sequence[float],
        *,
        physics_step: int,
    ) -> SlackBreakawayOutput:
        """Evaluate the force that must be applied during the next step."""
        step = int(physics_step)
        if step < 0:
            raise ValueError("physics_step must be non-negative")

        position_xy = self._xy(position, name="position")
        velocity_xy = self._xy(linear_velocity, name="linear_velocity")

        delta = position_xy - self.anchor_xy
        radial_distance = float(np.linalg.norm(delta))
        self.max_radial_distance = max(
            float(self.max_radial_distance),
            radial_distance,
        )

        if radial_distance > 1e-12:
            unit = delta / radial_distance
            radial_speed = float(np.dot(velocity_xy, unit))
        else:
            unit = np.zeros(2, dtype=np.float64)
            radial_speed = 0.0

        extension = max(
            0.0,
            radial_distance - float(self.config.slack_distance),
        )
        self.max_extension = max(float(self.max_extension), extension)

        if self.state == RELEASED:
            return self._zero_output(
                radial_distance=radial_distance,
                radial_speed=radial_speed,
                extension=extension,
                release_reason=self.release_reason,
            )

        if self.state == DORMANT and extension <= 0.0:
            return self._zero_output(
                radial_distance=radial_distance,
                radial_speed=radial_speed,
                extension=0.0,
            )

        engaged_now = False
        if self.state == DORMANT:
            self.state = ENGAGED
            self.engagement_physics_step = step
            engaged_now = True

        outward_speed = max(0.0, radial_speed)
        uncapped_tension = (
            float(self.config.spring_stiffness) * extension
            + float(self.config.radial_damping) * outward_speed
        )
        tension = min(
            float(self.config.max_tension),
            max(0.0, uncapped_tension),
        )
        self.max_tension = max(float(self.max_tension), tension)

        release_reason: Optional[str] = None
        if extension >= float(self.config.breakaway_extension):
            release_reason = "extension"
        elif tension >= float(self.config.breakaway_force):
            release_reason = "force"

        if release_reason is not None:
            self.state = RELEASED
            self.release_physics_step = step
            self.release_reason = release_reason
            return self._zero_output(
                radial_distance=radial_distance,
                radial_speed=radial_speed,
                extension=extension,
                engaged_now=engaged_now,
                released_now=True,
                release_reason=release_reason,
            )

        force_xy = -tension * unit
        force_xyz = np.asarray(
            [force_xy[0], force_xy[1], 0.0],
            dtype=np.float64,
        )

        self.last_radial_distance = radial_distance
        self.last_radial_speed = radial_speed
        self.last_extension = extension
        self.last_tension = tension
        self.last_force_xyz = force_xyz.copy()

        return SlackBreakawayOutput(
            state=self.state,
            force_xyz=force_xyz,
            radial_distance=radial_distance,
            radial_speed=radial_speed,
            extension=extension,
            tension=tension,
            engaged_now=engaged_now,
            released_now=False,
            release_reason=None,
        )

    def snapshot(self) -> Dict[str, Any]:
        return {
            "snapshot_version": self.SNAPSHOT_VERSION,
            "config": asdict(self.config),
            "anchor_xy": self.anchor_xy.astype(float).tolist(),
            "anchor_z": float(self.anchor_z),
            "state": str(self.state),
            "engagement_physics_step": self.engagement_physics_step,
            "release_physics_step": self.release_physics_step,
            "release_reason": self.release_reason,
            "max_radial_distance": float(self.max_radial_distance),
            "max_extension": float(self.max_extension),
            "max_tension": float(self.max_tension),
            "last_radial_distance": float(self.last_radial_distance),
            "last_radial_speed": float(self.last_radial_speed),
            "last_extension": float(self.last_extension),
            "last_tension": float(self.last_tension),
            "last_force_xyz": self.last_force_xyz.astype(float).tolist(),
        }

    @classmethod
    def from_snapshot(
        cls,
        snapshot: Dict[str, Any],
    ) -> "UnilateralSlackBreakaway":
        if snapshot.get("snapshot_version") != cls.SNAPSHOT_VERSION:
            raise ValueError(
                "unsupported slack snapshot version: "
                f"{snapshot.get('snapshot_version')!r}"
            )
        config = SlackBreakawayConfig(**dict(snapshot["config"]))
        anchor = [
            float(snapshot["anchor_xy"][0]),
            float(snapshot["anchor_xy"][1]),
            float(snapshot.get("anchor_z", 0.0)),
        ]
        instance = cls(config=config, anchor_position=anchor)
        instance.restore(snapshot)
        return instance

    def restore(self, snapshot: Dict[str, Any]) -> None:
        if snapshot.get("snapshot_version") != self.SNAPSHOT_VERSION:
            raise ValueError(
                "unsupported slack snapshot version: "
                f"{snapshot.get('snapshot_version')!r}"
            )

        config = SlackBreakawayConfig(**dict(snapshot["config"]))
        if config != self.config:
            raise ValueError("snapshot config does not match active config")

        anchor_xy = np.asarray(
            snapshot["anchor_xy"],
            dtype=np.float64,
        ).reshape(2)
        if not np.allclose(anchor_xy, self.anchor_xy, atol=0.0, rtol=0.0):
            raise ValueError("snapshot anchor does not match active anchor")

        state = str(snapshot["state"])
        if state not in _VALID_STATES:
            raise ValueError(f"invalid snapshot state: {state}")

        self.state = state
        self.engagement_physics_step = snapshot.get(
            "engagement_physics_step"
        )
        self.release_physics_step = snapshot.get(
            "release_physics_step"
        )
        self.release_reason = snapshot.get("release_reason")

        for field in (
            "max_radial_distance",
            "max_extension",
            "max_tension",
            "last_radial_distance",
            "last_radial_speed",
            "last_extension",
            "last_tension",
        ):
            value = float(snapshot[field])
            if not np.isfinite(value):
                raise ValueError(f"non-finite snapshot field: {field}")
            setattr(self, field, value)

        force = np.asarray(
            snapshot["last_force_xyz"],
            dtype=np.float64,
        ).reshape(3)
        if not np.all(np.isfinite(force)):
            raise ValueError("non-finite snapshot force")
        self.last_force_xyz = force.copy()
