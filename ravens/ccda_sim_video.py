"""Real PyBullet RGB video recording for CCDA simulation rollouts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import cv2
import numpy as np
import pybullet as p


def _vector(values: Sequence[float], size: int, name: str) -> List[float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size != size:
        raise ValueError(f"{name} must contain {size} values, got {array.size}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")
    return [float(value) for value in array]


def draw_status_overlay(frame_bgr: np.ndarray, lines: Sequence[str]) -> np.ndarray:
    """Draw a readable status panel without changing frame dimensions."""
    frame = np.asarray(frame_bgr, dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"expected BGR frame [H,W,3], got {frame.shape}")

    output = frame.copy()
    panel_height = max(54, 24 + 24 * len(lines))
    panel_height = min(panel_height, output.shape[0])
    panel = output[:panel_height].copy()
    dark = np.zeros_like(panel)
    cv2.addWeighted(panel, 0.28, dark, 0.72, 0.0, panel)
    output[:panel_height] = panel

    y = 23
    for line in lines:
        cv2.putText(
            output,
            str(line),
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 23
    return output


class CCDASimulationVideoRecorder:
    """Capture real RGB frames from the active PyBullet simulation."""

    def __init__(
        self,
        output_path: Path,
        task: Any,
        condition: str,
        group_id: str,
        seed: int,
        config: Mapping[str, Any],
    ) -> None:
        self.output_path = Path(output_path).resolve()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.output_path.with_suffix(".frames.json")
        self.task = task
        self.condition = str(condition)
        self.group_id = str(group_id)
        self.seed = int(seed)

        self.width = int(config.get("width", 640))
        self.height = int(config.get("height", 480))
        self.fps = float(config.get("fps", 20.0))
        self.record_call_stride = int(config.get("record_call_stride", 5))
        self.camera_eye = _vector(
            config.get("camera_eye", [0.68, -0.82, 0.62]), 3, "camera_eye"
        )
        self.camera_target = _vector(
            config.get("camera_target", [0.50, -0.02, 0.025]),
            3,
            "camera_target",
        )
        self.camera_up = _vector(
            config.get("camera_up", [0.0, 0.0, 1.0]), 3, "camera_up"
        )
        self.fov = float(config.get("fov", 50.0))
        self.near = float(config.get("near", 0.01))
        self.far = float(config.get("far", 3.0))
        self.renderer = str(config.get("renderer", "tiny"))
        self.overlay = bool(config.get("overlay", True))

        if self.width <= 0 or self.height <= 0:
            raise ValueError("video width and height must be positive")
        if self.fps <= 0:
            raise ValueError("video fps must be positive")
        if self.record_call_stride <= 0:
            raise ValueError("record_call_stride must be positive")
        if self.near <= 0 or self.far <= self.near:
            raise ValueError("camera clipping planes are invalid")
        if self.renderer not in {"tiny", "hardware", "auto"}:
            raise ValueError("renderer must be tiny, hardware, or auto")

        self._view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.camera_eye,
            cameraTargetPosition=self.camera_target,
            cameraUpVector=self.camera_up,
        )
        self._projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=float(self.width) / float(self.height),
            nearVal=self.near,
            farVal=self.far,
        )
        self._writer, self.codec = self._open_writer()
        self._record_calls = 0
        self._frames_written = 0
        self._closed = False
        self._frames: List[Dict[str, Any]] = []

    def _open_writer(self):
        attempts = []
        for codec in ("mp4v", "avc1"):
            writer = cv2.VideoWriter(
                str(self.output_path),
                cv2.VideoWriter_fourcc(*codec),
                self.fps,
                (self.width, self.height),
            )
            if writer.isOpened():
                return writer, codec
            writer.release()
            attempts.append(codec)
        raise RuntimeError(
            "OpenCV could not open an MP4 writer; attempted codecs: "
            + ", ".join(attempts)
        )

    def _renderer_candidates(self) -> List[int]:
        if self.renderer == "tiny":
            return [p.ER_TINY_RENDERER]
        if self.renderer == "hardware":
            return [p.ER_BULLET_HARDWARE_OPENGL]
        return [p.ER_BULLET_HARDWARE_OPENGL, p.ER_TINY_RENDERER]

    def _render_bgr(self) -> np.ndarray:
        errors = []
        for renderer in self._renderer_candidates():
            try:
                result = p.getCameraImage(
                    width=self.width,
                    height=self.height,
                    viewMatrix=self._view_matrix,
                    projectionMatrix=self._projection_matrix,
                    shadow=1,
                    renderer=renderer,
                )
                rgba = np.asarray(result[2], dtype=np.uint8).reshape(
                    self.height, self.width, 4
                )
                rgb = rgba[:, :, :3]
                if rgb.size == 0:
                    raise RuntimeError("renderer returned an empty frame")
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            except Exception as exc:  # pragma: no cover
                errors.append(repr(exc))
        raise RuntimeError("all PyBullet renderers failed: " + " | ".join(errors))

    def _phase(self) -> str:
        getter = getattr(self.task, "ccda_phase", None)
        if callable(getter):
            return str(getter())
        return str(getattr(self.task, "_phase", "unknown"))

    def _physics_step(self) -> int:
        getter = getattr(self.task, "physics_step_count", None)
        return int(getter()) if callable(getter) else -1

    def _contact(self) -> Dict[str, Any]:
        getter = getattr(self.task, "ccda_contact_observation", None)
        if not callable(getter):
            return {"force_norm": 0.0, "max_force_norm": 0.0, "active_beads": 0}
        value = dict(getter())
        return {
            "force_norm": float(value.get("force_norm", 0.0)),
            "max_force_norm": float(value.get("max_force_norm", 0.0)),
            "active_beads": int(value.get("active_beads", 0)),
        }

    def record(self, label: str = "") -> None:
        if self._closed:
            return
        self._record_calls += 1
        if self._record_calls != 1 and (
            (self._record_calls - 1) % self.record_call_stride != 0
        ):
            return

        frame = self._render_bgr()
        phase = self._phase()
        physics_step = self._physics_step()
        contact = self._contact()
        if self.overlay:
            lines = [
                f"{self.group_id} | {self.condition} | seed={self.seed}",
                f"phase={phase} | motion={label or 'sample'} | physics_step={physics_step}",
                (
                    f"contact_force_sum={contact['force_norm']:.5f} N | "
                    f"max_bead_force={contact['max_force_norm']:.5f} N | "
                    f"active_beads={contact['active_beads']}"
                ),
            ]
            frame = draw_status_overlay(frame, lines)

        self._writer.write(frame)
        self._frames.append(
            {
                "frame_index": int(self._frames_written),
                "record_call": int(self._record_calls),
                "label": str(label),
                "phase": phase,
                "physics_step": physics_step,
                "contact": contact,
            }
        )
        self._frames_written += 1

    def summary(self) -> Dict[str, Any]:
        return {
            "output_path": str(self.output_path),
            "manifest_path": str(self.manifest_path),
            "condition": self.condition,
            "group_id": self.group_id,
            "seed": self.seed,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "codec": self.codec,
            "renderer": self.renderer,
            "record_call_stride": self.record_call_stride,
            "record_calls": self._record_calls,
            "frames_written": self._frames_written,
            "camera_eye": self.camera_eye,
            "camera_target": self.camera_target,
            "camera_up": self.camera_up,
            "fov": self.fov,
            "near": self.near,
            "far": self.far,
        }

    def close(self) -> Dict[str, Any]:
        if self._closed:
            return self.summary()
        self._writer.release()
        self._closed = True
        payload = self.summary()
        payload["frames"] = self._frames
        self.manifest_path.write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return payload
