#!/usr/bin/env python
"""Continuous PyBullet video recorder for CCDA hidden-contact rollouts."""

import os
from typing import Optional, Sequence

import cv2
import numpy as np
import pybullet as p


class CCDAVideoRecorder(object):
    """Record a side-top PyBullet debug-camera MP4 during motion primitives.

    This recorder is intentionally independent from Ravens policy cameras. It
    uses p.getCameraImage from a side-top view so Phase1.1 videos can show the
    UR5, cable, and contact process inside a pick-place primitive.
    """

    def __init__(
        self,
        path,
        condition,
        visible_seed,
        fps=20,
        stride=4,
        image_width=960,
        image_height=720,
        camera_position=None,
        camera_target=None,
        camera_up=None,
        fov=50.0,
        near=0.01,
        far=3.0,
    ):
        self.path = os.path.abspath(path)
        self.condition = str(condition)
        self.visible_seed = str(visible_seed)
        self.fps = float(fps)
        self.stride = max(1, int(stride))
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.camera_position = list(camera_position or [0.55, -0.75, 0.45])
        self.camera_target = list(camera_target or [0.50, 0.00, 0.02])
        self.camera_up = list(camera_up or [0, 0, 1])
        self.fov = float(fov)
        self.near = float(near)
        self.far = float(far)
        self.action_step = 0
        self.call_count = 0
        self.frame_count = 0
        self.last_label = ""
        self._closed = False

        outdir = os.path.dirname(self.path)
        if outdir:
            os.makedirs(outdir, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            self.path,
            fourcc,
            self.fps,
            (self.image_width, self.image_height),
        )
        if not self.writer.isOpened():
            raise RuntimeError("Could not open VideoWriter for {}".format(self.path))

        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.camera_position,
            cameraTargetPosition=self.camera_target,
            cameraUpVector=self.camera_up,
        )
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=float(self.image_width) / float(self.image_height),
            nearVal=self.near,
            farVal=self.far,
        )

    def set_action_step(self, action_step):
        self.action_step = int(action_step)

    def record(self, label=""):
        """Render and write one frame if the stride permits it."""
        if self._closed:
            return

        if label:
            self.last_label = str(label)

        should_write = self.call_count == 0 or (self.call_count % self.stride == 0)
        self.call_count += 1
        if not should_write:
            return

        rgb = self._render_rgb()
        frame = self._overlay(rgb, self.last_label)
        self.writer.write(frame)
        self.frame_count += 1

    def close(self):
        if self._closed:
            return
        self.writer.release()
        self._closed = True

    def summary(self):
        return {
            "path": self.path,
            "condition": self.condition,
            "visible_seed": self.visible_seed,
            "fps": self.fps,
            "stride": self.stride,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "camera_position": self.camera_position,
            "camera_target": self.camera_target,
            "camera_up": self.camera_up,
            "record_calls": self.call_count,
            "frames_written": self.frame_count,
        }

    def _render_rgb(self):
        try:
            result = p.getCameraImage(
                width=self.image_width,
                height=self.image_height,
                viewMatrix=self.view_matrix,
                projectionMatrix=self.projection_matrix,
                shadow=1,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
            )
        except Exception:
            result = p.getCameraImage(
                width=self.image_width,
                height=self.image_height,
                viewMatrix=self.view_matrix,
                projectionMatrix=self.projection_matrix,
                shadow=1,
                renderer=p.ER_TINY_RENDERER,
            )

        rgba = np.asarray(result[2], dtype=np.uint8).reshape(
            self.image_height,
            self.image_width,
            4,
        )
        return rgba[:, :, :3]

    def _overlay(self, rgb, label):
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.rectangle(bgr, (0, 0), (self.image_width, 86), (0, 0, 0), -1)

        lines = [
            "condition: {}".format(self.condition),
            "visible_seed: {} | action step: {} | phase: {}".format(
                self.visible_seed,
                self.action_step,
                label or "",
            ),
            "debug camera: side-top | frame: {}".format(self.frame_count),
        ]
        y = 24
        for line in lines:
            cv2.putText(
                bgr,
                line,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            y += 26
        return bgr
