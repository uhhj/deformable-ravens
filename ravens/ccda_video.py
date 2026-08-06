"""Small synchronous video recorder for deterministic CCDA audits."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class CCDAVideoRecorder:
    """Capture environment RGB frames while preserving a PNG fallback."""

    def __init__(self, camera_config, output_path, fps=30, stride=4):
        self.camera_config = dict(camera_config)
        self.output_path = Path(output_path).resolve()
        self.frames_dir = self.output_path.parent / 'frames'
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps)
        self.stride = int(stride)
        if self.fps <= 0:
            raise ValueError('fps must be positive')
        if self.stride <= 0:
            raise ValueError('stride must be positive')
        self._environment = None
        self._writer = None
        self._codec = None
        self._video_error = None
        self._record_calls = 0
        self._frames_written = 0
        self._closed = False
        self._labels = []

    def bind(self, environment):
        self._environment = environment

    def reset_phase_label(self):
        """Environment restore hook; labels are supplied on later records."""

    def _open_writer(self, width, height):
        errors = []
        for codec in ('mp4v', 'avc1'):
            writer = cv2.VideoWriter(
                str(self.output_path), cv2.VideoWriter_fourcc(*codec),
                self.fps, (int(width), int(height)))
            if writer.isOpened():
                self._writer = writer
                self._codec = codec
                return
            writer.release()
            errors.append(codec)
        self._video_error = (
            'OpenCV could not open an MP4 writer; attempted codecs: '
            + ', '.join(errors))

    def record(self, label=''):
        if self._closed:
            return
        if self._environment is None:
            raise RuntimeError('video recorder is not bound to an environment')
        self._record_calls += 1
        if self._record_calls != 1 and (
                (self._record_calls - 1) % self.stride != 0):
            return

        color, _, _ = self._environment.render(self.camera_config)
        rgb = np.asarray(color, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise RuntimeError('environment render did not return RGB [H,W,3]')
        height, width = rgb.shape[:2]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        frame_path = self.frames_dir / 'frame_{:06d}.png'.format(
            self._frames_written)
        if not cv2.imwrite(str(frame_path), bgr):
            raise RuntimeError('failed to write video fallback frame')
        if self._writer is None and self._video_error is None:
            self._open_writer(width, height)
        if self._writer is not None:
            self._writer.write(bgr)
        self._labels.append(str(label))
        self._frames_written += 1

    def close(self):
        if not self._closed:
            if self._writer is not None:
                self._writer.release()
            self._closed = True
        return {
            'output_path': str(self.output_path),
            'frames_dir': str(self.frames_dir),
            'fps': self.fps,
            'stride': self.stride,
            'record_calls': self._record_calls,
            'frames_written': self._frames_written,
            'codec': self._codec,
            'video_available': bool(
                self._codec is not None and self.output_path.exists()),
            'error': self._video_error,
            'labels': list(self._labels),
        }
