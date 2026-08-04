import numpy as np
import pytest

from ravens.ccda_sim_video import draw_status_overlay


def test_draw_status_overlay_preserves_shape_and_dtype():
    frame = np.full((120, 180, 3), 127, dtype=np.uint8)
    output = draw_status_overlay(frame, ["line one", "line two"])
    assert output.shape == frame.shape
    assert output.dtype == np.uint8
    assert not np.array_equal(output, frame)


def test_draw_status_overlay_rejects_non_bgr_input():
    with pytest.raises(ValueError):
        draw_status_overlay(np.zeros((20, 20), dtype=np.uint8), ["bad"])
