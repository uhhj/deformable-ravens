import numpy as np

from ravens.tasks.ccda_hidden_hook_geometry import (
    HiddenHookGeometryConfig,
    compute_hidden_hook_layout,
)


def config():
    return HiddenHookGeometryConfig(
        center_ratio=0.45,
        mouth_offset=0.008,
        width=0.028,
        depth=0.02,
        thickness=0.003,
        height=0.025,
        probe_distance=0.012,
        main_pull_distance=0.08,
        workspace_x=(0.25, 0.75),
        workspace_y=(-0.45, 0.45),
    )


def beads():
    return np.column_stack((
        np.linspace(0.35, 0.65, 25),
        np.zeros(25),
        np.full(25, 0.01),
    ))


def test_layout_is_deterministic_and_has_three_boxes():
    first = compute_hidden_hook_layout(beads(), config())
    second = compute_hidden_hook_layout(beads(), config())
    assert first == second
    assert len(first["boxes"]) == 3


def test_layout_basis_is_orthonormal_and_targets_are_inside_workspace():
    layout = compute_hidden_hook_layout(beads(), config())
    tangent = np.asarray(layout["tangent_xy"])
    normal = np.asarray(layout["normal_xy"])
    assert np.linalg.norm(tangent) == 1.0
    assert np.linalg.norm(normal) == 1.0
    assert np.dot(tangent, normal) == 0.0
    for name in ("probe_target_xy", "main_target_xy"):
        x, y = layout[name]
        assert 0.25 < x < 0.75
        assert -0.45 < y < 0.45
