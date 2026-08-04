import pytest

from ravens.tasks.ccda_hidden_friction import NativeSegmentFrictionConfig


def test_native_snapshot():
    config = NativeSegmentFrictionConfig(0.15, 1.2, 0.0, 0.0, 0.0)
    value = config.snapshot()
    assert value["snapshot_version"] == "ccda_native_segment_friction_v1"
    assert value["config"]["hidden_lateral_friction"] == 1.2


def test_native_hidden_must_exceed_base():
    with pytest.raises(ValueError):
        NativeSegmentFrictionConfig(0.5, 0.5, 0.0, 0.0, 0.0)


def test_native_rejects_negative():
    with pytest.raises(ValueError):
        NativeSegmentFrictionConfig(-0.1, 1.0, 0.0, 0.0, 0.0)
