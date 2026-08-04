from __future__ import annotations

import json
import unittest

import numpy as np

from ravens import tasks
from ravens.tasks.ccda_hidden_friction import (
    HiddenFrictionConfig,
    HiddenPlanarFrictionPatch,
)


def make_patch(*, enabled=True, **overrides):
    values = {
        "patch_radius": 0.1,
        "viscous_gain": 2.0,
        "coulomb_force": 0.2,
        "max_force": 0.5,
        "speed_epsilon": 1e-4,
        "contact_height": 0.03,
    }
    values.update(overrides)
    return HiddenPlanarFrictionPatch(
        HiddenFrictionConfig(**values), center_xy=[0.5, 0.0], enabled=enabled
    )


class HiddenPlanarFrictionTest(unittest.TestCase):
    def test_outside_patch_is_zero(self):
        output = make_patch().evaluate([0.8, 0.0, 0.01], [1.0, 0.0, 0.0])
        np.testing.assert_array_equal(output.force_xyz, np.zeros(3))
        self.assertFalse(output.active)

    def test_disabled_is_zero(self):
        output = make_patch(enabled=False).evaluate([0.5, 0.0, 0.01], [1.0, 0.0, 0.0])
        np.testing.assert_array_equal(output.force_xyz, np.zeros(3))

    def test_stationary_is_zero(self):
        output = make_patch().evaluate([0.5, 0.0, 0.01], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(output.force_xyz, np.zeros(3))

    def test_force_opposes_velocity(self):
        velocity = np.array([0.3, -0.4, 0.0])
        output = make_patch().evaluate([0.5, 0.0, 0.01], velocity)
        self.assertTrue(output.active)
        self.assertLessEqual(float(np.dot(output.force_xyz[:2], velocity[:2])), 0.0)

    def test_force_is_capped(self):
        output = make_patch().evaluate([0.5, 0.0, 0.01], [100.0, 0.0, 0.0])
        self.assertLessEqual(output.force_norm, 0.5)
        self.assertAlmostEqual(output.force_norm, 0.5)

    def test_above_contact_height_is_zero(self):
        output = make_patch().evaluate([0.5, 0.0, 0.04], [1.0, 0.0, 0.0])
        np.testing.assert_array_equal(output.force_xyz, np.zeros(3))

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            make_patch(patch_radius=0.0)
        with self.assertRaises(ValueError):
            make_patch(coulomb_force=0.6, max_force=0.5)
        with self.assertRaises(ValueError):
            HiddenPlanarFrictionPatch(make_patch().config, [np.nan, 0.0], True)

    def test_snapshot_is_json_serializable(self):
        json.dumps(make_patch().snapshot(), sort_keys=True)

    def test_task_is_registered(self):
        self.assertIn("ccda-hidden-friction-cable", tasks.names)


if __name__ == "__main__":
    unittest.main()
