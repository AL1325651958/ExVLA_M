import unittest

from scripts.sim.env import ExcavatorSim


class ExcavatorSimCompatibilityTests(unittest.TestCase):
    def test_elevation_extent_alias_exposes_map_extent(self):
        sim = object.__new__(ExcavatorSim)
        sim.extent = 20.0

        self.assertEqual(sim.elevation_extent, 20.0)


if __name__ == "__main__":
    unittest.main()
