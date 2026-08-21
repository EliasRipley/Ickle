import unittest

from src.reality_check import collect_checks


class RealityCheckTests(unittest.TestCase):
    def test_collect_contains_core_rows(self):
        checks = collect_checks()
        areas = {c.area: c for c in checks}
        self.assertIn("core_model", areas)
        self.assertIn("autonomous_learning", areas)
        self.assertIn("epistemic_commons", areas)
        self.assertIn("public_swarm_discovery", areas)
        self.assertEqual(areas["epistemic_commons"].status, "implemented")
        self.assertEqual(areas["public_swarm_discovery"].status, "implemented")
        self.assertEqual(areas["autonomous_learning"].status, "prototype")


if __name__ == "__main__":
    unittest.main()
