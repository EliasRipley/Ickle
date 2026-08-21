import unittest

from src.reality_check import collect_checks


class RealityCheckTests(unittest.TestCase):
    def test_collect_contains_core_rows(self):
        checks = collect_checks()
        areas = {c.area: c for c in checks}
        self.assertIn("core_model", areas)
        self.assertIn("autonomous_learning", areas)
        self.assertEqual(areas["autonomous_learning"].status, "prototype")


if __name__ == "__main__":
    unittest.main()
