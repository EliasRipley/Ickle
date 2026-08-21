import tempfile
import unittest
from pathlib import Path

from src.scoped_knowledge import ScopedKnowledgeManager


class RemoveDeltaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.mgr = ScopedKnowledgeManager(data_dir=self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_remove_delta_deletes_registry_entry_and_version_directory(self):
        self.mgr.register_delta(
            delta_id="junk_baseline",
            domain_description="How AP reported in all formats from tornado-stricken regions",
            description="Model trained for 2999 steps on streaming English data",
        )
        self.mgr.registry.save_version("junk_baseline")
        version_dir = Path(self.tmp.name) / "junk_baseline"
        self.assertTrue(version_dir.exists())
        self.assertEqual(len(self.mgr.list_deltas()), 1)

        ok = self.mgr.remove_delta("junk_baseline")

        self.assertTrue(ok)
        self.assertEqual(self.mgr.list_deltas(), [])
        self.assertIsNone(self.mgr.registry.get("junk_baseline"))
        self.assertFalse(version_dir.exists())

    def test_remove_delta_on_unknown_id_returns_false(self):
        self.assertFalse(self.mgr.remove_delta("does_not_exist"))

    def test_remove_delta_also_clears_it_from_the_active_router(self):
        self.mgr.register_delta(delta_id="active_one", domain_description="topic")
        self.assertIn("active_one", self.mgr.router._deltas)

        self.mgr.remove_delta("active_one")

        self.assertNotIn("active_one", self.mgr.router._deltas)


if __name__ == "__main__":
    unittest.main()
