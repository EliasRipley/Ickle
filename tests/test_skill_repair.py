import gc
import os
import tempfile
import unittest
from pathlib import Path

from src.skill_repair import SkillRepairManager


class SkillRepairTests(unittest.TestCase):
    def test_incident_and_plan(self):
        with tempfile.TemporaryDirectory() as td:
            mgr = SkillRepairManager(root=td)
            mgr.record_incident("social_media", "missing_tool", "No posting tool")
            latest = mgr.latest_for_skill("social_media")
            self.assertIsNotNone(latest)
            plan = mgr.plan(latest.failure_type)
            self.assertGreaterEqual(len(plan), 1)
            self.assertIn("Create a user tool plugin scaffold", plan[0].step)
            del mgr
            gc.collect()

    def test_scaffold_tool(self):
        with tempfile.TemporaryDirectory() as td:
            cwd = Path.cwd()
            try:
                os.chdir(td)
                mgr = SkillRepairManager(root="data")
                msg = mgr.scaffold_tool("social_post", "post to platform")
                self.assertIn("Scaffold created", msg)
                self.assertTrue(Path("user_tools/social_post.py").exists())
                del mgr
                gc.collect()
            finally:
                os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
