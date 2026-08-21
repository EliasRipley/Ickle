import os
import unittest
import uuid
from pathlib import Path

from src.model_maintain import run_model_maintenance, run_data_maintenance


class ModelMaintainTests(unittest.TestCase):
    @staticmethod
    def _tmp_root() -> Path:
        root = Path("data") / ".tmp_tests" / f"model_maintain_{uuid.uuid4().hex}"
        (root / "models").mkdir(parents=True, exist_ok=True)
        return root

    def test_archives_older_models_and_prunes_checkpoints(self):
        root = self._tmp_root()
        models = root / "models"
        active = models / "active.pt"
        old = models / "old.pt"
        older = models / "older.pt"
        for p, content in [(active, b"a"), (old, b"b"), (older, b"c")]:
            p.write_bytes(content)
            (Path(str(p) + ".meta.json")).write_text("{\"steps\": 1}", encoding="utf-8")

        ck1 = models / "active.pt.checkpoint.pt"
        ck2 = models / "old.pt.checkpoint.pt"
        ck3 = models / "older.pt.checkpoint.pt"
        for p, content in [(ck1, b"x"), (ck2, b"y"), (ck3, b"z")]:
            p.write_bytes(content)

        now = os.path.getmtime(active)
        os.utime(old, (now - 5000, now - 5000))
        os.utime(older, (now - 9000, now - 9000))
        os.utime(ck2, (now - 5000, now - 5000))
        os.utime(ck3, (now - 9000, now - 9000))

        report = run_model_maintenance(
            models_root=str(models),
            archive_dir=str(models / "archive"),
            keep_recent=0,
            keep_names_csv="active.pt",
            checkpoint_keep_recent=1,
            checkpoint_ttl_days=0.0,
            apply=True,
        )

        self.assertTrue(active.exists())
        self.assertFalse(old.exists())
        self.assertFalse(older.exists())
        self.assertTrue((models / "archive" / "old.pt.gz").exists())
        self.assertTrue((models / "archive" / "older.pt.gz").exists())
        self.assertTrue((models / "archive" / "old.pt.meta.json.gz").exists())
        self.assertTrue((models / "archive" / "older.pt.meta.json.gz").exists())

        # Keep newest checkpoint, prune stale ones
        self.assertTrue(ck1.exists())
        self.assertFalse(ck2.exists())
        self.assertFalse(ck3.exists())

        self.assertIn("status_counts", report)
        self.assertGreaterEqual(report["status_counts"].get("done", 0), 2)

    def test_candidates_subfolder_is_swept_by_default(self):
        # Regression test: run_model_maintenance used to only glob the
        # top-level models_root, so anything in models/candidates/ (where
        # fresh training output actually lands) was invisible to cleanup.
        root = self._tmp_root()
        models = root / "models"
        candidates = models / "candidates"
        candidates.mkdir(parents=True, exist_ok=True)

        active = models / "active.pt"
        active.write_bytes(b"a")

        cand_new = candidates / "cand_new.pt"
        cand_old = candidates / "cand_old.pt"
        cand_new.write_bytes(b"n")
        cand_old.write_bytes(b"o")
        now = os.path.getmtime(cand_new)
        os.utime(cand_old, (now - 9000, now - 9000))

        report = run_model_maintenance(
            models_root=str(models),
            archive_dir=str(models / "archive"),
            keep_recent=1,
            keep_names_csv="active.pt",
            checkpoint_keep_recent=0,
            checkpoint_ttl_days=0.0,
            candidates_keep_recent=1,
            apply=True,
        )

        self.assertEqual(report.get("candidates_root"), str(candidates.resolve()))
        self.assertTrue(active.exists())
        self.assertTrue(cand_new.exists(), "most recent candidate should be kept")
        self.assertFalse(cand_old.exists(), "stale candidate should be archived away")
        self.assertTrue((models / "archive" / "candidates" / "cand_old.pt.gz").exists())

    def test_candidates_can_be_disabled(self):
        root = self._tmp_root()
        models = root / "models"
        candidates = models / "candidates"
        candidates.mkdir(parents=True, exist_ok=True)
        stale = candidates / "stale.pt"
        stale.write_bytes(b"s")
        now = os.path.getmtime(stale)
        os.utime(stale, (now - 9000, now - 9000))

        report = run_model_maintenance(
            models_root=str(models),
            archive_dir=str(models / "archive"),
            include_candidates=False,
            apply=True,
        )

        self.assertIsNone(report.get("candidates_root"))
        self.assertTrue(stale.exists(), "candidates dir should be untouched when disabled")

    def test_missing_candidates_dir_is_a_no_op(self):
        root = self._tmp_root()
        models = root / "models"
        (models / "active.pt").write_bytes(b"a")

        report = run_model_maintenance(
            models_root=str(models),
            archive_dir=str(models / "archive"),
            apply=False,
        )
        self.assertIsNone(report.get("candidates_root"))

    def test_run_data_maintenance_does_not_crash(self):
        # Regression test: run_data_maintenance called asdict() without
        # importing it, so it raised NameError the moment it ran for real.
        root = self._tmp_root()
        report = run_data_maintenance(
            data_root=str(root),
            continual_dir=str(root / "continual"),
            tasks_dir=str(root / "tasks"),
            runtime_dir=str(root / "runtime"),
            maintenance_dir=str(root / "maintenance"),
            apply=False,
        )
        self.assertIn("status_counts", report)


if __name__ == "__main__":
    unittest.main()

