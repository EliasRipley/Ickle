import os
import unittest
import uuid
from pathlib import Path

from src.training_maintain import run_training_maintenance


class TrainingMaintainTests(unittest.TestCase):
    @staticmethod
    def _tmp_root() -> Path:
        root = Path("data") / ".tmp_tests" / f"training_maintain_{uuid.uuid4().hex}"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def test_archives_old_training_file_and_compacts_queue(self):
        root = self._tmp_root()
        protected = root / "comprehensive_english.txt"
        protected.write_text("protected\n", encoding="utf-8")

        old_file = root / "scratch_notes.txt"
        old_file.write_text("x" * 200, encoding="utf-8")
        now = os.path.getmtime(old_file)
        os.utime(old_file, (now - 90000, now - 90000))

        queue = root / "queued_wikipedia_learning.txt"
        queue.write_text(
            "\n".join(
                [
                    "User: A",
                    "Ickle: B",
                    "",
                    "User: A",
                    "Ickle: B",
                    "",
                    "User: C",
                    "Ickle: D",
                    "",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        report = run_training_maintenance(
            training_root=str(root),
            archive_dir=str(root / "archive"),
            min_age_days=0.0,
            min_size_bytes=10,
            max_queue_lines=4,
            apply=True,
        )

        self.assertTrue(protected.exists())
        self.assertFalse(old_file.exists())
        self.assertTrue((root / "archive" / "files" / "scratch_notes.txt.gz").exists())

        compacted = queue.read_text(encoding="utf-8").splitlines()
        self.assertLessEqual(len(compacted), 4)
        self.assertIn("queue_stats", report)
        self.assertGreater(report["queue_stats"]["before_lines"], report["queue_stats"]["after_lines"])


if __name__ == "__main__":
    unittest.main()

