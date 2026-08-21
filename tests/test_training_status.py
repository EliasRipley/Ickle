import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.training_control import inspect_training_status


class TrainingStatusTests(unittest.TestCase):
    def test_old_running_heartbeat_is_reported_as_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "training.json"
            now = datetime(2026, 8, 12, tzinfo=timezone.utc)
            path.write_text(
                json.dumps(
                    {
                        "status": "running",
                        "timestamp_utc": (now - timedelta(hours=1)).isoformat(),
                        "step": 12,
                    }
                ),
                encoding="utf-8",
            )
            status = inspect_training_status(path, stale_after_seconds=180, now=now)
        self.assertEqual(status["status"], "stale")
        self.assertTrue(status["is_stale"])
        self.assertIn("heartbeat", status["stale_reason"])

    def test_completed_status_remains_completed_even_when_old(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "training.json"
            now = datetime(2026, 8, 12, tzinfo=timezone.utc)
            path.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "timestamp_utc": (now - timedelta(days=10)).isoformat(),
                    }
                ),
                encoding="utf-8",
            )
            status = inspect_training_status(path, now=now)
        self.assertEqual(status["status"], "completed")
        self.assertFalse(status["is_stale"])

    def test_missing_status_file_is_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            status = inspect_training_status(Path(tmp) / "missing.json")
        self.assertEqual(status["status"], "unavailable")
        self.assertFalse(status["exists"])


if __name__ == "__main__":
    unittest.main()
