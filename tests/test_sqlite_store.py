import gc
import tempfile
import unittest
from pathlib import Path

from src.sqlite_store import SQLiteEventStore


class SQLiteStoreTests(unittest.TestCase):
    def test_add_and_read_latest(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "ilm.db"
            schema = Path(td) / "schema.sql"
            schema.write_text(
                "CREATE TABLE IF NOT EXISTS skill_events (id INTEGER PRIMARY KEY AUTOINCREMENT, event_type TEXT, skill TEXT, payload TEXT, created_at_utc TEXT);",
                encoding="utf-8",
            )
            store = SQLiteEventStore(db_path=str(db), schema_path=str(schema))
            store.add_event("skill_incident", "social_media", {"failure": "missing_tool"})
            latest = store.latest_event("social_media", "skill_incident")
            self.assertIsNotNone(latest)
            self.assertEqual(latest["payload"]["failure"], "missing_tool")
            del store
            gc.collect()


if __name__ == "__main__":
    unittest.main()
