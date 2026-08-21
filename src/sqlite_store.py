import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class SQLiteEventStore:
    def __init__(self, db_path: str = "data/ilm.db", schema_path: str = "sql/skill_events.sql"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        schema_candidate = Path(schema_path)
        if schema_candidate.exists():
            self.schema_path = schema_candidate
        else:
            repo_root = Path(__file__).resolve().parent.parent
            self.schema_path = repo_root / schema_path

        self._conn: sqlite3.Connection | None = None
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
        return self._conn

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _init_schema(self):
        schema_sql = self.schema_path.read_text(encoding="utf-8")
        conn = self._connect()
        conn.executescript(schema_sql)
        conn.commit()

    def add_event(self, event_type: str, skill: str, payload: dict):
        row_payload = json.dumps(payload, ensure_ascii=False)
        ts = datetime.now(timezone.utc).isoformat()
        conn = self._connect()
        conn.execute(
            "INSERT INTO skill_events (event_type, skill, payload, created_at_utc) VALUES (?, ?, ?, ?)",
            (event_type, skill, row_payload, ts),
        )
        conn.commit()

    def latest_event(self, skill: str, event_type: str) -> dict | None:
        conn = self._connect()
        cur = conn.execute(
            "SELECT payload, created_at_utc FROM skill_events WHERE skill=? AND event_type=? ORDER BY id DESC LIMIT 1",
            (skill, event_type),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {"payload": json.loads(row[0]), "created_at_utc": row[1]}
