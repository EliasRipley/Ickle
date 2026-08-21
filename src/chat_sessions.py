from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


MAX_SESSIONS = 50
MAX_MESSAGES_PER_SESSION = 500


class ChatSessions:
    def __init__(self, storage_dir: str = "data/sessions"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _session_path(self, session_id: str) -> Path:
        return self.storage_dir / f"{session_id}.json"

    def list_sessions(self) -> list[dict[str, Any]]:
        sessions = []
        for f in sorted(self.storage_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                sessions.append({
                    "id": data.get("id", f.stem),
                    "title": data.get("title", "Untitled"),
                    "created_at": data.get("created_at", ""),
                    "updated_at": data.get("updated_at", ""),
                    "message_count": len(data.get("messages", [])),
                })
            except (json.JSONDecodeError, OSError):
                pass
        return sessions

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        path = self._session_path(session_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def create_session(self, title: str = "") -> dict[str, Any]:
        now = _utc_now()
        session_id = uuid.uuid4().hex[:12]
        session = {
            "id": session_id,
            "title": title or "New chat",
            "created_at": now,
            "updated_at": now,
            "messages": [],
        }
        self._save_session(session)
        self._prune_old_sessions()
        return session

    def add_message(
        self,
        session_id: str,
        role: str,
        text: str,
        thinking: str = "",
        model: str = "",
        epistemics: dict[str, Any] | None = None,
        low_confidence: bool = False,
    ) -> dict[str, Any] | None:
        session = self.get_session(session_id)
        if session is None:
            return None
        msg = {
            "role": role,
            "text": text,
            "thinking": thinking,
            "model": model,
            "at": _utc_now(),
        }
        if role == "assistant":
            msg["lowConfidence"] = bool(low_confidence)
            if isinstance(epistemics, dict):
                msg["epistemics"] = epistemics
        session.setdefault("messages", []).append(msg)
        if len(session["messages"]) > MAX_MESSAGES_PER_SESSION:
            session["messages"][:] = session["messages"][-MAX_MESSAGES_PER_SESSION:]
        if role == "user" and len(session["messages"]) == 1:
            session["title"] = text[:60]
        session["updated_at"] = _utc_now()
        self._save_session(session)
        return msg

    def delete_session(self, session_id: str) -> bool:
        path = self._session_path(session_id)
        if not path.exists():
            return False
        path.unlink()
        return True

    def _save_session(self, session: dict[str, Any]):
        path = self._session_path(session["id"])
        path.write_text(json.dumps(session, indent=2, ensure_ascii=False), encoding="utf-8")

    def _prune_old_sessions(self):
        files = sorted(self.storage_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
        while len(files) > MAX_SESSIONS:
            files.pop(0).unlink()
