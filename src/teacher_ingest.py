"""Trainer Integration Layer â€” Phase 2: Direct Teaching Hook (Teacher -> Ickle).

External models push "teaching" feedback directly into Ickle via a session API.

Flow:
  1. Teacher starts a session  -> POST /api/teach/session/start
  2. For each Q&A pair         -> POST /api/teach/session/<id>/turn
  3. Close session              -> POST /api/teach/session/<id>/close
  4. Build corpora              -> SFT corpus + DPO preference pairs
  5. Wire into task_actions     -> build_teacher_corpus / build_teacher_preferences / train_from_teacher
"""

from __future__ import annotations

import json
import random
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TeacherTurn:
    prompt: str
    ickle_answer: str
    teacher_feedback: str = ""
    improved_answer: str = ""
    score: float = 0.0
    tags: list[str] = field(default_factory=list)
    source_model: str = ""
    turn_index: int = 0


@dataclass
class TeacherSession:
    session_id: str
    topic: str = ""
    source_model: str = ""
    tags: list[str] = field(default_factory=list)
    turns: list[TeacherTurn] = field(default_factory=list)
    opened_at: str = field(default_factory=_utc_now)
    closed_at: str = ""
    status: str = "open"


class TeacherStore:
    """Stores teaching sessions and events. Builds SFT + DPO corpora."""

    def __init__(self, base_dir: str = "data/teacher"):
        self._dir = Path(base_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._events_path = self._dir / "teacher_events.jsonl"
        self._sessions_path = self._dir / "sessions.json"

    def _load_sessions(self) -> dict[str, Any]:
        if self._sessions_path.exists():
            return json.loads(self._sessions_path.read_text(encoding="utf-8"))
        return {}

    def _save_sessions(self, data: dict[str, Any]):
        self._sessions_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def start_session(
        self,
        *,
        topic: str = "",
        source_model: str = "",
        tags: list[str] | None = None,
    ) -> TeacherSession:
        sid = uuid.uuid4().hex[:12]
        session = TeacherSession(
            session_id=sid,
            topic=topic,
            source_model=source_model,
            tags=tags or [],
        )
        sessions = self._load_sessions()
        sessions[sid] = {
            "session_id": sid,
            "topic": topic,
            "source_model": source_model,
            "tags": tags or [],
            "opened_at": session.opened_at,
            "turn_count": 0,
            "status": "open",
        }
        self._save_sessions(sessions)
        return session

    def add_turn(
        self,
        session_id: str,
        *,
        prompt: str,
        ickle_answer: str,
        teacher_feedback: str = "",
        improved_answer: str = "",
        score: float = 0.0,
        tags: list[str] | None = None,
        source_model: str = "",
    ) -> dict[str, Any]:
        sessions = self._load_sessions()
        if session_id not in sessions:
            raise KeyError(f"Session not found: {session_id}")
        if sessions[session_id].get("status") != "open":
            raise ValueError(f"Session {session_id} is already closed")

        turn_index = int(sessions[session_id].get("turn_count", 0))
        turn = TeacherTurn(
            prompt=prompt,
            ickle_answer=ickle_answer,
            teacher_feedback=teacher_feedback,
            improved_answer=improved_answer,
            score=float(score),
            tags=tags or [],
            source_model=source_model,
            turn_index=turn_index,
        )

        event = {
            "session_id": session_id,
            "event_type": "turn",
            "prompt": turn.prompt,
            "ickle_answer": turn.ickle_answer,
            "teacher_feedback": turn.teacher_feedback,
            "improved_answer": turn.improved_answer,
            "score": turn.score,
            "tags": turn.tags,
            "source_model": turn.source_model or sessions[session_id].get("source_model", ""),
            "turn_index": turn_index,
            "timestamp_utc": _utc_now(),
        }

        with self._events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

        sessions[session_id]["turn_count"] = turn_index + 1
        sessions[session_id]["last_turn_utc"] = _utc_now()
        self._save_sessions(sessions)

        return event

    def close_session(self, session_id: str) -> dict[str, Any]:
        sessions = self._load_sessions()
        if session_id not in sessions:
            raise KeyError(f"Session not found: {session_id}")
        sessions[session_id]["status"] = "closed"
        sessions[session_id]["closed_at"] = _utc_now()
        self._save_sessions(sessions)

        turns = self._read_session_turns(session_id)
        return {
            "session_id": session_id,
            "status": "closed",
            "turn_count": len(turns),
            "closed_at": sessions[session_id]["closed_at"],
        }

    def _read_session_turns(self, session_id: str) -> list[dict[str, Any]]:
        if not self._events_path.exists():
            return []
        turns: list[dict[str, Any]] = []
        with self._events_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if ev.get("session_id") == session_id and ev.get("event_type") == "turn":
                    turns.append(ev)
        return sorted(turns, key=lambda e: int(e.get("turn_index", 0)))

    def list_sessions(self) -> list[dict[str, Any]]:
        sessions = self._load_sessions()
        return [{"session_id": k, **v} for k, v in sorted(sessions.items())]

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        sessions = self._load_sessions()
        base = sessions.get(session_id)
        if not base:
            return None
        base["turns"] = self._read_session_turns(session_id)
        return base

    def build_sft_corpus(self, out_path: str = "data/teacher/teacher_sft_corpus.txt", *, min_score: float = 0.0) -> dict[str, Any]:
        """Build an SFT corpus from teaching events (prompt -> improved_answer).

        If min_score > 0, only turns with score >= min_score are included.
        """
        if not self._events_path.exists():
            return {"pairs": 0, "out_path": out_path}

        pairs: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        skipped_low_score = 0
        with self._events_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                improved = str(ev.get("improved_answer", "")).strip()
                prompt = str(ev.get("prompt", "")).strip()
                score = float(ev.get("score", 0.0))
                feedback = str(ev.get("teacher_feedback", "")).strip()
                if not improved or not prompt:
                    continue
                if min_score > 0 and score < min_score:
                    skipped_low_score += 1
                    continue
                key = (prompt.lower(), improved.lower())
                if key in seen:
                    continue
                seen.add(key)
                pairs.append((prompt, improved, feedback))

        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        for prompt, answer, feedback in pairs:
            lines.append(f"User: {prompt}")
            if feedback:
                lines.append(f"Teacher feedback: {feedback}")
            lines.append(f"Ickle: {answer}")
            lines.append("")
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return {"pairs": len(pairs), "out_path": str(out), "skipped_low_score": skipped_low_score}

    def build_preference_pairs(self, out_path: str = "data/teacher/teacher_prefs.jsonl", *, min_score: float = 0.0) -> dict[str, Any]:
        """Build DPO preference pairs: prompt, chosen=improved_answer, rejected=ickle_answer.

        If min_score > 0, only turns with score >= min_score are included.
        """
        if not self._events_path.exists():
            return {"pairs": 0, "out_path": out_path}

        prefs: list[dict[str, str]] = []
        seen: set[str] = set()
        skipped_low_score = 0
        with self._events_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prompt = str(ev.get("prompt", "")).strip()
                chosen = str(ev.get("improved_answer", "")).strip()
                rejected = str(ev.get("ickle_answer", "")).strip()
                score = float(ev.get("score", 0.0))
                if not prompt or not chosen or not rejected or chosen == rejected:
                    continue
                if min_score > 0 and score < min_score:
                    skipped_low_score += 1
                    continue
                key = hashlib.md5(f"{prompt}|{chosen}|{rejected}".encode()).hexdigest()[:16]
                if key in seen:
                    continue
                seen.add(key)
                prefs.append({
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                })

        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for row in prefs:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return {"pairs": len(prefs), "out_path": str(out), "skipped_low_score": skipped_low_score}


import hashlib
