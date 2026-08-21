"""Shared feedback storage: rate a (prompt, response) pair for later use as
training data. Used by both src/hub.py (REPL `/feedback` command) and
src/serve_web.py (the web app's inline message rating), and consumed by
src/build_feedback_corpus.py and src/build_preference_pairs.py.

Kept separate from hub.py so serve_web.py (a server process) doesn't have to
import hub.py's module-level `import keyboard` and global F12-hotkey
registration, which is desktop-REPL-only behavior."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_FEEDBACK_PATH = "data/hub_feedback.jsonl"


@dataclass
class HubFeedback:
    prompt: str
    response: str
    rating: int
    notes: str
    created_at: str


def record_feedback(
    prompt: str,
    response: str,
    rating: int,
    notes: str = "",
    path: str = DEFAULT_FEEDBACK_PATH,
) -> HubFeedback:
    """Append one rated (prompt, response) pair to the feedback JSONL file.

    `rating` is clamped to 1-5, matching build_feedback_corpus.py's/
    build_preference_pairs.py's expected schema (both default to keeping
    rows rated >= 4).
    """
    fb = HubFeedback(
        prompt=str(prompt or "").strip(),
        response=str(response or "").strip(),
        rating=max(1, min(5, int(rating))),
        notes=str(notes or "").strip(),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    feedback_path = Path(path)
    feedback_path.parent.mkdir(parents=True, exist_ok=True)
    with feedback_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(fb), ensure_ascii=False) + "\n")
    return fb
