"""Turns adopted Epistemic Commons corrections into a real training signal.

Today a human correction (`EpistemicLedger` relation `correct`/`adopt`) only
ever becomes ephemeral prompt context (`EpistemicLedger.context_for_prompt`):
it nudges one future answer and is gone the moment the topic doesn't come up
again. It never reaches `continual_guard.run_guarded_step`, so it carries
*less* durable weight than raw scraped conversation logs, even though it is
the single highest-trust signal in the system -- an explicit, signed, local
human judgement, not a guess. This module closes that gap: it turns the
owner's own adopted corrections into a small oversampled dialog corpus that
`task_actions.run_continual_guard_task` folds into every guarded training
step, so a correction can become part of the model itself, still gated by
the same anti-forgetting promotion checks as everything else.

Only local-authored events are used (own corrections, or peer reviews the
owner explicitly adopted) -- matching `context_for_prompt`'s trust boundary.
A peer's un-adopted review never steers training, exactly as it never steers
a prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.federated.knowledge_commons import DEFAULT_COMMONS_DB, EpistemicLedger

DEFAULT_CORRECTIONS_CORPUS = "data/continual/verified_corrections.txt"
_LOCAL_RELATIONS = {"correct", "adopt"}


@dataclass(frozen=True)
class CorrectionPair:
    user: str
    assistant: str
    source: str


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def collect_verified_corrections(
    *,
    commons_db_path: str = DEFAULT_COMMONS_DB,
    limit: int = 2000,
    ledger: EpistemicLedger | None = None,
) -> list[CorrectionPair]:
    """Local-authored, active correct/adopt events, deduplicated.

    `ledger` allows tests to inject a ledger opened with a specific identity;
    production callers rely on the `commons_db_path` default, which opens
    with the same identity every other commons feature in the app uses.
    """
    ledger = ledger or EpistemicLedger(commons_db_path)
    out: list[CorrectionPair] = []
    seen: set[tuple[str, str]] = set()
    for event in ledger.list_events(limit=max(1, int(limit)), active_only=True):
        if event.relation not in _LOCAL_RELATIONS:
            continue
        if event.author_peer_id != ledger.identity.peer_id:
            continue
        claim = _clean(event.claim_text)
        correction = _clean(event.correction_text)
        if len(claim) < 4 or len(correction) < 4:
            continue
        key = (claim.lower(), correction.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(
            CorrectionPair(
                user=f'Is this accurate: "{claim}"?',
                assistant=correction,
                source=f"commons:{event.event_id}",
            )
        )
    return out


def build_verified_corrections_corpus_file(
    *,
    out_path: str = DEFAULT_CORRECTIONS_CORPUS,
    commons_db_path: str = DEFAULT_COMMONS_DB,
    oversample: int = 3,
    max_pairs: int = 900,
    ledger: EpistemicLedger | None = None,
) -> dict[str, Any]:
    """Writes adopted corrections as a `User:`/`Ickle:` dialog corpus.

    Each distinct correction is repeated `oversample` times: corrections are
    almost always vastly outnumbered by ordinary conversation pairs in the
    corpora `continual_guard` mixes together, so without oversampling they
    would be diluted to near-zero training influence despite being the most
    trustworthy examples available.
    """
    pairs = collect_verified_corrections(commons_db_path=commons_db_path, ledger=ledger)
    oversample = max(1, int(oversample))
    max_pairs = max(0, int(max_pairs))

    expanded: list[CorrectionPair] = []
    for pair in pairs:
        for _ in range(oversample):
            if max_pairs and len(expanded) >= max_pairs:
                break
            expanded.append(pair)
        if max_pairs and len(expanded) >= max_pairs:
            break

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for pair in expanded:
        lines.append(f"User: {pair.user}")
        lines.append(f"Ickle: {pair.assistant}")
        lines.append("")
    out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    return {
        "out_path": str(out),
        "distinct_corrections": len(pairs),
        "written_pairs": len(expanded),
        "oversample": oversample,
        "commons_db_path": str(commons_db_path),
    }


def verified_corrections_status(
    *, commons_db_path: str = DEFAULT_COMMONS_DB, ledger: EpistemicLedger | None = None
) -> dict[str, Any]:
    pairs = collect_verified_corrections(commons_db_path=commons_db_path, ledger=ledger)
    return {
        "eligible_corrections": len(pairs),
        "commons_db_path": str(commons_db_path),
        "sample": [
            {"claim": pair.user, "correction": pair.assistant, "source": pair.source}
            for pair in pairs[:5]
        ],
    }
