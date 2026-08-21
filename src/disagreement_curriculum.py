"""Turns disagreement between independently-trained peers into a curriculum.

Ickle's swarm deliberation (`src.epistemics.build_collective_view`, used by
both the live "ask the swarm too" chat path and periodic co-distillation
rounds) already detects when peers running different models on different
private data give conflicting answers. Today that detection is thrown away
the moment the HTTP response or the round's report file is overwritten --
it changes what one answer looks like and nothing else.

That is a real asset a single centrally-trained model cannot reproduce: a
model can be asked to doubt itself, but its blind spots are correlated with
itself, which is exactly why research on self-reported confidence finds it
unreliable (see docs/EPISTEMIC_COMMONS.md's citations). Independently
trained peers, on different data, are not correlated with each other in the
same way -- their disagreement is a much stronger uncertainty signal, and
Ickle already has the machinery to collect it.

This module gives that signal somewhere durable to live:

1. `record_conflicts` accumulates polarity-conflicting claim clusters from
   both the live-ask path and codistill rounds into one deduplicated,
   evidence-weighted queue (`data/commons/disagreements.json`).
2. `disagreement_queue`/`disagreement_status` rank that queue by how many
   independent peers actually diverge -- the network's own ranked list of
   what it is least sure about, for a human to look at, instead of a human
   guessing what to correct next.
3. `build_hedge_corpus_file` turns *currently unresolved* entries into
   training pairs that teach the model to hedge specifically on those
   claims ("independent peers disagree, treat this as unsettled") rather
   than confidently pick a side its own peers don't agree on -- folded into
   `continual_guard_step` the same way `verified_corrections.py` folds in
   resolved corrections.

The two corpora are deliberately mutually exclusive: the moment a claim is
resolved locally (the owner adopts/corrects it via the Epistemic Commons,
using the *same* claim text so the ids line up), it drops out of the hedge
queue here and becomes a confident correction there instead. A claim is
never trained to be both hedged and asserted at once.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from src.epistemics import stable_claim_id

DEFAULT_DISAGREEMENTS_PATH = "data/commons/disagreements.json"
DEFAULT_HEDGE_CORPUS_PATH = "data/continual/disagreement_hedges.txt"
MIN_PEER_COUNT_FOR_HEDGE = 2


def _load(path: str) -> dict[str, dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _save(path: str, data: dict[str, dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def record_conflicts(
    conflicts: list[dict[str, Any]],
    *,
    source: str,
    path: str = DEFAULT_DISAGREEMENTS_PATH,
) -> dict[str, Any]:
    """Upserts polarity-conflicting claim clusters (the `possible_conflicts`
    list from `build_collective_view`) into the durable queue, merging peer
    ids and variant wordings when the same disagreement is observed again --
    by a live chat ask, a scheduled co-distillation round, or both."""
    if not conflicts:
        return {"path": path, "new": 0, "updated": 0, "total": len(_load(path))}
    store = _load(path)
    now = time.time()
    new_count = 0
    updated_count = 0
    for cluster in conflicts:
        representative = str(cluster.get("representative", "")).strip()
        if not representative:
            continue
        claim_id = str(cluster.get("cluster_id") or stable_claim_id(representative))
        peer_ids = sorted({str(p).strip() for p in list(cluster.get("peer_ids", [])) if str(p).strip()})
        variants = [str(v).strip() for v in list(cluster.get("variants", [])) if str(v).strip()][:6]

        entry = store.get(claim_id)
        if entry is None:
            store[claim_id] = {
                "claim_id": claim_id,
                "representative": representative,
                "variants": variants,
                "peer_ids": peer_ids,
                "peer_count": len(peer_ids),
                "first_seen": now,
                "last_seen": now,
                "times_observed": 1,
                "sources": [source],
            }
            new_count += 1
            continue

        merged_peers = sorted(set(entry.get("peer_ids", [])) | set(peer_ids))
        merged_variants = list(dict.fromkeys(list(entry.get("variants", [])) + variants))[:8]
        merged_sources = sorted(set(entry.get("sources", [])) | {source})
        entry["representative"] = representative or entry.get("representative", "")
        entry["variants"] = merged_variants
        entry["peer_ids"] = merged_peers
        entry["peer_count"] = len(merged_peers)
        entry["last_seen"] = now
        entry["times_observed"] = int(entry.get("times_observed", 0)) + 1
        entry["sources"] = merged_sources
        updated_count += 1

    _save(path, store)
    return {"path": path, "new": new_count, "updated": updated_count, "total": len(store)}


def _resolved_claim_ids(*, ledger: Any = None) -> set[str]:
    """Claim ids the owner has already resolved locally (correct/adopt), so
    a disagreement drops out of the open queue the moment its exact claim
    text is corrected -- no second 'resolved' flag to keep in sync by hand."""
    if ledger is None:
        from src.federated.knowledge_commons import EpistemicLedger

        ledger = EpistemicLedger()
    resolved: set[str] = set()
    for event in ledger.list_events(limit=2000, active_only=True):
        if event.relation in {"correct", "adopt"} and event.author_peer_id == ledger.identity.peer_id:
            resolved.add(event.claim_id)
    return resolved


def disagreement_queue(
    *,
    path: str = DEFAULT_DISAGREEMENTS_PATH,
    limit: int = 20,
    min_peer_count: int = MIN_PEER_COUNT_FOR_HEDGE,
    ledger: Any = None,
) -> list[dict[str, Any]]:
    """Unresolved disagreements ranked by how many independent peers
    actually diverge, then how often the disagreement has recurred: the
    network's own ranked list of what it is least sure about."""
    store = _load(path)
    resolved = _resolved_claim_ids(ledger=ledger)
    rows = [
        entry
        for claim_id, entry in store.items()
        if claim_id not in resolved and int(entry.get("peer_count", 0)) >= max(1, int(min_peer_count))
    ]
    rows.sort(
        key=lambda e: (int(e.get("peer_count", 0)), int(e.get("times_observed", 0)), float(e.get("last_seen", 0))),
        reverse=True,
    )
    return rows[: max(1, int(limit))]


def disagreement_status(
    *,
    path: str = DEFAULT_DISAGREEMENTS_PATH,
    min_peer_count: int = MIN_PEER_COUNT_FOR_HEDGE,
    ledger: Any = None,
) -> dict[str, Any]:
    store = _load(path)
    resolved = _resolved_claim_ids(ledger=ledger)
    open_ids = {
        claim_id
        for claim_id, entry in store.items()
        if claim_id not in resolved and int(entry.get("peer_count", 0)) >= max(1, int(min_peer_count))
    }
    return {
        "total_tracked": len(store),
        "open_count": len(open_ids),
        "resolved_count": len(store) - len(open_ids),
        "top": disagreement_queue(path=path, limit=5, min_peer_count=min_peer_count, ledger=ledger),
    }


def build_hedge_corpus_file(
    *,
    out_path: str = DEFAULT_HEDGE_CORPUS_PATH,
    disagreements_path: str = DEFAULT_DISAGREEMENTS_PATH,
    oversample: int = 2,
    max_pairs: int = 300,
    min_peer_count: int = MIN_PEER_COUNT_FOR_HEDGE,
    ledger: Any = None,
) -> dict[str, Any]:
    """Writes currently-unresolved disagreements as hedge-training pairs.

    Deliberately lower oversample than `verified_corrections.py`'s default
    (2x vs 3x): a wrong hedge (hedging on something actually settled) is a
    much smaller mistake than a wrong confident assertion, so it doesn't
    need as much weight to compete with everyday conversation pairs -- and
    over-representing "I'm not sure" risks teaching a generically evasive
    model, which is exactly the failure mode `continual_guard`'s own
    evasiveness checks already watch for.
    """
    entries = disagreement_queue(path=disagreements_path, limit=1000, min_peer_count=min_peer_count, ledger=ledger)
    oversample = max(1, int(oversample))
    max_pairs = max(0, int(max_pairs))

    lines: list[str] = []
    written = 0
    for entry in entries:
        representative = str(entry.get("representative", "")).strip()
        if not representative:
            continue
        variants = [v for v in list(entry.get("variants", [])) if v and v != representative][:2]
        peer_count = int(entry.get("peer_count", 0))
        variant_note = f' Some peers describe it differently: {"; ".join(variants)}.' if variants else ""
        user = f'Is this accurate: "{representative}"?'
        assistant = (
            f"I'm not confident here -- {peer_count} independent peers disagree on this, "
            f"so I shouldn't state it as settled.{variant_note} Treat it as an open question "
            "rather than a confirmed fact until a human resolves it."
        )
        for _ in range(oversample):
            if max_pairs and written >= max_pairs:
                break
            lines.append(f"User: {user}")
            lines.append(f"Ickle: {assistant}")
            lines.append("")
            written += 1
        if max_pairs and written >= max_pairs:
            break

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    return {
        "out_path": str(out),
        "distinct_disagreements": len(entries),
        "written_pairs": written,
        "oversample": oversample,
        "disagreements_path": disagreements_path,
    }
