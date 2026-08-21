"""Inspectable epistemic structure for answers and swarm deliberation.

Language models normally collapse knowledge, uncertainty, and opinion into one
fluent string.  This module builds a deliberately modest *answer map* around
that string: candidate claims, related retrieved evidence, human review, and
areas where peers agree or differ.  The map never calls textual overlap
"truth" and never asks the model to grade its own confidence.

Everything here is deterministic and model-independent.  That matters for a
small local model: the transparency layer must still work when the model is too
small to reliably emit a complicated JSON schema of its own.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any, Protocol

from src.evidence_policy import content_tokens, jaccard_similarity, topic_relevance


MAX_ANSWER_CLAIMS = 10
_NEGATIONS = {"not", "never", "no", "none", "cannot", "can't", "isn't", "aren't", "won't", "without"}
_NON_CLAIM_PREFIXES = (
    "hi ",
    "hello ",
    "thanks",
    "thank you",
    "let me ",
    "i can ",
    "i cannot ",
    "i can't ",
    "would you ",
)
_ADVICE_MARKERS = (
    " should ",
    " recommend ",
    " consider ",
    " try ",
    " could ",
    " might want ",
)


class ReviewLookup(Protocol):
    def reviews_for_claim(self, claim_text: str, claim_id: str = "", limit: int = 40) -> list[dict[str, Any]]:
        ...


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_claim(text: str) -> str:
    value = re.sub(r"[`*_>#]", " ", str(text or ""))
    value = re.sub(r"\s+", " ", value).strip(" \t\r\n-:;,.!?")
    return value


def stable_claim_id(text: str) -> str:
    normalized = normalize_claim(text).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]


def _without_code_blocks(text: str) -> str:
    return re.sub(r"```.*?```", " ", str(text or ""), flags=re.DOTALL)


def _split_candidate_claims(text: str) -> list[str]:
    """Split prose conservatively without pretending decomposition is perfect.

    List items are useful boundaries.  A semicolon is only treated as a
    boundary when both sides are long enough to stand on their own.  The UI
    calls these "candidate claims" for exactly this reason.
    """

    cleaned = _without_code_blocks(text).replace("\r", "\n")
    chunks: list[str] = []
    for raw_line in cleaned.split("\n"):
        line = re.sub(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", "", raw_line).strip()
        if not line:
            continue
        if line.startswith("#") or (line.endswith(":") and len(line.split()) <= 8):
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", line):
            sentence = sentence.strip()
            if ";" in sentence:
                parts = [part.strip() for part in sentence.split(";")]
                if all(len(part.split()) >= 5 for part in parts):
                    chunks.extend(parts)
                    continue
            chunks.append(sentence)
    return chunks


def _claim_kind(text: str) -> str:
    folded = f" {text.casefold()} "
    if any(marker in folded for marker in _ADVICE_MARKERS):
        return "advice"
    if re.search(r"\b(?:may|might|could|likely|possibly|probably)\b", folded):
        return "qualified"
    return "statement"


def extract_candidate_claims(text: str, max_claims: int = MAX_ANSWER_CLAIMS) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in _split_candidate_claims(text):
        claim = normalize_claim(candidate)
        folded = claim.casefold()
        words = re.findall(r"[\w']+", claim, flags=re.UNICODE)
        if len(words) < 4 or len(claim) < 18 or len(claim) > 420:
            continue
        if claim.endswith("?") or any(folded.startswith(prefix) for prefix in _NON_CLAIM_PREFIXES):
            continue
        # A bare navigation/meta sentence is not useful to inspect.
        if folded.startswith(("here are ", "the following ", "in summary ", "this answer ")):
            continue
        claim_key = claim.casefold()
        if claim_key in seen:
            continue
        seen.add(claim_key)
        claims.append(
            {
                "claim_id": stable_claim_id(claim),
                "text": claim,
                "kind": _claim_kind(claim),
            }
        )
        if len(claims) >= max(1, int(max_claims)):
            break
    return claims


def _clean_evidence_items(evidence_items: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for raw in list(evidence_items or [])[:80]:
        if not isinstance(raw, dict):
            continue
        evidence_claim = normalize_claim(str(raw.get("claim", "")))
        url = str(raw.get("source_url", raw.get("url", ""))).strip()
        title = str(raw.get("source_title", raw.get("title", ""))).strip()
        if not evidence_claim:
            continue
        key = (evidence_claim.casefold(), url)
        if key in seen:
            continue
        seen.add(key)
        try:
            score = max(0.0, min(1.0, float(raw.get("score", 0.0))))
        except (TypeError, ValueError):
            score = 0.0
        cleaned.append(
            {
                "claim": evidence_claim,
                "source_url": url,
                "source_title": title or url,
                "score": round(score, 4),
            }
        )
    return cleaned


def _evidence_matches(claim: str, evidence_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    claim_tokens = content_tokens(claim)
    for evidence in evidence_items:
        evidence_claim = str(evidence.get("claim", ""))
        overlap = jaccard_similarity(claim, evidence_claim)
        coverage = topic_relevance(claim, evidence_claim)
        reverse_coverage = topic_relevance(evidence_claim, claim)
        # Require at least two substantive shared tokens as well as a useful
        # ratio.  A source being "related" is intentionally weaker than it
        # entailing or proving the generated claim.
        shared = len(claim_tokens.intersection(content_tokens(evidence_claim)))
        relatedness = max(overlap, min(coverage, reverse_coverage))
        if shared < 2 or relatedness < 0.16:
            continue
        row = dict(evidence)
        row["relatedness"] = round(relatedness, 4)
        matches.append(row)
    matches.sort(key=lambda row: (float(row.get("relatedness", 0.0)), float(row.get("score", 0.0))), reverse=True)
    return matches[:3]


def _review_summary(reviews: list[dict[str, Any]]) -> dict[str, Any]:
    active = [row for row in reviews if str(row.get("relation", "")) != "retract"]
    local = [row for row in active if bool(row.get("is_local", False))]
    shared = [row for row in active if not bool(row.get("is_local", False))]
    corrections = [row for row in active if str(row.get("relation", "")) in {"correct", "adopt"}]
    disputes = [row for row in active if str(row.get("relation", "")) == "dispute"]
    supports = [row for row in active if str(row.get("relation", "")) == "support"]
    return {
        "local_reviews": len(local),
        "peer_reviews": len(shared),
        "supports": len(supports),
        "local_supports": sum(1 for row in local if str(row.get("relation", "")) == "support"),
        "local_disputes": sum(1 for row in local if str(row.get("relation", "")) == "dispute"),
        "local_corrections": sum(1 for row in local if str(row.get("relation", "")) in {"correct", "adopt"}),
        "peer_positions": len(shared),
        "disputes": len(disputes),
        "corrections": [
            {
                "text": str(row.get("correction_text", "")).strip(),
                "source_url": str(row.get("source_url", "")).strip(),
                "is_local": bool(row.get("is_local", False)),
            }
            for row in corrections
            if str(row.get("correction_text", "")).strip()
        ][:4],
    }


def build_answer_map(
    *,
    prompt: str,
    response: str,
    evidence_items: list[dict[str, Any]] | None = None,
    review_lookup: ReviewLookup | None = None,
    low_confidence: bool = False,
) -> dict[str, Any]:
    """Build the JSON-serializable epistemic passport shown with an answer."""

    cleaned_evidence = _clean_evidence_items(evidence_items)
    claims = extract_candidate_claims(response)
    counts = {"source_linked": 0, "human_reviewed": 0, "contested": 0, "open": 0, "advice": 0}
    mapped: list[dict[str, Any]] = []
    for claim in claims:
        claim_text = str(claim["text"])
        linked = _evidence_matches(claim_text, cleaned_evidence)
        reviews: list[dict[str, Any]] = []
        if review_lookup is not None:
            try:
                reviews = review_lookup.reviews_for_claim(claim_text, str(claim["claim_id"]), limit=40)
            except Exception:  # The answer must survive a damaged optional ledger.
                reviews = []
        review = _review_summary(reviews)

        if review["local_corrections"]:
            status = "corrected"
            counts["human_reviewed"] += 1
        elif review["local_disputes"]:
            status = "contested"
            counts["contested"] += 1
        elif review["local_supports"]:
            status = "human_reviewed"
            counts["human_reviewed"] += 1
        elif review["peer_positions"]:
            status = "peer_perspective"
            counts["contested"] += 1
        elif linked:
            status = "source_linked"
            counts["source_linked"] += 1
        elif claim["kind"] == "advice":
            status = "advice"
            counts["advice"] += 1
        else:
            status = "open"
            counts["open"] += 1

        basis: list[str] = []
        if linked:
            source_count = len({row.get("source_url") or row.get("source_title") for row in linked})
            plural = "s" if source_count != 1 else ""
            basis.append(
                f"Related text was retrieved from {source_count} source{plural}; "
                "this is not an entailment check."
            )
        if review["local_reviews"]:
            review_plural = "s" if review["local_reviews"] != 1 else ""
            basis.append(f"Reviewed {review['local_reviews']} time{review_plural} on this device.")
        if review["peer_reviews"]:
            peer_plural = "s" if review["peer_reviews"] != 1 else ""
            basis.append(
                f"{review['peer_reviews']} signed peer perspective{peer_plural}; "
                "peer agreement is not treated as proof."
            )
        if not basis:
            basis.append("No retrieved source or human review is linked to this candidate claim yet.")

        mapped.append(
            {
                **claim,
                "status": status,
                "basis": basis,
                "sources": linked,
                "reviews": review,
            }
        )

    answer_hash = hashlib.sha256(str(response or "").encode("utf-8")).hexdigest()[:24]
    return {
        "version": 1,
        "answer_hash": answer_hash,
        "generated_at": _utc_now(),
        "prompt_claim_count": len(content_tokens(prompt)),
        "claims": mapped,
        "counts": counts,
        "low_confidence_signal": bool(low_confidence),
        "method": "deterministic claim candidates + retrieved-evidence links + signed human reviews",
        "caveat": "This map exposes what an answer rests on. It does not certify that a claim is true.",
    }


def _claim_polarity(text: str) -> str:
    tokens = {token.casefold() for token in re.findall(r"[A-Za-z']+", str(text or ""))}
    return "negative" if tokens.intersection(_NEGATIONS) else "positive"


def _cluster_key_tokens(text: str) -> set[str]:
    return {token for token in content_tokens(text) if token not in _NEGATIONS}


def build_collective_view(responses: list[dict[str, Any]]) -> dict[str, Any]:
    """Preserve peer common ground and differences at candidate-claim level.

    This is descriptive, not a vote.  Repeated claims are called common
    ground, unique claims remain visible, and simple polarity clashes are
    flagged without declaring a winner.
    """

    clusters: list[dict[str, Any]] = []
    for response in responses:
        peer_id = str(response.get("peer_id", response.get("teacher_peer_id", ""))).strip()
        text = str(response.get("response", "")).strip()
        for claim in extract_candidate_claims(text, max_claims=8):
            tokens = _cluster_key_tokens(str(claim["text"]))
            if not tokens:
                continue
            chosen: dict[str, Any] | None = None
            best_overlap = 0.0
            for cluster in clusters:
                representative_tokens = set(cluster["_tokens"])
                overlap = len(tokens.intersection(representative_tokens)) / max(1, len(tokens.union(representative_tokens)))
                if overlap >= 0.34 and overlap > best_overlap:
                    chosen = cluster
                    best_overlap = overlap
            if chosen is None:
                chosen = {
                    "cluster_id": stable_claim_id(str(claim["text"])),
                    "representative": str(claim["text"]),
                    "variants": [],
                    "peer_ids": [],
                    "polarities": [],
                    "_tokens": sorted(tokens),
                }
                clusters.append(chosen)
            if peer_id and peer_id not in chosen["peer_ids"]:
                chosen["peer_ids"].append(peer_id)
            variant = str(claim["text"])
            if variant not in chosen["variants"]:
                chosen["variants"].append(variant)
            chosen["polarities"].append(_claim_polarity(variant))

    for cluster in clusters:
        polarities = set(cluster.pop("polarities", []))
        cluster.pop("_tokens", None)
        cluster["peer_count"] = len(cluster["peer_ids"])
        cluster["polarity_conflict"] = len(polarities) > 1
        cluster["variants"] = cluster["variants"][:6]
    clusters.sort(key=lambda row: (int(row["peer_count"]), len(str(row["representative"]))), reverse=True)

    common = [row for row in clusters if int(row["peer_count"]) >= 2]
    distinct = [row for row in clusters if int(row["peer_count"]) == 1]
    conflicts = [row for row in clusters if bool(row["polarity_conflict"])]
    return {
        "method": "claim-level descriptive clustering",
        "common_ground": common,
        "distinct_contributions": distinct,
        "possible_conflicts": conflicts,
        "summary": {
            "peers": len({str(row.get("peer_id", "")) for row in responses if str(row.get("peer_id", ""))}),
            "common_claims": len(common),
            "distinct_claims": len(distinct),
            "possible_conflicts": len(conflicts),
        },
        "caveat": "Agreement measures repeated wording or meaning, not correctness. Minority claims are retained.",
    }
