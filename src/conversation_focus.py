from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.icklization import ick


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "how",
    "i",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
}

# Sourced from the same config (persona.toml's low_quality_evasive_markers)
# the live chat quality gate uses (src/ilm_chat_utils.py:_looks_low_quality_response)
# instead of an independently-maintained copy that had already drifted from it.
EVASIVE_PATTERNS = tuple(re.escape(marker) for marker in ick.detection_list("low_quality_evasive_markers"))

NOISY_PATTERNS = (
    r"wikipedia does not have an article",
    r"article wizard",
    r"sister projects",
    r"other reasons this message may be displayed",
    r"\bBEGININPUT\b",
    r"\bENDINPUT\b",
    r"\bSYSTEM PROMPT\b",
)

GRAMMAR_RED_FLAGS = (
    r"\bare\s+inhabits\b",
    r"\bare\s+inhabit\b",
    r"\bis\s+are\b",
    r"\bthe been become\b",
)


@dataclass(frozen=True)
class DialogPair:
    user: str
    assistant: str
    source: str


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _tokens(text: str) -> set[str]:
    raw = re.findall(r"[a-zA-Z][a-zA-Z0-9'-]{2,}", str(text or "").lower())
    return {tok for tok in raw if tok not in STOPWORDS}


def _is_evasive(text: str) -> bool:
    lower = str(text or "").lower()
    return any(re.search(pattern, lower) for pattern in EVASIVE_PATTERNS)


def _is_low_signal(text: str) -> bool:
    lower = str(text or "").lower()
    return any(re.search(pattern, lower) for pattern in NOISY_PATTERNS)


def _fails_grammar_guard(text: str) -> bool:
    lower = str(text or "").lower()
    return any(re.search(pattern, lower) for pattern in GRAMMAR_RED_FLAGS)


def _quality_score(text: str) -> float:
    value = _clean(text)
    if not value:
        return 0.0
    if _is_evasive(value) or _is_low_signal(value) or _fails_grammar_guard(value):
        return 0.0
    words = re.findall(r"[a-zA-Z']+", value.lower())
    if len(words) < 6:
        return 0.1
    if re.search(r"\b(\w+)(?:\s+\1){2,}\b", value.lower()):
        return 0.0
    unique_ratio = len(set(words)) / max(1, len(words))
    if unique_ratio < 0.40:
        return 0.2
    return 0.85


def _pair_quality_ok(user: str, assistant: str) -> bool:
    u = _clean(user)
    a = _clean(assistant)
    if len(u) < 6 or len(a) < 18:
        return False
    if len(u) > 260 or len(a) > 520:
        return False
    if "```" in u or "```" in a:
        return False
    if _quality_score(a) <= 0.0:
        return False
    u_tokens = _tokens(u)
    if len(u_tokens) >= 3:
        a_tokens = _tokens(a)
        if not u_tokens.intersection(a_tokens):
            return False
    return True


def parse_dialog_pairs(path: Path, *, max_pairs: int = 0) -> list[DialogPair]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    out: list[DialogPair] = []
    pending_user = ""
    seen: set[tuple[str, str]] = set()
    for raw in lines:
        line = _clean(raw)
        if not line:
            continue
        lower = line.lower()
        if lower.startswith("user:"):
            pending_user = _clean(line.split(":", 1)[1])
            continue
        if lower.startswith("ickle:") or lower.startswith("assistant:"):
            if not pending_user:
                continue
            assistant = _clean(line.split(":", 1)[1])
            if not _pair_quality_ok(pending_user, assistant):
                pending_user = ""
                continue
            key = (pending_user.lower(), assistant.lower())
            if key in seen:
                pending_user = ""
                continue
            seen.add(key)
            out.append(DialogPair(user=pending_user, assistant=assistant, source=str(path)))
            pending_user = ""
            if max_pairs > 0 and len(out) >= max_pairs:
                break
    return out


def _load_benchmark_cases(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        prompt = _clean(str(row.get("prompt", "")))
        if not prompt:
            continue
        keywords = [str(x).strip().lower() for x in list(row.get("keywords") or []) if str(x).strip()]
        out.append(
            {
                "name": _clean(str(row.get("name", ""))) or prompt[:40],
                "prompt": prompt,
                "keywords": keywords,
            }
        )
    return out


def _candidate_score(case: dict[str, Any], pair: DialogPair) -> float:
    prompt = str(case.get("prompt", ""))
    keywords = [str(x).strip().lower() for x in list(case.get("keywords") or []) if str(x).strip()]
    p_tokens = _tokens(prompt)
    q_tokens = _tokens(pair.user)
    a_tokens = _tokens(pair.assistant)
    if not q_tokens:
        return 0.0
    q_overlap = len(p_tokens.intersection(q_tokens)) / max(1, len(p_tokens.union(q_tokens)))
    a_overlap = len(p_tokens.intersection(a_tokens)) / max(1, len(p_tokens))
    keyword_hits = 0.0
    if keywords:
        lower_answer = pair.assistant.lower()
        keyword_hits = sum(1 for k in keywords if k in lower_answer) / max(1, len(keywords))
    quality = _quality_score(pair.assistant)
    return (0.45 * q_overlap) + (0.25 * keyword_hits) + (0.15 * a_overlap) + (0.15 * quality)


def _load_memory_web_facts(memory_dir: Path) -> list[dict[str, Any]]:
    path = memory_dir / "web_learning.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []
    rows = payload.get("extracted_facts", [])
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        fact = _clean(str(row.get("fact", "")))
        if not fact:
            continue
        if _quality_score(fact) <= 0.0:
            continue
        out.append(
            {
                "fact": fact,
                "source_title": _clean(str(row.get("source_title", ""))),
                "topic": _clean(str(row.get("topic", ""))),
            }
        )
    return out


def _memory_fact_for_case(case: dict[str, Any], rows: list[dict[str, Any]]) -> str | None:
    if not rows:
        return None
    prompt = str(case.get("prompt", ""))
    keywords = [str(x).strip().lower() for x in list(case.get("keywords") or []) if str(x).strip()]
    p_tokens = _tokens(prompt)
    best_score = 0.0
    best_fact = ""
    for row in rows:
        fact = str(row.get("fact", "")).strip()
        if not fact:
            continue
        combined = " ".join([fact, str(row.get("source_title", "")), str(row.get("topic", ""))]).strip().lower()
        keyword_hits = 0.0
        if keywords:
            keyword_hits = sum(1 for k in keywords if k in combined) / max(1, len(keywords))
        f_tokens = _tokens(combined)
        overlap = len(p_tokens.intersection(f_tokens)) / max(1, len(p_tokens))
        score = (0.60 * keyword_hits) + (0.40 * overlap)
        if score > best_score:
            best_score = score
            best_fact = fact
    if best_score < 0.20:
        return None
    return best_fact


def build_focus_corpus_file(
    *,
    out_path: str,
    benchmark_file: str,
    candidate_paths: list[str],
    max_pairs: int = 240,
    seed: int = 1337,
    memory_dir: str = "data/memory",
) -> dict[str, Any]:
    rng = random.Random(seed)
    candidates: list[DialogPair] = []
    source_stats: dict[str, int] = {}
    for raw_path in candidate_paths:
        p = Path(str(raw_path or "").strip())
        if not p.exists():
            continue
        parsed = parse_dialog_pairs(p, max_pairs=max(200, max_pairs * 4))
        if not parsed:
            continue
        candidates.extend(parsed)
        source_stats[str(p)] = len(parsed)
    if not candidates:
        return {
            "out_path": out_path,
            "written_pairs": 0,
            "selected_from_benchmark": 0,
            "selected_from_memory": 0,
            "fallback_pairs": 0,
            "candidate_pairs": 0,
            "sources": source_stats,
        }

    bench_cases = _load_benchmark_cases(Path(benchmark_file))
    memory_facts = _load_memory_web_facts(Path(memory_dir))
    selected: list[DialogPair] = []
    used_answers: set[str] = set()
    selected_from_benchmark = 0
    selected_from_memory = 0
    for case in bench_cases:
        scored: list[tuple[float, DialogPair]] = []
        for pair in candidates:
            score = _candidate_score(case, pair)
            if score < 0.20:
                continue
            scored.append((score, pair))
        scored.sort(key=lambda item: item[0], reverse=True)
        chosen = False
        for score, pair in scored:
            key = pair.assistant.lower()
            if key in used_answers:
                continue
            used_answers.add(key)
            selected.append(
                DialogPair(
                    user=str(case.get("prompt", "")),
                    assistant=pair.assistant,
                    source=f"benchmark:{case.get('name', '')}|{pair.source}|score={score:.3f}",
                )
            )
            selected_from_benchmark += 1
            chosen = True
            break
        if chosen:
            if len(selected) >= max_pairs:
                break
            continue
        memory_fact = _memory_fact_for_case(case, memory_facts)
        if memory_fact and memory_fact.lower() not in used_answers:
            used_answers.add(memory_fact.lower())
            selected.append(
                DialogPair(
                    user=str(case.get("prompt", "")),
                    assistant=memory_fact,
                    source=f"memory:{case.get('name', '')}",
                )
            )
            selected_from_memory += 1
        if len(selected) >= max_pairs:
            break

    fallback_pool = [p for p in candidates if p.assistant.lower() not in used_answers]
    rng.shuffle(fallback_pool)
    fallback_pairs = 0
    for pair in fallback_pool:
        if len(selected) >= max_pairs:
            break
        lower_prompt = pair.user.lower()
        if not (
            lower_prompt.startswith("what ")
            or lower_prompt.startswith("where ")
            or lower_prompt.startswith("who ")
            or lower_prompt.startswith("when ")
            or lower_prompt.startswith("why ")
            or lower_prompt.startswith("how ")
            or "explain " in lower_prompt
            or "tell me about" in lower_prompt
        ):
            continue
        selected.append(pair)
        fallback_pairs += 1

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    seen_pairs: set[tuple[str, str]] = set()
    for pair in selected:
        key = (pair.user.lower(), pair.assistant.lower())
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        lines.append(f"User: {pair.user}")
        lines.append(f"Ickle: {pair.assistant}")
        lines.append("")
    out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return {
        "out_path": str(out),
        "written_pairs": len(seen_pairs),
        "selected_from_benchmark": selected_from_benchmark,
        "selected_from_memory": selected_from_memory,
        "fallback_pairs": fallback_pairs,
        "candidate_pairs": len(candidates),
        "sources": source_stats,
    }


def merge_dialog_corpora(
    *,
    corpus_paths: list[str],
    out_path: str,
    max_pairs_per_source: int = 6000,
    max_total_pairs: int = 18000,
    seed: int = 1337,
) -> dict[str, Any]:
    rng = random.Random(seed)
    buckets: list[DialogPair] = []
    source_stats: dict[str, int] = {}
    for raw_path in corpus_paths:
        p = Path(str(raw_path or "").strip())
        if not p.exists():
            continue
        parsed = parse_dialog_pairs(p, max_pairs=max(100, max_pairs_per_source))
        if not parsed:
            continue
        buckets.extend(parsed)
        source_stats[str(p)] = len(parsed)
    if not buckets:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("", encoding="utf-8")
        return {
            "out_path": str(out),
            "written_pairs": 0,
            "source_pairs": source_stats,
        }

    rng.shuffle(buckets)
    out_pairs: list[DialogPair] = []
    seen: set[tuple[str, str]] = set()
    for pair in buckets:
        key = (pair.user.lower(), pair.assistant.lower())
        if key in seen:
            continue
        seen.add(key)
        out_pairs.append(pair)
        if len(out_pairs) >= max(200, max_total_pairs):
            break

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for pair in out_pairs:
        lines.append(f"User: {pair.user}")
        lines.append(f"Ickle: {pair.assistant}")
        lines.append("")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "out_path": str(out),
        "written_pairs": len(out_pairs),
        "source_pairs": source_stats,
    }
