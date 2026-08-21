#!/usr/bin/env python3
"""Build a higher-signal corpus focused on factual QA + practical responses."""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from src.data_quality import build_quality_filtered_corpus, has_repeated_word_run
from src.workspace_paths import get_training_root


@dataclass(frozen=True)
class DialogPair:
    user: str
    assistant: str
    source: str
    category: str


CODE_RE = re.compile(
    r"\b("
    r"import\s+\w+|from\s+\w+\s+import|def\s+\w+\(|class\s+\w+\(?|"
    r"function\s+\w+\(|const\s+\w+\s*=|let\s+\w+\s*=|var\s+\w+\s*=|"
    r"package\s+main|public\s+static\s+void|#include\s*<|"
    r"select\s+.+\s+from|insert\s+into|update\s+\w+\s+set|delete\s+from"
    r")\b",
    flags=re.IGNORECASE,
)

STYLE_RE = re.compile(
    r"\b("
    r"respond using the words/style of|"
    r"in the style of|"
    r"pretend to be|"
    r"roleplay|"
    r"act as|"
    r"this is a chat between|"
    r"write (?:a|an) .* in the style of"
    r")\b",
    flags=re.IGNORECASE,
)

LOW_SIGNAL_RE = re.compile(
    r"\b("
    r"i can help with that\. share the exact outcome|"
    r"wikipedia does not have an article|"
    r"article wizard|"
    r"sister projects|"
    r"other reasons this message may be displayed|"
    r"\bplan:\b|"
    r"#e\d+|"
    r":evidence\d+|"
    r"duckduckgo\[|"
    r"google(search)?\[|"
    r"yahoosearch\[|"
    r"infosearch\[|"
    r"qaengine\[|"
    r"textsummarizer\[|"
    r"diagramfinder\[|"
    r"encyclopedia(search)?\[|"
    r"are\s+inhabits|"
    r"are\s+inhabit\b"
    r")\b",
    flags=re.IGNORECASE,
)

AI_DISCLAIMER_RE = re.compile(
    r"\b("
    r"as an ai(?: language model| assistant| model)?|"
    r"i do not have personal (?:beliefs|opinions|experiences)|"
    r"i don't have personal (?:beliefs|opinions|experiences)|"
    r"i am not capable of"
    r")\b",
    flags=re.IGNORECASE,
)

META_ASSISTANT_RE = re.compile(
    r"^(?:yes|sure|understood)\.\s+i\s+(?:will|can|should)\b",
    flags=re.IGNORECASE,
)

QUESTION_KNOWLEDGE_STARTS = (
    "what ",
    "where ",
    "who ",
    "when ",
    "why ",
    "how ",
    "which ",
    "define ",
    "explain ",
    "tell me about",
)

REASONING_HINTS = (
    "calculate",
    "probability",
    "solve",
    "step by step",
    "show your steps",
    "logic",
    "equation",
    "math",
    "reason",
    "analyze",
    "compare",
)

CONTENT_STOPWORDS = {
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
    "me",
    "my",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
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


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _content_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9'-]{2,}", text.lower())
    return {t for t in tokens if t not in CONTENT_STOPWORDS}


def _symbol_density(text: str) -> float:
    if not text:
        return 0.0
    symbols = sum(1 for ch in text if ch in "{}[]()<>=`$;\\")
    return symbols / max(1, len(text))


def _pair_quality_ok(user: str, assistant: str) -> bool:
    u = _clean(user)
    a = _clean(assistant)
    if len(u) < 6 or len(a) < 18:
        return False
    if len(u) > 260 or len(a) > 520:
        return False
    if STYLE_RE.search(u) or STYLE_RE.search(a):
        return False
    if LOW_SIGNAL_RE.search(a):
        return False
    if AI_DISCLAIMER_RE.search(a):
        return False
    if META_ASSISTANT_RE.search(a) and len(a) < 140:
        return False
    lower_a = a.lower()
    if "plan:" in lower_a and ("[" in a or "#e" in lower_a or ":evidence" in lower_a):
        return False
    if CODE_RE.search(a):
        return False
    if _symbol_density(a) > 0.05:
        return False
    if "```" in u or "```" in a:
        return False
    if u.lower() == a.lower():
        return False
    if has_repeated_word_run(a.lower()):
        return False
    overlap = _content_tokens(u).intersection(_content_tokens(a))
    if len(_content_tokens(u)) >= 3 and not overlap:
        return False
    return True


def _categorize_pair(user: str) -> str:
    lower = _clean(user).lower()
    if lower.startswith(QUESTION_KNOWLEDGE_STARTS):
        return "knowledge"
    if any(h in lower for h in REASONING_HINTS):
        return "reasoning"
    return "assistant"


def _parse_pairs(path: Path, max_pairs: int) -> list[DialogPair]:
    out: list[DialogPair] = []
    if not path.exists():
        return out
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
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
            answer = _clean(line.split(":", 1)[1])
            key = (pending_user.lower(), answer.lower())
            if key in seen:
                pending_user = ""
                continue
            seen.add(key)
            if _pair_quality_ok(pending_user, answer):
                out.append(
                    DialogPair(
                        user=pending_user,
                        assistant=answer,
                        source=str(path),
                        category=_categorize_pair(pending_user),
                    )
                )
                if len(out) >= max_pairs:
                    break
            pending_user = ""
    return out


def _parse_dictionary_style_pairs(path: Path, max_pairs: int) -> list[DialogPair]:
    out: list[DialogPair] = []
    if not path.exists():
        return out
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    i = 0
    seen: set[tuple[str, str]] = set()
    while i < len(lines):
        line = _clean(lines[i])
        i += 1
        if not line:
            continue
        m = re.match(r"^what does ([a-zA-Z][a-zA-Z0-9' -]{1,40}) mean\?$", line, flags=re.IGNORECASE)
        if not m:
            continue
        word = _clean(m.group(1))
        answer = ""
        while i < len(lines):
            candidate = _clean(lines[i])
            i += 1
            if not candidate:
                continue
            lower = candidate.lower()
            if lower.startswith(f"{word.lower()} means "):
                answer = candidate
                break
            if lower.startswith("the definition of ") and " is " in lower:
                answer = candidate
                break
            if re.match(r"^what does .+ mean\?$", lower):
                i -= 1
                break
        if not answer:
            continue
        user = f"What does {word} mean?"
        if not _pair_quality_ok(user, answer):
            continue
        key = (user.lower(), answer.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(DialogPair(user=user, assistant=answer, source=str(path), category="knowledge"))
        if len(out) >= max_pairs:
            break
    return out


def build_smart_corpus(
    *,
    training_root: Path,
    max_pairs: int,
    seed: int,
    include_legacy_dictionary: bool = False,
    include_topic_queues: bool = False,
    quality_filter: bool = False,
) -> tuple[list[DialogPair], dict[str, int]]:
    source_paths = [
        Path("data/ickle_curated_only.txt"),
        Path("data/ickle_reasoning_booster.txt"),
        training_root / "open_oasst1_stream.txt",
        training_root / "open_openhermes_2_5_stream.txt",
    ]
    if include_topic_queues:
        source_paths.append(training_root / "queued_wikipedia_learning.txt")
        topic_dir = training_root / "topic_queues"
        if topic_dir.exists():
            for p in sorted(topic_dir.glob("*.txt")):
                source_paths.append(p)
    dictionary_sources = []
    if include_legacy_dictionary:
        dictionary_sources = [
            training_root / "ickle_v3_corpus.txt",
            training_root / "comprehensive_english.txt",
        ]
    existing = [p for p in source_paths if p.exists()]

    harvested: list[DialogPair] = []
    source_counts: dict[str, int] = {}
    for p in existing:
        pairs = _parse_pairs(p, max_pairs=max(1000, max_pairs))
        harvested.extend(pairs)
        source_counts[str(p)] = len(pairs)

    for p in dictionary_sources:
        if not p.exists():
            continue
        pairs = _parse_dictionary_style_pairs(p, max_pairs=max(500, max_pairs // 2))
        harvested.extend(pairs)
        source_counts[str(p)] = len(pairs)

    deduped: list[DialogPair] = []
    seen: set[tuple[str, str]] = set()
    for pair in harvested:
        key = (pair.user.lower(), pair.assistant.lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(pair)

    if quality_filter:
        pairs_as_tuples = [(p.user, p.assistant) for p in deduped]
        filtered_tuples, qstats = build_quality_filtered_corpus(pairs_as_tuples)
        filtered_map = {(u.lower(), a.lower()): (u, a) for u, a in filtered_tuples}
        deduped = [
            p for p in deduped
            if (p.user.lower(), p.assistant.lower()) in filtered_map
        ]

    rng = random.Random(seed)
    by_cat: dict[str, list[DialogPair]] = {"knowledge": [], "reasoning": [], "assistant": []}
    for pair in deduped:
        by_cat.setdefault(pair.category, []).append(pair)
    for rows in by_cat.values():
        rng.shuffle(rows)

    knowledge_target = int(max_pairs * 0.60)
    reasoning_target = int(max_pairs * 0.25)
    assistant_target = max(0, max_pairs - knowledge_target - reasoning_target)

    def _take_without_replacement(pool: list[DialogPair], n: int) -> list[DialogPair]:
        if n <= 0 or not pool:
            return []
        return list(pool[: min(len(pool), n)])

    selected: list[DialogPair] = []
    selected.extend(_take_without_replacement(by_cat.get("knowledge", []), knowledge_target))
    selected.extend(_take_without_replacement(by_cat.get("reasoning", []), reasoning_target))
    selected.extend(_take_without_replacement(by_cat.get("assistant", []), assistant_target))

    if len(selected) < max_pairs:
        used_keys = {(p.user.lower(), p.assistant.lower()) for p in selected}
        leftovers = [
            p
            for p in deduped
            if (p.user.lower(), p.assistant.lower()) not in used_keys
        ]
        rng.shuffle(leftovers)
        need = max_pairs - len(selected)
        selected.extend(leftovers[:need])

    rng.shuffle(selected)
    stats = {
        "files_used": len(existing),
        "harvested_pairs": len(harvested),
        "deduped_pairs": len(deduped),
        "selected_pairs": len(selected),
        "knowledge_pairs": sum(1 for p in selected if p.category == "knowledge"),
        "reasoning_pairs": sum(1 for p in selected if p.category == "reasoning"),
        "assistant_pairs": sum(1 for p in selected if p.category == "assistant"),
    }
    if quality_filter:
        for k, v in qstats.items():
            stats[f"quality_{k}"] = v
    for source, count in source_counts.items():
        stats[f"source::{source}"] = count
    return selected, stats


def main():
    parser = argparse.ArgumentParser(description="Build high-signal smart corpus for model training.")
    parser.add_argument("--training-root", default=str(get_training_root()))
    parser.add_argument("--out", default="data/ickle_smart_corpus.txt")
    parser.add_argument("--stats-out", default="data/maintenance/smart_corpus_stats.json")
    parser.add_argument("--max-pairs", type=int, default=22000)
    parser.add_argument("--include-legacy-dictionary", action="store_true")
    parser.add_argument("--include-topic-queues", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--quality-filter", action="store_true", help="Apply MinHash dedup, language, length/entropy filters")
    args = parser.parse_args()

    training_root = Path(args.training_root)
    selected, stats = build_smart_corpus(
        training_root=training_root,
        max_pairs=max(2000, int(args.max_pairs)),
        seed=int(args.seed),
        include_legacy_dictionary=bool(args.include_legacy_dictionary),
        include_topic_queues=bool(args.include_topic_queues),
        quality_filter=bool(args.quality_filter),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for pair in selected:
        lines.append(f"User: {pair.user}")
        lines.append(f"Ickle: {pair.assistant}")
        lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    stats_out = Path(args.stats_out)
    stats_out.parent.mkdir(parents=True, exist_ok=True)
    stats_out.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    report = {
        "out": str(out_path.resolve()),
        "stats_out": str(stats_out.resolve()),
        "stats": stats,
    }
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(
            f"smart corpus ready pairs={stats['selected_pairs']} "
            f"knowledge={stats['knowledge_pairs']} reasoning={stats['reasoning_pairs']} assistant={stats['assistant_pairs']}"
        )


if __name__ == "__main__":
    main()
