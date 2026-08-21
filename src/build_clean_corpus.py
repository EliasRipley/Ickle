#!/usr/bin/env python3
"""Build a cleaner training corpus by filtering noisy patterns and adding curated dialogs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from src.data_quality import (
    build_quality_filtered_corpus,
    dialogue_pair_fails_content_checks,
    dialogue_pair_has_structural_noise,
)
from src.workspace_paths import get_training_root

# The "Elias Ripley"/"Roman Empire" entries below overlap with
# data_quality.CORE_CONTAMINATION_PHRASES (the shared source-attribution/wiki-leak
# list also used by sanitize_training_data.py and build_honest_context_package.py);
# the rest of this list is structural formatting noise specific to this builder's
# source files, not contamination phrases, so it isn't merged into the shared list.
NOISE_PATTERNS = [
    r"\bNATURAL FLOW\b",
    r"\bCONVERSATION EXAMPLES\b",
    r"\bWEB REQUESTS\b",
    r"\bBASIC GREETINGS\b",
    r"\bOWNER QUESTIONS\b",
    r"\bMEMORY QUESTIONS\b",
    r"\bElias Ripley\b",
    r"\bRoman Empire\b",
    r"\bBBC Weather\b",
    r"text\.replace\(",
    r"aria-label",
    r"HEADLINE:",
    r"PARAGRAPH:",
    r"ITEM:",
]

LOW_SIGNAL_PATTERNS = [
    "wikipedia does not have an article",
    "look for ",
    "article wizard",
    "sister projects",
    "the free encyclopedia",
]

CURATED_DIALOGS: list[tuple[str, str]] = [
    ("Hello Ickle.", "Hi. What are we working on today?"),
    ("Can you help me think through a problem?", "Yes. Share the goal and constraints and I will help step by step."),
    ("I need a quick summary of this topic.", "Sure. Send the topic or source text and I will keep the summary concise."),
    ("I have a vague idea but no clear plan.", "No problem. Tell me the outcome you want and I will suggest a concrete plan."),
    ("Remember that I prefer concise answers.", "Understood. I will keep answers concise unless you ask for depth."),
    ("What do you remember about my preferences?", "I can recall stored preferences when you ask and use them when relevant."),
    ("Can you check this page https://example.com?", "Yes. I can read that page, extract key points, and summarize what matters."),
    ("I need facts from a website, not guesses.", "Agreed. I will read the source and separate observed facts from assumptions."),
    ("What if you are not sure about an answer?", "I will say what is uncertain and suggest how to verify it."),
    ("Can you continue from our previous topic?", "Yes. I can use recent context and memory to continue where we left off."),
    ("Please remember that my project deadline is Friday.", "Saved. I will keep that deadline in mind for planning."),
    ("How should you respond to unclear requests?", "I should ask a focused clarification question before making assumptions."),
    ("Can you browse and then report key points only?", "Yes. I will return the main points and avoid filler."),
    ("What should you do before using a tool?", "I should confirm the tool is relevant and then use it only when needed."),
    ("Can you use memory without repeating irrelevant old details?", "Yes. I should retrieve only details that help with your request."),
    ("If a web page fails to load, what should you do?", "I should report the error clearly and suggest the next useful attempt."),
    ("Can you give me options with tradeoffs?", "Yes. I will list practical options and note the consequences of each."),
    ("How do you handle long tasks?", "I break them into small steps, execute, and verify results as we go."),
    ("Can you be honest about limits?", "Yes. If I cannot do something reliably, I should say so directly."),
    ("I need a direct answer first, then details.", "Understood. I will lead with the answer, then provide supporting detail."),
    ("Ickle can you please tell me the time in Japan?", "I cannot read a live clock directly. Japan uses UTC plus 9, and I can check the current time if tools are enabled."),
    ("What time is it in Tokyo right now?", "I cannot guarantee live time without a tool call. Tokyo is in Japan Standard Time, UTC plus 9."),
    ("Can you help me debug a Python script that crashes on startup?", "Yes. Share the traceback and startup code path, and I will narrow the cause step by step."),
    ("How should I debug a crash quickly?", "Reproduce the issue, capture the full error, isolate the failing function, and test one fix at a time."),
    ("Can you write code without guessing missing details?", "Yes. I will ask for missing constraints when needed and avoid pretending certainty."),
    ("How should you answer if a request is ambiguous?", "I should ask one focused clarification question before acting."),
    ("Can you summarize this article in five bullets?", "Yes. Provide the text or URL and I will return a concise bullet summary."),
    ("I want a plan for learning machine learning in 8 weeks.", "I can create a week by week plan with goals, exercises, and checkpoints."),
    ("What do you do when information might be outdated?", "I should verify with a current source before answering confidently."),
    ("Can you propose three options for this architecture choice?", "Yes. I will give three options and list pros, cons, and risk for each."),
    ("How should you use memory in conversation?", "I should store durable user preferences and retrieve only what is relevant to the current request."),
    ("Can you keep your response practical?", "Yes. I will focus on actionable steps and keep the wording direct."),
    ("Should you invent citations if none are available?", "No. I should never invent citations and should clearly mark uncertainty."),
    ("Can you break this task into a queue?", "Yes. I can split it into ordered tasks with dependencies and status updates."),
    ("When should you run tools?", "Only when a tool materially improves accuracy or execution for the request."),
    ("Can you explain a concept with an example?", "Yes. I will define the concept briefly, then show a concrete example."),
    ("I need a short answer now.", "Understood. I will answer briefly first and add detail only if you ask."),
    ("Can you self-check before finalizing an answer?", "Yes. I should verify the result against the request and catch obvious mistakes."),
    ("How do you handle failed commands?", "I should report the failure, explain likely cause, and propose the next safe retry."),
    ("Can you compare two implementation options?", "Yes. I can compare complexity, performance, maintainability, and risk."),
    ("I need help writing tests for this module.", "I can suggest focused test cases for happy path, edge cases, and failures."),
    ("What should you avoid in responses?", "I should avoid filler, unsupported claims, and unrelated context."),
    ("Can you keep a collaborative tone?", "Yes. I should be direct, supportive, and easy to work with."),
]


def _clean_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def _is_noise(line: str) -> bool:
    if not line:
        return True
    if len(line) < 4:
        return True
    if len(line) > 900:
        return True
    if line.count("<") > 4 and line.count(">") > 4:
        return True
    if re.fullmatch(r"[A-Z][A-Z0-9 _-]{6,}", line):
        return True
    return any(re.search(pattern, line, re.IGNORECASE) for pattern in NOISE_PATTERNS)


def _is_low_quality_pair(user: str, assistant: str) -> bool:
    u = _clean_line(user)
    a = _clean_line(assistant)
    if _is_noise(u) or _is_noise(a):
        return True
    if len(u) > 220 or len(a) > 280:
        return True
    if dialogue_pair_has_structural_noise(u, a):
        return True
    low_a = a.lower()
    if any(p in low_a for p in LOW_SIGNAL_PATTERNS):
        return True
    if low_a.count("|") >= 4:
        return True
    return dialogue_pair_fails_content_checks(u, a)


def _load_text_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []


def _parse_dialogue_pairs(lines: list[str], max_pairs: int) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    pending_user: str | None = None

    for raw in lines:
        line = _clean_line(raw)
        if not line:
            continue

        lower = line.lower()
        if lower.startswith("user:"):
            candidate = _clean_line(line.split(":", 1)[1])
            pending_user = candidate if not _is_noise(candidate) else None
            continue

        if lower.startswith("ickle:") or lower.startswith("assistant:"):
            if not pending_user:
                continue
            candidate = _clean_line(line.split(":", 1)[1])
            if _is_low_quality_pair(pending_user, candidate):
                pending_user = None
                continue
            key = (pending_user.lower(), candidate.lower())
            if key in seen:
                pending_user = None
                continue
            seen.add(key)
            out.append((pending_user, candidate))
            pending_user = None
            if len(out) >= max_pairs:
                break

    return out


def _collect_dialogue_pairs(paths: list[Path], max_pairs: int) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for path in paths:
        remaining = max_pairs - len(out)
        if remaining <= 0:
            break
        parsed = _parse_dialogue_pairs(_load_text_lines(path), max_pairs=remaining)
        for user, assistant in parsed:
            key = (user.lower(), assistant.lower())
            if key in seen:
                continue
            seen.add(key)
            out.append((user, assistant))
            if len(out) >= max_pairs:
                break
    return out


def _collect_dictionary_dialogs(path: Path, max_items: int = 0) -> list[str]:
    if max_items <= 0 or not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []
    if not isinstance(raw, dict):
        return []

    out: list[str] = []
    seen_words: set[str] = set()
    for word in sorted(raw.keys()):
        if len(out) >= max_items * 3:
            break
        if not isinstance(word, str):
            continue
        definition = raw.get(word)
        if not isinstance(definition, str):
            continue
        w = _clean_line(word)
        d = _clean_line(definition)
        if not w or not d:
            continue
        if len(w) < 2 or len(d) < 25:
            continue
        key = w.lower()
        if key in seen_words:
            continue
        seen_words.add(key)
        d = d[:320]
        out.append(f"User: What does {w} mean?")
        out.append(f"Ickle: {w} means {d}")
        out.append("")
    return out


def _pairs_to_lines(pairs: list[tuple[str, str]]) -> list[str]:
    out: list[str] = []
    for user, assistant in pairs:
        out.append(f"User: {user}")
        out.append(f"Ickle: {assistant}")
        out.append("")
    return out


def build_corpus(
    training_root: Path,
    max_lines: int = 20000,
    dictionary_items: int = 0,
    include_open_stream: bool = False,
    quality_filter: bool = False,
) -> tuple[list[str], dict[str, int]]:
    max_pairs = max(200, max_lines // 3)

    candidate_files = [
        Path("data/hub_feedback_corpus.txt"),
        Path("data/training_queue/queued_wikipedia_learning.txt"),
        training_root / "queued_wikipedia_learning.txt",
        training_root / "open_oasst1_stream.txt",
        training_root / "open_openhermes_2_5_stream.txt",
        training_root / "natural_conversation_training.txt",
        training_root / "web_conversation_training.txt",
        training_root / "memory_aware_training.txt",
        training_root / "focused_memory_training.txt",
        training_root / "override_conversation_training.txt",
    ]
    if include_open_stream:
        candidate_files.extend(
            [
                training_root / "open_fineweb_stream.txt",
                training_root / "open_fineweb_edu_stream.txt",
            ]
        )
        for p in sorted(training_root.glob("open_*_stream.txt")):
            if p not in candidate_files:
                candidate_files.append(p)

    existing_files = [p for p in candidate_files if p.exists()]
    harvested_pairs = _collect_dialogue_pairs(existing_files, max_pairs=max_pairs)

    curated_lines = _pairs_to_lines(CURATED_DIALOGS)
    harvested_lines = _pairs_to_lines(harvested_pairs)
    dictionary_lines = _collect_dictionary_dialogs(
        training_root / "webster_dictionary.json",
        max_items=dictionary_items,
    )

    combined = curated_lines + harvested_lines + dictionary_lines
    stats = {
        "files_used": len(existing_files),
        "curated_pairs": len(CURATED_DIALOGS),
        "harvested_pairs": len(harvested_pairs),
        "harvested_lines": len(harvested_lines),
        "dictionary_lines": len(dictionary_lines),
        "curated_lines": len(curated_lines),
        "total_lines": len(combined),
    }

    if quality_filter:
        combined_before = len(combined)
        pairs = _parse_dialogue_pairs(combined, max_pairs=len(combined) // 2 + 1)
        filtered_pairs, qstats = build_quality_filtered_corpus(pairs)
        combined = _pairs_to_lines(filtered_pairs)
        stats["quality_filter_input_pairs"] = len(pairs)
        for k, v in qstats.items():
            stats[f"quality_{k}"] = v
        stats["total_lines"] = len(combined)

    return combined, stats


def main():
    parser = argparse.ArgumentParser(description="Build cleaned corpus for Ickle retraining.")
    parser.add_argument("--training-root", default=str(get_training_root()))
    parser.add_argument("--out", default="data/ickle_clean_corpus.txt")
    parser.add_argument("--max-lines", type=int, default=20000)
    parser.add_argument("--dictionary-items", type=int, default=0)
    parser.add_argument(
        "--include-open-stream",
        action="store_true",
        help="Include open web stream files (can add topical breadth but lower consistency).",
    )
    parser.add_argument(
        "--quality-filter",
        action="store_true",
        help="Apply MinHash dedup, language detection, length/entropy filters to output pairs.",
    )
    args = parser.parse_args()

    lines, stats = build_corpus(
        training_root=Path(args.training_root),
        max_lines=max(1200, args.max_lines),
        dictionary_items=max(0, args.dictionary_items),
        include_open_stream=bool(args.include_open_stream),
        quality_filter=bool(args.quality_filter),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"saved cleaned corpus: {out_path}")
    print(
        "stats:",
        f"files={stats['files_used']}",
        f"curated_pairs={stats['curated_pairs']}",
        f"harvested_pairs={stats['harvested_pairs']}",
        f"harvested_lines={stats['harvested_lines']}",
        f"dictionary={stats['dictionary_lines']}",
        f"total={stats['total_lines']}",
    )


if __name__ == "__main__":
    main()
