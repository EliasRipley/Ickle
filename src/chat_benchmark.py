#!/usr/bin/env python3
"""User-visible chat benchmark for candidate model promotion."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from src.ilm_chat import extract_response_text, generate_response
from src.promotion_gate import has_repeated_word_run, repetition_quality_ladder, token_overlap_score


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
    "in",
    "is",
    "it",
    "its",
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


@dataclass
class BenchmarkCase:
    name: str
    prompt: str
    keywords: list[str]
    required_keywords: list[str]
    forbidden_keywords: list[str]


def _quality_score(text: str) -> float:
    value = str(text or "").strip()
    if not value:
        return 0.0
    lower = value.lower()
    if "user:" in lower or "assistant:" in lower:
        return 0.05
    words = re.findall(r"[a-zA-Z']+", lower)
    if len(words) < 5 and re.search(r"\d\s*[+\-*/×÷^]\s*\d.*=\s*-?\d", value):
        return 0.8
    if len(words) < 5:
        return 0.2
    if has_repeated_word_run(lower):
        return 0.0
    unique_ratio = len(set(words)) / max(1, len(words))
    return repetition_quality_ladder(unique_ratio, high_value=0.8)


_TOKEN_PATTERN = r"[a-zA-Z][a-zA-Z0-9'-]{2,}"


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(_TOKEN_PATTERN, str(text or "").lower()) if t not in STOPWORDS}


def _topic_overlap(prompt: str, response: str) -> float:
    return token_overlap_score(
        prompt, response, stopwords=STOPWORDS, token_pattern=_TOKEN_PATTERN, min_token_len=0
    )


def _keyword_score(response: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    text = str(response or "").lower()
    hits = 0
    for kw in keywords:
        if str(kw).lower() in text:
            hits += 1
    return hits / max(1, len(keywords))


def _keyword_stuffing_penalty(response: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    lower = str(response or "").lower()
    total_hits = 0
    unique_hits = 0
    for kw in keywords:
        k = str(kw).strip().lower()
        if not k:
            continue
        count = len(re.findall(re.escape(k), lower))
        if count > 0:
            unique_hits += 1
            total_hits += count
    if unique_hits == 0:
        return 0.0
    if total_hits > unique_hits * 3:
        return 0.35
    if total_hits > unique_hits * 2:
        return 0.18
    return 0.0


def _load_cases(path: Path) -> list[BenchmarkCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        return []
    out: list[BenchmarkCase] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        prompt = str(row.get("prompt", "")).strip()
        if not prompt:
            continue
        name = str(row.get("name", "")).strip() or prompt[:40]
        keywords = [str(x).strip() for x in list(row.get("keywords") or []) if str(x).strip()]
        required_keywords = [
            str(x).strip() for x in list(row.get("required_keywords") or []) if str(x).strip()
        ]
        forbidden_keywords = [
            str(x).strip() for x in list(row.get("forbidden_keywords") or []) if str(x).strip()
        ]
        out.append(
            BenchmarkCase(
                name=name,
                prompt=prompt,
                keywords=keywords,
                required_keywords=required_keywords,
                forbidden_keywords=forbidden_keywords,
            )
        )
    return out


def _chat(model: str, prompt: str, *, enable_memory: bool, enable_web_tools: bool) -> str:
    args = SimpleNamespace(
        model=model,
        prompt=prompt,
        max_new=240,
        max_new_limit=320,
        temperature=0.25,
        top_k=1,
        torch_threads=4,
        skill="",
        enable_memory=bool(enable_memory),
        enable_web_tools=bool(enable_web_tools),
    )
    return extract_response_text(generate_response(args))


def _chat_once(model: str, prompt: str, *, enable_memory: bool = False) -> str:
    """Same generation call as _chat(), fixed to enable_web_tools=False. Used by the
    coherence probe below (also called from train_autopilot.py and
    train_intelligence_stack.py) which don't need web tools during the sanity-check
    chat exchange."""
    return _chat(model, prompt, enable_memory=enable_memory, enable_web_tools=False)


def _coherence_probe(model: str) -> dict[str, Any]:
    """Quick sanity check that a candidate model gives non-generic answers and can
    carry short-term memory across a two-turn follow-up. Shared by train_autopilot.py
    and train_intelligence_stack.py, which previously each carried their own
    byte-for-byte copy of this function."""
    prompts = [
        "Where do gorillas live?",
        "What is photosynthesis?",
        "What causes earthquakes?",
        "I have 2 hours and 5 tasks. How should I prioritize?",
    ]
    rows: list[dict[str, Any]] = []
    generic_hits = 0
    for prompt in prompts:
        response = _chat_once(model, prompt, enable_memory=False)
        is_generic = (
            "I can help with that. Share the exact outcome you want and any constraints." in response
            or "I may not have enough reliable local knowledge for that yet." in response
        )
        if is_generic:
            generic_hits += 1
        rows.append({"prompt": prompt, "response": response, "is_generic": is_generic})

    followup_1 = _chat_once(model, "Ickle can you please tell me the time in Japan?", enable_memory=True)
    followup_2 = _chat_once(model, "And what is that relative to UTC?", enable_memory=True)
    rows.append({"prompt": "Ickle can you please tell me the time in Japan?", "response": followup_1, "is_generic": False})
    rows.append({"prompt": "And what is that relative to UTC?", "response": followup_2, "is_generic": False})

    followup_ok = "utc" in followup_2.lower() or "+09" in followup_2.lower()
    passed = generic_hits == 0 and followup_ok
    return {
        "passed": passed,
        "generic_hits": generic_hits,
        "followup_ok": followup_ok,
        "rows": rows,
    }


def _score_response(
    prompt: str,
    response: str,
    keywords: list[str],
    required_keywords: list[str] | None = None,
    forbidden_keywords: list[str] | None = None,
) -> dict[str, float]:
    quality = _quality_score(response)
    keyword = _keyword_score(response, keywords)
    overlap = _topic_overlap(prompt, response)
    stuffing_penalty = _keyword_stuffing_penalty(response, keywords)
    lower = str(response or "").lower()
    required = [str(x).strip().lower() for x in (required_keywords or []) if str(x).strip()]
    forbidden = [str(x).strip().lower() for x in (forbidden_keywords or []) if str(x).strip()]
    required_score = (
        sum(1 for item in required if item in lower) / len(required)
        if required
        else 1.0
    )
    forbidden_hits = sum(1 for item in forbidden if item in lower)
    score = (0.40 * keyword) + (0.35 * overlap) + (0.25 * quality) - stuffing_penalty
    if keyword >= 0.7 and overlap < 0.2:
        score -= 0.2
    if required and required_score < 1.0:
        score = min(score, 0.2 * required_score)
    if forbidden_hits:
        score = min(score, 0.05)
    score = max(0.0, min(1.0, score))
    return {
        "score": round(score, 4),
        "keyword_score": round(keyword, 4),
        "overlap_score": round(overlap, 4),
        "quality_score": round(quality, 4),
        "stuffing_penalty": round(stuffing_penalty, 4),
        "required_score": round(required_score, 4),
        "forbidden_hits": float(forbidden_hits),
    }


def run_benchmark(
    *,
    model: str,
    cases: list[BenchmarkCase],
    enable_memory: bool,
    enable_web_tools: bool,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    total = 0.0
    for case in cases:
        response = _chat(model, case.prompt, enable_memory=enable_memory, enable_web_tools=enable_web_tools)
        score = _score_response(
            case.prompt,
            response,
            case.keywords,
            case.required_keywords,
            case.forbidden_keywords,
        )
        total += float(score["score"])
        rows.append(
            {
                "name": case.name,
                "prompt": case.prompt,
                "response": response,
                "keywords": case.keywords,
                "required_keywords": case.required_keywords,
                "forbidden_keywords": case.forbidden_keywords,
                **score,
            }
        )
    avg = total / max(1, len(rows))
    return {
        "model": model,
        "case_count": len(rows),
        "avg_score": round(avg, 4),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description="Run user-facing chat benchmark for Ickle models.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--baseline-model", default="")
    parser.add_argument("--benchmark-file", default="data/maintenance/user_chat_benchmark.json")
    parser.add_argument("--enable-memory", action="store_true")
    parser.add_argument("--enable-web-tools", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    bench_path = Path(args.benchmark_file)
    if not bench_path.exists():
        raise SystemExit(f"benchmark file not found: {bench_path}")

    cases = _load_cases(bench_path)
    if not cases:
        raise SystemExit("benchmark file has no valid cases.")

    candidate = run_benchmark(
        model=args.model,
        cases=cases,
        enable_memory=bool(args.enable_memory),
        enable_web_tools=bool(args.enable_web_tools),
    )
    out: dict[str, Any] = {"candidate": candidate, "benchmark_file": str(bench_path)}

    baseline_model = str(args.baseline_model or "").strip()
    if baseline_model:
        baseline = run_benchmark(
            model=baseline_model,
            cases=cases,
            enable_memory=bool(args.enable_memory),
            enable_web_tools=bool(args.enable_web_tools),
        )
        out["baseline"] = baseline
        out["delta"] = round(candidate["avg_score"] - baseline["avg_score"], 4)

    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        if baseline_model:
            print(
                f"candidate={candidate['avg_score']:.4f} "
                f"baseline={out['baseline']['avg_score']:.4f} delta={out['delta']:.4f}"
            )
        else:
            print(f"candidate={candidate['avg_score']:.4f}")


if __name__ == "__main__":
    main()
