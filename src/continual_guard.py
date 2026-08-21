from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from src.ilm_chat import extract_response_text, generate_response
from src.resource_defaults import DEFAULT_CPU_PCT, DEFAULT_GPU_PCT, DEFAULT_RAM_PCT, add_resource_pct_args
from src.promotion_gate import (
    PromotionGateConfig,
    has_repeated_word_run,
    keyword_stuffing_penalty,
    repetition_quality_ladder,
    run_promotion_cycle,
    token_overlap_score,
)
from src.train_invoke import build_train_command, run_train_command


@dataclass
class DialogPair:
    user: str
    assistant: str
    source: str = ""


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def parse_dialog_pairs(corpus_path: str, max_pairs: int = 0) -> list[DialogPair]:
    p = Path(corpus_path)
    if not p.exists():
        return []
    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    out: list[DialogPair] = []
    pending_user = ""
    for raw in lines:
        line = _clean_text(raw)
        if not line:
            continue
        lower = line.lower()
        if lower.startswith("user:"):
            pending_user = _clean_text(line.split(":", 1)[1])
            continue
        if lower.startswith("ickle:") or lower.startswith("assistant:"):
            if not pending_user:
                continue
            assistant = _clean_text(line.split(":", 1)[1])
            if len(pending_user) < 4 or len(assistant) < 4:
                pending_user = ""
                continue
            out.append(DialogPair(user=pending_user, assistant=assistant, source=str(p)))
            pending_user = ""
            if max_pairs > 0 and len(out) >= max_pairs:
                break
    return out


def load_replay_buffer(path: str) -> list[DialogPair]:
    p = Path(path)
    if not p.exists():
        return []
    out: list[DialogPair] = []
    with p.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            user = _clean_text(row.get("user", ""))
            assistant = _clean_text(row.get("assistant", ""))
            source = _clean_text(row.get("source", ""))
            if user and assistant:
                out.append(DialogPair(user=user, assistant=assistant, source=source))
    return out


def save_replay_buffer(path: str, pairs: list[DialogPair]):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for pair in pairs:
            row = {"user": pair.user, "assistant": pair.assistant, "source": pair.source}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _pair_key(pair: DialogPair) -> tuple[str, str]:
    return (pair.user.lower(), pair.assistant.lower())


def update_replay_buffer(
    replay_pairs: list[DialogPair],
    new_pairs: list[DialogPair],
    *,
    max_size: int,
    seed: int,
) -> list[DialogPair]:
    rng = random.Random(seed)
    reservoir: list[DialogPair] = []
    seen: set[tuple[str, str]] = set()

    # Start with existing pairs.
    for pair in replay_pairs:
        key = _pair_key(pair)
        if key in seen:
            continue
        seen.add(key)
        reservoir.append(pair)
        if len(reservoir) >= max_size:
            break

    seen_total = len(reservoir)
    for pair in new_pairs:
        key = _pair_key(pair)
        if key in seen:
            continue
        seen.add(key)
        seen_total += 1
        if len(reservoir) < max_size:
            reservoir.append(pair)
            continue
        idx = rng.randint(0, max(0, seen_total - 1))
        if idx < max_size:
            reservoir[idx] = pair

    return reservoir


def _sample_pairs(
    pairs: list[DialogPair],
    n: int,
    rng: random.Random,
    *,
    allow_repeat: bool = False,
) -> list[DialogPair]:
    if n <= 0 or not pairs:
        return []
    if n >= len(pairs):
        if not allow_repeat:
            return list(pairs)
        return [pairs[rng.randrange(len(pairs))] for _ in range(n)]
    return rng.sample(pairs, n)


def build_compartment_mixture(
    *,
    core_pairs: list[DialogPair],
    replay_pairs: list[DialogPair],
    new_pairs: list[DialogPair],
    total_pairs: int,
    core_ratio: float,
    replay_ratio: float,
    new_ratio: float,
    seed: int,
) -> list[DialogPair]:
    rng = random.Random(seed)
    total_pairs = max(100, int(total_pairs))

    ratio_sum = max(1e-9, float(core_ratio) + float(replay_ratio) + float(new_ratio))
    core_ratio = float(core_ratio) / ratio_sum
    replay_ratio = float(replay_ratio) / ratio_sum
    new_ratio = float(new_ratio) / ratio_sum

    core_n = int(round(total_pairs * core_ratio))
    replay_n = int(round(total_pairs * replay_ratio))
    new_n = max(0, total_pairs - core_n - replay_n)

    core_pick = _sample_pairs(core_pairs, core_n, rng, allow_repeat=True)
    replay_pick = _sample_pairs(replay_pairs, replay_n, rng, allow_repeat=True)
    new_pick = _sample_pairs(new_pairs, new_n, rng, allow_repeat=True)

    mixed = core_pick + replay_pick + new_pick
    if not mixed:
        return []
    rng.shuffle(mixed)
    return mixed


def write_pairs_as_corpus(path: str, pairs: list[DialogPair]):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for pair in pairs:
        lines.append(f"User: {pair.user}")
        lines.append(f"Ickle: {pair.assistant}")
        lines.append("")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _quality_score(text: str) -> float:
    value = _clean_text(text)
    if not value:
        return 0.0
    if _is_evasive_response(value):
        return 0.05
    lower = value.lower()
    if "user:" in lower or "assistant:" in lower or "ickle:" in lower:
        return 0.1
    words = re.findall(r"[a-zA-Z']+", lower)
    if len(words) < 5:
        return 0.15
    if has_repeated_word_run(lower):
        return 0.0
    unique_ratio = len(set(words)) / max(1, len(words))
    return repetition_quality_ladder(unique_ratio)


_STOPWORDS = {
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

_EVASIVE_PATTERNS = (
    r"i can help with that\. share the exact outcome",
    r"share the exact outcome you want",
    r"i may not have enough reliable local knowledge",
    r"if you want, i can research",
    r"as an ai",
)


def _is_evasive_response(text: str) -> bool:
    lower = str(text or "").lower()
    return any(re.search(pattern, lower) for pattern in _EVASIVE_PATTERNS)


def _topic_overlap_score(prompt: str, response: str) -> float:
    return token_overlap_score(prompt, response, stopwords=_STOPWORDS, empty_a_fallback=0.4)


def _expected_overlap_score(expected: str, response: str) -> float:
    return token_overlap_score(expected, response, stopwords=_STOPWORDS, empty_a_fallback=0.0)


def _score_prompt_response(prompt: str, response: str) -> float:
    quality = _quality_score(response)
    overlap = _topic_overlap_score(prompt, response)
    return 0.65 * quality + 0.35 * overlap


def _score_pair_response(prompt: str, expected: str, response: str) -> float:
    quality = _quality_score(response)
    prompt_overlap = _topic_overlap_score(prompt, response)
    expected_overlap = _expected_overlap_score(expected, response)
    return 0.40 * quality + 0.20 * prompt_overlap + 0.40 * expected_overlap


def _chat_once(model: str, prompt: str) -> str:
    args = SimpleNamespace(
        model=model,
        prompt=prompt,
        max_new=220,
        max_new_limit=320,
        temperature=0.4,
        top_k=25,
        torch_threads=4,
        skill="",
        enable_memory=False,
        enable_web_tools=False,
    )
    return extract_response_text(generate_response(args))


def evaluate_model(model: str, prompts: list[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if not prompts:
        return {"avg_score": 0.0, "avg_quality": 0.0, "count": 0, "rows": rows}

    score_sum = 0.0
    quality_sum = 0.0
    for prompt in prompts:
        response = _chat_once(model, prompt)
        score = _score_prompt_response(prompt, response)
        quality = _quality_score(response)
        rows.append(
            {
                "prompt": prompt,
                "response": response,
                "score": round(score, 4),
                "quality": round(quality, 4),
            }
        )
        score_sum += score
        quality_sum += quality
    count = len(rows)
    return {
        "avg_score": round(score_sum / max(1, count), 4),
        "avg_quality": round(quality_sum / max(1, count), 4),
        "count": count,
        "rows": rows,
    }


def evaluate_model_pairs(model: str, pairs: list[DialogPair]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if not pairs:
        return {"avg_score": 0.0, "avg_quality": 0.0, "count": 0, "rows": rows}

    score_sum = 0.0
    quality_sum = 0.0
    for pair in pairs:
        prompt = pair.user
        expected = pair.assistant
        response = _chat_once(model, prompt)
        score = _score_pair_response(prompt, expected, response)
        quality = _quality_score(response)
        rows.append(
            {
                "prompt": prompt,
                "expected": expected,
                "response": response,
                "score": round(score, 4),
                "quality": round(quality, 4),
            }
        )
        score_sum += score
        quality_sum += quality
    count = len(rows)
    return {
        "avg_score": round(score_sum / max(1, count), 4),
        "avg_quality": round(quality_sum / max(1, count), 4),
        "count": count,
        "rows": rows,
    }


def _pick_pairs(pairs: list[DialogPair], n: int, seed: int) -> list[DialogPair]:
    rng = random.Random(seed)
    eligible = [p for p in pairs if p.user and p.assistant]
    if len(eligible) <= n:
        return eligible
    return rng.sample(eligible, n)


def _load_user_benchmark_cases(path: str) -> list[dict[str, Any]]:
    p = Path(path)
    if not path or not p.exists():
        return []
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        prompt = _clean_text(row.get("prompt", ""))
        if not prompt:
            continue
        keywords = [str(x).strip().lower() for x in list(row.get("keywords") or []) if str(x).strip()]
        out.append(
            {
                "name": _clean_text(row.get("name", "")) or prompt[:40],
                "prompt": prompt,
                "keywords": keywords,
            }
        )
    return out


def _keyword_match_score(response: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    lower = str(response or "").lower()
    hits = sum(1 for k in keywords if k in lower)
    return hits / max(1, len(keywords))


def evaluate_model_user_cases(model: str, cases: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if not cases:
        return {
            "avg_score": 0.0,
            "avg_quality": 0.0,
            "count": 0,
            "rows": rows,
            "min_case_score": 0.0,
            "evasive_count": 0,
        }

    score_sum = 0.0
    quality_sum = 0.0
    min_case_score = 1.0
    evasive_count = 0
    for row in cases:
        prompt = str(row.get("prompt", "")).strip()
        keywords = [str(x).strip().lower() for x in list(row.get("keywords") or []) if str(x).strip()]
        response = _chat_once(model, prompt)
        evasive = _is_evasive_response(response)
        if evasive:
            evasive_count += 1
        quality = _quality_score(response)
        overlap = _topic_overlap_score(prompt, response)
        keyword = _keyword_match_score(response, keywords)
        stuffing = keyword_stuffing_penalty(response, keywords)
        score = (0.55 * keyword) + (0.25 * overlap) + (0.20 * quality) - stuffing
        score = max(0.0, score)
        if evasive:
            score *= 0.2
        min_case_score = min(min_case_score, score)
        rows.append(
            {
                "name": str(row.get("name", "")),
                "prompt": prompt,
                "keywords": keywords,
                "response": response,
                "score": round(score, 4),
                "keyword_score": round(keyword, 4),
                "overlap_score": round(overlap, 4),
                "quality": round(quality, 4),
                "stuffing_penalty": round(stuffing, 4),
                "evasive": evasive,
            }
        )
        score_sum += score
        quality_sum += quality
    count = len(rows)
    return {
        "avg_score": round(score_sum / max(1, count), 4),
        "avg_quality": round(quality_sum / max(1, count), 4),
        "count": count,
        "rows": rows,
        "min_case_score": round(min_case_score if count else 0.0, 4),
        "evasive_count": int(evasive_count),
    }


def run_training_command(
    *,
    data_path: str,
    out_model: str,
    init_model: str,
    steps: int,
    lr: float,
    warmup_steps: int,
    cpu_pct: int = DEFAULT_CPU_PCT,
    ram_pct: int = DEFAULT_RAM_PCT,
    gpu_pct: int = DEFAULT_GPU_PCT,
    checkpoint_path: str,
    resume_if_possible: bool,
    torch_threads: int = 0,
    batch_size: int = 0,
    grad_accum_steps: int = 1,
    line_cb: Callable[[str], None] | None = None,
) -> list[str]:
    cmd = build_train_command(
        data_path=data_path,
        out_model=out_model,
        init_model=init_model,
        steps=steps,
        lr=lr,
        warmup_steps=warmup_steps,
        cpu_pct=cpu_pct,
        ram_pct=ram_pct,
        gpu_pct=gpu_pct,
        checkpoint_path=checkpoint_path,
        resume_if_possible=resume_if_possible,
        torch_threads=torch_threads,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
    )
    return run_train_command(cmd, line_cb=line_cb, tail_lines=40, error_label="continual training")


def run_guarded_step(args, progress_cb: Callable[[str], None] | None = None) -> dict[str, Any]:
    def _emit(message: str):
        if progress_cb is None:
            return
        progress_cb(str(message))

    core_pairs = parse_dialog_pairs(args.core_corpus, max_pairs=max(200, args.max_core_pairs))
    new_pairs = parse_dialog_pairs(args.new_corpus, max_pairs=max(200, args.max_new_pairs))
    replay_pairs = load_replay_buffer(args.replay_buffer)

    if not core_pairs:
        raise ValueError("core corpus is empty or invalid.")
    if not new_pairs:
        raise ValueError("new corpus is empty or invalid.")

    replay_pairs = update_replay_buffer(
        replay_pairs,
        new_pairs,
        max_size=max(500, int(args.replay_max_size)),
        seed=int(args.seed),
    )
    save_replay_buffer(args.replay_buffer, replay_pairs)

    mixed_pairs = build_compartment_mixture(
        core_pairs=core_pairs,
        replay_pairs=replay_pairs,
        new_pairs=new_pairs,
        total_pairs=max(600, int(args.total_pairs)),
        core_ratio=float(args.core_ratio),
        replay_ratio=float(args.replay_ratio),
        new_ratio=float(args.new_ratio),
        seed=int(args.seed),
    )
    if not mixed_pairs:
        raise RuntimeError("failed to build mixed continual corpus.")
    write_pairs_as_corpus(args.mixed_corpus_out, mixed_pairs)

    _emit(
        "continual-train-start: "
        f"steps={int(args.steps)} core_pairs={len(core_pairs)} "
        f"new_pairs={len(new_pairs)} mixed_pairs={len(mixed_pairs)}"
    )

    def _on_train_line(line: str):
        low = str(line).lower()
        if low.startswith("step=") or "checkpoint saved" in low or low.startswith("saved model:"):
            _emit(f"train: {line[:320]}")

    train_tail = run_training_command(
        data_path=args.mixed_corpus_out,
        out_model=args.out_model,
        init_model=args.baseline_model,
        steps=int(args.steps),
        lr=float(args.lr),
        warmup_steps=int(args.warmup_steps),
        cpu_pct=int(getattr(args, "cpu_pct", 80) or 80),
        ram_pct=int(getattr(args, "ram_pct", 80) or 80),
        gpu_pct=int(getattr(args, "gpu_pct", 80) or 80),
        checkpoint_path=args.checkpoint_path,
        resume_if_possible=bool(args.resume_if_possible),
        torch_threads=int(getattr(args, "torch_threads", 0) or 0),
        batch_size=int(getattr(args, "batch_size", 0) or 0),
        grad_accum_steps=int(getattr(args, "grad_accum_steps", 1) or 1),
        line_cb=_on_train_line,
    )

    core_eval_pairs = _pick_pairs(core_pairs, n=max(5, int(args.eval_core_prompts)), seed=int(args.seed) + 11)
    new_eval_pairs = _pick_pairs(new_pairs, n=max(5, int(args.eval_new_prompts)), seed=int(args.seed) + 29)

    baseline_core = evaluate_model_pairs(args.baseline_model, core_eval_pairs)
    candidate_core = evaluate_model_pairs(args.out_model, core_eval_pairs)
    baseline_new = evaluate_model_pairs(args.baseline_model, new_eval_pairs)
    candidate_new = evaluate_model_pairs(args.out_model, new_eval_pairs)

    core_drop = baseline_core["avg_score"] - candidate_core["avg_score"]
    new_gain = candidate_new["avg_score"] - baseline_new["avg_score"]
    core_score = float(candidate_core["avg_score"])
    quality_floor = float(candidate_core["avg_quality"])

    core_ok = core_drop <= float(args.max_core_drop)
    new_ok = new_gain >= float(args.min_new_gain)
    core_score_ok = core_score >= float(args.min_core_score)
    quality_ok = quality_floor >= float(args.min_core_quality)

    user_cases = _load_user_benchmark_cases(str(args.user_benchmark_file or ""))
    baseline_user = evaluate_model_user_cases(args.baseline_model, user_cases) if user_cases else None
    candidate_user = evaluate_model_user_cases(args.out_model, user_cases) if user_cases else None
    user_ok = True
    user_delta = 0.0
    user_score = 0.0
    user_min_case_score = 0.0
    user_evasive_count = 0
    required_user_score = float(getattr(args, "min_user_score", -1.0))
    required_user_case_score = float(getattr(args, "min_user_case_score", -1.0))
    allowed_user_evasive = int(getattr(args, "max_user_evasive", -1))
    if baseline_user and candidate_user:
        baseline_user_score = float(baseline_user.get("avg_score", 0.0))
        baseline_user_case = float(baseline_user.get("min_case_score", 0.0))
        baseline_user_evasive = int(baseline_user.get("evasive_count", 0))
        if required_user_score < 0:
            required_user_score = baseline_user_score + max(0.0, float(args.min_user_delta))
        if required_user_case_score < 0:
            required_user_case_score = baseline_user_case
        if allowed_user_evasive < 0:
            allowed_user_evasive = baseline_user_evasive

        user_delta = float(candidate_user["avg_score"]) - float(baseline_user["avg_score"])
        user_score = float(candidate_user["avg_score"])
        user_min_case_score = float(candidate_user.get("min_case_score", 0.0))
        user_evasive_count = int(candidate_user.get("evasive_count", 0))
        user_ok = (
            (user_delta >= float(args.min_user_delta))
            and (user_score >= required_user_score)
            and (user_min_case_score >= required_user_case_score)
            and (user_evasive_count <= allowed_user_evasive)
        )

    passed = core_ok and new_ok and core_score_ok and quality_ok and user_ok

    promotion_gate_result: dict[str, Any] | None = None
    if bool(getattr(args, "promotion_gate", False)):
        promotion_gate_result = run_promotion_cycle(
            chat_fn=generate_response,
            candidate_model=args.out_model,
            baseline_model=args.baseline_model,
            report_path=str(getattr(args, "promotion_report_path", "") or ""),
        )
        gate_passed = promotion_gate_result.get("promotion_gate", {}).get("passed", False)
        if not gate_passed:
            passed = False

    promoted = False
    if passed and args.promote_to:
        src = Path(args.out_model)
        dst = Path(args.promote_to)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        src_meta = Path(str(src) + ".meta.json")
        dst_meta = Path(str(dst) + ".meta.json")
        if src_meta.exists():
            shutil.copyfile(src_meta, dst_meta)
        promoted = True

    _emit(
        "continual-eval: "
        f"passed={passed} promoted={promoted} "
        f"core_drop={round(core_drop, 4)} new_gain={round(new_gain, 4)} "
        f"user_delta={round(user_delta, 4)}"
    )

    report = {
        "baseline_model": args.baseline_model,
        "candidate_model": args.out_model,
        "promote_to": args.promote_to or None,
        "promoted": promoted,
        "passed": passed,
        "gates": {
            "max_core_drop": float(args.max_core_drop),
            "min_new_gain": float(args.min_new_gain),
            "min_core_score": float(args.min_core_score),
            "min_core_quality": float(args.min_core_quality),
            "min_user_delta": float(args.min_user_delta),
            "min_user_score": required_user_score,
            "min_user_case_score": required_user_case_score,
            "max_user_evasive": allowed_user_evasive,
        },
        "scores": {
            "baseline_core": baseline_core,
            "candidate_core": candidate_core,
            "baseline_new": baseline_new,
            "candidate_new": candidate_new,
            "core_drop": round(core_drop, 4),
            "new_gain": round(new_gain, 4),
            "baseline_user": baseline_user,
            "candidate_user": candidate_user,
            "user_delta": round(user_delta, 4),
            "user_score": round(user_score, 4),
            "user_min_case_score": round(user_min_case_score, 4),
            "user_evasive_count": int(user_evasive_count),
        },
        "mixing": {
            "core_pairs": len(core_pairs),
            "replay_pairs": len(replay_pairs),
            "new_pairs": len(new_pairs),
            "mixed_pairs": len(mixed_pairs),
            "ratios": {
                "core_ratio": float(args.core_ratio),
                "replay_ratio": float(args.replay_ratio),
                "new_ratio": float(args.new_ratio),
            },
        },
        "training_tail": train_tail,
        "promotion_gate": promotion_gate_result,
        "paths": {
            "replay_buffer": args.replay_buffer,
            "mixed_corpus": args.mixed_corpus_out,
            "user_benchmark_file": str(args.user_benchmark_file or ""),
        },
    }

    if args.report_path:
        report_path = Path(args.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    return report


def main():
    parser = argparse.ArgumentParser(description="Continual learning guardrail tools for Ickle.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_update = sub.add_parser("update-replay")
    p_update.add_argument("--source-corpus", required=True)
    p_update.add_argument("--replay-buffer", default="data/continual/replay_buffer.jsonl")
    p_update.add_argument("--max-size", type=int, default=20000)
    p_update.add_argument("--seed", type=int, default=1337)
    p_update.add_argument("--json", action="store_true")

    p_build = sub.add_parser("build-mix")
    p_build.add_argument("--core-corpus", required=True)
    p_build.add_argument("--new-corpus", required=True)
    p_build.add_argument("--replay-buffer", default="data/continual/replay_buffer.jsonl")
    p_build.add_argument("--out", default="data/continual/continual_mix.txt")
    p_build.add_argument("--total-pairs", type=int, default=12000)
    p_build.add_argument("--core-ratio", type=float, default=0.45)
    p_build.add_argument("--replay-ratio", type=float, default=0.35)
    p_build.add_argument("--new-ratio", type=float, default=0.20)
    p_build.add_argument("--seed", type=int, default=1337)
    p_build.add_argument("--json", action="store_true")

    p_run = sub.add_parser("run-step")
    p_run.add_argument("--core-corpus", default="data/ickle_curated_only.txt")
    p_run.add_argument("--new-corpus", default="data/ickle_clean_corpus.txt")
    p_run.add_argument("--replay-buffer", default="data/continual/replay_buffer.jsonl")
    p_run.add_argument("--mixed-corpus-out", default="data/continual/continual_mix.txt")
    p_run.add_argument("--baseline-model", default="models/ickle_clean.pt")
    p_run.add_argument("--out-model", default="models/ickle_continual_candidate.pt")
    p_run.add_argument("--checkpoint-path", default="models/ickle_continual_candidate.pt.checkpoint.pt")
    p_run.add_argument("--promote-to", default="models/ickle_clean.pt")
    p_run.add_argument("--report-path", default="data/continual/guard_step_report.json")
    p_run.add_argument("--steps", type=int, default=1200)
    p_run.add_argument("--lr", type=float, default=8e-6)
    p_run.add_argument("--warmup-steps", type=int, default=80)
    add_resource_pct_args(p_run)
    p_run.add_argument("--replay-max-size", type=int, default=20000)
    p_run.add_argument("--total-pairs", type=int, default=12000)
    p_run.add_argument("--max-core-pairs", type=int, default=6000)
    p_run.add_argument("--max-new-pairs", type=int, default=10000)
    p_run.add_argument("--core-ratio", type=float, default=0.45)
    p_run.add_argument("--replay-ratio", type=float, default=0.35)
    p_run.add_argument("--new-ratio", type=float, default=0.20)
    p_run.add_argument("--eval-core-prompts", type=int, default=16)
    p_run.add_argument("--eval-new-prompts", type=int, default=16)
    p_run.add_argument("--max-core-drop", type=float, default=0.03)
    p_run.add_argument("--min-new-gain", type=float, default=0.00)
    p_run.add_argument("--min-core-score", type=float, default=0.38)
    p_run.add_argument("--min-core-quality", type=float, default=0.35)
    p_run.add_argument("--user-benchmark-file", default="data/maintenance/user_chat_benchmark.json")
    p_run.add_argument("--min-user-delta", type=float, default=0.00)
    p_run.add_argument("--min-user-score", type=float, default=-1.0)
    p_run.add_argument("--min-user-case-score", type=float, default=-1.0)
    p_run.add_argument("--max-user-evasive", type=int, default=-1)
    p_run.add_argument("--seed", type=int, default=1337)
    p_run.add_argument("--promotion-gate", action="store_true", help="Require benchmark suite pass before promotion")
    p_run.add_argument("--promotion-report-path", default="data/continual/promotion_gate_report.json")
    p_run.add_argument("--resume-if-possible", dest="resume_if_possible", action="store_true")
    p_run.add_argument("--no-resume-if-possible", dest="resume_if_possible", action="store_false")
    p_run.set_defaults(resume_if_possible=True)
    p_run.add_argument("--json", action="store_true")

    args = parser.parse_args()

    if args.command == "update-replay":
        replay = load_replay_buffer(args.replay_buffer)
        source_pairs = parse_dialog_pairs(args.source_corpus)
        merged = update_replay_buffer(replay, source_pairs, max_size=max(100, int(args.max_size)), seed=int(args.seed))
        save_replay_buffer(args.replay_buffer, merged)
        report = {
            "source_pairs": len(source_pairs),
            "previous_replay_pairs": len(replay),
            "updated_replay_pairs": len(merged),
            "replay_buffer": args.replay_buffer,
        }
        print(json.dumps(report, indent=2, ensure_ascii=False) if args.json else report)
        return

    if args.command == "build-mix":
        core_pairs = parse_dialog_pairs(args.core_corpus)
        new_pairs = parse_dialog_pairs(args.new_corpus)
        replay_pairs = load_replay_buffer(args.replay_buffer)
        mixed = build_compartment_mixture(
            core_pairs=core_pairs,
            replay_pairs=replay_pairs,
            new_pairs=new_pairs,
            total_pairs=max(100, int(args.total_pairs)),
            core_ratio=float(args.core_ratio),
            replay_ratio=float(args.replay_ratio),
            new_ratio=float(args.new_ratio),
            seed=int(args.seed),
        )
        write_pairs_as_corpus(args.out, mixed)
        report = {
            "core_pairs": len(core_pairs),
            "replay_pairs": len(replay_pairs),
            "new_pairs": len(new_pairs),
            "mixed_pairs": len(mixed),
            "out": args.out,
        }
        print(json.dumps(report, indent=2, ensure_ascii=False) if args.json else report)
        return

    report = run_guarded_step(args)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"passed={report['passed']} promoted={report['promoted']}")
        print(
            "core_drop=",
            report["scores"]["core_drop"],
            "new_gain=",
            report["scores"]["new_gain"],
        )
        print(f"report_path={args.report_path}")


if __name__ == "__main__":
    main()
