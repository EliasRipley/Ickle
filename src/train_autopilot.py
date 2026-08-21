#!/usr/bin/env python3
"""Autopilot training loop that keeps training until chat quality gates pass."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from src.build_smart_corpus import build_smart_corpus
from src.chat_benchmark import _coherence_probe, _load_cases, run_benchmark
from src.resource_defaults import add_resource_pct_args
from src.train_invoke import build_train_command, run_train_command
from src.workspace_paths import get_training_corpus_path, get_training_root


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _checkpoint_step(path: Path) -> int:
    if not path.exists():
        return -1
    try:
        payload = torch.load(str(path), map_location="cpu")
        return int(payload.get("step", -1))
    except Exception:  # noqa: BLE001
        return -1


def _write_corpus(path: Path, pairs: list[Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for pair in pairs:
        lines.append(f"User: {pair.user}")
        lines.append(f"Ickle: {pair.assistant}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_train(
    *,
    corpus_path: str,
    out_model: str,
    checkpoint_path: str,
    init_model: str,
    cpu_pct: int,
    ram_pct: int,
    gpu_pct: int,
    lr: float,
    warmup_steps: int,
    cycle_steps: int,
    spm_vocab_size: int,
) -> dict[str, Any]:
    ckpt_step = _checkpoint_step(Path(checkpoint_path))
    total_steps = max(int(cycle_steps), ckpt_step + 1 + int(cycle_steps))

    cmd = build_train_command(
        data_path=corpus_path,
        out_model=out_model,
        steps=total_steps,
        cpu_pct=cpu_pct,
        ram_pct=ram_pct,
        gpu_pct=gpu_pct,
        lr=lr,
        warmup_steps=warmup_steps,
        init_model=init_model,
        checkpoint_path=checkpoint_path,
        resume_if_possible=True,
        tokenizer="sentencepiece",
        spm_vocab_size=max(512, int(spm_vocab_size)),
        spm_model_type="bpe",
    )

    out_lines = run_train_command(cmd, tail_lines=60, error_label="training")

    return {
        "checkpoint_step_before": ckpt_step,
        "target_total_steps": total_steps,
        "log_tail": out_lines[-40:],
        "command": cmd,
    }


def main():
    parser = argparse.ArgumentParser(description="Autopilot model growth loop for Ickle.")
    parser.add_argument("--training-root", default=str(get_training_root()))
    parser.add_argument("--corpus-out", default=str(get_training_corpus_path("ickle_smart_corpus.txt")))
    parser.add_argument("--corpus-stats-out", default="data/maintenance/smart_corpus_stats.json")
    parser.add_argument("--max-pairs", type=int, default=12000)
    parser.add_argument("--include-legacy-dictionary", action="store_true")
    parser.add_argument("--candidate-model", default="models/ickle_autopilot_candidate.pt")
    parser.add_argument("--checkpoint-path", default="models/ickle_autopilot_candidate.pt.checkpoint.pt")
    parser.add_argument("--baseline-model", default="")
    parser.add_argument("--init-model", default="")
    add_resource_pct_args(parser)
    parser.add_argument("--cycle-steps", type=int, default=700)
    parser.add_argument("--max-cycles", type=int, default=12)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--spm-vocab-size", type=int, default=2048)
    parser.add_argument("--benchmark-file", default="data/maintenance/user_chat_benchmark.json")
    parser.add_argument("--target-score", type=float, default=0.42)
    parser.add_argument("--target-delta", type=float, default=0.05)
    parser.add_argument("--promote-if-pass", action="store_true")
    parser.add_argument("--promote-to-model", default="")
    parser.add_argument("--log-path", default="data/maintenance/train_autopilot_log.jsonl")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if not args.baseline_model:
        from src.model_resolver import resolve_default_model

        try:
            args.baseline_model = resolve_default_model()
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from None
    if not args.promote_to_model:
        from src.model_resolver import resolve_default_model

        try:
            args.promote_to_model = resolve_default_model()
        except FileNotFoundError:
            pass

    training_root = Path(args.training_root)
    corpus_out = Path(args.corpus_out)
    corpus_stats_out = Path(args.corpus_stats_out)
    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    bench_cases = _load_cases(Path(args.benchmark_file))
    if not bench_cases:
        raise SystemExit(f"No benchmark cases found in {args.benchmark_file}")

    cycle_reports: list[dict[str, Any]] = []
    passed = False
    promoted = False

    for cycle in range(1, max(1, int(args.max_cycles)) + 1):
        pairs, corpus_stats = build_smart_corpus(
            training_root=training_root,
            max_pairs=max(2000, int(args.max_pairs)),
            seed=int(args.seed) + cycle,
            include_legacy_dictionary=bool(args.include_legacy_dictionary),
        )
        _write_corpus(corpus_out, pairs)
        corpus_stats_out.parent.mkdir(parents=True, exist_ok=True)
        corpus_stats_out.write_text(json.dumps(corpus_stats, indent=2, ensure_ascii=False), encoding="utf-8")

        train_report = _run_train(
            corpus_path=str(corpus_out),
            out_model=str(args.candidate_model),
            checkpoint_path=str(args.checkpoint_path),
            init_model=str(args.init_model or ""),
            cpu_pct=int(args.cpu_pct),
            ram_pct=int(args.ram_pct),
            gpu_pct=int(args.gpu_pct),
            lr=float(args.lr),
            warmup_steps=int(args.warmup_steps),
            cycle_steps=int(args.cycle_steps),
            spm_vocab_size=int(args.spm_vocab_size),
        )

        candidate_eval = run_benchmark(
            model=str(args.candidate_model),
            cases=bench_cases,
            enable_memory=False,
            enable_web_tools=False,
        )
        baseline_eval = run_benchmark(
            model=str(args.baseline_model),
            cases=bench_cases,
            enable_memory=False,
            enable_web_tools=False,
        )
        delta = float(candidate_eval["avg_score"]) - float(baseline_eval["avg_score"])

        probe = _coherence_probe(str(args.candidate_model))
        passed = (
            float(candidate_eval["avg_score"]) >= float(args.target_score)
            and delta >= float(args.target_delta)
            and bool(probe["passed"])
        )

        cycle_report = {
            "timestamp_utc": _utc_now(),
            "cycle": cycle,
            "passed": passed,
            "candidate_avg_score": candidate_eval["avg_score"],
            "baseline_avg_score": baseline_eval["avg_score"],
            "delta": round(delta, 4),
            "coherence_probe": probe,
            "corpus_stats": corpus_stats,
            "train": train_report,
        }
        cycle_reports.append(cycle_report)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(cycle_report, ensure_ascii=False) + "\n")

        if passed:
            break

    if passed and bool(args.promote_if_pass):
        src = Path(str(args.candidate_model))
        dst = Path(str(args.promote_to_model))
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        src_meta = Path(str(src) + ".meta.json")
        dst_meta = Path(str(dst) + ".meta.json")
        if src_meta.exists():
            shutil.copyfile(src_meta, dst_meta)
        promoted = True

    final = {
        "passed": passed,
        "promoted": promoted,
        "candidate_model": str(args.candidate_model),
        "baseline_model": str(args.baseline_model),
        "promote_to_model": str(args.promote_to_model),
        "cycles_ran": len(cycle_reports),
        "last_cycle": cycle_reports[-1] if cycle_reports else None,
        "log_path": str(log_path),
        "corpus_out": str(corpus_out),
        "corpus_stats_out": str(corpus_stats_out),
    }

    if args.json:
        print(json.dumps(final, indent=2, ensure_ascii=False))
    else:
        print(
            f"passed={final['passed']} promoted={final['promoted']} "
            f"cycles={final['cycles_ran']} log={final['log_path']}"
        )


if __name__ == "__main__":
    main()
