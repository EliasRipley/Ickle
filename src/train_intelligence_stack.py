#!/usr/bin/env python3
"""Two-stage training stack: base LM pretraining + instruction refinement."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.build_base_lm_corpus import build_base_corpus
from src.build_smart_corpus import build_smart_corpus
from src.chat_benchmark import _coherence_probe, _load_cases, run_benchmark
from src.open_dataset_ingest import check_training_internet
from src.resource_defaults import add_resource_pct_args
from src.train_invoke import build_train_command, run_train_command
from src.workspace_paths import get_training_corpus_path, get_training_root


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_smart_corpus(path: Path, pairs: list[Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for pair in pairs:
        lines.append(f"User: {pair.user}")
        lines.append(f"Ickle: {pair.assistant}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Train Ickle with a two-stage intelligence stack.")
    parser.add_argument("--training-root", default=str(get_training_root()))
    parser.add_argument("--base-corpus-out", default=str(get_training_root() / "base_lm_corpus.txt"))
    parser.add_argument("--smart-corpus-out", default=str(get_training_corpus_path("ickle_smart_corpus.txt")))
    parser.add_argument("--smart-corpus-max-pairs", type=int, default=12000)
    parser.add_argument("--base-model-out", default="models/ickle_brain_base.pt")
    parser.add_argument("--base-checkpoint", default="models/ickle_brain_base.pt.checkpoint.pt")
    parser.add_argument("--brain-model-out", default="models/ickle_brain_candidate.pt")
    parser.add_argument("--brain-checkpoint", default="models/ickle_brain_candidate.pt.checkpoint.pt")
    parser.add_argument("--baseline-model", default="")
    add_resource_pct_args(parser)
    parser.add_argument("--base-steps", type=int, default=2600)
    parser.add_argument("--brain-steps", type=int, default=1200)
    parser.add_argument("--base-lr", type=float, default=2.5e-4)
    parser.add_argument("--brain-lr", type=float, default=8e-5)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--spm-vocab-size", type=int, default=3072)
    parser.add_argument("--max-local-records", type=int, default=25000)
    parser.add_argument("--max-wikitext", type=int, default=25000)
    parser.add_argument("--max-fineweb-edu", type=int, default=8000)
    parser.add_argument("--benchmark-file", default="data/maintenance/user_chat_benchmark.json")
    parser.add_argument("--target-score", type=float, default=0.42)
    parser.add_argument("--target-delta", type=float, default=0.05)
    parser.add_argument("--promote-if-pass", action="store_true")
    parser.add_argument("--promote-to-model", default="")
    parser.add_argument("--report-out", default="data/maintenance/intelligence_stack_report.json")
    parser.add_argument("--check-internet", action="store_true")
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
    base_corpus_out = Path(args.base_corpus_out)
    smart_corpus_out = Path(args.smart_corpus_out)
    internet_status = check_training_internet() if bool(args.check_internet) else None

    base_corpus_report = build_base_corpus(
        out_path=base_corpus_out,
        training_root=training_root,
        max_local_records=max(0, int(args.max_local_records)),
        max_wikitext=max(0, int(args.max_wikitext)),
        max_fineweb_edu=max(0, int(args.max_fineweb_edu)),
        max_chars=900,
    )

    base_cmd = build_train_command(
        data_path=str(base_corpus_out),
        out_model=str(args.base_model_out),
        steps=max(200, int(args.base_steps)),
        cpu_pct=args.cpu_pct,
        ram_pct=args.ram_pct,
        gpu_pct=args.gpu_pct,
        lr=max(1e-7, float(args.base_lr)),
        warmup_steps=max(0, int(args.warmup_steps)),
        checkpoint_path=str(args.base_checkpoint),
        resume_if_possible=True,
        tokenizer="sentencepiece",
        spm_vocab_size=max(1024, int(args.spm_vocab_size)),
        spm_model_type="bpe",
    )
    base_train_tail = run_train_command(base_cmd, error_label="base-stage training")

    smart_pairs, smart_stats = build_smart_corpus(
        training_root=training_root,
        max_pairs=max(2000, int(args.smart_corpus_max_pairs)),
        seed=1337,
        include_legacy_dictionary=False,
    )
    _write_smart_corpus(smart_corpus_out, smart_pairs)

    brain_cmd = build_train_command(
        data_path=str(smart_corpus_out),
        out_model=str(args.brain_model_out),
        steps=max(200, int(args.brain_steps)),
        cpu_pct=args.cpu_pct,
        ram_pct=args.ram_pct,
        gpu_pct=args.gpu_pct,
        lr=max(1e-7, float(args.brain_lr)),
        warmup_steps=max(0, int(args.warmup_steps)),
        init_model=str(args.base_model_out),
        checkpoint_path=str(args.brain_checkpoint),
        resume_if_possible=True,
    )
    brain_train_tail = run_train_command(brain_cmd, error_label="brain-stage training")

    bench_cases = _load_cases(Path(args.benchmark_file))
    if not bench_cases:
        raise SystemExit(f"No benchmark cases found in {args.benchmark_file}")
    candidate_eval = run_benchmark(
        model=str(args.brain_model_out),
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
    probe = _coherence_probe(str(args.brain_model_out))

    passed = (
        float(candidate_eval["avg_score"]) >= float(args.target_score)
        and delta >= float(args.target_delta)
        and bool(probe["passed"])
    )
    promoted = False
    if passed and bool(args.promote_if_pass):
        src = Path(str(args.brain_model_out))
        dst = Path(str(args.promote_to_model))
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        src_meta = Path(str(src) + ".meta.json")
        dst_meta = Path(str(dst) + ".meta.json")
        if src_meta.exists():
            shutil.copyfile(src_meta, dst_meta)
        promoted = True

    report = {
        "timestamp_utc": _utc_now(),
        "passed": passed,
        "promoted": promoted,
        "candidate_model": str(args.brain_model_out),
        "baseline_model": str(args.baseline_model),
        "promote_to_model": str(args.promote_to_model),
        "base_corpus": base_corpus_report,
        "smart_corpus_stats": smart_stats,
        "train": {
            "base_cmd": base_cmd,
            "base_tail": base_train_tail,
            "brain_cmd": brain_cmd,
            "brain_tail": brain_train_tail,
        },
        "evaluation": {
            "candidate_avg_score": candidate_eval["avg_score"],
            "baseline_avg_score": baseline_eval["avg_score"],
            "delta": round(delta, 4),
            "coherence_probe": probe,
        },
    }
    if internet_status is not None:
        report["internet_status"] = internet_status

    report_out = Path(args.report_out)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(
            f"passed={passed} promoted={promoted} "
            f"candidate={candidate_eval['avg_score']:.4f} baseline={baseline_eval['avg_score']:.4f} "
            f"delta={delta:.4f}"
        )


if __name__ == "__main__":
    main()
