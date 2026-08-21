#!/usr/bin/env python3
"""Run repeated build/train/smoke-test cycles for Ickle."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from src.build_clean_corpus import build_corpus
from src.ilm_chat import extract_response_text, generate_response
from src.resource_defaults import add_resource_pct_args
from src.train_invoke import build_train_command, run_train_command
from src.workspace_paths import get_training_root


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_corpus(out_path: Path, training_root: Path, max_lines: int) -> dict[str, int]:
    lines, stats = build_corpus(training_root=training_root, max_lines=max_lines)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return stats


def _smoke_test(model_path: str) -> list[dict[str, str]]:
    prompts = [
        "Hello Ickle.",
        "Remember that I prefer concise coding explanations.",
        "What do you remember about my coding preference?",
        "Can you help me break down a coding task into steps?",
    ]
    rows: list[dict[str, str]] = []
    for prompt in prompts:
        args = SimpleNamespace(
            model=model_path,
            prompt=prompt,
            max_new=200,
            max_new_limit=500,
            temperature=0.55,
            top_k=40,
            torch_threads=4,
            skill="",
        )
        response = generate_response(args)
        rows.append({"prompt": prompt, "response": extract_response_text(response)})
    return rows


def main():
    parser = argparse.ArgumentParser(description="Repeated training/evaluation cycle for Ickle.")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--sleep-minutes", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=1200)
    add_resource_pct_args(parser)
    parser.add_argument("--training-root", default=str(get_training_root()))
    parser.add_argument("--max-lines", type=int, default=20000)
    parser.add_argument("--corpus-out", default="data/ickle_clean_corpus.txt")
    parser.add_argument("--model-prefix", default="models/ickle_cycle")
    parser.add_argument("--latest-model", default="models/ickle_brain_candidate_v2.pt")
    parser.add_argument("--log-path", default="data/train_cycle_log.jsonl")
    args = parser.parse_args()

    training_root = Path(args.training_root)
    corpus_out = Path(args.corpus_out)
    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    for cycle in range(1, args.iterations + 1):
        cycle_model = Path(f"{args.model_prefix}_{cycle:03d}.pt")
        stats = _write_corpus(corpus_out, training_root=training_root, max_lines=max(2000, args.max_lines))

        cmd = build_train_command(
            data_path=str(corpus_out),
            out_model=str(cycle_model),
            steps=args.steps,
            cpu_pct=args.cpu_pct,
            ram_pct=args.ram_pct,
            gpu_pct=args.gpu_pct,
            bootstrap_english=True,
        )
        run_train_command(cmd, error_label="train-cycle")

        latest = Path(args.latest_model)
        latest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cycle_model, latest)
        meta_src = Path(str(cycle_model) + ".meta.json")
        meta_dst = Path(str(latest) + ".meta.json")
        if meta_src.exists():
            shutil.copyfile(meta_src, meta_dst)

        smoke_rows = _smoke_test(str(latest))
        log_entry = {
            "timestamp": _utc_now(),
            "cycle": cycle,
            "model_path": str(cycle_model),
            "latest_model": str(latest),
            "corpus_stats": stats,
            "smoke_test": smoke_rows,
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        print(f"cycle {cycle} complete -> {cycle_model}")
        if cycle < args.iterations and args.sleep_minutes > 0:
            time.sleep(args.sleep_minutes * 60.0)


if __name__ == "__main__":
    main()
