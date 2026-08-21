"""Deprecated: superseded by continual_guard.py.

This module's row-count-threshold trigger has no replay buffer, compartment
mixing, or promotion gating -- continual_guard.py does all of that and should
be used instead (`python -m src.app continual-guard`). Kept for backward
compatibility with existing automation that still calls `continual-learn`
directly; not recommended for new use.
"""

import argparse
import json
import sys
from pathlib import Path
import subprocess

from src.autodidact import build_python_autodidact_corpus
from src.resource_defaults import add_resource_pct_args


def load_state(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {"trained_rows": 0}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_state(path: str, payload: dict):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def should_trigger_retrain(previous_rows: int, current_rows: int, min_new_rows: int) -> bool:
    return current_rows - previous_rows >= min_new_rows


def run_training(corpus_path: str, out_model: str, steps: int, cpu_pct: int, ram_pct: int, gpu_pct: int):
    cmd = [
        "python",
        "-m",
        "src.train",
        "--data",
        corpus_path,
        "--out",
        out_model,
        "--steps",
        str(steps),
        "--cpu-pct",
        str(cpu_pct),
        "--ram-pct",
        str(ram_pct),
        "--gpu-pct",
        str(gpu_pct),
        "--bootstrap-english",
    ]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Autonomous continual-learning loop for local ILM")
    parser.add_argument("--attempt-log", default="data/python_attempts.jsonl")
    parser.add_argument("--corpus", default="data/python_autodidact_corpus.txt")
    parser.add_argument("--state", default="data/continual_state.json")
    parser.add_argument("--out-model", default="models/tiny_continual.pt")
    parser.add_argument("--min-new-rows", type=int, default=25)
    parser.add_argument("--steps", type=int, default=1200)
    add_resource_pct_args(parser)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(
        "Warning: continual-learn is deprecated (no replay buffer, compartment "
        "mixing, or promotion gating). Use `continual-guard` instead.",
        file=sys.stderr,
    )

    current_rows = build_python_autodidact_corpus(args.attempt_log, args.corpus)
    state = load_state(args.state)
    previous_rows = int(state.get("trained_rows", 0))

    if not should_trigger_retrain(previous_rows, current_rows, args.min_new_rows):
        print(
            f"No retrain triggered: previous_rows={previous_rows}, current_rows={current_rows}, min_new_rows={args.min_new_rows}"
        )
        return

    if args.dry_run:
        print("Retrain would be triggered now (dry-run).")
        return

    run_training(args.corpus, args.out_model, args.steps, args.cpu_pct, args.ram_pct, args.gpu_pct)
    save_state(args.state, {"trained_rows": current_rows, "last_model": args.out_model})
    print(f"Continual retrain complete. Updated model: {args.out_model}")


if __name__ == "__main__":
    main()
