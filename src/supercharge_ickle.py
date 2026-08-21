from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.build_clean_corpus import build_corpus
from src.build_honest_context_package import build_honest_context_package
from src.build_smart_corpus import build_smart_corpus
from src.ilm_memory import get_memory
from src.open_dataset_ingest import check_training_internet
from src.resource_defaults import DEFAULT_CPU_PCT, DEFAULT_GPU_PCT, DEFAULT_RAM_PCT
from src.sanitize_training_data import run_sanitize_training_data
from src.train_invoke import build_train_command, run_train_command
from src.workspace_paths import get_training_root


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_smart_corpus(path: Path, pairs: list[Any]):
    lines: list[str] = []
    for pair in pairs:
        lines.append(f"User: {pair.user}")
        lines.append(f"Ickle: {pair.assistant}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_quick_train(
    *,
    data_path: Path,
    model_in: str,
    model_out: str,
    steps: int,
    cpu_pct: int,
    ram_pct: int,
    gpu_pct: int,
) -> dict[str, Any]:
    cmd = build_train_command(
        data_path=str(data_path),
        out_model=str(model_out),
        steps=max(20, int(steps)),
        cpu_pct=cpu_pct,
        ram_pct=ram_pct,
        gpu_pct=gpu_pct,
        init_model=model_in,
    )
    lines = run_train_command(cmd, error_label="quick train")
    return {"command": cmd, "log_tail": lines[-40:], "out_model": model_out}


def run_supercharge(
    *,
    training_root: Path,
    apply_sanitize: bool,
    clear_memory: bool,
    check_internet: bool,
    build_clean: bool,
    build_smart: bool,
    build_honest: bool,
    quick_train_steps: int,
    quick_train_model_in: str,
    quick_train_model_out: str,
    quick_train_cpu_pct: int,
    quick_train_ram_pct: int,
    quick_train_gpu_pct: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "timestamp_utc": _utc_now(),
        "training_root": str(training_root),
    }

    if check_internet:
        report["internet_status"] = check_training_internet()

    report["sanitize_training"] = run_sanitize_training_data(
        training_root=training_root,
        apply=bool(apply_sanitize),
        drop_qa_templates=True,
        include_files=None,
    )

    if clear_memory:
        memory = get_memory()
        report["memory_prune"] = memory.prune_nonsense(
            clear_short_term=True,
            min_fact_confidence=0.45,
            min_research_confidence=0.5,
        )

    if build_clean:
        clean_lines, clean_stats = build_corpus(
            training_root=training_root,
            max_lines=22000,
            dictionary_items=0,
            include_open_stream=True,
        )
        clean_out = Path("data/ickle_clean_corpus_supercharged.txt")
        clean_out.write_text("\n".join(clean_lines) + "\n", encoding="utf-8")
        report["clean_corpus"] = {"out_path": str(clean_out.resolve()), "stats": clean_stats}

    if build_smart:
        smart_pairs, smart_stats = build_smart_corpus(
            training_root=training_root,
            max_pairs=22000,
            seed=1337,
            include_legacy_dictionary=False,
            include_topic_queues=True,
        )
        smart_out = Path("data/ickle_smart_corpus_supercharged.txt")
        _write_smart_corpus(smart_out, smart_pairs)
        report["smart_corpus"] = {"out_path": str(smart_out.resolve()), "stats": smart_stats}

    if build_honest:
        honest_report = build_honest_context_package(
            training_root=training_root,
            out_dir=Path("data/training_packages/honest_context_v2"),
            max_source_pairs=2200,
            target_sft_pairs=1300,
            seed=2026,
            include_open_streams=True,
        )
        report["honest_context_package"] = honest_report

    if quick_train_steps > 0:
        data_path = Path("data/ickle_smart_corpus_supercharged.txt")
        if not data_path.exists():
            raise FileNotFoundError(f"quick train data not found: {data_path}")
        report["quick_train"] = _run_quick_train(
            data_path=data_path,
            model_in=quick_train_model_in,
            model_out=quick_train_model_out,
            steps=quick_train_steps,
            cpu_pct=quick_train_cpu_pct,
            ram_pct=quick_train_ram_pct,
            gpu_pct=quick_train_gpu_pct,
        )

    return report


def main():
    parser = argparse.ArgumentParser(description="Supercharge Ickle: sanitize data, prune memory, and build stronger corpora.")
    parser.add_argument("--training-root", default=str(get_training_root()))
    parser.add_argument("--apply-sanitize", action="store_true")
    parser.add_argument("--clear-memory", action="store_true")
    parser.add_argument("--check-internet", action="store_true")
    parser.add_argument("--skip-clean", action="store_true")
    parser.add_argument("--skip-smart", action="store_true")
    parser.add_argument("--skip-honest", action="store_true")
    parser.add_argument("--quick-train-steps", type=int, default=0)
    parser.add_argument("--quick-train-model-in", default="")
    parser.add_argument("--quick-train-model-out", default="models/ickle_supercharged_candidate.pt")
    parser.add_argument("--quick-train-cpu-pct", type=int, default=DEFAULT_CPU_PCT, help="Percentage of CPU cores to use (10-100)")
    parser.add_argument("--quick-train-ram-pct", type=int, default=DEFAULT_RAM_PCT, help="Percentage of RAM to use (10-100)")
    parser.add_argument("--quick-train-gpu-pct", type=int, default=DEFAULT_GPU_PCT, help="Percentage of GPU to use (0-100, 0 = CPU-only)")
    parser.add_argument("--report-out", default="data/maintenance/supercharge_report.json")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if not args.quick_train_model_in:
        from src.model_resolver import resolve_default_model

        try:
            args.quick_train_model_in = resolve_default_model()
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from None

    report = run_supercharge(
        training_root=Path(args.training_root).resolve(),
        apply_sanitize=bool(args.apply_sanitize),
        clear_memory=bool(args.clear_memory),
        check_internet=bool(args.check_internet),
        build_clean=not bool(args.skip_clean),
        build_smart=not bool(args.skip_smart),
        build_honest=not bool(args.skip_honest),
        quick_train_steps=int(args.quick_train_steps),
        quick_train_model_in=str(args.quick_train_model_in),
        quick_train_model_out=str(args.quick_train_model_out),
        quick_train_cpu_pct=int(args.quick_train_cpu_pct),
        quick_train_ram_pct=int(args.quick_train_ram_pct),
        quick_train_gpu_pct=int(args.quick_train_gpu_pct),
    )

    report_path = Path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report["report_out"] = str(report_path.resolve())

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return
    print(f"supercharge complete report={report['report_out']}")


if __name__ == "__main__":
    main()

