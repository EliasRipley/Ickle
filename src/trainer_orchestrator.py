"""Trainer Integration Layer â€” Phase 3: Trainer Operator Mode.

External agents (like OpenCode) run full train/eval/promote loops via a program runner API.

Accepts a JSON training plan with:
- topic selection, corpora list, model path, eval thresholds, promotion target
- Executes as a task graph: collect -> build corpus -> train -> eval -> promote/rollback
- Promotion never overwrites promote_to unless every eval_thresholds gate passes
  (min_avg_score, min_delta, max_regression); on failure with rollback_on_fail set,
  the failed candidate checkpoint is quarantined to data/trainer/rollback/ rather
  than left in place under output_model, so it can't be mistaken for a live candidate

Advanced, CLI-only surface: managed via `python -m src.app trainer-operator`
(see src/trainer_orchestrator_cli.py: submit/run/list/get). There is no web UI
or control-server HTTP endpoint for this -- it is not exposed anywhere under
/api/, unlike the main chat/training control API in serve_control.py.
"""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.resource_defaults import DEFAULT_CHECKPOINT_EVERY, DEFAULT_CPU_PCT, DEFAULT_GPU_PCT, DEFAULT_RAM_PCT


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TrainerProgram:
    """A training plan submitted by an external operator agent."""

    program_id: str = ""
    mode: str = "operator"
    topic: str = ""
    corpus_paths: list[str] = field(default_factory=list)
    model_path: str = "models/ickle_brain_candidate_v2.pt"
    output_model: str = "models/ickle_operator_candidate.pt"
    steps: int = 1200
    cpu_pct: int = DEFAULT_CPU_PCT
    ram_pct: int = DEFAULT_RAM_PCT
    gpu_pct: int = DEFAULT_GPU_PCT
    lr: float = 8e-6
    eval_thresholds: dict[str, float] = field(default_factory=lambda: {
        "min_avg_score": 0.40,
        "min_delta": 0.0,
        "max_regression": 0.03,
    })
    promote_to: str = ""
    rollback_on_fail: bool = True
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainRunStatus:
    run_id: str
    program: TrainerProgram
    status: str = "queued"
    steps: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    error: str = ""


STEP_ORDER = [
    "validate_program",
    "build_corpus",
    "train_model",
    "evaluate_model",
    "promote_or_rollback",
]


class TrainerOperator:
    """Runs training programs as a task graph with checkpoint/rollback."""

    def __init__(
        self,
        runs_dir: str = "data/trainer/runs",
        *,
        build_corpus_fn: Callable | None = None,
        train_fn: Callable | None = None,
        eval_fn: Callable | None = None,
    ):
        self._runs_dir = Path(runs_dir)
        self._runs_dir.mkdir(parents=True, exist_ok=True)
        self._runs: dict[str, TrainRunStatus] = {}
        self._build_corpus_fn = build_corpus_fn
        self._train_fn = train_fn
        self._eval_fn = eval_fn
        self._load_runs()

    def _run_path(self, run_id: str) -> Path:
        return self._runs_dir / f"{run_id}.json"

    def _load_runs(self):
        for p in sorted(self._runs_dir.glob("*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                prog_data = data.get("program", {})
                program = TrainerProgram(
                    program_id=prog_data.get("program_id", ""),
                    mode=prog_data.get("mode", "operator"),
                    topic=prog_data.get("topic", ""),
                    corpus_paths=prog_data.get("corpus_paths", []),
                    model_path=prog_data.get("model_path", ""),
                    output_model=prog_data.get("output_model", ""),
                    steps=prog_data.get("steps", 1200),
                    cpu_pct=prog_data.get("cpu_pct", DEFAULT_CPU_PCT),
                    ram_pct=prog_data.get("ram_pct", DEFAULT_RAM_PCT),
                    gpu_pct=prog_data.get("gpu_pct", DEFAULT_GPU_PCT),
                    lr=prog_data.get("lr", 8e-6),
                    eval_thresholds=prog_data.get("eval_thresholds", {}),
                    promote_to=prog_data.get("promote_to", ""),
                    rollback_on_fail=prog_data.get("rollback_on_fail", True),
                    checkpoint_every=prog_data.get("checkpoint_every", DEFAULT_CHECKPOINT_EVERY),
                    tags=prog_data.get("tags", []),
                    metadata=prog_data.get("metadata", {}),
                )
                run = TrainRunStatus(
                    run_id=data.get("run_id", ""),
                    program=program,
                    status=data.get("status", "unknown"),
                    steps=data.get("steps", {}),
                    result=data.get("result", {}),
                    created_at=data.get("created_at", ""),
                    updated_at=data.get("updated_at", ""),
                    error=data.get("error", ""),
                )
                self._runs[run.run_id] = run
            except Exception:
                pass

    def _save_run(self, run: TrainRunStatus):
        run.updated_at = _utc_now()
        self._runs[run.run_id] = run
        self._run_path(run.run_id).write_text(
            json.dumps({
                "run_id": run.run_id,
                "status": run.status,
                "program": {
                    "program_id": run.program.program_id,
                    "mode": run.program.mode,
                    "topic": run.program.topic,
                    "corpus_paths": run.program.corpus_paths,
                    "model_path": run.program.model_path,
                    "output_model": run.program.output_model,
                    "steps": run.program.steps,
                    "cpu_pct": run.program.cpu_pct,
                    "ram_pct": run.program.ram_pct,
                    "gpu_pct": run.program.gpu_pct,
                    "lr": run.program.lr,
                    "eval_thresholds": run.program.eval_thresholds,
                    "promote_to": run.program.promote_to,
                    "rollback_on_fail": run.program.rollback_on_fail,
                    "checkpoint_every": run.program.checkpoint_every,
                    "tags": run.program.tags,
                    "metadata": run.program.metadata,
                },
                "steps": run.steps,
                "result": run.result,
                "created_at": run.created_at,
                "updated_at": run.updated_at,
                "error": run.error,
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def submit_program(self, program: TrainerProgram) -> TrainRunStatus:
        if not program.program_id:
            program.program_id = uuid.uuid4().hex[:12]
        run = TrainRunStatus(
            run_id=program.program_id,
            program=program,
            status="queued",
        )
        self._save_run(run)
        return run

    def get_run(self, run_id: str) -> TrainRunStatus | None:
        return self._runs.get(run_id)

    def list_runs(self, status: str = "") -> list[dict[str, Any]]:
        runs = sorted(self._runs.values(), key=lambda r: r.created_at, reverse=True)
        if status:
            runs = [r for r in runs if r.status == status]
        return [{
            "run_id": r.run_id,
            "status": r.status,
            "topic": r.program.topic,
            "model_path": r.program.model_path,
            "output_model": r.program.output_model,
            "created_at": r.created_at,
            "updated_at": r.updated_at,
            "error": r.error,
            "tags": r.program.tags,
        } for r in runs]

    def execute_run(
        self,
        run_id: str,
        progress_cb: Callable[[str], None] | None = None,
    ) -> TrainRunStatus:
        run = self._runs.get(run_id)
        if not run:
            raise KeyError(f"Run not found: {run_id}")

        def _emit(msg: str):
            if progress_cb:
                progress_cb(msg)

        try:
            run.status = "running"
            self._save_run(run)

            prog = run.program

            for step_name in STEP_ORDER:
                _emit(f"Step: {step_name}")
                step_result, step_ok = self._execute_step(step_name, prog, _emit)
                run.steps[step_name] = step_result
                if not step_ok:
                    run.status = "failed"
                    run.error = f"Step '{step_name}' failed: {step_result.get('error', 'unknown')}"
                    self._save_run(run)
                    return run

            run.status = "completed"
            run.result = {
                "output_model": prog.output_model,
                "promoted": prog.promote_to if run.steps.get("promote_or_rollback", {}).get("promoted") else False,
                "steps_completed": len(run.steps),
            }
            self._save_run(run)
            _emit(f"Run {run_id} completed")
            return run

        except Exception as exc:
            run.status = "failed"
            run.error = str(exc)
            self._save_run(run)
            return run

    def _execute_step(self, step_name: str, prog: TrainerProgram, emit: Callable[[str], None]) -> tuple[dict[str, Any], bool]:
        if step_name == "validate_program":
            return self._step_validate(prog)
        elif step_name == "build_corpus":
            return self._step_build_corpus(prog, emit)
        elif step_name == "train_model":
            return self._step_train(prog, emit)
        elif step_name == "evaluate_model":
            return self._step_evaluate(prog, emit)
        elif step_name == "promote_or_rollback":
            return self._step_promote_or_rollback(prog)
        return {"error": f"Unknown step: {step_name}"}, False

    def _step_validate(self, prog: TrainerProgram) -> tuple[dict[str, Any], bool]:
        errors: list[str] = []
        if not Path(prog.model_path).exists():
            errors.append(f"Model not found: {prog.model_path}")
        if not prog.corpus_paths:
            errors.append("No corpus paths provided")
        if prog.steps < 10:
            errors.append(f"Steps too low: {prog.steps}")
        if errors:
            return {"errors": errors}, False
        return {"valid": True, "model_path": prog.model_path, "corpus_count": len(prog.corpus_paths)}, True

    def _step_build_corpus(self, prog: TrainerProgram, emit: Callable[[str], None]) -> tuple[dict[str, Any], bool]:
        if self._build_corpus_fn:
            try:
                result = self._build_corpus_fn(prog)
                return result, True
            except Exception as exc:
                return {"error": str(exc)}, False

        merged_path = f"data/trainer/corpus_{prog.program_id}.txt"
        merged = Path(merged_path)
        merged.parent.mkdir(parents=True, exist_ok=True)

        total_lines = 0
        seen: set[str] = set()
        with merged.open("w", encoding="utf-8") as out:
            for cp in prog.corpus_paths:
                if not Path(cp).exists():
                    emit(f"Warning: corpus not found: {cp}")
                    continue
                with open(cp, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line in seen:
                            continue
                        seen.add(line)
                        out.write(line + "\n")
                        total_lines += 1

        return {"corpus_path": merged_path, "lines": total_lines}, total_lines > 10

    def _step_train(self, prog: TrainerProgram, emit: Callable[[str], None]) -> tuple[dict[str, Any], bool]:
        corpus_path = prog.corpus_paths[0] if prog.corpus_paths else ""
        if not corpus_path or not Path(corpus_path).exists():
            return {"error": f"Training corpus not found: {corpus_path}"}, False

        if self._train_fn:
            try:
                result = self._train_fn(prog, emit)
                return result, True
            except Exception as exc:
                return {"error": str(exc)}, False

        from src.train_invoke import build_train_command, run_train_command

        checkpoint_path = f"{prog.output_model}.checkpoint.pt" if prog.checkpoint_every > 0 else ""
        cmd = build_train_command(
            data_path=corpus_path,
            out_model=prog.output_model,
            steps=prog.steps,
            cpu_pct=prog.cpu_pct,
            ram_pct=prog.ram_pct,
            gpu_pct=prog.gpu_pct,
            lr=prog.lr,
            init_model=prog.model_path,
            checkpoint_path=checkpoint_path,
            checkpoint_every=prog.checkpoint_every,
            resume_if_possible=True,
        )

        emit(f"Running: {' '.join(cmd)}")
        try:
            run_train_command(cmd, line_cb=emit, error_label="trainer-operator training")
        except RuntimeError as exc:
            return {"error": str(exc)[-500:]}, False
        return {"output_model": prog.output_model, "steps": prog.steps}, True

    def _step_evaluate(self, prog: TrainerProgram, emit: Callable[[str], None]) -> tuple[dict[str, Any], bool]:
        if self._eval_fn:
            try:
                result = self._eval_fn(prog, emit)
                return result, True
            except Exception as exc:
                return {"error": str(exc)}, False

        from src.promotion_gate import evaluate_model_on_benchmark, PromotionGateConfig
        from src.ilm_chat import extract_response_text, generate_response

        def _chat_fn(args):
            return extract_response_text(generate_response(args))

        baseline_result = evaluate_model_on_benchmark(_chat_fn, prog.model_path, PromotionGateConfig().benchmark_cases)
        candidate_result = evaluate_model_on_benchmark(_chat_fn, prog.output_model, PromotionGateConfig().benchmark_cases)

        baseline_avg = baseline_result["avg_score"]
        candidate_avg = candidate_result["avg_score"]
        delta = candidate_avg - baseline_avg

        min_delta = prog.eval_thresholds.get("min_delta", 0.0)
        min_avg_score = prog.eval_thresholds.get("min_avg_score", 0.0)
        max_regression = prog.eval_thresholds.get("max_regression")

        passed = delta >= min_delta and candidate_avg >= min_avg_score
        if max_regression is not None:
            passed = passed and delta >= -max_regression

        return {
            "baseline_avg": baseline_avg,
            "candidate_avg": candidate_avg,
            "delta": round(delta, 4),
            "passed": passed,
            "baseline_detail": baseline_result,
            "candidate_detail": candidate_result,
        }, True

    def _step_promote_or_rollback(self, prog: TrainerProgram) -> tuple[dict[str, Any], bool]:
        # NOTE: this must not use dict.get(key, TrainRunStatus(...)) -- Python
        # evaluates the default argument eagerly even when the key is found,
        # and TrainRunStatus requires run_id, so that form crashed this step
        # on every single run.
        existing_run = self._runs.get(prog.program_id)
        eval_result = existing_run.steps.get("evaluate_model", {}) if existing_run else {}
        passed = eval_result.get("passed", False)

        if passed and prog.promote_to:
            src = Path(prog.output_model)
            dst = Path(prog.promote_to)
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                from src.model_accumulate import Accumulator
                candidate_score = float(eval_result.get("candidate_avg", 0.0))
                accumulator = Accumulator()
                acc_report = accumulator.try_accumulate(
                    master_path=str(dst),
                    candidate_path=str(src),
                    candidate_score=candidate_score,
                )
                if not acc_report.get("merged"):
                    shutil.copyfile(src, dst)
                src_meta = Path(str(src) + ".meta.json")
                dst_meta = Path(str(dst) + ".meta.json")
                if src_meta.exists():
                    shutil.copyfile(src_meta, dst_meta)
                return {"promoted": True, "promoted_to": prog.promote_to, "accumulated": acc_report.get("merged", False), "accumulation_method": acc_report.get("method", "none")}, True

        if not passed and prog.rollback_on_fail:
            return self._quarantine_candidate(prog)

        return {"promoted": False}, True

    def _quarantine_candidate(self, prog: TrainerProgram) -> tuple[dict[str, Any], bool]:
        """Move a failed candidate (and its checkpoint) out of output_model so
        it can never be mistaken for a live model by a later run or by anyone
        reading model_path/output_model off disk -- this is the actual
        "revert" the rollback_on_fail policy promises, not just a status flag."""
        quarantine_dir = self._runs_dir.parent / "rollback"
        quarantine_dir.mkdir(parents=True, exist_ok=True)

        moved: list[str] = []
        for suffix in ("", ".meta.json", ".checkpoint.pt"):
            src = Path(f"{prog.output_model}{suffix}")
            if not src.exists():
                continue
            dst = quarantine_dir / f"{prog.program_id}_{src.name}"
            shutil.move(str(src), str(dst))
            moved.append(str(dst))

        return {
            "promoted": False,
            "rollback": True,
            "reason": "eval thresholds not met",
            "quarantined_files": moved,
        }, True


def operator_from_dict(data: dict[str, Any]) -> TrainerProgram:
    return TrainerProgram(
        program_id=data.get("program_id", ""),
        mode=data.get("mode", "operator"),
        topic=data.get("topic", ""),
        corpus_paths=data.get("corpus_paths", []),
        model_path=data.get("model_path", "models/ickle_brain_candidate_v2.pt"),
        output_model=data.get("output_model", "models/ickle_operator_candidate.pt"),
        steps=data.get("steps", 1200),
        cpu_pct=data.get("cpu_pct", DEFAULT_CPU_PCT),
        ram_pct=data.get("ram_pct", DEFAULT_RAM_PCT),
        gpu_pct=data.get("gpu_pct", DEFAULT_GPU_PCT),
        lr=data.get("lr", 8e-6),
        eval_thresholds=data.get("eval_thresholds", {}),
        promote_to=data.get("promote_to", ""),
        rollback_on_fail=data.get("rollback_on_fail", True),
        checkpoint_every=data.get("checkpoint_every", DEFAULT_CHECKPOINT_EVERY),
        tags=data.get("tags", []),
        metadata=data.get("metadata", {}),
    )
