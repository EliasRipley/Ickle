"""Shared helper for invoking `python -m src.train` as a subprocess.

Every training-loop caller (continual_guard.py, train_autopilot.py,
train_intelligence_stack.py, train_cycle.py, trainer_orchestrator.py) used to
hand-build this argv list independently. They had drifted apart on which
flags they passed -- some supported --resume-from-checkpoint, some (notably
trainer_orchestrator.py) didn't, meaning a crashed or interrupted run through
that path always restarted from scratch instead of resuming. Centralizing
the flag-building here means a fix (or a new flag) only needs to land once.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Callable

from src.resource_defaults import DEFAULT_CHECKPOINT_EVERY, DEFAULT_CPU_PCT, DEFAULT_GPU_PCT, DEFAULT_RAM_PCT


def build_train_command(
    *,
    data_path: str,
    out_model: str,
    steps: int,
    cpu_pct: int = DEFAULT_CPU_PCT,
    ram_pct: int = DEFAULT_RAM_PCT,
    gpu_pct: int = DEFAULT_GPU_PCT,
    lr: float | None = None,
    warmup_steps: int | None = None,
    init_model: str = "",
    checkpoint_path: str = "",
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
    resume_if_possible: bool = True,
    torch_threads: int = 0,
    batch_size: int = 0,
    grad_accum_steps: int = 1,
    tokenizer: str = "",
    spm_vocab_size: int = 0,
    spm_model_type: str = "",
    extra_args: list[str] | None = None,
) -> list[str]:
    """Build the argv for a `src.train` subprocess invocation.

    Resume-vs-init precedence: if a checkpoint exists at checkpoint_path and
    resume_if_possible is set, --resume-from-checkpoint wins over
    --init-model (matches src/train.py's own precedence). Otherwise falls
    back to --init-model, then to a fresh --tokenizer/--spm-* setup for a
    from-scratch run.
    """
    cmd = [
        sys.executable, "-u", "-m", "src.train",
        "--data", str(data_path),
        "--out", str(out_model),
        "--steps", str(max(10, int(steps))),
        "--cpu-pct", str(int(cpu_pct)),
        "--ram-pct", str(int(ram_pct)),
        "--gpu-pct", str(int(gpu_pct)),
    ]
    if lr is not None:
        cmd.extend(["--lr", str(max(1e-7, float(lr)))])
    if warmup_steps is not None:
        cmd.extend(["--warmup-steps", str(max(0, int(warmup_steps)))])
    if checkpoint_path:
        cmd.extend(["--checkpoint-every", str(max(1, int(checkpoint_every)))])
        cmd.extend(["--checkpoint-path", str(checkpoint_path)])
    if int(torch_threads) > 0:
        cmd.extend(["--torch-threads", str(int(torch_threads))])
    if int(batch_size) > 0:
        cmd.extend(["--batch-size", str(int(batch_size))])
    if int(grad_accum_steps) > 1:
        cmd.extend(["--grad-accum-steps", str(int(grad_accum_steps))])

    checkpoint_exists = bool(checkpoint_path and Path(checkpoint_path).exists())
    if resume_if_possible and checkpoint_exists:
        cmd.extend(["--resume-from-checkpoint", str(checkpoint_path)])
    elif init_model:
        cmd.extend(["--init-model", str(init_model)])
    elif tokenizer:
        cmd.extend(["--tokenizer", tokenizer])
        if spm_vocab_size:
            cmd.extend(["--spm-vocab-size", str(max(1, int(spm_vocab_size)))])
        if spm_model_type:
            cmd.extend(["--spm-model-type", spm_model_type])

    if extra_args:
        cmd.extend(extra_args)
    return cmd


def run_train_command(
    cmd: list[str],
    *,
    line_cb: Callable[[str], None] | None = None,
    tail_lines: int = 60,
    error_label: str = "training",
) -> list[str]:
    """Run a training subprocess to completion, optionally streaming each
    output line to line_cb as it arrives. Returns the last `tail_lines`
    non-blank lines of combined stdout/stderr. Raises RuntimeError on
    non-zero exit. If interrupted (e.g. Ctrl+C propagating as a raised
    exception while waiting), terminates -- and if needed kills -- the
    subprocess rather than leaving an orphaned training process running."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    out_lines: list[str] = []
    try:
        if proc.stdout is not None:
            for raw_line in proc.stdout:
                line = raw_line.strip()
                if not line:
                    continue
                out_lines.append(line)
                if line_cb:
                    line_cb(line)
        proc.wait()
    except BaseException:
        try:
            proc.terminate()
        except Exception:  # noqa: BLE001
            pass
        try:
            proc.wait(timeout=3)
        except Exception:  # noqa: BLE001
            try:
                proc.kill()
                proc.wait(timeout=3)
            except Exception:  # noqa: BLE001
                pass
        raise
    finally:
        if proc.stdout is not None:
            proc.stdout.close()

    if proc.returncode != 0:
        excerpt = "\n".join(out_lines[-tail_lines:])
        raise RuntimeError(f"{error_label} failed (exit {proc.returncode})\n{excerpt}")
    return out_lines[-tail_lines:]
