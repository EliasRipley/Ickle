"""Shared resource-budget defaults for training/inference CLI scripts.

DEFAULT_CPU_PCT / DEFAULT_RAM_PCT / DEFAULT_GPU_PCT (all 80) and
DEFAULT_CHECKPOINT_EVERY (100) used to be bare literals repeated across a
dozen argparse blocks, several dataclasses, and payload.get() fallbacks in
task_actions.py. Centralizing them means changing a default is a one-line
edit instead of a grep-and-replace across the whole training surface, and
`add_resource_pct_args` collapses the three-line --cpu-pct/--ram-pct/--gpu-pct
argparse boilerplate (duplicated identically in ~10 files) into one call.
"""

from __future__ import annotations

DEFAULT_CPU_PCT = 80
DEFAULT_RAM_PCT = 80
DEFAULT_GPU_PCT = 80
DEFAULT_CHECKPOINT_EVERY = 100


def add_resource_pct_args(
    parser,
    *,
    cpu_default: int = DEFAULT_CPU_PCT,
    ram_default: int = DEFAULT_RAM_PCT,
    gpu_default: int = DEFAULT_GPU_PCT,
) -> None:
    """Add the standard --cpu-pct/--ram-pct/--gpu-pct trio to an argparse parser."""
    parser.add_argument("--cpu-pct", type=int, default=cpu_default, help="Percentage of CPU cores to use (10-100)")
    parser.add_argument("--ram-pct", type=int, default=ram_default, help="Percentage of RAM to use (10-100)")
    parser.add_argument("--gpu-pct", type=int, default=gpu_default, help="Percentage of GPU to use (0-100, 0 = CPU-only)")
