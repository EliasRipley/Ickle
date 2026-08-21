import os
from dataclasses import dataclass

from src.resource_defaults import DEFAULT_CPU_PCT, DEFAULT_GPU_PCT, DEFAULT_RAM_PCT, add_resource_pct_args


@dataclass
class ResourceConfig:
    cpu_cores: int
    ram_gb: float
    gpu_available: bool
    cpu_percent: int = DEFAULT_CPU_PCT
    ram_percent: int = DEFAULT_RAM_PCT
    gpu_percent: int = DEFAULT_GPU_PCT

    @property
    def effective_cores(self) -> int:
        return max(1, int(self.cpu_cores * self.cpu_percent / 100))

    @property
    def effective_ram_gb(self) -> float:
        return max(0.5, self.ram_gb * self.ram_percent / 100)

    @property
    def torch_threads(self) -> int:
        return max(1, min(self.effective_cores, 16))

    @property
    def block_size(self) -> int:
        cores = self.effective_cores
        ram = self.effective_ram_gb
        base = max(256, min(2048, int(256 + ram * 48 + cores * 16)))
        return (base // 64) * 64

    @property
    def n_embd(self) -> int:
        cores = self.effective_cores
        ram = self.effective_ram_gb
        base = max(64, min(768, int(48 + ram * 32 + cores * 8)))
        return (base // 16) * 16

    @property
    def n_head(self) -> int:
        embd = self.n_embd
        possible = [h for h in [2, 4, 6, 8, 12, 16] if embd % h == 0]
        return possible[len(possible) // 2] if possible else max(2, embd // 16)

    @property
    def n_layer(self) -> int:
        cores = self.effective_cores
        ram = self.effective_ram_gb
        return max(2, min(16, int(2 + cores * 0.5 + ram * 0.3)))

    @property
    def batch_size(self) -> int:
        cores = self.effective_cores
        ram = self.effective_ram_gb
        if self.gpu_available and self.gpu_percent > 0:
            return max(8, min(64, int(cores * 2 + ram * 2)))
        return max(4, min(32, int(cores * 1.5 + ram)))

    def summary(self) -> str:
        lines = [
            f"Resource config ({self.cpu_percent}% CPU / {self.ram_percent}% RAM / {self.gpu_percent}% GPU):",
            f"  CPU: {self.cpu_cores} cores -> {self.effective_cores} effective ({self.torch_threads} threads)",
            f"  RAM: {self.ram_gb:.1f} GB -> {self.effective_ram_gb:.1f} GB effective",
            f"  GPU: {'available' if self.gpu_available else 'none'}",
            f"  Model: block={self.block_size} embd={self.n_embd} heads={self.n_head} layers={self.n_layer} batch={self.batch_size}",
        ]
        return "\n".join(lines)


def detect_resources() -> ResourceConfig:
    cpu = os.cpu_count() or 4
    ram = 2.0
    try:
        import psutil
        ram = psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        pass
    gpu = False
    try:
        from src.device_bridge import is_accelerator_available
        gpu = is_accelerator_available()
    except Exception:
        pass
    return ResourceConfig(cpu_cores=cpu, ram_gb=ram, gpu_available=gpu)


def resource_arg_parser(parser, defaults: bool = True):
    if defaults:
        add_resource_pct_args(parser)
    else:
        parser.add_argument("--cpu-pct", type=int, default=0, help="Percentage of CPU cores to use (0 = detect)")
        parser.add_argument("--ram-pct", type=int, default=0, help="Percentage of RAM to use (0 = detect)")
        parser.add_argument("--gpu-pct", type=int, default=0, help="Percentage of GPU to use (0 = detect)")
    parser.add_argument("--block-size", type=int, default=0, help="Override model block size (0 = auto)")
    parser.add_argument("--n-embd", type=int, default=0, help="Override model embedding dim (0 = auto)")
    parser.add_argument("--n-head", type=int, default=0, help="Override model attention heads (0 = auto)")
    parser.add_argument("--n-layer", type=int, default=0, help="Override model layers (0 = auto)")
    parser.add_argument("--batch-size", type=int, default=0, help="Override batch size (0 = auto)")
    parser.add_argument("--torch-threads", type=int, default=0, help="Override torch thread count (0 = auto)")


def resolve_resource_config(args) -> ResourceConfig:
    rc = detect_resources()
    cpu_pct = getattr(args, "cpu_pct", None) if args else None
    ram_pct = getattr(args, "ram_pct", None) if args else None
    gpu_pct = getattr(args, "gpu_pct", None) if args else None
    if cpu_pct is not None and 10 <= cpu_pct <= 100:
        rc.cpu_percent = cpu_pct
    if ram_pct is not None and 10 <= ram_pct <= 100:
        rc.ram_percent = ram_pct
    if gpu_pct is not None and 0 <= gpu_pct <= 100:
        rc.gpu_percent = gpu_pct
    return rc


def apply_cpu_thread_budget(n_threads: int):
    """Bound *every* CPU-bound library to the same thread budget, not just torch.

    `torch.set_num_threads()` alone only governs torch's own intra-op pool.
    NumPy/BLAS-backed code elsewhere (tokenizer merges, federated aggregation,
    psutil-adjacent numeric work) reads OMP_NUM_THREADS/MKL_NUM_THREADS/
    OPENBLAS_NUM_THREADS instead, and those default to *all* logical cores if
    unset. That silently breaks the `--cpu-pct` promise ("leave some CPU free
    for the rest of the machine") for everything except torch matmuls, and
    causes oversubscription when several thread pools each grab every core at
    once (worst during concurrent inference-serving, where multiple requests
    run on a shared process). Setting these before the first parallel region
    runs keeps every backend inside the same budget.
    """
    n = max(1, int(n_threads))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(n)
    try:
        import torch
        torch.set_num_threads(n)
        try:
            torch.set_num_interop_threads(max(1, min(4, n)))
        except RuntimeError:
            pass  # already-started interop pool can't be resized; not fatal
    except ImportError:
        pass


def resolve_model_config(args, rc: ResourceConfig):
    block = getattr(args, "block_size", 0) if args else 0
    embd = getattr(args, "n_embd", 0) if args else 0
    head = getattr(args, "n_head", 0) if args else 0
    layer = getattr(args, "n_layer", 0) if args else 0
    batch = getattr(args, "batch_size", 0) if args else 0
    return {
        "block_size": block if block > 0 else rc.block_size,
        "n_embd": embd if embd > 0 else rc.n_embd,
        "n_head": head if head > 0 else rc.n_head,
        "n_layer": layer if layer > 0 else rc.n_layer,
        "batch_size": batch if batch > 0 else rc.batch_size,
    }
