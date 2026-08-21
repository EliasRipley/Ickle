import os
from dataclasses import dataclass


@dataclass
class ILMProfile:
    name: str
    block_size: int
    n_embd: int
    n_head: int
    n_layer: int
    batch_size: int
    torch_threads: int
    description: str


@dataclass
class ResourceConfig:
    cpu_cores: int
    ram_gb: float
    gpu_available: bool
    cpu_percent: int = 80
    ram_percent: int = 80
    gpu_percent: int = 80

    @property
    def effective_cores(self):
        return max(1, int(self.cpu_cores * self.cpu_percent / 100))

    @property
    def effective_ram_gb(self):
        return max(0.5, self.ram_gb * self.ram_percent / 100)

    @property
    def torch_threads(self):
        return max(1, min(self.effective_cores, 8))

    @property
    def block_size(self):
        cores = self.effective_cores
        ram = self.effective_ram_gb
        base = max(256, min(1024, int(256 + ram * 30 + cores * 10)))
        return (base // 64) * 64

    @property
    def n_embd(self):
        cores = self.effective_cores
        ram = self.effective_ram_gb
        base = max(64, min(384, int(48 + ram * 20 + cores * 6)))
        return (base // 16) * 16

    @property
    def n_head(self):
        embd = self.n_embd
        possible = [h for h in [2, 4, 6, 8] if embd % h == 0]
        return possible[len(possible) // 2] if possible else max(2, embd // 16)

    @property
    def n_layer(self):
        cores = self.effective_cores
        ram = self.effective_ram_gb
        return max(2, min(10, int(2 + cores * 0.3 + ram * 0.2)))

    @property
    def batch_size(self):
        cores = self.effective_cores
        return max(4, min(24, int(cores * 1.5)))

    def summary(self):
        return (
            f"cores={self.cpu_cores}({self.cpu_percent}%) ram={self.ram_gb:.1f}GB({self.ram_percent}%) "
            f"gpu={self.gpu_available} "
            f"model: block={self.block_size} embd={self.n_embd} "
            f"heads={self.n_head} layers={self.n_layer} batch={self.batch_size}"
        )


def detect_resources():
    cpu = os.cpu_count() or 4
    if cpu > 8:
        cpu = 8
    ram = 4.0
    gpu = False
    return ResourceConfig(cpu_cores=cpu, ram_gb=ram, gpu_available=gpu)


def get_profile(name):
    rc = detect_resources()
    return ILMProfile(
        name=name,
        block_size=rc.block_size,
        n_embd=rc.n_embd,
        n_head=rc.n_head,
        n_layer=rc.n_layer,
        batch_size=rc.batch_size,
        torch_threads=rc.torch_threads,
        description=rc.summary(),
    )


def resolve_resource_config(args):
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
