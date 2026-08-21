from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class AcceleratorInfo:
    backend: str
    device: torch.device
    available: bool
    count: int
    names: list[str]
    amp_supported: bool
    amp_device_type: str
    supports_bf16: bool
    supports_fp16: bool
    supports_compile: bool

    @property
    def is_cpu(self) -> bool:
        return self.backend == "cpu"


def _try_cuda() -> AcceleratorInfo | None:
    if not torch.cuda.is_available():
        return None
    count = torch.cuda.device_count()
    names: list[str] = []
    for i in range(count):
        try:
            names.append(torch.cuda.get_device_name(i))
        except Exception:
            names.append(f"cuda:{i}")
    return AcceleratorInfo(
        backend="cuda",
        device=torch.device("cuda"),
        available=True,
        count=count,
        names=names,
        amp_supported=True,
        amp_device_type="cuda",
        supports_bf16=True,
        supports_fp16=True,
        supports_compile=True,
    )


def _try_directml() -> AcceleratorInfo | None:
    try:
        import torch_directml
    except ImportError:
        return None
    try:
        count = torch_directml.device_count()
    except Exception:
        count = 1
    if count == 0:
        return None
    names: list[str] = []
    for i in range(count):
        try:
            names.append(torch_directml.device_name(i))
        except Exception:
            try:
                d = torch_directml.device(i)
                names.append(str(d))
            except Exception:
                names.append(f"directml:{i}")
    try:
        device = torch_directml.device(0)
    except Exception:
        device = torch.device("privateuseone:0")
    return AcceleratorInfo(
        backend="directml",
        device=device,
        available=True,
        count=count,
        names=names,
        amp_supported=False,
        amp_device_type="privateuseone",
        supports_bf16=False,
        supports_fp16=True,
        supports_compile=False,
    )


def _try_mps() -> AcceleratorInfo | None:
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return None
    return AcceleratorInfo(
        backend="mps",
        device=torch.device("mps"),
        available=True,
        count=1,
        names=["Apple MPS"],
        amp_supported=True,
        amp_device_type="mps",
        supports_bf16=False,
        supports_fp16=True,
        supports_compile=False,
    )


def _cpu_fallback() -> AcceleratorInfo:
    return AcceleratorInfo(
        backend="cpu",
        device=torch.device("cpu"),
        available=False,
        count=0,
        names=[],
        amp_supported=False,
        amp_device_type="cpu",
        supports_bf16=False,
        supports_fp16=False,
        supports_compile=False,
    )


_detected: AcceleratorInfo | None = None


def detect_accelerator() -> AcceleratorInfo:
    global _detected
    if _detected is not None:
        return _detected
    for probe in (_try_cuda, _try_directml, _try_mps):
        info = probe()
        if info is not None and info.available:
            _detected = info
            return _detected
    _detected = _cpu_fallback()
    return _detected


def get_device() -> torch.device:
    return detect_accelerator().device


def is_accelerator_available() -> bool:
    return detect_accelerator().available


def get_gpu_info() -> dict[str, Any]:
    info = detect_accelerator()
    return {
        "available": info.available,
        "backend": info.backend,
        "count": info.count,
        "names": info.names,
        "amp_supported": info.amp_supported,
        "amp_device_type": info.amp_device_type,
        "supports_bf16": info.supports_bf16,
        "supports_fp16": info.supports_fp16,
    }


def get_amp_device_type() -> str:
    return detect_accelerator().amp_device_type


def resolve_amp_dtype(amp_flag: str) -> tuple[torch.dtype | None, torch.cuda.amp.GradScaler | None]:
    info = detect_accelerator()
    requested = (amp_flag or "").strip().lower()
    if not requested or not info.amp_supported:
        return (None, None)

    prefer_bf16 = requested in ("bf16", "bfloat16")
    prefer_fp16 = requested in ("fp16", "float16")

    if prefer_bf16 and info.supports_bf16:
        return (torch.bfloat16, None)
    if prefer_bf16 and not info.supports_bf16 and info.supports_fp16:
        warnings.warn(f"--amp bf16 requested but {info.backend} only supports fp16; using fp16 instead.")
        return (torch.float16, None)
    if prefer_fp16 and info.supports_fp16:
        scaler: torch.cuda.amp.GradScaler | None = None
        if info.backend == "cuda":
            scaler = torch.cuda.amp.GradScaler(enabled=True)
        return (torch.float16, scaler)

    if info.supports_bf16:
        return (torch.bfloat16, None)
    if info.supports_fp16:
        scaler = torch.cuda.amp.GradScaler(enabled=True) if info.backend == "cuda" else None
        return (torch.float16, scaler)
    return (None, None)
