from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.0
    target_modules: tuple[str, ...] = ("q", "k", "v", "proj", "w1", "w2", "w3", "lm_head")

    def as_dict(self) -> dict:
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": list(self.target_modules),
        }

    @staticmethod
    def from_dict(payload: dict) -> "LoRAConfig":
        return LoRAConfig(
            rank=int(payload.get("rank", 8)),
            alpha=int(payload.get("alpha", 16)),
            dropout=float(payload.get("dropout", 0.0)),
            target_modules=tuple(payload.get("target_modules", ("q", "k", "v", "proj", "w1", "w2", "w3", "lm_head"))),
        )


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: int, dropout: float):
        super().__init__()
        self.base = base
        self.rank = int(rank)
        self.alpha = int(alpha)
        self.scale = float(alpha) / max(1, int(rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.gate = 1.0

        in_features = int(base.in_features)
        out_features = int(base.out_features)

        self.lora_a = nn.Parameter(torch.zeros(self.rank, in_features))
        self.lora_b = nn.Parameter(torch.zeros(out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)
        nn.init.zeros_(self.lora_b)

        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        if self.gate <= 0.0:
            return base_out
        low_rank = F.linear(self.dropout(x), self.lora_a)
        lora_out = F.linear(low_rank, self.lora_b) * self.scale
        return base_out + self.gate * lora_out


def _matches_target(module_name: str, targets: tuple[str, ...]) -> bool:
    return any(tag in module_name for tag in targets)


def inject_lora(model: nn.Module, cfg: LoRAConfig) -> list[str]:
    replaced: list[str] = []
    module_items = list(model.named_modules())
    for parent_name, parent_module in module_items:
        for child_name, child_module in list(parent_module.named_children()):
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if isinstance(child_module, nn.Linear) and _matches_target(full_name, cfg.target_modules):
                setattr(
                    parent_module,
                    child_name,
                    LoRALinear(
                        base=child_module,
                        rank=cfg.rank,
                        alpha=cfg.alpha,
                        dropout=cfg.dropout,
                    ),
                )
                replaced.append(full_name)
    return replaced


def set_only_lora_trainable(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_a.requires_grad = True
            module.lora_b.requires_grad = True


def set_lora_gates(model: nn.Module, gate: float):
    """Set all LoRA adapter gates in the model to the same value."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.gate = max(0.0, min(1.0, float(gate)))


def set_lora_gates_per_module(model: nn.Module, gate_map: dict[str, float]):
    """Set LoRA gates per module name prefix. Keys are module name fragments (e.g. 'q', 'v', 'w1').
    The gate_map values override the default gate=1.0 for matching layers."""
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            matched = 1.0
            for fragment, gate in gate_map.items():
                if fragment in name:
                    matched = gate
                    break
            module.gate = max(0.0, min(1.0, float(matched)))


def reset_lora_gates(model: nn.Module):
    """Reset all LoRA gates to full activation (1.0)."""
    set_lora_gates(model, 1.0)


def get_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    payload: dict[str, torch.Tensor] = {}
    for module_name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            payload[f"{module_name}.lora_a"] = module.lora_a.detach().cpu().clone()
            payload[f"{module_name}.lora_b"] = module.lora_b.detach().cpu().clone()
    return payload


def _module_index(model: nn.Module) -> dict[str, nn.Module]:
    return dict(model.named_modules())


def load_lora_state_dict(model: nn.Module, state: dict[str, torch.Tensor], strict: bool = True):
    modules = _module_index(model)
    missing: list[str] = []
    for key, tensor in state.items():
        if key.endswith(".lora_a"):
            name = key[: -len(".lora_a")]
            mod = modules.get(name)
            if isinstance(mod, LoRALinear):
                mod.lora_a.data.copy_(tensor.to(mod.lora_a.device, dtype=mod.lora_a.dtype))
            else:
                missing.append(key)
        elif key.endswith(".lora_b"):
            name = key[: -len(".lora_b")]
            mod = modules.get(name)
            if isinstance(mod, LoRALinear):
                mod.lora_b.data.copy_(tensor.to(mod.lora_b.device, dtype=mod.lora_b.dtype))
            else:
                missing.append(key)
        else:
            missing.append(key)
    if strict and missing:
        missing_str = ", ".join(missing[:10])
        raise KeyError(f"Missing LoRA tensors in model: {missing_str}")


def zero_lora_state_like(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: torch.zeros_like(tensor) for name, tensor in get_lora_state_dict(model).items()}


def add_states(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    keys = sorted(set(a.keys()) | set(b.keys()))
    out: dict[str, torch.Tensor] = {}
    for key in keys:
        left = a.get(key)
        right = b.get(key)
        if left is None:
            out[key] = right.clone()
        elif right is None:
            out[key] = left.clone()
        else:
            out[key] = left + right
    return out


def subtract_states(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    keys = sorted(set(a.keys()) | set(b.keys()))
    out: dict[str, torch.Tensor] = {}
    for key in keys:
        left = a.get(key)
        right = b.get(key)
        if left is None or right is None:
            raise KeyError(f"Cannot subtract missing state key '{key}'")
        out[key] = left - right
    return out

