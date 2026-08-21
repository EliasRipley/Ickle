from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class GlobalOptimizerConfig:
    lr: float = 1.0
    beta: float = 0.9
    nesterov: bool = True


class GlobalOptimizer:
    """Server-side DiLoCo optimizer with Nesterov momentum.

    The server aggregates client model deltas and applies a momentum-based
    outer-loop step instead of naive addition:

        d_t     = mean(client_model - global_model)
        v_t     = beta * v_{t-1} + d_t             (velocity / momentum)
        w_{t+1} = w_t + lr * (d_t + beta * v_t)   (Nesterov correction when nesterov=True)
        w_{t+1} = w_t + lr * v_t                   (standard momentum when nesterov=False)

    Client updates use the conventional ``end_state - start_state`` delta.
    Consequently the outer optimizer must *add* that direction.  Treating it
    as a gradient and subtracting it would move the global adapter away from
    every locally trained model.

    This is the core DiLoCo insight: 500× communication reduction via
    dual-optimizer (local AdamW + global momentum).
    """

    def __init__(self, cfg: GlobalOptimizerConfig):
        self.cfg = cfg
        self.velocity: dict[str, torch.Tensor] = {}

    def state_dict(self) -> dict[str, Any]:
        return {
            "cfg": {"lr": self.cfg.lr, "beta": self.cfg.beta, "nesterov": self.cfg.nesterov},
            "velocity": {k: v.clone() for k, v in self.velocity.items()},
        }

    def load_state_dict(self, state: dict[str, Any]):
        cfg = state.get("cfg", {})
        self.cfg.lr = float(cfg.get("lr", 1.0))
        self.cfg.beta = float(cfg.get("beta", 0.9))
        self.cfg.nesterov = bool(cfg.get("nesterov", True))
        self.velocity = {k: v.clone() for k, v in state.get("velocity", {}).items()}

    def step(
        self,
        params: dict[str, torch.Tensor],
        model_delta: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Apply one DiLoCo global optimizer step.

        Args:
            params: Current global adapter parameters.
            model_delta: Aggregated ``client_after - client_before`` delta.

        Returns:
            Updated parameters after momentum step.
        """
        lr = self.cfg.lr
        beta = self.cfg.beta
        use_nesterov = self.cfg.nesterov

        out: dict[str, torch.Tensor] = {}
        for key in params:
            delta = model_delta.get(key, torch.zeros_like(params[key]))
            prev_v = self.velocity.get(key, torch.zeros_like(params[key]))
            v = beta * prev_v + delta
            self.velocity[key] = v
            if use_nesterov:
                out[key] = params[key] + lr * (delta + beta * v)
            else:
                out[key] = params[key] + lr * v
        return out


def save_global_optimizer(optimizer: GlobalOptimizer, path: str):
    """Save optimizer state to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(optimizer.state_dict(), path)


def load_global_optimizer(path: str, cfg: GlobalOptimizerConfig | None = None) -> GlobalOptimizer:
    """Load optimizer state from disk, creating a fresh one if missing."""
    p = Path(path)
    if p.exists():
        state = torch.load(p, map_location="cpu")
        opt = GlobalOptimizer(cfg or GlobalOptimizerConfig())
        opt.load_state_dict(state)
        return opt
    return GlobalOptimizer(cfg or GlobalOptimizerConfig())
