from dataclasses import dataclass, field
from typing import Any
import torch

from src.resource_defaults import DEFAULT_CPU_PCT, DEFAULT_GPU_PCT, DEFAULT_RAM_PCT


@dataclass
class TrainingConfig:
    steps: int = 2000
    batch_size: int = 0
    grad_accum_steps: int = 1
    lr: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 100
    eval_every: int = 200
    eval_iters: int = 20

    optimizer: str = "adamw"
    lr_schedule: str = "cosine"
    lr_hold_steps: int = 0
    lr_min_ratio: float = 0.1

    contrastive_coeff: float = 0.0
    embed_norm: bool = False
    curriculum_sort: bool = False

    ema_decay: float = 0.0
    llrd_decay: float = 1.0

    cpu_pct: int = DEFAULT_CPU_PCT
    ram_pct: int = DEFAULT_RAM_PCT
    gpu_pct: int = DEFAULT_GPU_PCT
    torch_threads: int = 0
    seed: int = 1337

    block_size: int = 0
    n_embd: int = 0
    n_head: int = 0
    n_layer: int = 0
    n_kv_heads: int = 0

    amp: str = ""
    compile: bool = False
    compile_mode: str = "default"
    use_checkpoint: bool = False

    z_loss_coeff: float = 1e-4
    n_pred_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps, "batch_size": self.batch_size,
            "grad_accum_steps": self.grad_accum_steps, "lr": self.lr,
            "weight_decay": self.weight_decay, "grad_clip": self.grad_clip,
            "warmup_steps": self.warmup_steps, "eval_every": self.eval_every,
            "eval_iters": self.eval_iters, "optimizer": self.optimizer,
            "lr_schedule": self.lr_schedule, "lr_hold_steps": self.lr_hold_steps,
            "lr_min_ratio": self.lr_min_ratio, "contrastive_coeff": self.contrastive_coeff,
            "embed_norm": self.embed_norm, "curriculum_sort": self.curriculum_sort,
            "ema_decay": self.ema_decay, "llrd_decay": self.llrd_decay,
            "cpu_pct": self.cpu_pct, "ram_pct": self.ram_pct, "gpu_pct": self.gpu_pct,
            "torch_threads": self.torch_threads, "seed": self.seed,
            "block_size": self.block_size, "n_embd": self.n_embd,
            "n_head": self.n_head, "n_layer": self.n_layer,
            "n_kv_heads": self.n_kv_heads, "amp": self.amp,
            "compile": self.compile, "compile_mode": self.compile_mode,
            "use_checkpoint": self.use_checkpoint,
            "z_loss_coeff": self.z_loss_coeff, "n_pred_tokens": self.n_pred_tokens,
        }

    @staticmethod
    def from_args(args) -> "TrainingConfig":
        return TrainingConfig(
            steps=getattr(args, "steps", 2000),
            batch_size=getattr(args, "batch_size", 0),
            grad_accum_steps=getattr(args, "grad_accum_steps", 1),
            lr=getattr(args, "lr", 3e-4),
            weight_decay=getattr(args, "weight_decay", 0.1),
            grad_clip=getattr(args, "grad_clip", 1.0),
            warmup_steps=getattr(args, "warmup_steps", 100),
            eval_every=getattr(args, "eval_every", 200),
            eval_iters=getattr(args, "eval_iters", 20),
            optimizer=getattr(args, "optimizer", "adamw"),
            lr_schedule=getattr(args, "lr_schedule", "cosine"),
            lr_hold_steps=getattr(args, "lr_hold_steps", 0),
            lr_min_ratio=getattr(args, "lr_min_ratio", 0.1),
            contrastive_coeff=getattr(args, "contrastive_coeff", 0.0),
            embed_norm=getattr(args, "embed_norm", False),
            curriculum_sort=getattr(args, "curriculum_sort", False),
            ema_decay=getattr(args, "ema_decay", 0.0),
            llrd_decay=getattr(args, "llrd_decay", 1.0),
            cpu_pct=getattr(args, "cpu_pct", DEFAULT_CPU_PCT),
            ram_pct=getattr(args, "ram_pct", DEFAULT_RAM_PCT),
            gpu_pct=getattr(args, "gpu_pct", DEFAULT_GPU_PCT),
            torch_threads=getattr(args, "torch_threads", 0),
            seed=getattr(args, "seed", 1337),
            block_size=getattr(args, "block_size", 0),
            n_embd=getattr(args, "n_embd", 0),
            n_head=getattr(args, "n_head", 0),
            n_layer=getattr(args, "n_layer", 0),
            n_kv_heads=getattr(args, "n_kv_heads", 0),
            amp=getattr(args, "amp", ""),
            compile=getattr(args, "compile", False),
            compile_mode=getattr(args, "compile_mode", "default"),
            use_checkpoint=getattr(args, "use_checkpoint", False),
            z_loss_coeff=getattr(args, "z_loss_coeff", 1e-4),
            n_pred_tokens=getattr(args, "n_pred_tokens", 0),
        )

def build_training_optimizer(model: torch.nn.Module, tc: TrainingConfig):
    if tc.optimizer == "muon":
        from src.muon import MuonWithAuxAdam
        muon_2d = []
        adam_1d = []
        for p in model.parameters():
            if not p.requires_grad:
                continue
            if p.ndim >= 2:
                muon_2d.append(p)
            else:
                adam_1d.append(p)
        groups = []
        if muon_2d:
            groups.append(dict(params=muon_2d, use_muon=True, lr=tc.lr * 20.0, weight_decay=tc.weight_decay))
        if adam_1d:
            groups.append(dict(params=adam_1d, use_muon=False, lr=tc.lr, betas=(0.9, 0.95), weight_decay=0.0))
        if not groups:
            groups.append(dict(params=list(model.parameters()), use_muon=False, lr=tc.lr, betas=(0.9, 0.95), weight_decay=tc.weight_decay))
        return MuonWithAuxAdam(groups)
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=tc.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=tc.weight_decay,
    )


def compute_lr(tc: TrainingConfig, step: int, total_steps: int) -> float:
    from src.train import linear_onecycle_lr, constant_lr, warmup_stable_cosine_lr
    if tc.lr_schedule == "linear":
        return linear_onecycle_lr(step, total_steps, tc.lr, tc.warmup_steps, tc.lr_min_ratio)
    elif tc.lr_schedule == "constant":
        return constant_lr(step, total_steps, tc.lr, tc.warmup_steps, tc.lr_min_ratio)
    return warmup_stable_cosine_lr(step, total_steps, tc.lr, tc.warmup_steps, tc.lr_hold_steps, tc.lr_min_ratio)


def apply_embedding_norm(model: torch.nn.Module):
    with torch.no_grad():
        emb = model.token_embedding_table.weight.data
        emb_norm = torch.nn.functional.normalize(emb, dim=-1)
        emb.copy_(emb_norm)
