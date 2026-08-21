from dataclasses import dataclass, field
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TinyConfig:
    vocab_size: int
    block_size: int = 512
    n_embd: int = 256
    n_head: int = 8
    n_kv_heads: int = 0
    n_layer: int = 6
    dropout: float = 0.1
    attn_dropout: float = 0.0
    rope_base: int = 10000
    rope_scaling: dict | None = None
    use_compile: bool = False
    amp_dtype: str = ""
    qk_norm: bool = False
    z_loss_coeff: float = 1e-4
    layer_drop_rate: float = 0.0
    n_pred_tokens: int = 0
    use_checkpoint: bool = False
    contrastive_coeff: float = 0.0
    _extra: dict[str, Any] = field(default_factory=dict)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * (x * rms)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0, scaling: dict | None = None):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")

        if scaling is not None:
            scale_type = scaling.get("type", "")
            factor = float(scaling.get("factor", 1.0))
            if scale_type == "yarn":
                max_seq_len = int(max_seq_len * factor)
                base = base * (factor ** (head_dim / (head_dim - 2)))
            elif scale_type == "linear":
                max_seq_len = int(max_seq_len * factor)

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def apply_rotary(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        t = x.size(-2)
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        cos = self.cos[offset : offset + t].unsqueeze(0).unsqueeze(0)
        sin = self.sin[offset : offset + t].unsqueeze(0).unsqueeze(0)
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
        return torch.stack((out_even, out_odd), dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TinyConfig):
        super().__init__()
        if cfg.n_embd % cfg.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = cfg.n_head
        self.n_kv_heads = cfg.n_kv_heads if cfg.n_kv_heads > 0 else cfg.n_head
        if cfg.n_head % self.n_kv_heads != 0:
            raise ValueError("n_head must be divisible by n_kv_heads")
        self.head_dim = cfg.n_embd // cfg.n_head
        self.n_rep = self.n_head // self.n_kv_heads

        kv_dim = self.n_kv_heads * self.head_dim
        self.q = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.k = nn.Linear(cfg.n_embd, kv_dim, bias=False)
        self.v = nn.Linear(cfg.n_embd, kv_dim, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.attn_dropout = cfg.attn_dropout
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.rope = RotaryEmbedding(self.head_dim, cfg.block_size, cfg.rope_base, cfg.rope_scaling)
        self.qk_norm = nn.LayerNorm(self.head_dim) if cfg.qk_norm else None

    def forward(
        self, x: torch.Tensor, past_kv: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        b, t, c = x.shape
        q = self.q(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            kv_offset = past_k.size(2)
            if self.qk_norm is not None:
                q = self.qk_norm(q)
                k = self.qk_norm(k)
            q = self.rope.apply_rotary(q, offset=kv_offset)
            k = self.rope.apply_rotary(k, offset=kv_offset)
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
            attn_mask = torch.triu(
                torch.full((t, k.size(2)), float("-inf"), device=x.device, dtype=x.dtype),
                diagonal=1 + kv_offset,
            )
        else:
            if self.qk_norm is not None:
                q = self.qk_norm(q)
                k = self.qk_norm(k)
            q = self.rope.apply_rotary(q)
            k = self.rope.apply_rotary(k)
            attn_mask = None

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=(attn_mask is None),
        )
        out = attn_out.transpose(1, 2).contiguous().view(b, -1, c)
        return self.resid_dropout(self.proj(out)), (k, v)


class FeedForward(nn.Module):
    def __init__(self, cfg: TinyConfig):
        super().__init__()
        hidden = int((8 * cfg.n_embd) / 3)
        self.w1 = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w2 = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w3 = nn.Linear(hidden, cfg.n_embd, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.w1(x)) * self.w2(x)
        return self.dropout(self.w3(x))


class MultiTokenHead(nn.Module):
    def __init__(self, n_embd: int, vocab_size: int):
        super().__init__()
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class Block(nn.Module):
    def __init__(self, cfg: TinyConfig, layer_idx: int = 0):
        super().__init__()
        self.attn = CausalSelfAttention(cfg)
        self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg.n_embd)
        self.norm2 = RMSNorm(cfg.n_embd)
        self.layer_idx = layer_idx
        self.use_checkpoint = cfg.use_checkpoint

    def forward(
        self, x: torch.Tensor, past_kv: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if self.use_checkpoint and self.training and past_kv is None:
            def _attn_forward(x_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                a, kv = self.attn(self.norm1(x_in), past_kv=None)
                return a, kv
            a, _ = torch.utils.checkpoint.checkpoint(_attn_forward, x, use_reentrant=False)
            x = x + a
            def _ff_forward(x_in: torch.Tensor) -> torch.Tensor:
                return self.ff(self.norm2(x_in))
            ff_out = torch.utils.checkpoint.checkpoint(_ff_forward, x, use_reentrant=False)
            x = x + ff_out
            return x, (None, None)
        a, kv = self.attn(self.norm1(x), past_kv=past_kv)
        x = x + a
        x = x + self.ff(self.norm2(x))
        return x, kv


def migrate_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not any("qkv.weight" in k for k in state_dict):
        return state_dict

    new_sd = {}
    for k, v in state_dict.items():
        if "qkv.weight" in k:
            prefix = k.replace("qkv.weight", "")
            split_dim = v.size(0) // 3
            new_sd[f"{prefix}q.weight"] = v[:split_dim]
            new_sd[f"{prefix}k.weight"] = v[split_dim : 2 * split_dim]
            new_sd[f"{prefix}v.weight"] = v[2 * split_dim :]
        else:
            new_sd[k] = v
    return new_sd


_AMP_DTYPE_MAP: dict[str, torch.dtype] = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _resolve_amp_dtype(amp_dtype: str, device: torch.device) -> torch.dtype | None:
    if not amp_dtype:
        return None
    dtype = _AMP_DTYPE_MAP.get(amp_dtype.lower())
    if dtype is None:
        return None
    if dtype == torch.bfloat16 and device.type != "cuda":
        return None
    return dtype


def _apply_repetition_penalty(logits: torch.Tensor, generated: torch.Tensor, penalty: float) -> None:
    """In-place CTRL-style repetition penalty (Keskar et al. 2019, the same
    mechanism as HF transformers' `repetition_penalty`): scores for tokens
    already present in the generated sequence are divided (if positive) or
    multiplied (if negative) by `penalty`, discouraging -- not blocking --
    their reselection. Without this, plain temperature/top_k sampling on a
    small or undertrained model reliably collapses into repeating the same
    locally-high-probability phrase forever once it appears once; confirmed
    live with the default chat settings (temperature=0.25, top_k=20), which
    reproduced a genuine infinite repetition loop on a freshly trained model."""
    if penalty == 1.0:
        return
    for b in range(generated.size(0)):
        seen = torch.unique(generated[b])
        seen = seen[(seen >= 0) & (seen < logits.size(-1))]
        if seen.numel() == 0:
            continue
        scores = logits[b, seen]
        logits[b, seen] = torch.where(scores < 0, scores * penalty, scores / penalty)


def _apply_no_repeat_ngram(logits: torch.Tensor, generated: torch.Tensor, ngram_size: int) -> None:
    """Hard-block any continuation that would repeat an n-gram already seen
    in this generation (the same technique HF transformers' `generate()`
    calls `no_repeat_ngram_size`). This is the complement to
    _apply_repetition_penalty: the penalty discourages short repeat cycles
    (e.g. "X. X. X.") but a locally-dominant 2-4 token phrase can still win
    even with it applied -- outright blocking exact n-gram repeats closes
    that gap, confirmed live to remove a residual repeat loop the penalty
    alone left in place."""
    if ngram_size <= 0:
        return
    seq_len = generated.size(1)
    if seq_len < ngram_size:
        return
    for b in range(generated.size(0)):
        seq = generated[b].tolist()
        prefix = tuple(seq[-(ngram_size - 1):]) if ngram_size > 1 else ()
        banned: set[int] = set()
        for i in range(seq_len - ngram_size + 1):
            if tuple(seq[i : i + ngram_size - 1]) == prefix:
                banned.add(seq[i + ngram_size - 1])
        if banned:
            idxs = torch.tensor(sorted(banned), device=logits.device, dtype=torch.long)
            logits[b, idxs] = float("-inf")


class ILM(nn.Module):
    def __init__(self, cfg: TinyConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg, layer_idx=i) for i in range(cfg.n_layer)])
        self.ln_f = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        self.mtp_heads: nn.ModuleList | None = None
        if cfg.n_pred_tokens > 0:
            heads = [MultiTokenHead(cfg.n_embd, cfg.vocab_size) for _ in range(cfg.n_pred_tokens)]
            self.mtp_heads = nn.ModuleList(heads)

        self.lm_head.weight = self.token_embedding_table.weight

        self.apply(self._init_weights)
        self._amp_dtype: torch.dtype | None = None
        self._amp_device_type: str = "cuda"
        self._compiled_forward = None

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            std = 0.02 / math.sqrt(max(1, self.cfg.n_layer / 3))
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True, assign: bool = False):
        migrated = migrate_state_dict(state_dict)
        return super().load_state_dict(migrated, strict=strict, assign=assign)

    def configure_compile(self, *, enable: bool = True, mode: str = "default"):
        if enable and hasattr(torch, "compile"):
            compiled = torch.compile(self, mode=mode)
            object.__setattr__(compiled, "cfg", self.cfg)
            object.__setattr__(compiled, "_compiled_forward", compiled)
            return compiled
        return self

    def configure_amp(self, amp_dtype: str, device: torch.device):
        self._amp_dtype = _resolve_amp_dtype(amp_dtype, device)
        self._amp_device_type = device.type

    @property
    def amp_autocast_ctx(self):
        if self._amp_dtype is not None:
            device_type = getattr(self, "_amp_device_type", "cuda")
            return torch.autocast(device_type=device_type, dtype=self._amp_dtype)
        from contextlib import nullcontext
        return nullcontext()

    def _compute_loss(
        self, logits: torch.Tensor, targets: torch.Tensor, loss_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        b, t, c = logits.shape
        logits_flat = logits.reshape(b * t, c).float()
        targets_flat = targets.reshape(b * t)
        if loss_mask is not None:
            mask_flat = loss_mask.reshape(b * t)
            losses = F.cross_entropy(logits_flat, targets_flat, reduction="none")
            loss = losses.masked_select(mask_flat).mean()
        else:
            loss = F.cross_entropy(logits_flat, targets_flat)

        if self.cfg.z_loss_coeff > 0.0:
            logits_max = logits_flat.max(dim=-1, keepdim=True).values
            logsumexp = torch.log(torch.exp(logits_flat - logits_max).sum(dim=-1)) + logits_max.squeeze(-1)
            z_loss = logsumexp.pow(2).mean() * self.cfg.z_loss_coeff
            loss = loss + z_loss

        return loss

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, loss_mask: torch.Tensor | None = None):
        with self.amp_autocast_ctx:
            tok_emb = self.token_embedding_table(idx)
            x = self.dropout(tok_emb)

            layer_drop = self.cfg.layer_drop_rate
            for block in self.blocks:
                if self.training and layer_drop > 0.0 and block.layer_idx > 0:
                    if torch.rand(1).item() < layer_drop:
                        continue
                x, _ = block(x)

            x = self.ln_f(x)
            logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = self._compute_loss(logits, targets, loss_mask)

            if self.cfg.contrastive_coeff > 0.0 and x.size(1) >= 4:
                mid = x.size(1) // 2
                h1 = x[:, :mid, :].mean(dim=1)
                h2 = x[:, mid:, :].mean(dim=1)
                h1_norm = F.normalize(h1.float(), dim=-1)
                h2_norm = F.normalize(h2.float(), dim=-1)
                contrastive_loss = -(h1_norm * h2_norm).sum(dim=-1).mean()
                loss = loss + self.cfg.contrastive_coeff * contrastive_loss

            n_pred = self.cfg.n_pred_tokens
            if n_pred > 0 and self.mtp_heads is not None and targets is not None:
                b, t = idx.shape
                for k in range(n_pred):
                    if t > k + 1:
                        mtp_targets = targets[:, k + 1:]
                        mtp_shift = x[:, :-k - 1]
                        if mtp_shift is not None and k < len(self.mtp_heads):
                            mtp_logits = self.mtp_heads[k](mtp_shift)
                            mtp_mask = loss_mask[:, k + 1:] if loss_mask is not None else None
                            mtp_loss = self._compute_loss(mtp_logits, mtp_targets, mtp_mask)
                            loss = loss + mtp_loss / (n_pred + 1)

        return logits, loss

    def _forward_with_kv(
        self, idx: torch.Tensor, past_kv: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None
    ) -> tuple[torch.Tensor, None, list[tuple[torch.Tensor, torch.Tensor]]]:
        with self.amp_autocast_ctx:
            tok_emb = self.token_embedding_table(idx)
            x = self.dropout(tok_emb)
            new_past_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
            for i, block in enumerate(self.blocks):
                kv = past_kv[i] if past_kv is not None and i < len(past_kv) else None
                x, kv_out = block(x, past_kv=kv)
                new_past_kv.append(kv_out)
            x = self.ln_f(x)
            logits = self.lm_head(x)
        return logits, None, new_past_kv

    def _trim_past_kv(
        self,
        past_kv: list[tuple[torch.Tensor, torch.Tensor] | None] | None,
    ) -> list[tuple[torch.Tensor, torch.Tensor] | None] | None:
        if past_kv is None:
            return None
        max_cache_tokens = max(1, int(self.cfg.block_size) - 1)
        trimmed: list[tuple[torch.Tensor, torch.Tensor] | None] = []
        for kv in past_kv:
            if kv is None:
                trimmed.append(None)
                continue
            k, v = kv
            if k is None or v is None:
                trimmed.append((k, v))
                continue
            if k.size(2) > max_cache_tokens:
                k = k[:, :, -max_cache_tokens:, :]
                v = v[:, :, -max_cache_tokens:, :]
            trimmed.append((k, v))
        return trimmed

    @torch.inference_mode()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 80,
        temperature: float = 0.8,
        top_k: int = 40,
        repetition_penalty: float = 1.3,
        no_repeat_ngram_size: int = 3,
    ) -> torch.Tensor:
        past_kv: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None
        for _ in range(max_new_tokens):
            if past_kv is None:
                idx_input = idx[:, -self.cfg.block_size :]
            else:
                past_kv = self._trim_past_kv(past_kv)
                idx_input = idx[:, -1:]
            logits, _, past_kv = self._forward_with_kv(idx_input, past_kv=past_kv)
            logits = logits[:, -1, :] / max(temperature, 1e-4)
            _apply_repetition_penalty(logits, idx, repetition_penalty)
            _apply_no_repeat_ngram(logits, idx, no_repeat_ngram_size)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    @torch.inference_mode()
    def generate_streaming(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 80,
        temperature: float = 0.8,
        top_k: int = 40,
        repetition_penalty: float = 1.3,
        no_repeat_ngram_size: int = 3,
    ):
        past_kv: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None
        for _ in range(max_new_tokens):
            if past_kv is None:
                idx_input = idx[:, -self.cfg.block_size :]
            else:
                past_kv = self._trim_past_kv(past_kv)
                idx_input = idx[:, -1:]
            logits, _, past_kv = self._forward_with_kv(idx_input, past_kv=past_kv)
            logits = logits[:, -1, :] / max(temperature, 1e-4)
            _apply_repetition_penalty(logits, idx, repetition_penalty)
            _apply_no_repeat_ngram(logits, idx, no_repeat_ngram_size)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            yield idx_next
