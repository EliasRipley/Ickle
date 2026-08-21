import argparse
import json
import math
import os
import random
import shutil
import tempfile
import string
import time
from pathlib import Path
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.device_bridge import detect_accelerator, get_amp_device_type, resolve_amp_dtype
from src.workspace_paths import get_training_root, get_training_corpus_dir

from src.federated.lora import LoRAConfig, inject_lora, get_lora_state_dict, load_lora_state_dict
from src.ilm_profile import apply_cpu_thread_budget, detect_resources, resolve_resource_config, resolve_model_config, ResourceConfig
from src.model import TinyConfig, ILM
from src.resource_defaults import add_resource_pct_args
from src.muon import MuonWithAuxAdam
from src.tokenizer import (
    BaseTokenizer,
    CharTokenizer,
    SentencePieceTokenizer,
    TokenizerError,
    sanitize_text_for_tokenizer,
    tokenizer_from_checkpoint,
)




def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def train_sentencepiece_tokenizer_with_retry(
    text: str,
    *,
    spm_vocab_size: int,
    model_type: str,
) -> tuple[SentencePieceTokenizer, int]:
    """Train a SentencePiece tokenizer, halving the requested vocab size on
    failure (small/repetitive corpora can't support a large vocab) down to a
    128-token floor. Returns (tokenizer, actual_vocab_size_used) so the
    caller can report when it had to reduce from what was requested.

    target_size is clamped to the 128 floor immediately, and the loop always
    attempts exactly 128 before giving up -- an earlier version could halve
    straight past 128 (e.g. 150 -> 75) without ever re-checking the raise
    condition at the new size, exiting the loop silently with no tokenizer
    assigned and no exception raised (reproduced with a small/repetitive
    corpus, not a hypothetical). Raises SystemExit with an actionable message
    if even the floor fails, instead of a raw traceback.
    """
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", encoding="utf-8", delete=False) as f:
        f.write(text)
        tmp_corpus = f.name
    try:
        target_size = max(128, spm_vocab_size)
        while True:
            try:
                tokenizer = SentencePieceTokenizer.train_from_corpus(
                    corpus_path=tmp_corpus,
                    vocab_size=target_size,
                    model_type=model_type,
                )
                return tokenizer, target_size
            except (TokenizerError, RuntimeError):
                if target_size <= 128:
                    raise
                target_size = max(128, target_size // 2)
    except (TokenizerError, RuntimeError) as exc:
        # RuntimeError is what sentencepiece itself raises (e.g. even the
        # 128-token floor is too high for a near-empty/degenerate corpus) --
        # must be converted to a clean SystemExit here too, not just
        # TokenizerError, or it surfaces as a raw traceback.
        raise SystemExit(
            f"Could not train a SentencePiece tokenizer on this corpus, even at the "
            f"minimum vocab size (128): {exc}. The corpus is likely too small or "
            f"repetitive for subword tokenization -- try more/varied training text, "
            f"or pass --tokenizer char to use the character-level fallback instead."
        ) from exc
    finally:
        try:
            os.unlink(tmp_corpus)
        except OSError:
            pass


class _SafeFormatDict(dict):
    def __missing__(self, key):
        return ""


def _render_stream_template(template: str, row: dict[str, Any]) -> str:
    safe_row: dict[str, Any] = {}
    for key, value in dict(row or {}).items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe_row[str(key)] = "" if value is None else str(value)
        else:
            safe_row[str(key)] = json.dumps(value, ensure_ascii=False)
    formatter = string.Formatter()
    field_names = [name for _, name, _, _ in formatter.parse(template) if name]
    if field_names and not any(name in safe_row for name in field_names):
        return ""
    try:
        rendered = template.format_map(_SafeFormatDict(safe_row))
    except Exception:
        return ""
    return rendered.strip()


def _parse_stream_role_map(role_map_expr: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for raw_pair in str(role_map_expr or "").split(","):
        raw_pair = raw_pair.strip()
        if not raw_pair or "=" not in raw_pair:
            continue
        old, new = raw_pair.split("=", 1)
        old, new = old.strip(), new.strip()
        if old and new:
            pairs.append((old, new))
    return pairs


def apply_stream_role_map(text: str, role_map: list[tuple[str, str]]) -> str:
    """Remaps dialogue role markers (e.g. 'Human:'/'Assistant:' -> 'User:'/'Ickle:')
    so streamed chat/preference text actually matches Ickle's own template and
    build_loss_mask()'s response boundary instead of silently training in
    unmasked raw-text mode. See --stream-role-map's help text for why."""
    for old, new in role_map:
        text = text.replace(f"{old}:", f"{new}:")
    return text


def build_loss_mask(tokens: list[int], tokenizer, text: str, response_prefix: str = "Ickle:") -> list[bool]:
    """Build a boolean mask: True for response tokens (train on these), False for prompt tokens.
    If no response_prefix or 'User:' markers found, all tokens are masked True (raw text mode).

    Works entirely in token space. This used to index the raw `text` string
    with a variable (`j`) that actually walks `tokens` -- harmless only for
    a hypothetical one-char-per-token tokenizer; with the actual default
    (SentencePiece, where token count != character count), `text[j]` reads
    an essentially arbitrary character, so the "User:" boundary this exists
    to find effectively never matched. `text` is kept as a parameter for
    call-site compatibility but is no longer used.
    """
    del text  # kept only for backward-compatible call signature; see docstring
    response_ids = tokenizer.encode(response_prefix)
    prefix_len = len(response_ids)
    user_markers = [ids for ids in (tokenizer.encode("\nUser:"), tokenizer.encode("User:")) if ids]
    mask = [False] * len(tokens)
    i = 0
    while i < len(tokens):
        if tokens[i : i + prefix_len] == response_ids:
            j = i + prefix_len
            while j < len(tokens) and not any(tokens[j : j + len(m)] == m for m in user_markers):
                j += 1
            for k in range(i, min(j, len(tokens))):
                mask[k] = True
            i = j
        else:
            i += 1
    if not any(mask):
        return [True] * len(tokens)
    return mask


def shuffle_in_chunks(encoded: torch.Tensor, *, chunk_size: int, seed: int) -> torch.Tensor:
    """Shuffle a token tensor by fixed-size contiguous chunk, preserving
    local structure within each chunk while randomizing macro-order.

    Training text is built by concatenating sources in a fixed sequence
    (streamed dataset, then a second streamed dataset). The train/val split
    below takes the *positional* last 10% of the token stream as val_data --
    so whatever was concatenated last dominates validation; if any source
    is small and repetitive relative to the rest, that split alone can give
    a deceptively low val_loss/perplexity reflecting memorization of that
    one source rather than real generalization. (This module used to also
    unconditionally append a fixed ~9KB "bootstrap English" block repeated
    200x -- ~1.8M characters of the same passage -- to every training task
    the web UI created, which was a much worse version of exactly this
    problem: confirmed live to dominate what small runs actually learned,
    leaking its own literal sentences into unrelated answers regardless of
    the prompt. That mechanism has been removed entirely, not just
    disabled, so nothing can silently reintroduce it.)

    get_batch() already samples random block_size windows regardless of
    buffer order, so reordering here doesn't change training semantics at
    all -- it only changes which characters end up on which side of the
    train/val split, which is the actual bug.
    """
    if chunk_size <= 0 or len(encoded) <= chunk_size:
        return encoded
    chunks = [encoded[i : i + chunk_size] for i in range(0, len(encoded), chunk_size)]
    random.Random(seed).shuffle(chunks)
    return torch.cat(chunks)


def get_batch(data, mask, block_size: int, batch_size: int, device):
    if len(data) <= block_size + 1:
        raise ValueError(
            f"Dataset too small for block_size={block_size}. Need > {block_size + 1} tokens, got {len(data)}"
        )
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    m = torch.stack([torch.tensor(mask[i : i + block_size], dtype=torch.bool) for i in ix])
    return x.to(device), y.to(device), m.to(device)


def warmup_stable_cosine_lr(
    step: int, total_steps: int, base_lr: float,
    warmup_steps: int, hold_steps: int = 0, min_lr_ratio: float = 0.1,
) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    adjusted_step = step - warmup_steps
    if adjusted_step < hold_steps:
        return base_lr
    decay_step = adjusted_step - hold_steps
    decay_total = max(1, total_steps - warmup_steps - hold_steps)
    progress = min(decay_step / decay_total, 1.0)
    min_lr = base_lr * min_lr_ratio
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def linear_onecycle_lr(
    step: int, total_steps: int, base_lr: float,
    warmup_steps: int, min_lr_ratio: float = 0.01,
) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    decay_step = step - warmup_steps
    decay_total = max(1, total_steps - warmup_steps)
    progress = min(decay_step / decay_total, 1.0)
    min_lr = base_lr * min_lr_ratio
    return base_lr + (min_lr - base_lr) * progress


def constant_lr(step: int, total_steps: int, base_lr: float,
                warmup_steps: int, min_lr_ratio: float = 0.1) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    return base_lr


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, data: torch.Tensor, mask: list[bool] | None, block_size: int):
        self.data = data
        self.mask = torch.tensor(mask, dtype=torch.bool) if mask is not None else None
        self.block_size = block_size

    def __len__(self) -> int:
        return max(0, len(self.data) - self.block_size - 1)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.data[idx: idx + self.block_size]
        y = self.data[idx + 1: idx + self.block_size + 1]
        if self.mask is not None:
            m = self.mask[idx: idx + self.block_size]
        else:
            m = torch.ones(self.block_size, dtype=torch.bool)
        return x, y, m


def estimate_loss(model, data, mask, cfg, batch_size, device, eval_iters: int = 20):
    model.eval()
    losses = []
    for _ in range(eval_iters):
        xb, yb, mb = get_batch(data, mask, cfg.block_size, batch_size, device)
        with torch.no_grad():
            _, loss = model(xb, yb, loss_mask=mb)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def estimate_loss_advanced(
    model, data, mask, cfg, batch_size, device, eval_iters: int = 20,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    top1_correct = 0
    top5_correct = 0
    total_tokens = 0
    for _ in range(eval_iters):
        xb, yb, mb = get_batch(data, mask, cfg.block_size, batch_size, device)
        with torch.no_grad():
            logits, _ = model(xb, yb, loss_mask=mb)
            logits_flat = logits.view(-1, logits.size(-1)).float()
            targets_flat = yb.view(-1)
            mask_flat = mb.view(-1) if mb is not None else torch.ones_like(targets_flat, dtype=torch.bool)

            losses_batch = F.cross_entropy(logits_flat, targets_flat, reduction="none")
            losses_batch = losses_batch.masked_select(mask_flat)
            losses.extend(losses_batch.cpu().tolist())

            _, top1 = logits_flat.topk(1, dim=-1)
            _, top5 = logits_flat.topk(min(5, logits_flat.size(-1)), dim=-1)
            top1_match = top1.squeeze(-1) == targets_flat
            top5_match = top5.eq(targets_flat.unsqueeze(-1)).any(dim=-1)
            top1_correct += top1_match.masked_select(mask_flat).sum().item()
            top5_correct += top5_match.masked_select(mask_flat).sum().item()
            total_tokens += mask_flat.sum().item()

    model.train()
    avg_loss = sum(losses) / max(1, len(losses))
    return {
        "loss": avg_loss,
        "perplexity": math.exp(min(avg_loss, 20.0)),
        "acc_top1": top1_correct / max(1, total_tokens),
        "acc_top5": top5_correct / max(1, total_tokens),
        "tokens": total_tokens,
    }


def _save_bundle(
    path: str,
    *,
    model,
    cfg: TinyConfig,
    tokenizer: BaseTokenizer,
    optimizer=None,
    step: int | None = None,
    include_optimizer: bool = False,
    lora_state: dict[str, torch.Tensor] | None = None,
    ema_state: dict | None = None,
    training_metrics: dict[str, Any] | None = None,
):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "config": vars(cfg),
    }
    payload.update(tokenizer.checkpoint_payload())
    if step is not None:
        payload["step"] = step
    if include_optimizer and optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if lora_state is not None:
        payload["lora_state"] = lora_state
    if ema_state is not None:
        payload["ema_state"] = ema_state
    if training_metrics is not None:
        payload["training_metrics"] = dict(training_metrics)
    target = Path(path)
    fd, temp_path = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent or Path(".")))
    os.close(fd)
    try:
        torch.save(payload, temp_path)
        os.replace(temp_path, target)
    finally:
        try:
            Path(temp_path).unlink(missing_ok=True)
        except OSError:
            pass


def _read_best_validation_loss(path: str) -> float | None:
    """Return a persisted best validation loss, or None for legacy/invalid bundles."""
    try:
        bundle = torch.load(path, map_location="cpu", weights_only=True)
        metrics = bundle.get("training_metrics", {}) if isinstance(bundle, dict) else {}
        value = metrics.get("validation_loss") if isinstance(metrics, dict) else None
        loss = float(value)
        if math.isfinite(loss) and loss >= 0:
            return loss
    except (OSError, RuntimeError, TypeError, ValueError):
        pass
    return None


class EMATracker:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if name in self.shadow and param.requires_grad:
                self.shadow[name].lerp_(param.data, 1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model: nn.Module):
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, sd: dict):
        self.decay = sd["decay"]
        self.shadow = sd["shadow"]


def build_llrd_param_groups(
    model: nn.Module,
    base_lr: float,
    weight_decay: float,
    decay_factor: float,
) -> list[dict]:
    if decay_factor >= 1.0:
        return [
            {
                "params": [p for p in model.parameters() if p.requires_grad],
                "lr": base_lr,
                "weight_decay": weight_decay,
            }
        ]

    groups: list[dict] = []
    decay_params = []
    no_decay_params = []
    seen = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        seen.add(id(param))
        depth = name.count("blocks.") + 1
        layer_lr = base_lr * (decay_factor ** depth)
        group = {
            "params": [param],
            "lr": layer_lr,
            "weight_decay": 0.0 if "norm" in name or "bias" in name else weight_decay,
        }
        groups.append(group)

    return groups if groups else [
        {
            "params": [p for p in model.parameters() if p.requires_grad],
            "lr": base_lr,
            "weight_decay": weight_decay,
        }
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="", help="Instruction data (User:/Ickle: format) for fine-tuning, or max chars when --stream-dataset is set")
    parser.add_argument("--pretrain-data", default="", help="Raw text data for language model pretraining stage")
    parser.add_argument("--pretrain-steps", type=int, default=0, help="Number of pretraining steps (0 = skip pretraining)")
    parser.add_argument("--stream-dataset", default="", help="HF dataset ID to stream from instead of reading --data as a file")
    parser.add_argument("--stream-field", default="text", help="Field name in the streamed dataset containing text")
    parser.add_argument("--stream-dataset-2", default="", help="Second HF dataset ID to stream and mix in")
    parser.add_argument("--stream-field-2", default="text", help="Field name in the second streamed dataset containing text")
    parser.add_argument("--stream-template-2", default="", help="Python f-string template for the second stream (e.g. '{row[lemma]}: {row[gloss]}')")
    parser.add_argument("--stream-max-chars-2", type=int, default=0, help="Max chars to consume from second stream (0 = same as stream-max-chars)")
    parser.add_argument(
        "--stream-template",
        default="",
        help="Optional format template for streamed rows, e.g. 'User: {instruction}\\n{context}\\n\\nIckle: {response}'.",
    )
    parser.add_argument("--stream-filter", default="", help="Python filter expr, e.g. row.get('language')=='English'")
    parser.add_argument("--stream-config", default="", help="Dataset config/subset")
    parser.add_argument("--stream-max-chars", type=int, default=2000000, help="Max chars to accumulate when streaming")
    parser.add_argument(
        "--stream-shuffle-buffer",
        type=int,
        default=10000,
        help="Approximate shuffle buffer size for streamed datasets (0 disables shuffling and reads raw storage order)",
    )
    parser.add_argument(
        "--stream-shuffle-seed",
        type=int,
        default=-1,
        help="Shuffle seed for streamed datasets (-1 = a fresh random seed each run, so repeated/continued runs "
        "sample a different slice of the dataset instead of always the same leading rows)",
    )
    parser.add_argument(
        "--stream-role-map",
        default="Human=User,Assistant=Ickle",
        help="Comma-separated OldRole=NewRole pairs remapped in streamed dialogue text before accumulation. "
        "Many chat/preference HF datasets format multi-turn text as 'Human: ...\\n\\nAssistant: ...' "
        "(e.g. Anthropic/hh-rlhf's chosen/rejected fields); Ickle's own template and build_loss_mask() "
        "only recognize 'User:'/'Ickle:', so without this a dataset in that format never actually matches "
        "Ickle's response-masking boundary (silently falling back to unmasked raw-text training) and the "
        "model can pick up literal 'Human:'-shaped fragments as if they were ordinary content. Empty string "
        "disables remapping (safe no-op for plain text sources like fineweb, which have no such markers).",
    )
    parser.add_argument("--out", default=str(Path("models") / "ickle.pt"))
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--pretrain-lr", type=float, default=0.0, help="Learning rate for pretraining (0 = use --lr)")
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--lr-hold-steps", type=int, default=0, help="Hold LR at peak for N steps before decay")
    parser.add_argument("--lr-min-ratio", type=float, default=0.1, help="Min LR as fraction of peak LR")
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "muon"], help="Optimizer: adamw or muon (Muon for 2D weights + AdamW for 1D)")
    parser.add_argument("--lr-schedule", default="cosine", choices=["cosine", "linear", "constant"], help="LR schedule: cosine (smooth), linear (1-cycle, faster convergence), constant (warmup only)")
    parser.add_argument("--contrastive-coeff", type=float, default=0.0, help="Contrastive auxiliary loss coefficient (0 = disabled, 0.1 recommended)")
    parser.add_argument("--embed-norm", action="store_true", help="Normalize token embeddings to unit norm after each optimizer step")
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--max-train-chars", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)
    add_resource_pct_args(parser)
    parser.add_argument("--block-size", type=int, default=0)
    parser.add_argument("--n-embd", type=int, default=0)
    parser.add_argument("--n-head", type=int, default=0)
    parser.add_argument("--n-kv-heads", type=int, default=0, help="KV heads for GQA (0 = MHA, same as n_head)")
    parser.add_argument("--n-layer", type=int, default=0)
    parser.add_argument("--z-loss-coeff", type=float, default=1e-4, help="Z-loss regularization coefficient (0 = disabled)")
    parser.add_argument("--layer-drop-rate", type=float, default=0.0, help="Stochastic depth / layer drop probability (0 = disabled)")
    parser.add_argument("--n-pred-tokens", type=int, default=0, help="Multi-token prediction auxiliary heads count (0 = disabled, Meta 2024)")
    parser.add_argument("--use-checkpoint", action="store_true", help="Enable gradient checkpointing to reduce memory")
    parser.add_argument("--ema-decay", type=float, default=0.0, help="EMA decay rate for shadow weights (0 = disabled, 0.999 recommended)")
    parser.add_argument("--llrd-decay", type=float, default=1.0, help="Layer-wise LR decay factor per layer (1.0 = uniform). Earlier layers get lr * decay^depth.")
    parser.add_argument("--init-model", default="")
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--resume-from-checkpoint", default="")
    parser.add_argument("--best-model-path", default="", help="Save best model (lowest val loss) to this path")
    parser.add_argument(
        "--replace-existing-best",
        action="store_true",
        help="Allow replacing a legacy best-model file that has no comparable validation metric.",
    )
    parser.add_argument("--save-best-on-interrupt", action="store_true", help="Save best model on Ctrl+C")
    parser.add_argument("--target-loss", type=float, default=0.0, help="Stop training early when val_loss reaches this target (0 = disabled)")
    parser.add_argument("--max-target-steps", type=int, default=50000, help="Maximum steps when using --target-loss (safety cap)")
    parser.add_argument("--auto-register", action="store_true", default=True, help="After training, evaluate and register model as a scoped knowledge delta")
    parser.add_argument("--no-auto-register", dest="auto_register", action="store_false", help="Skip post-training evaluation and delta registration")
    parser.add_argument(
        "--tokenizer",
        default="sentencepiece",
        choices=["sentencepiece", "char", "auto"],
        help="Tokenizer type for new training runs. Default sentencepiece (BPE). Use char for legacy compatibility.",
    )
    parser.add_argument("--spm-vocab-size", type=int, default=0, help="SentencePiece vocabulary size. 0 = auto based on profile.")
    parser.add_argument("--spm-model-type", default="bpe", choices=["bpe", "unigram"], help="SentencePiece model type.")
    parser.add_argument("--lora", action="store_true", help="Use LoRA adapter training on top of a frozen base model.")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank (r).")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha scaling factor.")
    parser.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout rate.")
    parser.add_argument("--lora-targets", default="q,k,v,proj,w1,w2,w3,lm_head", help="Comma-separated LoRA target module tags.")
    parser.add_argument("--lora-load", default="", help="Path to a previously trained LoRA state dict (.pt or checkpoint) to resume fine-tuning.")
    parser.add_argument(
        "--lora-adapter-only",
        action="store_true",
        help="When --lora is enabled, save only adapter tensors + metadata instead of full model weights.",
    )
    parser.add_argument("--amp", default="", choices=["", "bf16", "fp16"], help="Enable automatic mixed precision (bf16 recommended on Ampere+)")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile on the model")
    parser.add_argument("--compile-mode", default="default", choices=["default", "reduce-overhead", "max-autotune"], help="torch.compile mode")
    parser.add_argument("--auto-threads", action="store_true", help="Auto-tune torch thread count from CPU count")
    parser.add_argument("--status-file", default="", help="Path to write live training status JSON (updated every eval step)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    rc = resolve_resource_config(args)
    print(rc.summary())
    torch_threads = args.torch_threads if args.torch_threads > 0 else rc.torch_threads
    apply_cpu_thread_budget(torch_threads)
    accel = detect_accelerator()
    device = accel.device
    print(f"Using accelerator backend: {accel.backend} ({', '.join(accel.names) if accel.names else 'CPU'})")

    amp_enabled = bool(args.amp)
    amp_dtype: torch.dtype | None = None
    scaler: torch.cuda.amp.GradScaler | None = None
    if amp_enabled:
        amp_dtype, scaler = resolve_amp_dtype(args.amp)
    amp_device_type = get_amp_device_type()

    if args.stream_dataset:
        print(f"streaming from HF dataset: {args.stream_dataset}")
        from datasets import load_dataset
        ds_kwargs = {"path": args.stream_dataset, "split": "train", "streaming": True}
        ds_config = args.stream_config.strip() if hasattr(args, "stream_config") and args.stream_config else None
        if ds_config:
            ds_kwargs["name"] = ds_config
        try:
            stream = load_dataset(**ds_kwargs)
        except ValueError as exc:
            msg = str(exc).lower()
            if "namespace/name" in msg or "repository id must be" in msg:
                raise ValueError(
                    f"'{args.stream_dataset}' isn't a valid Hugging Face dataset id -- it needs an "
                    f"organization/user prefix (e.g. 'Salesforce/wikitext', not 'wikitext'). "
                    f"Original error: {exc}"
                ) from exc
            if not ds_config and "config" in msg:
                raise ValueError(
                    f"Dataset '{args.stream_dataset}' requires a config/subset name "
                    f"(e.g. --stream-config wikitext-2-raw-v1). Original error: {exc}"
                ) from exc
            raise

        # Without this, `for row in stream` reads a streamed dataset's raw
        # storage order -- typically grouped by source/topic/time, not
        # random -- so a bounded slice (however large) is a biased sample of
        # whatever happens to be first, not a representative one, and every
        # run/continuation reads the *same* leading rows since nothing else
        # varies the starting point. HF's buffer-based `.shuffle()` works on
        # an IterableDataset without downloading the whole thing first.
        shuffle_buffer = max(0, int(args.stream_shuffle_buffer))
        if shuffle_buffer > 0:
            shuffle_seed = int(args.stream_shuffle_seed)
            if shuffle_seed < 0:
                shuffle_seed = random.randint(0, 2**31 - 1)
            stream = stream.shuffle(seed=shuffle_seed, buffer_size=shuffle_buffer)
            print(f"shuffling stream (buffer={shuffle_buffer}, seed={shuffle_seed})")

        text_field = args.stream_field.strip() or "text"
        stream_template = (args.stream_template or "").strip()
        max_chars = int(args.stream_max_chars)
        row_filter_expr = (args.stream_filter or "").strip()
        role_map = _parse_stream_role_map(args.stream_role_map)
        text = ""
        consumed = 0
        for row in stream:
            if not isinstance(row, dict):
                continue
            if row_filter_expr:
                try:
                    if not eval(row_filter_expr, {"__builtins__": {}}, {"row": row}):
                        continue
                except Exception:
                    continue
            if stream_template:
                value = _render_stream_template(stream_template, row)
            else:
                value = row.get(text_field, "")
            if not isinstance(value, str) or len(value) < 80:
                continue
            if role_map:
                value = apply_stream_role_map(value, role_map)
            text += value + "\n\n"
            consumed += 1
            if len(text) >= max_chars:
                break
        print(f"streamed {consumed} records, accumulated {len(text)} chars")
        if len(text) < 120:
            raise ValueError(f"Streamed dataset {args.stream_dataset} produced too little text ({len(text)} chars). Check --stream-filter and --stream-field.")
        if args.stream_dataset_2:
            ds2 = args.stream_dataset_2.strip()
            field2 = args.stream_field_2.strip() or "text"
            template2 = (args.stream_template_2 or "").strip()
            max_chars_2 = int(args.stream_max_chars_2) if args.stream_max_chars_2 > 0 else max_chars
            stream2 = load_dataset(ds2, split="train", streaming=True)
            if shuffle_buffer > 0:
                stream2 = stream2.shuffle(seed=shuffle_seed, buffer_size=shuffle_buffer)
            consumed2 = 0
            text2 = ""
            for row in stream2:
                if not isinstance(row, dict):
                    continue
                if template2:
                    value = _render_stream_template(template2, row)
                else:
                    value = row.get(field2, "")
                if not isinstance(value, str) or len(value) < 3:
                    continue
                if role_map:
                    value = apply_stream_role_map(value, role_map)
                text2 += value + "\n\n"
                consumed2 += 1
                if len(text2) >= max_chars_2:
                    break
            text = text + "\n\n" + text2
            print(f"streamed {consumed2} records from {ds2}, accumulated {len(text2)} chars")
    elif args.data:
        text = load_text(args.data)
    else:
        raise ValueError("Either --data or --stream-dataset must be provided.")
    if args.max_train_chars > 0:
        text = text[: args.max_train_chars]

    resume_bundle: dict[str, Any] | None = None
    resume_path = args.resume_from_checkpoint.strip()
    if resume_path:
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        resume_bundle = torch.load(resume_path, map_location=device)
        print(f"resumed from checkpoint: {resume_path}")

    init_bundle: dict[str, Any] | None = None
    init_path = args.init_model.strip()
    if not resume_bundle and init_path:
        if not os.path.exists(init_path):
            raise FileNotFoundError(f"init model not found: {init_path}")
        init_bundle = torch.load(init_path, map_location=device)
        print(f"initialized from model: {init_path}")

    # Resolved once, unconditionally: the resume/init branches load their model
    # config from the checkpoint itself and never touched this, but
    # effective_batch below reads mc["batch_size"] regardless of which branch
    # ran -- leaving it unset on the resume/init paths raised
    # UnboundLocalError the moment anyone combined --init-model with the
    # default (unset) --batch-size.
    mc = resolve_model_config(args, rc)

    start_step = 0
    if resume_bundle:
        cfg_dict = resume_bundle.get("config") or {}
        cfg = TinyConfig(**cfg_dict)
        tokenizer = tokenizer_from_checkpoint(resume_bundle)
        text_for_training = sanitize_text_for_tokenizer(text, tokenizer)
        if len(text_for_training) < 2:
            raise ValueError("No compatible training text left after applying resume tokenizer.")
        encoded = torch.tensor(tokenizer.encode(text_for_training), dtype=torch.long)
        model = ILM(cfg).to(device)
        model.load_state_dict(resume_bundle["model_state"])
        start_step = int(resume_bundle.get("step", -1)) + 1
    elif init_bundle:
        cfg_dict = init_bundle.get("config") or {}
        cfg = TinyConfig(**cfg_dict)
        tokenizer = tokenizer_from_checkpoint(init_bundle)
        text_for_training = sanitize_text_for_tokenizer(text, tokenizer)
        if len(text_for_training) < 2:
            raise ValueError("No compatible training text left after applying init-model tokenizer.")
        encoded = torch.tensor(tokenizer.encode(text_for_training), dtype=torch.long)
        model = ILM(cfg).to(device)
        model.load_state_dict(init_bundle["model_state"])
    else:
        tokenizer_choice = args.tokenizer.lower()
        if tokenizer_choice == "auto":
            try:
                import sentencepiece as _spm
                tokenizer_choice = "sentencepiece"
            except ImportError:
                tokenizer_choice = "char"
        if tokenizer_choice == "sentencepiece":
            spm_vocab_size = int(args.spm_vocab_size)
            if spm_vocab_size <= 0:
                spm_vocab_size = max(2048, min(32768, rc.block_size * 6))
            tokenizer, actual_vocab_size = train_sentencepiece_tokenizer_with_retry(
                text, spm_vocab_size=spm_vocab_size, model_type=args.spm_model_type
            )
            if actual_vocab_size != spm_vocab_size:
                print(f"SPM vocab size adjusted: {spm_vocab_size} -> {actual_vocab_size} (corpus too small)")
        else:
            tokenizer = CharTokenizer.from_text(text)
        text_for_training = text
        print("Encoding training text...", flush=True)
        encoded = torch.tensor(tokenizer.encode(text_for_training), dtype=torch.long)
        print(f"Encoded: {len(encoded)} tokens", flush=True)
        cfg = TinyConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=mc["block_size"],
            n_embd=mc["n_embd"],
            n_head=mc["n_head"],
            n_kv_heads=args.n_kv_heads,
            n_layer=mc["n_layer"],
            z_loss_coeff=args.z_loss_coeff,
            layer_drop_rate=args.layer_drop_rate,
            n_pred_tokens=args.n_pred_tokens,
            use_checkpoint=args.use_checkpoint,
            contrastive_coeff=args.contrastive_coeff,
        )
        print(f"Config: vocab={cfg.vocab_size} block={cfg.block_size} n_embd={cfg.n_embd} n_head={cfg.n_head} n_kv_heads={cfg.n_kv_heads} n_layer={cfg.n_layer}", flush=True)
        model = ILM(cfg).to(device)
        print(f"Model created, moving to device...", flush=True)

    if args.compile and hasattr(torch, "compile"):
        model = model.configure_compile(enable=True, mode=args.compile_mode)
        print(f"torch.compile enabled (mode={args.compile_mode})")
    if amp_dtype is not None:
        model.configure_amp(args.amp, torch.device(device))

    lora_state: dict[str, torch.Tensor] | None = None
    lora_cfg: LoRAConfig | None = None
    if args.lora:
        lora_targets = tuple(t.strip() for t in args.lora_targets.split(",") if t.strip())
        lora_cfg = LoRAConfig(
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=lora_targets,
        )

    effective_batch = args.batch_size if args.batch_size > 0 else mc["batch_size"]
    grad_accum_steps = max(1, int(args.grad_accum_steps))

    pretrain_data_path = args.pretrain_data.strip()
    has_pretrain = bool(pretrain_data_path) and args.pretrain_steps > 0 and not resume_bundle
    if has_pretrain:
        pretrain_text = load_text(pretrain_data_path)
        if args.max_train_chars > 0:
            pretrain_text = pretrain_text[:args.max_train_chars]
        pretrain_encoded = torch.tensor(tokenizer.encode(pretrain_text), dtype=torch.long)
        pn = int(0.9 * len(pretrain_encoded))
        pretrain_train = pretrain_encoded[:pn]
        pretrain_val = pretrain_encoded[pn:]
        pretrain_lr = args.pretrain_lr if args.pretrain_lr > 0.0 else args.lr
        pretrain_optimizer = torch.optim.AdamW(
            model.parameters(), lr=pretrain_lr, betas=(0.9, 0.95),
            eps=1e-8, weight_decay=args.weight_decay,
        )

        def _pretrain_estimate_loss(data, cfg, batch_size, device, eval_iters=20):
            model.eval()
            losses = []
            for _ in range(eval_iters):
                ix = torch.randint(len(data) - cfg.block_size - 1, (batch_size,))
                xb = torch.stack([data[i:i + cfg.block_size] for i in ix]).to(device)
                yb = torch.stack([data[i + 1:i + cfg.block_size + 1] for i in ix]).to(device)
                with torch.no_grad():
                    _, loss = model(xb, yb)
                losses.append(loss.item())
            model.train()
            return sum(losses) / len(losses)

        print(f"Phase 1: Pretraining on raw text ({len(pretrain_encoded)} tokens, {args.pretrain_steps} steps, lr={pretrain_lr})")
        for pstep in range(args.pretrain_steps):
            plr = warmup_stable_cosine_lr(pstep, args.pretrain_steps, pretrain_lr, max(1, args.warmup_steps), args.lr_hold_steps, args.lr_min_ratio)
            for pg in pretrain_optimizer.param_groups:
                pg["lr"] = plr
            pretrain_optimizer.zero_grad(set_to_none=True)
            plosses = []
            for _ in range(grad_accum_steps):
                ix = torch.randint(len(pretrain_train) - cfg.block_size - 1, (effective_batch,))
                xb = torch.stack([pretrain_train[i:i + cfg.block_size] for i in ix]).to(device)
                yb = torch.stack([pretrain_train[i + 1:i + cfg.block_size + 1] for i in ix]).to(device)
                with torch.autocast(device_type=amp_device_type, dtype=amp_dtype) if amp_dtype else nullcontext():
                    _, mloss = model(xb, yb)
                if scaler:
                    scaler.scale(mloss / grad_accum_steps).backward()
                else:
                    (mloss / grad_accum_steps).backward()
                plosses.append(float(mloss.item()))
            if scaler:
                scaler.unscale_(pretrain_optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if scaler:
                scaler.step(pretrain_optimizer)
                scaler.update()
            else:
                pretrain_optimizer.step()
            if pstep % args.eval_every == 0 or pstep == args.pretrain_steps - 1:
                pv = _pretrain_estimate_loss(pretrain_val, cfg, effective_batch, device, args.eval_iters)
                print(f"  pretrain step={pstep} train_loss={sum(plosses)/len(plosses):.4f} val_loss={pv:.4f} lr={plr:.6f}")
        pt_ckpt = f"{args.out}.pretrain.pt"
        _save_bundle(pt_ckpt, model=model, cfg=cfg, tokenizer=tokenizer, optimizer=pretrain_optimizer, step=args.pretrain_steps - 1, include_optimizer=True)
        print(f"Phase 1 complete: {pt_ckpt}")
        print("Phase 2: Instruction fine-tuning with loss masking")

    if args.lora and lora_cfg is not None:
        if args.lora_load:
            lora_bundle = torch.load(args.lora_load, map_location=device)
            if "lora_state" in lora_bundle:
                load_lora_state_dict(model, lora_bundle["lora_state"])
            elif "model_state" in lora_bundle:
                load_lora_state_dict(model, lora_bundle["model_state"])
            else:
                load_lora_state_dict(model, lora_bundle)
        replaced = inject_lora(model, lora_cfg)
        print(f"LoRA injected: {len(replaced)} layers replaced (rank={lora_cfg.rank}, alpha={lora_cfg.alpha})")

    # See shuffle_in_chunks()'s docstring: without this, the positional
    # "last 10%" split below is dominated by whatever was concatenated
    # last, giving a val_loss that reflects memorizing that source rather
    # than real generalization.
    encoded = shuffle_in_chunks(encoded, chunk_size=max(cfg.block_size * 4, 1024), seed=int(args.seed))

    n = int(0.9 * len(encoded))
    train_data = encoded[:n]
    val_data = encoded[n:]
    raw_text = text_for_training if resume_bundle or init_bundle else text
    # Previously hardcoded to all-True here (the default full-training path,
    # unlike lora_train.py, never actually masked the prompt out of the
    # loss) -- now applies the same build_loss_mask() LoRA training already
    # uses, so "User: ...\nIckle: ..." prompt text isn't trained on as if it
    # were the desired output, for both training paths equally.
    loss_mask = build_loss_mask(encoded.tolist(), tokenizer, "")
    train_mask = loss_mask[:n]
    val_mask = loss_mask[n:]

    use_dataloader = len(train_data) > cfg.block_size * 100
    if use_dataloader:
        train_dataset = TextDataset(train_data, train_mask, cfg.block_size)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=effective_batch,
            shuffle=True,
            num_workers=0,
            pin_memory=device.type == "cuda",
            drop_last=True,
        )
        train_iter = iter(train_loader)
    else:
        train_loader = None
        train_iter = None

    # Previously indented one level too far (inside the `else:` above), so
    # these two lines silently never printed whenever use_dataloader was
    # True -- the common case for any corpus bigger than ~100 blocks.
    masked_true = sum(1 for m in loss_mask if m)
    print(f"Loss mask: {masked_true}/{len(loss_mask)} tokens trained on", flush=True)
    print("Building optimizer...", flush=True)
    param_groups = build_llrd_param_groups(
        model,
        base_lr=args.lr,
        weight_decay=args.weight_decay,
        decay_factor=args.llrd_decay,
    )
    lora_params = [p for p in model.parameters() if p.requires_grad] if args.lora else None
    if lora_params is not None:
        param_groups = [{
            "params": lora_params,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        }]
    use_muon = args.optimizer == "muon"
    if use_muon:
        muon_weight_params = []
        adam_1d_params = []
        for pg in param_groups:
            muon_2d = [p for p in pg["params"] if p.ndim >= 2]
            adam_1d = [p for p in pg["params"] if p.ndim < 2]
            if muon_2d:
                muon_weight_params.append(dict(
                    params=muon_2d, use_muon=True,
                    lr=pg.get("lr", args.lr) * 20.0,
                    weight_decay=pg.get("weight_decay", args.weight_decay),
                ))
            if adam_1d:
                adam_1d_params.append(dict(
                    params=adam_1d, use_muon=False,
                    lr=pg.get("lr", args.lr),
                    weight_decay=pg.get("weight_decay", 0.0),
                ))
        if not adam_1d_params:
            adam_1d_params.append(dict(
                params=[], use_muon=False,
                lr=args.lr * 0.01,
                betas=(0.9, 0.95), weight_decay=0.0,
            ))
        optimizer = MuonWithAuxAdam(muon_weight_params + adam_1d_params)
        print(f"Muon optimizer: {sum(len(pg['params']) for pg in muon_weight_params)} weight groups + {sum(len(pg['params']) for pg in adam_1d_params)} scalar groups")
    else:
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=args.lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=args.weight_decay,
        )
    if resume_bundle and isinstance(resume_bundle.get("optimizer_state"), dict):
        optimizer.load_state_dict(resume_bundle["optimizer_state"])

    ema: EMATracker | None = None
    if args.ema_decay > 0.0 and not args.lora:
        ema = EMATracker(model, decay=args.ema_decay)
        if resume_bundle and "ema_state" in resume_bundle:
            ema.load_state_dict(resume_bundle["ema_state"])
            print(f"resumed EMA state (decay={ema.decay})")

    if args.lora and resume_bundle and "lora_state" in resume_bundle:
        lora_state = resume_bundle["lora_state"]
    elif args.lora:
        lora_state = get_lora_state_dict(model)

    effective_tokens_per_update = int(effective_batch) * int(grad_accum_steps)
    checkpoint_every = max(0, int(args.checkpoint_every))
    checkpoint_path = args.checkpoint_path.strip() or f"{args.out}.checkpoint.pt"
    best_model_path = args.best_model_path.strip() or ""

    best_val_loss: float = float("inf")
    protect_existing_best = False
    if best_model_path and Path(best_model_path).is_file() and not args.replace_existing_best:
        persisted_best_loss = _read_best_validation_loss(best_model_path)
        if persisted_best_loss is None:
            protect_existing_best = True
            print(
                f"existing best model protected (no comparable validation metric): {best_model_path}; "
                "use --replace-existing-best to replace it explicitly"
            )
        else:
            best_val_loss = persisted_best_loss
            print(f"existing best validation loss: {best_val_loss:.4f} ({best_model_path})")

    def save_checkpoint(step: int, reason: str):
        if not checkpoint_path:
            return
        _save_bundle(
            checkpoint_path,
            model=model,
            cfg=cfg,
            tokenizer=tokenizer,
            optimizer=optimizer,
            step=step,
            include_optimizer=True,
            lora_state=get_lora_state_dict(model) if args.lora else None,
            ema_state=ema.state_dict() if ema is not None else None,
        )
        print(f"checkpoint saved: {checkpoint_path} step={step} reason={reason}")

    def save_best(step: int, val_loss: float):
        nonlocal best_val_loss
        if not best_model_path or protect_existing_best:
            return
        if val_loss < best_val_loss:
            existing_best = Path(best_model_path)
            if existing_best.is_file():
                backup_path = Path(f"{best_model_path}.previous.pt")
                if not backup_path.exists():
                    backup_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(existing_best, backup_path)
            best_val_loss = val_loss
            if ema is not None:
                saved_sd = {key: value.detach().clone() for key, value in model.state_dict().items()}
                ema.apply_to(model)
                _save_bundle(
                    best_model_path,
                    model=model,
                    cfg=cfg,
                    tokenizer=tokenizer,
                    step=step,
                    lora_state=get_lora_state_dict(model) if args.lora else None,
                    training_metrics={"validation_loss": float(val_loss), "best_step": int(step)},
                )
                model.load_state_dict(saved_sd)
            else:
                _save_bundle(
                    best_model_path,
                    model=model,
                    cfg=cfg,
                    tokenizer=tokenizer,
                    step=step,
                    lora_state=get_lora_state_dict(model) if args.lora else None,
                    ema_state=ema.state_dict() if ema is not None else None,
                    training_metrics={"validation_loss": float(val_loss), "best_step": int(step)},
                )
            print(f"best model saved: {best_model_path} step={step} val_loss={val_loss:.4f}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_steps = args.steps
    if args.target_loss > 0:
        total_steps = max(args.steps, args.max_target_steps)
        print(f"Target loss mode: will train up to {total_steps} steps or until val_loss <= {args.target_loss}")
    print(f"Model: {total_params:,} params ({trainable_params:,} trainable)", flush=True)
    print(f"Device: {device} | Effective batch: {effective_tokens_per_update} | Steps: {total_steps}", flush=True)
    print(f"Starting training loop...", flush=True)
    model.train()
    last_step = start_step - 1
    latest_step = start_step
    _train_start_time = time.time()
    _train_started_at_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _training_run_id = f"train_{os.getpid()}_{int(_train_start_time)}"
    last_eval_metrics: dict[str, Any] = {}
    last_step_train_loss: float | None = None
    last_lr: float | None = None

    def _write_status(path: str, data: dict) -> None:
        payload = dict(data)
        payload.update(
            {
                "run_id": _training_run_id,
                "pid": os.getpid(),
                "started_at_utc": _train_started_at_utc,
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "out_model": str(args.out),
                "checkpoint_path": str(checkpoint_path),
            }
        )
        try:
            status_path = Path(path)
            status_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = status_path.with_suffix(status_path.suffix + ".tmp")
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
            os.replace(tmp_path, status_path)
        except OSError:
            pass

    if args.status_file:
        _write_status(
            args.status_file,
            {
                "step": max(0, start_step),
                "total_steps": total_steps,
                "elapsed_seconds": 0.0,
                "status": "running",
            },
        )
    try:
        for step in range(start_step, total_steps):
            if args.lr_schedule == "linear":
                curr_lr = linear_onecycle_lr(step, total_steps, args.lr, args.warmup_steps, args.lr_min_ratio)
            elif args.lr_schedule == "constant":
                curr_lr = constant_lr(step, total_steps, args.lr, args.warmup_steps, args.lr_min_ratio)
            else:
                curr_lr = warmup_stable_cosine_lr(step, total_steps, args.lr, args.warmup_steps, args.lr_hold_steps, args.lr_min_ratio)
            for param_group in optimizer.param_groups:
                param_group["lr"] = curr_lr

            optimizer.zero_grad(set_to_none=True)
            micro_losses: list[float] = []
            for _ in range(grad_accum_steps):
                if use_dataloader and train_iter is not None:
                    try:
                        xb, yb, mb = next(train_iter)
                    except StopIteration:
                        train_iter = iter(train_loader)
                        xb, yb, mb = next(train_iter)
                    xb = xb.to(device)
                    yb = yb.to(device)
                    mb = mb.to(device)
                else:
                    xb, yb, mb = get_batch(train_data, train_mask, cfg.block_size, effective_batch, device)

                with torch.autocast(device_type=amp_device_type, dtype=amp_dtype) if amp_dtype else nullcontext():
                    _, micro_loss = model(xb, yb, loss_mask=mb)
                if scaler:
                    scaler.scale(micro_loss / grad_accum_steps).backward()
                else:
                    (micro_loss / grad_accum_steps).backward()
                micro_losses.append(float(micro_loss.item()))

            if scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            if args.embed_norm:
                with torch.no_grad():
                    emb = model.token_embedding_table.weight.data
                    emb_norm = torch.nn.functional.normalize(emb, dim=-1)
                    emb.copy_(emb_norm)

            if ema is not None:
                ema.update(model)

            last_step = step
            latest_step = step
            step_train_loss = sum(micro_losses) / max(1, len(micro_losses))
            last_step_train_loss = step_train_loss
            last_lr = curr_lr

            if step % args.eval_every == 0 or step == total_steps - 1:
                eval_results = estimate_loss_advanced(model, val_data, val_mask, cfg, effective_batch, device, args.eval_iters)
                save_best(step, eval_results["loss"])
                eval_metrics = {
                    "val_loss": round(float(eval_results["loss"]), 4),
                    "perplexity": round(float(eval_results["perplexity"]), 2),
                    "acc_top1": round(float(eval_results["acc_top1"]), 3),
                    "acc_top5": round(float(eval_results["acc_top5"]), 3),
                    "best_val_loss": round(float(best_val_loss) if best_val_loss < float("inf") else 0, 4),
                }
                last_eval_metrics = dict(eval_metrics)
                print(
                    f"step={step} train_loss={step_train_loss:.4f}"
                    f" val_loss={eval_results['loss']:.4f}"
                    f" ppl={eval_results['perplexity']:.2f}"
                    f" acc_top1={eval_results['acc_top1']:.3f}"
                    f" acc_top5={eval_results['acc_top5']:.3f}"
                    f" lr={curr_lr:.6f}"
                    f" grad_accum={grad_accum_steps} effective_batch={effective_tokens_per_update}",
                    flush=True,
                )
                if args.target_loss > 0 and float(eval_results["loss"]) <= args.target_loss:
                    print(f"\n  Target loss {args.target_loss} reached at step {step}! (val_loss={eval_results['loss']:.4f})")
                    if args.status_file:
                        status_payload = {
                            "step": step, "total_steps": args.steps,
                            "train_loss": round(step_train_loss, 4),
                            "lr": round(curr_lr, 6),
                            "elapsed_seconds": round(time.time() - _train_start_time, 1) if "_train_start_time" in dir() else 0,
                            "target_reached": True, "status": "completed",
                        }
                        status_payload.update(eval_metrics)
                        _write_status(args.status_file, status_payload)
                    break
                if args.status_file:
                    status_payload = {
                        "step": step,
                        "total_steps": total_steps,
                        "train_loss": round(step_train_loss, 4),
                        "lr": round(curr_lr, 6),
                        "elapsed_seconds": round(time.time() - _train_start_time, 1) if "_train_start_time" in dir() else 0,
                        "status": "running",
                    }
                    status_payload.update(eval_metrics)
                    _write_status(args.status_file, status_payload)
            elif step % 10 == 0:
                if args.status_file:
                    status_payload = {
                        "step": step,
                        "total_steps": total_steps,
                        "train_loss": round(step_train_loss, 4),
                        "lr": round(curr_lr, 6),
                        "elapsed_seconds": round(time.time() - _train_start_time, 1) if "_train_start_time" in dir() else 0,
                        "status": "running",
                    }
                    status_payload.update(last_eval_metrics)
                    _write_status(args.status_file, status_payload)
                print(f"  step={step} loss={step_train_loss:.4f}", flush=True)

            if checkpoint_every > 0 and ((step + 1) % checkpoint_every == 0 or step == args.steps - 1):
                save_checkpoint(step, reason="interval")
        if args.status_file:
            final_status = {
                "step": latest_step,
                "total_steps": args.steps,
                "elapsed_seconds": round(time.time() - _train_start_time, 1) if "_train_start_time" in dir() else 0,
                "status": "completed",
            }
            if last_step_train_loss is not None:
                final_status["train_loss"] = round(last_step_train_loss, 4)
            if last_lr is not None:
                final_status["lr"] = round(last_lr, 6)
            final_status.update(last_eval_metrics)
            _write_status(args.status_file, final_status)
    except KeyboardInterrupt:
        interrupt_step = max(start_step, last_step)
        save_checkpoint(interrupt_step, reason="interrupt")
        if args.save_best_on_interrupt and best_model_path and best_val_loss < float("inf"):
            save_best(interrupt_step, best_val_loss)
        print("training interrupted by user")
        if args.status_file:
            interrupt_status = {
                "step": interrupt_step,
                "total_steps": args.steps,
                "elapsed_seconds": round(time.time() - _train_start_time, 1) if "_train_start_time" in dir() else 0,
                "status": "interrupted",
            }
            if last_step_train_loss is not None:
                interrupt_status["train_loss"] = round(last_step_train_loss, 4)
            if last_lr is not None:
                interrupt_status["lr"] = round(last_lr, 6)
            interrupt_status.update(last_eval_metrics)
            _write_status(args.status_file, interrupt_status)
        raise SystemExit(130)
    except Exception as exc:
        if args.status_file:
            failure_status: dict[str, Any] = {
                "step": max(start_step, last_step),
                "total_steps": total_steps,
                "elapsed_seconds": round(time.time() - _train_start_time, 1),
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}"[:500],
            }
            if last_step_train_loss is not None:
                failure_status["train_loss"] = round(last_step_train_loss, 4)
            if last_lr is not None:
                failure_status["lr"] = round(last_lr, 6)
            failure_status.update(last_eval_metrics)
            _write_status(args.status_file, failure_status)
        raise

    if start_step >= args.steps:
        print(f"no training steps executed; checkpoint already at step={start_step - 1}")

    if args.lora and bool(args.lora_adapter_only):
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        lora_payload: dict[str, Any] = {
            "config": vars(cfg),
            "lora_state": get_lora_state_dict(model),
            "lora": {
                "rank": args.lora_rank,
                "alpha": args.lora_alpha,
                "dropout": args.lora_dropout,
                "target_modules": [t.strip() for t in args.lora_targets.split(",") if t.strip()],
            },
            "base_model": init_path or resume_path or None,
        }
        lora_payload.update(tokenizer.checkpoint_payload())
        torch.save(lora_payload, args.out)
    else:
        _save_bundle(
            args.out,
            model=model,
            cfg=cfg,
            tokenizer=tokenizer,
            lora_state=get_lora_state_dict(model) if args.lora else None,
            ema_state=ema.state_dict() if ema is not None else None,
        )

    meta_path = args.out + ".meta.json"
    meta_payload = {
        "vocab_size": tokenizer.vocab_size,
        "steps": args.steps,
        "batch_size": effective_batch,
        "grad_accum_steps": grad_accum_steps,
        "effective_batch": effective_tokens_per_update,
        "cpu_pct": rc.cpu_percent,
        "ram_pct": rc.ram_percent,
        "gpu_pct": rc.gpu_percent,
        "learning_rate": args.lr,
        "warmup_steps": args.warmup_steps,
        "lr_hold_steps": args.lr_hold_steps,
        "lr_min_ratio": args.lr_min_ratio,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "n_kv_heads": args.n_kv_heads,
        "z_loss_coeff": args.z_loss_coeff,
        "layer_drop_rate": args.layer_drop_rate,
        "n_pred_tokens": args.n_pred_tokens,
        "use_checkpoint": args.use_checkpoint,
        "contrastive_coeff": args.contrastive_coeff,
        "ema_decay": args.ema_decay,
        "llrd_decay": args.llrd_decay,
        "seed": args.seed,
        "optimizer": args.optimizer,
        "lr_schedule": args.lr_schedule,
        "embed_norm": bool(args.embed_norm),
        "torch_threads": torch_threads,
        "start_step": start_step,
        "checkpoint_path": checkpoint_path,
        "resume_from_checkpoint": resume_path or None,
        "init_model": init_path or None,
        "tokenizer_kind": tokenizer.kind,
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "amp": args.amp or None,
        "amp_dtype": str(amp_dtype) if amp_dtype else None,
        "compile": bool(args.compile),
        "compile_mode": args.compile_mode if args.compile else None,
        "auto_threads": bool(args.auto_threads),
    }
    if args.lora:
        meta_payload["lora"] = {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": [t.strip() for t in args.lora_targets.split(",") if t.strip()],
            "lora_load": args.lora_load or None,
            "adapter_only_bundle": bool(args.lora_adapter_only),
        }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_payload, f, indent=2)

    print(f"saved model: {args.out}")

    if args.auto_register and latest_step > 0:
        print("\nPost-training: evaluating and registering as knowledge delta...")
        try:
            from src.scoped_knowledge import get_scoped_manager
            from src.knowledge_extraction import extract_structured_knowledge
            mgr = get_scoped_manager()
            delta_id = str(Path(args.out).stem).replace(".", "_").replace(" ", "_")[:60]
            corpus_sample = text[:3000] if len(text) > 100 else "English language training data"
            knowledge = extract_structured_knowledge(corpus_sample)
            provenance_label = getattr(args, "stream_dataset", "") or "a local corpus"
            if knowledge.get("method") == "fallback":
                # No teacher was configured, so there is no real topic
                # classification to show -- the fallback's "domain_description"
                # is just an echo of whatever text happened to be first in the
                # corpus sample (see knowledge_extraction._fallback_extraction),
                # which reads as a garbled, misleading title (e.g. a scraped
                # news snippet) rather than an honest "we don't know the topic."
                domain_desc = f"Uncategorized -- no topic classifier configured (trained on {len(text)} chars from {provenance_label})"
            else:
                domain_desc = knowledge.get("domain_description") or f"Trained on {len(text)} chars from {provenance_label}"
            mgr.register_delta(
                delta_id=delta_id,
                domain_description=domain_desc[:300],
                description=f"Model trained for {latest_step} steps on streaming English data",
                adapter_path=args.out,
                memory_entries=knowledge.get("facts", [])[:20],
                confidence=0.5,
                provenance=f"stream:{getattr(args, 'stream_dataset', 'local') or 'local'}",
            )
            print(f"  Registered delta: {delta_id}")
            print(f"  Domain: {domain_desc[:100]}...")
            mgr.registry.save_version(delta_id)
        except Exception as e:
            print(f"  Delta registration skipped: {e}")


if __name__ == "__main__":
    main()
