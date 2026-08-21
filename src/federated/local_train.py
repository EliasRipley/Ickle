from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import torch

from src.federated.lora import (
    LoRAConfig,
    get_lora_state_dict,
    inject_lora,
    load_lora_state_dict,
    set_only_lora_trainable,
    subtract_states,
)
from src.ilm_profile import apply_cpu_thread_budget
from src.model import ILM, TinyConfig
from src.tokenizer import sanitize_text_for_tokenizer, tokenizer_from_checkpoint


@dataclass
class LocalTrainConfig:
    steps: int = 120
    batch_size: int = 12
    lr: float = 2e-3
    grad_clip: float = 1.0
    torch_threads: int = 4
    max_seconds: float = 0.0
    min_steps: int = 1
    optimizer: str = "adamw"
    embed_norm: bool = False


@dataclass
class StreamDataConfig:
    dataset: str = "HuggingFaceFW/fineweb-edu"
    config: str | None = None
    split: str = "train"
    text_field: str = "text"
    row_filter_expr: str = ""
    max_chars: int = 1_200_000
    min_chars: int = 120


def _load_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _load_stream_text(cfg: StreamDataConfig) -> tuple[str, int]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "The 'datasets' package is required for streaming training data. "
            "Install with: pip install datasets"
        ) from exc

    ds_kwargs: dict[str, Any] = {
        "path": cfg.dataset,
        "split": cfg.split or "train",
        "streaming": True,
    }
    if cfg.config:
        ds_kwargs["name"] = cfg.config
    stream = load_dataset(**ds_kwargs)

    text_field = (cfg.text_field or "text").strip() or "text"
    row_filter_expr = (cfg.row_filter_expr or "").strip()
    max_chars = max(256, int(cfg.max_chars))
    min_chars = max(32, int(cfg.min_chars))

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
        value = row.get(text_field, "")
        if not isinstance(value, str) or len(value) < 80:
            continue
        text += value + "\n\n"
        consumed += 1
        if len(text) >= max_chars:
            break

    if len(text) < min_chars:
        raise RuntimeError(
            f"Streamed dataset {cfg.dataset} produced too little text ({len(text)} chars). "
            "Adjust --stream-field/--stream-filter or increase --stream-max-chars."
        )
    return text, consumed


def load_training_text(
    *,
    local_data_path: str = "",
    stream_data: StreamDataConfig | None = None,
) -> tuple[str, dict[str, Any]]:
    if stream_data is not None:
        text, consumed = _load_stream_text(stream_data)
        return text, {
            "data_source": "stream",
            "stream_dataset": stream_data.dataset,
            "stream_config": stream_data.config or "",
            "stream_split": stream_data.split,
            "stream_field": stream_data.text_field,
            "stream_records": consumed,
            "stream_chars": len(text),
        }

    if not str(local_data_path or "").strip():
        raise ValueError("No training data source provided (local_data_path/stream_data).")
    source_path = Path(local_data_path).resolve()
    text = _load_text(str(source_path))
    return text, {
        "data_source": "local_file",
        "local_data_path": str(source_path),
        "stream_records": 0,
        "stream_chars": len(text),
    }


def _get_batch(data: torch.Tensor, block_size: int, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    if len(data) <= block_size + 1:
        raise ValueError(
            f"Dataset too small for block_size={block_size}. Need > {block_size + 1} tokens, got {len(data)}"
        )
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


def load_base_with_lora(base_model_path: str, lora_cfg: LoRAConfig) -> tuple[ILM, dict[str, Any]]:
    ckpt = torch.load(base_model_path, map_location="cpu")
    cfg = TinyConfig(**ckpt["config"])
    model = ILM(cfg)
    model.load_state_dict(ckpt["model_state"])
    replaced = inject_lora(model, lora_cfg)
    if not replaced:
        raise RuntimeError("No target modules were wrapped with LoRA; check target_modules in config.")
    set_only_lora_trainable(model)
    model.train()
    return model, ckpt


def train_local_delta(
    *,
    base_model_path: str,
    lora_cfg: LoRAConfig,
    global_adapter_state: dict[str, torch.Tensor],
    local_data_path: str = "",
    stream_data: StreamDataConfig | None = None,
    train_text: str = "",
    train_text_meta: dict[str, Any] | None = None,
    train_cfg: LocalTrainConfig,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    apply_cpu_thread_budget(train_cfg.torch_threads)

    model, ckpt = load_base_with_lora(base_model_path, lora_cfg)
    load_lora_state_dict(model, global_adapter_state, strict=True)
    start_state = get_lora_state_dict(model)

    if train_text.strip():
        text = train_text
        source_meta = dict(train_text_meta or {
            "data_source": "provided_text",
            "stream_records": 0,
            "stream_chars": len(text),
        })
    else:
        text, source_meta = load_training_text(local_data_path=local_data_path, stream_data=stream_data)
    tokenizer = tokenizer_from_checkpoint(ckpt)
    text = sanitize_text_for_tokenizer(text, tokenizer)
    ids = tokenizer.encode(text)
    data = torch.tensor(ids, dtype=torch.long)

    from src.training_config import TrainingConfig, build_training_optimizer, apply_embedding_norm
    tc = TrainingConfig(lr=train_cfg.lr, optimizer=train_cfg.optimizer, embed_norm=train_cfg.embed_norm, weight_decay=0.0)
    optimizer = build_training_optimizer(model, tc)
    final_loss = 0.0
    steps_budget = max(1, int(train_cfg.steps))
    min_steps = max(1, int(train_cfg.min_steps))
    max_seconds = max(0.0, float(train_cfg.max_seconds))
    started = time.monotonic()
    steps_ran = 0
    for _ in range(steps_budget):
        if max_seconds > 0.0 and steps_ran >= min_steps:
            elapsed = time.monotonic() - started
            if elapsed >= max_seconds:
                break
        xb, yb = _get_batch(data, block_size=model.cfg.block_size, batch_size=train_cfg.batch_size)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
        if train_cfg.embed_norm:
            apply_embedding_norm(model)
        final_loss = float(loss.item())
        steps_ran += 1

    end_state = get_lora_state_dict(model)
    delta = subtract_states(end_state, start_state)
    elapsed_sec = max(0.0, time.monotonic() - started)
    metrics = {
        "final_loss": final_loss,
        "token_count": float(len(ids)),
        "steps": float(steps_ran),
        "steps_budget": float(steps_budget),
        "steps_ran": float(steps_ran),
        "elapsed_sec": round(elapsed_sec, 4),
    }
    metrics.update(source_meta)
    return delta, metrics


def evaluate_adapter_loss(
    *,
    base_model_path: str,
    lora_cfg: LoRAConfig,
    adapter_state: dict[str, torch.Tensor],
    eval_data_path: str = "",
    eval_text: str = "",
    stream_data: StreamDataConfig | None = None,
    eval_iters: int = 12,
    batch_size: int = 12,
    torch_threads: int = 4,
) -> float:
    apply_cpu_thread_budget(torch_threads)
    model, ckpt = load_base_with_lora(base_model_path, lora_cfg)
    load_lora_state_dict(model, adapter_state, strict=True)
    model.eval()

    if eval_text.strip():
        text = eval_text
    elif eval_data_path.strip():
        text = _load_text(eval_data_path)
    elif stream_data is not None:
        text, _ = load_training_text(local_data_path="", stream_data=stream_data)
    else:
        raise ValueError("No eval source provided (eval_data_path/eval_text/stream_data).")
    tokenizer = tokenizer_from_checkpoint(ckpt)
    text = sanitize_text_for_tokenizer(text, tokenizer)
    ids = tokenizer.encode(text)
    data = torch.tensor(ids, dtype=torch.long)

    losses: list[float] = []
    for _ in range(eval_iters):
        xb, yb = _get_batch(data, block_size=model.cfg.block_size, batch_size=batch_size)
        with torch.no_grad():
            _, loss = model(xb, yb)
        losses.append(float(loss.item()))
    return sum(losses) / len(losses)
