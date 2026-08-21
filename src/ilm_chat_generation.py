"""Model loading and text generation for ILM chat."""

from __future__ import annotations

import os as _os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator

import torch

from src.device_bridge import detect_accelerator, get_amp_device_type
from src.icklization import ick
from src.ilm_chat_utils import _extract_response_text
from src.model import ILM, TinyConfig
from src.speculative_decode import speculative_generate, speculative_generate_simple
from src.system_limits import SystemLimits, clamp_new_tokens
from src.tokenizer import BaseTokenizer, tokenizer_from_checkpoint


@dataclass
class GenerationResult:
    text: str
    reasoning: str = ""
    token_count: int = 0

_MODEL_CACHE: dict[str, tuple[int, ILM, BaseTokenizer]] = {}
_DRAFT_CACHE: dict[str, tuple[int, ILM, BaseTokenizer]] = {}
_MAX_MODEL_CACHE = 3


def _detect_auto_torch_threads() -> int:
    cpu_count = _os.cpu_count() or 4
    return max(1, min(32, cpu_count * 3 // 4))


def _resolve_default_model() -> str:
    for candidate in ick.default_model_search_paths():
        if Path(candidate).exists():
            return candidate
    return ick.fallback_model()


def _load_model_bundle(model_path: str, *, use_compile: bool = False, amp_dtype: str = "") -> tuple[ILM, BaseTokenizer]:
    path = Path(model_path)
    resolved = str(path.resolve())
    mtime_ns = int(path.stat().st_mtime_ns)
    cached = _MODEL_CACHE.get(resolved)
    if cached and cached[0] == mtime_ns:
        return (cached[1], cached[2])

    ckpt = torch.load(model_path, map_location="cpu")
    cfg_dict = ckpt["config"]
    if use_compile:
        cfg_dict = dict(cfg_dict)
        cfg_dict["use_compile"] = True
    cfg = TinyConfig(**cfg_dict)
    tokenizer = tokenizer_from_checkpoint(ckpt)

    model = ILM(cfg)
    if ckpt.get("quantized"):
        model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    accel = detect_accelerator()
    device = accel.device
    model = model.to(device)

    if use_compile and hasattr(torch, "compile") and accel.supports_compile:
        model = model.configure_compile(enable=True)

    if amp_dtype and accel.amp_supported:
        model.configure_amp(amp_dtype, device)

    while len(_MODEL_CACHE) >= _MAX_MODEL_CACHE:
        oldest = next(iter(_MODEL_CACHE))
        del _MODEL_CACHE[oldest]
    _MODEL_CACHE[resolved] = (mtime_ns, model, tokenizer)
    return (model, tokenizer)


def _load_draft_model(draft_model_path: str, target_vocab_size: int) -> tuple[ILM, BaseTokenizer] | None:
    if not draft_model_path:
        return None
    path = Path(draft_model_path)
    if not path.exists():
        return None
    resolved = str(path.resolve())
    mtime_ns = int(path.stat().st_mtime_ns)
    cached = _DRAFT_CACHE.get(resolved)
    if cached and cached[0] == mtime_ns:
        return (cached[1], cached[2])

    ckpt = torch.load(draft_model_path, map_location="cpu")
    cfg = TinyConfig(**ckpt["config"])
    tokenizer = tokenizer_from_checkpoint(ckpt)
    if tokenizer.vocab_size != target_vocab_size:
        return None

    model = ILM(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    accel = detect_accelerator()
    device = accel.device
    model = model.to(device)

    while len(_DRAFT_CACHE) >= _MAX_MODEL_CACHE:
        oldest = next(iter(_DRAFT_CACHE))
        del _DRAFT_CACHE[oldest]
    _DRAFT_CACHE[resolved] = (mtime_ns, model, tokenizer)
    return (model, tokenizer)


def _generate_model_response(
    model: ILM,
    tokenizer: BaseTokenizer,
    prompt_text: str,
    args,
    limits: SystemLimits,
) -> str:
    tokens = tokenizer.encode(prompt_text)
    x = torch.tensor([tokens], dtype=torch.long, device=next(model.parameters()).device)

    max_new = clamp_new_tokens(args.max_new, limits.max_new_tokens)
    with torch.inference_mode():
        if getattr(args, "speculative", False):
            draft_model_path = str(getattr(args, "draft_model", "") or "")
            draft_bundle = _load_draft_model(draft_model_path, tokenizer.vocab_size)
            if draft_bundle is not None:
                draft_model, _ = draft_bundle
                out = speculative_generate(
                    model,
                    draft_model,
                    x,
                    max_new_tokens=max_new,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    gamma=getattr(args, "speculative_gamma", 4),
                )[0].tolist()
            else:
                out = speculative_generate_simple(
                    model,
                    x,
                    max_new_tokens=max_new,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    gamma=getattr(args, "speculative_gamma", 4),
                )[0].tolist()
        else:
            out = model.generate(
                x,
                max_new_tokens=max_new,
                temperature=args.temperature,
                top_k=args.top_k,
            )[0].tolist()

    generated_ids = out[len(tokens) :]
    raw_generated = tokenizer.decode(generated_ids)
    return _extract_response_text(raw_generated)


def _warm_model(model: ILM, tokenizer: BaseTokenizer, device: torch.device):
    """Run a minimal forward pass to warm CUDA kernels and cache allocator."""
    warm_tokens = tokenizer.encode("Hello")
    warm_input = torch.tensor([warm_tokens[: min(len(warm_tokens), 8)]], dtype=torch.long, device=device)
    model.eval()
    with torch.inference_mode():
        for _ in range(3):
            _ = model(warm_input)


def prewarm_cache(model_path: str | None = None, *, use_compile: bool = False, amp_dtype: str = ""):
    """Load and warm the default model into cache at startup."""
    if model_path is None:
        model_path = _resolve_default_model()
    model, tokenizer = _load_model_bundle(model_path, use_compile=use_compile, amp_dtype=amp_dtype)
    device = next(model.parameters()).device
    _warm_model(model, tokenizer, device)


def _generate_model_response_streaming(
    model: ILM,
    tokenizer: BaseTokenizer,
    prompt_text: str,
    args,
    limits: SystemLimits,
) -> Generator[str, None, str]:
    tokens = tokenizer.encode(prompt_text)
    x = torch.tensor([tokens], dtype=torch.long, device=next(model.parameters()).device)
    max_new = clamp_new_tokens(args.max_new, limits.max_new_tokens)

    accumulated_ids: list[int] = []
    last_text = ""
    with torch.inference_mode():
        for token_id in model.generate_streaming(
            x,
            max_new_tokens=max_new,
            temperature=args.temperature,
            top_k=args.top_k,
        ):
            accumulated_ids.append(token_id.item() if hasattr(token_id, "item") else int(token_id))
            full_text = tokenizer.decode(list(accumulated_ids))
            delta = full_text[len(last_text):]
            last_text = full_text
            if delta:
                yield delta

    return _extract_response_text(last_text)


def generate_reasoning_text(
    model: ILM,
    tokenizer: BaseTokenizer,
    prompt_text: str,
    *,
    max_tokens: int,
    temperature: float = 0.4,
    top_k: int = 40,
) -> tuple[str, int]:
    """Run one reasoning pre-pass and return (cleaned_reasoning_text, token_count).

    Shared by `_generate_with_reasoning` below (single-pass "Show thinking")
    and `agent_loop`'s `reasoning_enabled` branch (multi-step agent
    thinking) -- these used to be two independently-maintained copies of the
    same logic, and had already drifted: this one correctly calls `.item()`
    per streamed token, while agent_loop's copy still did
    `torch.cat(reason_ids).tolist()` over a list of `[1,1]` tensors, which
    produces nested lists (`[[id], [id], ...]`) that `tokenizer.decode()`
    can't `int()` over -- a real crash whenever agent mode and thinking mode
    were both enabled, found while wiring up agent mode's UI toggle.
    """
    tokens = tokenizer.encode(prompt_text)
    x = torch.tensor([tokens], dtype=torch.long, device=next(model.parameters()).device)

    reason_ids: list[int] = []
    with torch.inference_mode():
        for token_id in model.generate_streaming(
            x,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
        ):
            reason_ids.append(token_id.item() if hasattr(token_id, "item") else int(token_id))

    reasoning_raw = tokenizer.decode(reason_ids)
    reasoning_raw = reasoning_raw.split("</reason>")[0] if "</reason>" in reasoning_raw else reasoning_raw
    reasoning = re.sub(r'<[^>]+>', '', reasoning_raw).strip()
    return reasoning, len(reason_ids)


def _generate_with_reasoning(
    model: ILM,
    tokenizer: BaseTokenizer,
    prompt_text: str,
    args,
    limits: SystemLimits,
) -> GenerationResult:
    if not getattr(args, "thinking_mode", False):
        text = _generate_model_response(model, tokenizer, prompt_text, args, limits)
        return GenerationResult(text=text)

    thinking_prefix = f"\n\n{ick.assistant_label()} (reasoning step by step):\n<reason>"
    reasoning_prompt = prompt_text + thinking_prefix
    max_reason_tokens = min(120, clamp_new_tokens(args.max_new, limits.max_new_tokens))

    reasoning, token_count = generate_reasoning_text(
        model, tokenizer, reasoning_prompt, max_tokens=max_reason_tokens, temperature=0.4, top_k=args.top_k
    )

    answer_prompt = (
        f"{prompt_text}\n\nReasoning:\n{reasoning}\n\n{ick.assistant_label()}:"
    )
    answer = _generate_model_response(model, tokenizer, answer_prompt, args, limits)

    return GenerationResult(text=answer, reasoning=reasoning, token_count=token_count)


def _stream_delta_decode(
    tokenizer: BaseTokenizer,
    token_stream,
    initial_decoded: str = "",
) -> Generator[dict[str, Any], None, None]:
    accumulated_ids: list[int] = []
    last_text = str(initial_decoded or "")
    for token_id in token_stream:
        accumulated_ids.append(token_id.item() if hasattr(token_id, "item") else int(token_id))
        full_text = tokenizer.decode(list(accumulated_ids))
        delta = full_text[len(last_text):]
        last_text = full_text
        if delta:
            yield {"text": delta}
    yield {"final": last_text}


def generate_events_stream(
    model: ILM,
    tokenizer: BaseTokenizer,
    prompt_text: str,
    args,
    limits: SystemLimits,
):
    """Yield SSE-serializable events for streaming reasoning + response."""
    thinking_mode = getattr(args, "thinking_mode", False)

    if thinking_mode:
        thinking_prefix = f"\n\n{ick.assistant_label()} (reasoning step by step):\n<reason>"
        reasoning_prompt = prompt_text + thinking_prefix
        tokens = tokenizer.encode(reasoning_prompt)
        x = torch.tensor([tokens], dtype=torch.long, device=next(model.parameters()).device)
        max_reason = min(120, clamp_new_tokens(args.max_new, limits.max_new_tokens))

        yield {"type": "reasoning_start"}
        with torch.inference_mode():
            reason_stream = model.generate_streaming(
                x, max_new_tokens=max_reason, temperature=0.4, top_k=args.top_k,
            )
            for event in _stream_delta_decode(tokenizer, reason_stream):
                if "text" in event:
                    yield {"type": "reasoning", "text": event["text"]}
                elif "final" in event:
                    reasoning_raw = event["final"]
        reasoning_raw = reasoning_raw.split("</reason>")[0] if "</reason>" in reasoning_raw else reasoning_raw
        reasoning = re.sub(r'<[^>]+>', '', reasoning_raw).strip()
        yield {"type": "reasoning_end", "text": reasoning}

        answer_prompt = f"{prompt_text}\n\nReasoning:\n{reasoning}\n\n{ick.assistant_label()}:"
        tokens = tokenizer.encode(answer_prompt)
        x = torch.tensor([tokens], dtype=torch.long, device=next(model.parameters()).device)
    else:
        tokens = tokenizer.encode(prompt_text)
        x = torch.tensor([tokens], dtype=torch.long, device=next(model.parameters()).device)

    max_new = clamp_new_tokens(args.max_new, limits.max_new_tokens)
    yield {"type": "text_start"}
    with torch.inference_mode():
        gen_stream = model.generate_streaming(
            x, max_new_tokens=max_new, temperature=args.temperature, top_k=args.top_k,
        )
        for event in _stream_delta_decode(tokenizer, gen_stream):
            if "text" in event:
                yield {"type": "text", "text": event["text"]}
    yield {"type": "done"}


def _attempt_repair_response(
    *,
    model: ILM,
    tokenizer: BaseTokenizer,
    prompt_sections: list[str],
    args,
    limits: SystemLimits,
) -> str:
    repair_sections = list(prompt_sections)
    repair_sections.insert(1, ick.repair_text())
    repair_prompt = "\n\n".join(repair_sections)
    return _generate_model_response(model, tokenizer, repair_prompt, args, limits)
