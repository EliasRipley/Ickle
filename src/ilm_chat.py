#!/usr/bin/env python3
"""Ickle chat runtime with lightweight tool and memory routing."""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any
import sys

import torch

from src.agent import LocalAgent
from src.agent_loop import AgentResult, agent_loop
from src.autonomy import default_mode_name, get_mode
from src.device_bridge import detect_accelerator
from src.dynamic_web_reader import read_url_dynamic
from src.tools.image_reader import ImageToolsUnavailable, describe_image, extract_text_from_image
from src.ilm_profile import apply_cpu_thread_budget
from src.evidence_policy import claim_signature, evidence_score, jaccard_similarity, topic_relevance
from src.icklization import ick
from src.ilm_chat_generation import _generate_with_reasoning
from src.ilm_chat_utils import (
    _TOPIC_GENERIC_TOKENS,
    _extract_response_text,
    _is_noisy_memory_fact,
    _is_time_sensitive_request,
    _looks_like_knowledge_request,
    _looks_like_recall_request,
    _looks_low_quality_response,
    _needs_recent_context,
    _prompt_relevance_score,
    _token_set,
    _topic_hint_from_prompt,
    _topic_overlap_count,
)
from src.ilm_memory import get_memory
from src.knowledge_modules import (
    ResolvedKnowledgeModule,
    apply_lora_modules_to_model,
    default_registry_path,
    parse_module_ids,
    resolve_runtime_modules,
)
from src.local_reasoning import local_reasoning_response
from src.model import ILM, TinyConfig
from src.speculative_decode import speculative_generate_simple
from src.skill_system import SkillRegistry
from src.system_limits import SystemLimits, clamp_new_tokens
from src.think_loop import ThinkingLoop, ThinkConfig, assess_self_knowledge, score_confidence
from src.tokenizer import BaseTokenizer, tokenizer_from_checkpoint
from src.web_topic_lookup import collect_topic_web_evidence, wikipedia_payload_from_url

SYSTEM_PROMPT = ick.system_prompt()

# Real behavioral effect for AutonomyMode.tone (src/autonomy.py) -- it used
# to be stored and printed by /policy but never influenced generation at
# all, a config knob that looked functional and wasn't. "neutral" adds no
# guidance (balanced mode's default voice already matches SYSTEM_PROMPT).
_AUTONOMY_TONE_GUIDANCE = {
    "brief": "Tone: be concise. Prefer short, direct sentences over elaboration.",
    "technical": (
        "Tone: address the user as a competent, technical operator. Skip "
        "beginner explanations and hedging; use precise, domain-appropriate "
        "terminology."
    ),
}

URL_PATTERN = re.compile(r"(https?://[^\s<>\"]+)", re.IGNORECASE)
DOMAIN_PATTERN = re.compile(
    r"(?:visit|open|go to|check|read|browse|fetch|look up)\s+"
    r"([a-z0-9][a-z0-9.-]+\.[a-z]{2,}(?:/[^\s]*)?)",
    re.IGNORECASE,
)
REMEMBER_PATTERN = re.compile(r"^\s*(?:please\s+)?remember(?:\s+that)?\s+(.+?)\s*$", re.IGNORECASE)
NAME_PATTERN = re.compile(r"\bmy name is\s+([A-Za-z][A-Za-z \-']{0,48})", re.IGNORECASE)

# Cache model bundles by absolute path and file mtime.
_MODEL_CACHE: dict[str, tuple[int, ILM, BaseTokenizer]] = {}
_COMPOSED_MODEL_CACHE: dict[str, tuple[ILM, BaseTokenizer]] = {}
_MAX_MODEL_CACHE = 3
def _resolve_default_model() -> str:
    from src.model_resolver import resolve_default_model

    return resolve_default_model()


def detect_web_request(prompt: str) -> str | None:
    """Detect URL-like web requests."""
    direct = URL_PATTERN.search(prompt)
    if direct:
        return direct.group(1).rstrip(".,!?;:)]}")

    by_domain = DOMAIN_PATTERN.search(prompt)
    if not by_domain:
        return None

    raw = by_domain.group(1).rstrip(".,!?;:)]}")
    if not raw.startswith(("http://", "https://")):
        raw = f"https://{raw}"
    return raw


def _extract_timer_name(prompt: str) -> str:
    name_match = re.search(r"\b(?:named|called)\s+([A-Za-z0-9][A-Za-z0-9 _-]{0,40})", str(prompt), re.IGNORECASE)
    if name_match:
        return re.sub(r"\s+", " ", name_match.group(1)).strip(" .,:;!?")
    return ""


def _extract_timer_duration_text(prompt: str) -> str:
    duration_match = re.search(
        r"\b(?:for|in)\s+([0-9][A-Za-z0-9\s.,:_-]{0,80})",
        str(prompt),
        re.IGNORECASE,
    )
    if duration_match:
        return re.sub(r"\s+", " ", duration_match.group(1)).strip(" .,:;!?")
    return str(prompt or "").strip()


def _maybe_route_local_tools(prompt: str, limits: SystemLimits) -> str | None:
    text = str(prompt or "").strip()
    if not text:
        return None
    lower = text.lower()

    def _agent() -> LocalAgent:
        return LocalAgent(limits=limits)

    # Explicit command surface for deterministic local tool execution.
    if lower.startswith("/timer-set "):
        payload = text[len("/timer-set ") :].strip()
        duration_text, name = (payload.split("|", 1) + [""])[:2] if "|" in payload else (payload, "")
        try:
            return _agent().timer_set(duration_text=duration_text.strip(), name=name.strip())
        except Exception as exc:  # noqa: BLE001
            return f"Timer error: {exc}"
    if lower == "/timer-list":
        return _agent().timer_list()
    if lower.startswith("/timer-check "):
        return _agent().timer_check(text[len("/timer-check ") :].strip())
    if lower.startswith("/timer-pause "):
        return _agent().timer_pause(text[len("/timer-pause ") :].strip())
    if lower.startswith("/timer-resume "):
        return _agent().timer_resume(text[len("/timer-resume ") :].strip())
    if lower.startswith("/timer-cancel "):
        return _agent().timer_cancel(text[len("/timer-cancel ") :].strip())
    if lower.startswith("/desktop "):
        payload = text[len("/desktop ") :].strip()
        if not payload:
            return "Desktop usage: /desktop <action> <json-payload>"
        action, payload_json = (payload.split(" ", 1) + ["{}"])[:2]
        try:
            return _agent().desktop_control(action=action.strip(), payload_json=payload_json.strip() or "{}")
        except Exception as exc:  # noqa: BLE001
            return f"Desktop error: {exc}"

    # Light natural-language routing for timers and read-only desktop calls.
    if "timer" in lower and re.search(r"\b(?:set|start)\b", lower):
        try:
            return _agent().timer_set(
                duration_text=_extract_timer_duration_text(text),
                name=_extract_timer_name(text),
            )
        except Exception as exc:  # noqa: BLE001
            return f"Timer error: {exc}"
    if re.search(r"\b(?:list|show)\s+(?:all\s+)?timers?\b", lower):
        return _agent().timer_list()
    if "timer" in lower and re.search(r"\b(?:check|status|remaining)\b", lower):
        name = _extract_timer_name(text)
        if name:
            return _agent().timer_check(name)
    if "timer" in lower and re.search(r"\b(?:pause|resume|cancel|stop)\b", lower):
        name = _extract_timer_name(text)
        if not name:
            return "Please provide the timer name, for example: 'pause timer called study'."
        if "pause" in lower:
            return _agent().timer_pause(name)
        if "resume" in lower:
            return _agent().timer_resume(name)
        return _agent().timer_cancel(name)

    if re.search(r"\b(?:take|capture|get)\s+(?:a\s+)?screenshot\b", lower):
        try:
            return _agent().desktop_control(action="screenshot", payload_json=json.dumps({}))
        except Exception as exc:  # noqa: BLE001
            return f"Desktop error: {exc}"
    if "mouse position" in lower:
        try:
            return _agent().desktop_control(action="get_mouse_position", payload_json=json.dumps({}))
        except Exception as exc:  # noqa: BLE001
            return f"Desktop error: {exc}"
    if "screen size" in lower:
        try:
            return _agent().desktop_control(action="get_screen_size", payload_json=json.dumps({}))
        except Exception as exc:  # noqa: BLE001
            return f"Desktop error: {exc}"

    return None


def _load_base_model_bundle(model_path: str) -> tuple[ILM, BaseTokenizer]:
    path = Path(model_path)
    resolved = str(path.resolve())
    mtime_ns = int(path.stat().st_mtime_ns)
    cached = _MODEL_CACHE.get(resolved)
    if cached and cached[0] == mtime_ns:
        return (cached[1], cached[2])

    ckpt = torch.load(model_path, map_location="cpu")
    cfg = TinyConfig(**ckpt["config"])
    tokenizer = tokenizer_from_checkpoint(ckpt)

    model = ILM(cfg)
    if ckpt.get("quantized"):
        model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    # Quantized (INT8 dynamic) models are CPU-only by design -- moving them to
    # an accelerator device would break torch.ao.quantization's kernels, so
    # only route unquantized models to whatever detect_accelerator() finds.
    if not ckpt.get("quantized"):
        model = model.to(detect_accelerator().device)

    while len(_MODEL_CACHE) >= _MAX_MODEL_CACHE:
        oldest = next(iter(_MODEL_CACHE))
        del _MODEL_CACHE[oldest]
    _MODEL_CACHE[resolved] = (mtime_ns, model, tokenizer)
    return (model, tokenizer)


def _module_signature(module_specs: list[ResolvedKnowledgeModule]) -> str:
    parts: list[str] = []
    for item in module_specs:
        module_path = Path(item.path)
        try:
            resolved = str(module_path.resolve())
            mtime = int(module_path.stat().st_mtime_ns)
        except Exception:  # noqa: BLE001
            resolved = str(module_path)
            mtime = 0
        parts.append(f"{item.module_id}|{resolved}|{mtime}|{float(item.weight):.6f}")
    return "||".join(parts)


def _load_model_bundle(
    model_path: str,
    *,
    module_specs: list[ResolvedKnowledgeModule] | None = None,
) -> tuple[ILM, BaseTokenizer]:
    if not module_specs:
        return _load_base_model_bundle(model_path)

    path = Path(model_path)
    resolved = str(path.resolve())
    mtime_ns = int(path.stat().st_mtime_ns)
    signature = _module_signature(module_specs)
    cache_key = f"{resolved}|{mtime_ns}|{signature}"
    cached = _COMPOSED_MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    base_model, tokenizer = _load_base_model_bundle(model_path)
    composed_model = copy.deepcopy(base_model)
    report = apply_lora_modules_to_model(composed_model, module_specs)
    if int(report.get("applied_layers", 0)) <= 0:
        return base_model, tokenizer

    while len(_COMPOSED_MODEL_CACHE) >= _MAX_MODEL_CACHE:
        oldest = next(iter(_COMPOSED_MODEL_CACHE))
        del _COMPOSED_MODEL_CACHE[oldest]
    _COMPOSED_MODEL_CACHE[cache_key] = (composed_model, tokenizer)
    return composed_model, tokenizer


def _extract_key_facts_from_web(web_data: dict[str, Any], max_items: int = 8) -> list[str]:
    facts: list[str] = []
    for item in web_data.get("evidence_items", [])[: max_items * 2]:
        text = str(item.get("claim", "")).strip()
        if text:
            facts.append(text)
            if len(facts) >= max_items:
                return facts
    for headline in web_data.get("headlines", [])[:max_items]:
        text = str(headline.get("text", "")).strip()
        if text:
            facts.append(text)
            if len(facts) >= max_items:
                return facts
    return facts


def _build_image_context(image_path: str) -> str:
    """Run OCR + captioning on an attached image and format the results as
    prompt context, the same "external tool result -> plain text context"
    pattern format_web_content_for_ilm below already uses for web pages.
    Ickle's own model stays text-only; this is a bolt-on tool like
    web_read/news_research, not a multimodal core model. Both the OCR and
    captioning tools are optional (requirements-vision.txt) -- a missing
    dependency surfaces as an honest note in the context, not a crash."""
    parts: list[str] = []
    try:
        text = extract_text_from_image(image_path)
        if text:
            parts.append(f"Text found in the image:\n{text}")
    except ImageToolsUnavailable as exc:
        parts.append(f"(Could not read text from the image: {exc})")
    except Exception as exc:  # noqa: BLE001
        parts.append(f"(Error reading text from the image: {exc})")

    try:
        description = describe_image(image_path)
        if description:
            parts.append(f"Image description: {description}")
    except ImageToolsUnavailable as exc:
        parts.append(f"(Could not describe the image: {exc})")
    except Exception as exc:  # noqa: BLE001
        parts.append(f"(Error describing the image: {exc})")

    return "\n\n".join(parts)


def format_web_content_for_ilm(web_data: dict[str, Any]) -> str:
    """Format web data for prompt context."""
    if not web_data.get("success"):
        return f"Web read failed: {web_data.get('error', 'unknown error')}"

    lines = [
        f"URL: {web_data.get('url', '')}",
        f"Title: {web_data.get('title', 'Untitled page')}",
    ]

    description = str(web_data.get("description", "")).strip()
    if description:
        lines.append(f"Description: {description}")

    source_quality = web_data.get("source_quality", {}) or {}
    quality_score = float(source_quality.get("score", 0.0))
    if quality_score > 0:
        lines.append(
            "Source quality: "
            f"{quality_score:.2f} "
            f"(domain={float(source_quality.get('domain_score', 0.0)):.2f}, "
            f"freshness={float(source_quality.get('freshness_score', 0.0)):.2f})"
        )

    structure = web_data.get("structure", {}) or {}
    structure_bits = []
    if structure.get("has_main"):
        structure_bits.append("main")
    if structure.get("has_header"):
        structure_bits.append("header")
    if structure.get("has_nav"):
        structure_bits.append("navigation")
    if structure.get("has_aside"):
        structure_bits.append("sidebar")
    if structure.get("has_footer"):
        structure_bits.append("footer")
    if structure.get("framework_detected"):
        structure_bits.append(f"framework={structure['framework_detected']}")
    if structure_bits:
        lines.append("Structure: " + ", ".join(structure_bits))

    evidence_items = list(web_data.get("evidence_items", []))
    if evidence_items:
        lines.append("Evidence candidates:")
        for item in evidence_items[:10]:
            claim = str(item.get("claim", "")).strip()
            if not claim:
                continue
            score = float(item.get("score", 0.0))
            corr = int(item.get("corroboration_count", 0))
            kind = str(item.get("kind", "content")).strip()
            lines.append(f"- [{kind} s={score:.2f} c={corr}] {claim}")
    else:
        key_facts = _extract_key_facts_from_web(web_data, max_items=10)
        if key_facts:
            lines.append("Headlines:")
            lines.extend(f"- {item}" for item in key_facts)

    body = str(web_data.get("content", "")).strip()
    if body:
        clipped = body[:2500]
        lines.append("Extracted content sample:")
        lines.append(clipped)

    return "\n".join(lines)


def format_topic_web_context_for_ilm(topic_data: dict[str, Any]) -> str:
    if not topic_data.get("success"):
        return f"Topic web evidence failed: {topic_data.get('error', 'unknown error')}"

    topic = str(topic_data.get("topic", "")).strip()
    lines = [f"Topic: {topic}" if topic else "Topic evidence"]
    sources = list(topic_data.get("sources", []))
    if not sources:
        return "Topic web evidence has no sources."

    lines.append("Web sources:")
    for row in sources[:4]:
        title = str(row.get("title", "")).strip() or "Untitled"
        url = str(row.get("url", "")).strip()
        relevance = float(row.get("relevance", 0.0))
        quality = row.get("quality", {}) if isinstance(row.get("quality"), dict) else {}
        quality_score = float(quality.get("score", 0.0))
        lines.append(f"- {title} | r={relevance:.2f} q={quality_score:.2f} | {url}")
        facts = [str(item).strip() for item in list(row.get("facts", [])) if str(item).strip()]
        for fact in facts[:3]:
            lines.append(f"  - {fact}")

    facts = [str(item).strip() for item in list(topic_data.get("facts", [])) if str(item).strip()]
    if facts:
        lines.append("Merged evidence:")
        for fact in facts[:12]:
            lines.append(f"- {fact}")
    return "\n".join(lines)


def _format_factual_response(entries: list[dict[str, Any]], *, prefix: str) -> str | None:
    if not entries:
        return None
    lines: list[str] = []
    for row in entries:
        fact = str(row.get("fact", "")).strip()
        if not fact:
            continue
        if _is_noisy_memory_fact(fact):
            continue
        if len(fact) < 30:
            continue
        source = str(row.get("source_title", "")).strip() or str(row.get("topic", "")).strip()
        if source:
            lines.append(f"{source}: {fact}")
        else:
            lines.append(fact)
        if len(lines) >= 3:
            break
    if not lines:
        return None
    return f"{prefix} " + " ".join(lines)




def _memory_knowledge_response(memory, prompt: str) -> str | None:
    evidence = _collect_memory_evidence(memory, prompt, max_items=4)
    if not evidence:
        return None
    lines: list[str] = []
    for row in evidence[:3]:
        claim = str(row.get("claim", "")).strip()
        source = str(row.get("source", "")).strip()
        score = float(row.get("score", 0.0))
        if not claim:
            continue
        if source:
            lines.append(f"[{score:.2f}] {source}: {claim}")
        else:
            lines.append(f"[{score:.2f}] {claim}")
    if not lines:
        return None
    return "From stored evidence: " + " ".join(lines)


def _collect_memory_evidence(memory, prompt: str, max_items: int = 8) -> list[dict[str, Any]]:
    topic_hint = _topic_hint_from_prompt(prompt)
    rows: list[dict[str, Any]] = []

    for row in memory.search_web_facts(prompt, limit=8, topic_hint=topic_hint):
        claim = str(row.get("fact", "")).strip()
        if not claim or _is_noisy_memory_fact(claim):
            continue
        relevance = max(_prompt_relevance_score(prompt, claim), topic_relevance(topic_hint or prompt, claim))
        score = evidence_score(
            relevance=relevance,
            source_quality=0.74,
            confidence=float(row.get("confidence", row.get("score", 0.72))),
            corroboration_count=int(row.get("corroboration_count", 0)),
        )
        rows.append(
            {
                "claim": claim,
                "source": str(row.get("source_title", "") or row.get("source_url", "")).strip(),
                "kind": "web_fact",
                "score": score,
                "relevance": relevance,
                "signature": claim_signature(claim),
            }
        )

    for row in memory.search_facts(prompt, limit=6, topic_hint=topic_hint):
        claim = str(row.get("fact", "")).strip()
        if not claim or _is_noisy_memory_fact(claim):
            continue
        relevance = _prompt_relevance_score(prompt, claim)
        score = evidence_score(
            relevance=relevance,
            source_quality=0.62,
            confidence=float(row.get("confidence", 0.65)),
            corroboration_count=0,
        )
        rows.append(
            {
                "claim": claim,
                "source": str(row.get("source", "") or row.get("category", "")).strip(),
                "kind": "fact",
                "score": score,
                "relevance": relevance,
                "signature": claim_signature(claim),
            }
        )

    for row in memory.search_research_notes(prompt, limit=6, topic_hint=topic_hint):
        claim = str(row.get("finding", "")).strip()
        if not claim or _is_noisy_memory_fact(claim):
            continue
        relevance = _prompt_relevance_score(prompt, claim)
        score = evidence_score(
            relevance=relevance,
            source_quality=0.76,
            confidence=float(row.get("confidence", 0.7)),
            corroboration_count=0,
        )
        rows.append(
            {
                "claim": claim,
                "source": str(row.get("source_title", "") or row.get("source_url", "")).strip(),
                "kind": "research_note",
                "score": score,
                "relevance": relevance,
                "signature": claim_signature(claim),
            }
        )

    # Add cross-memory corroboration bonus.
    for idx, row in enumerate(rows):
        corr = 0
        for j, other in enumerate(rows):
            if idx == j:
                continue
            if str(row.get("source", "")).strip().lower() == str(other.get("source", "")).strip().lower():
                continue
            if jaccard_similarity(str(row.get("claim", "")), str(other.get("claim", ""))) >= 0.72:
                corr += 1
        if corr > 0:
            row["score"] = evidence_score(
                relevance=float(row.get("relevance", 0.0)),
                source_quality=0.72 if row.get("kind") == "web_fact" else 0.68,
                confidence=0.78 if row.get("kind") == "web_fact" else 0.70,
                corroboration_count=corr,
            )
            row["corroboration_count"] = corr

    rows.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    selected: list[dict[str, Any]] = []
    seen_claims: set[str] = set()
    for row in rows:
        claim = str(row.get("claim", "")).strip()
        key = claim.lower()
        if not claim or key in seen_claims:
            continue
        if float(row.get("score", 0.0)) < 0.35:
            continue
        seen_claims.add(key)
        selected.append(row)
        if len(selected) >= max(1, max_items):
            break
    return selected


def _memory_evidence_context(memory, prompt: str, max_items: int = 6) -> str:
    evidence = _collect_memory_evidence(memory, prompt, max_items=max_items)
    if not evidence:
        return ""
    lines = []
    for row in evidence:
        claim = str(row.get("claim", "")).strip()
        source = str(row.get("source", "")).strip()
        score = float(row.get("score", 0.0))
        corr = int(row.get("corroboration_count", 0))
        kind = str(row.get("kind", "")).strip()
        if source:
            lines.append(f"- [{kind} s={score:.2f} c={corr}] {claim} (source: {source})")
        else:
            lines.append(f"- [{kind} s={score:.2f} c={corr}] {claim}")
    return "Memory evidence:\n" + "\n".join(lines)


def _memory_recall_response(memory, prompt: str) -> str | None:
    if not _looks_like_recall_request(prompt):
        return None
    response = _memory_knowledge_response(memory, prompt)
    if response:
        return response
    return "I do not have a stored memory for that yet."




def _max_entry_overlap_ratio(prompt: str, rows: list[dict[str, Any]], keys: tuple[str, ...]) -> float:
    p_tokens = _token_set(prompt)
    if not p_tokens:
        return 0.0
    best = 0.0
    for row in rows:
        parts = [str(row.get(k, "")).strip() for k in keys]
        text = " ".join(x for x in parts if x)
        if not text:
            continue
        overlap = len(p_tokens.intersection(_token_set(text))) / max(1, len(p_tokens))
        if overlap > best:
            best = overlap
    return best


def _try_memory_write(memory, prompt: str) -> str | None:
    """Handle explicit memory-save requests deterministically."""
    name_match = NAME_PATTERN.search(prompt)
    if name_match:
        owner_name = name_match.group(1).strip()
        memory.set_owner_info(name=owner_name)
        return f"Understood. I will remember your name is {owner_name}."

    remember_match = REMEMBER_PATTERN.match(prompt)
    if remember_match:
        fact = remember_match.group(1).strip(" .")
        if fact:
            memory.add_fact(fact=fact, category="user_note", source="user", confidence=1.0)
            return "Understood. I saved that to memory."

    return None


def _build_memory_context(memory, prompt: str) -> str:
    sections: list[str] = []
    evidence_block = _memory_evidence_context(memory, prompt, max_items=6)
    if evidence_block:
        sections.append(evidence_block)

    lower = prompt.lower()
    if "my name" in lower or "who am i" in lower:
        owner = memory.get_owner_info()
        if owner.get("name"):
            sections.append(f"Known owner name: {owner['name']}")

    if _needs_recent_context(prompt):
        recent = memory.get_recent_context(limit=2)
        if recent:
            lines: list[str] = []
            for item in recent:
                user_input = str(item.get("user_input", "")).strip()
                response = str(item.get("ickle_response", "")).strip()
                if user_input:
                    lines.append(f"User: {user_input[:160]}")
                if response:
                    lines.append(f"Ickle: {response[:180]}")
            if lines:
                sections.append("Recent context:\n" + "\n".join(lines))

    return "\n\n".join(sections)


def _required_topic_overlap(prompt: str) -> int:
    p_tokens = {t for t in _token_set(prompt) if t not in _TOPIC_GENERIC_TOKENS}
    if not p_tokens:
        return 1
    # One specific shared term is enough for concise factual answers; requiring
    # two rejects valid paraphrases (for example, an astrolabe definition).
    return 1


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
    with torch.no_grad():
        if bool(getattr(args, "speculative", False)):
            gamma = max(1, int(getattr(args, "speculative_gamma", 3) or 3))
            out = speculative_generate_simple(
                model=model,
                idx=x,
                max_new_tokens=max_new,
                temperature=args.temperature,
                top_k=args.top_k,
                gamma=gamma,
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


def _attempt_repair_response(
    *,
    model: ILM,
    tokenizer: BaseTokenizer,
    prompt_sections: list[str],
    args,
    limits: SystemLimits,
) -> str:
    repair_sections = list(prompt_sections)
    repair_sections.insert(
        1,
        (
            "Response repair mode:\n"
            "- Answer the user's latest question directly in 1-4 sentences.\n"
            "- Stay on topic and include concrete content.\n"
            "- Do not output meta commentary about being an AI or about instructions.\n"
            "- Do not output turn markers such as 'User:' or 'Ickle:'."
        ),
    )
    repair_prompt = "\n\n".join(repair_sections)
    repair_args = argparse.Namespace(**vars(args))
    repair_args.temperature = min(float(getattr(args, "temperature", 0.55)), 0.2)
    repair_args.top_k = 1
    repair_args.max_new = min(int(getattr(args, "max_new", 120)), 120)
    return _generate_model_response(model, tokenizer, repair_prompt, repair_args, limits)


def _make_result(response: str, *, reasoning: str = "", model: str = "", low_confidence: bool = False) -> dict:
    return {"response": response, "reasoning": reasoning, "model": model, "low_confidence": low_confidence}


def _answer_evidence_items(
    web_data: dict[str, Any] | None,
    topic_web_data: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Flatten retrieved evidence for the web UI's inspectable answer map.

    This metadata is not presented as proof.  ``src.epistemics`` performs a
    second, conservative text-overlap link from each generated candidate
    claim to these source passages and labels the result "source linked".
    """

    rows: list[dict[str, Any]] = []
    if web_data and web_data.get("success"):
        fallback_url = str(web_data.get("url", "")).strip()
        fallback_title = str(web_data.get("title", "")).strip()
        for raw in list(web_data.get("evidence_items", []))[:24]:
            if not isinstance(raw, dict):
                continue
            row = dict(raw)
            row.setdefault("source_url", fallback_url)
            row.setdefault("source_title", fallback_title)
            rows.append(row)
    if topic_web_data and topic_web_data.get("success"):
        for source in list(topic_web_data.get("sources", []))[:8]:
            if not isinstance(source, dict):
                continue
            fallback_url = str(source.get("url", "")).strip()
            fallback_title = str(source.get("title", "")).strip()
            for raw in list(source.get("evidence_items", []))[:16]:
                if not isinstance(raw, dict):
                    continue
                row = dict(raw)
                row.setdefault("source_url", fallback_url)
                row.setdefault("source_title", fallback_title)
                rows.append(row)
    return rows[:64]


def extract_response_text(result: Any) -> str:
    """Normalize generate_response output into plain assistant text."""
    value = result
    if isinstance(result, dict):
        value = result.get("response", "")
    return str(value or "").strip()


def _format_agent_trace(agent_result: AgentResult) -> str:
    """Render agent_loop's step-by-step tool-call trace as readable text for
    the "thinking" block, instead of discarding it and keeping only the
    single reasoning string (agent_result.tool_calls was already being
    populated with every call/param/result -- it just never left this
    function before)."""
    lines: list[str] = []
    if agent_result.reasoning:
        lines.append(agent_result.reasoning.strip())
    for i, call in enumerate(agent_result.tool_calls, start=1):
        tool = call.get("tool", "?")
        params = call.get("params", {})
        params_str = ", ".join(f"{k}={v}" for k, v in params.items())
        lines.append(f"Step {i}: called {tool}({params_str})")
        if "error" in call:
            lines.append(f"  -> error: {call['error']}")
        elif "result" in call:
            result_preview = str(call["result"])[:300]
            lines.append(f"  -> {result_preview}")
    return "\n".join(lines)


def generate_response(args):
    """Generate response with optional tool, memory, and reasoning context.

    Returns a dict with keys: response, reasoning, model
    """
    memory_enabled = bool(getattr(args, "enable_memory", True))
    web_enabled = bool(getattr(args, "enable_web_tools", True))
    thinking_mode = bool(getattr(args, "thinking", False) or getattr(args, "thinking_mode", False))
    agent_mode = bool(getattr(args, "agent", False) or getattr(args, "agent_mode", False))
    memory = get_memory() if memory_enabled else None
    limits = SystemLimits(max_new_tokens=args.max_new_limit, torch_threads=args.torch_threads)
    apply_cpu_thread_budget(limits.torch_threads)
    time_sensitive = _is_time_sensitive_request(args.prompt)
    _model = str(getattr(args, "model", ""))
    knowledge_registry = str(getattr(args, "knowledge_registry", default_registry_path()))
    knowledge_max_modules = max(1, int(getattr(args, "knowledge_max_modules", 2) or 2))
    knowledge_auto = bool(getattr(args, "knowledge_auto", True))
    auto_web_knowledge = bool(getattr(args, "auto_web_knowledge", True))
    web_knowledge_max_sources = max(1, int(getattr(args, "web_knowledge_max_sources", 3) or 3))
    web_knowledge_timeout_ms = max(5000, int(getattr(args, "web_knowledge_timeout_ms", 20000) or 20000))
    explicit_module_ids = parse_module_ids(str(getattr(args, "knowledge_modules", "")))
    selected_modules: list[ResolvedKnowledgeModule] = []
    try:
        selected_modules = resolve_runtime_modules(
            prompt=str(args.prompt or ""),
            base_model=str(args.model or ""),
            registry_path=knowledge_registry,
            explicit_module_ids=explicit_module_ids,
            max_modules=knowledge_max_modules,
            auto_select=knowledge_auto,
        )
    except Exception:  # noqa: BLE001
        selected_modules = []

    if memory:
        memory_write_ack = _try_memory_write(memory, args.prompt)
        if memory_write_ack:
            memory.remember_conversation(args.prompt, memory_write_ack, {"memory_write": True})
            return _make_result(memory_write_ack, model=_model)

        direct_memory_recall = _memory_recall_response(memory, args.prompt)
        if direct_memory_recall:
            memory.remember_conversation(args.prompt, direct_memory_recall, {"memory_recall": True})
            return _make_result(direct_memory_recall, model=_model)

    local_reasoning = local_reasoning_response(args.prompt)
    if local_reasoning:
        if memory:
            memory.remember_conversation(args.prompt, local_reasoning, {"local_reasoning": True})
        return _make_result(local_reasoning, model=_model)

    autonomy_mode = str(getattr(args, "autonomy_mode", None) or default_mode_name())
    guardrail_agent = LocalAgent(limits=limits, autonomy_mode=autonomy_mode)

    if limits.require_clarification_on_vague:
        clarification = guardrail_agent.maybe_request_clarification(str(args.prompt or ""))
        if clarification.needs_clarification:
            if memory:
                memory.remember_conversation(args.prompt, clarification.question, {"clarification_requested": True})
            return _make_result(clarification.question, model=_model)

    tool_reply = _maybe_route_local_tools(args.prompt, limits)
    if tool_reply:
        if memory:
            memory.remember_conversation(args.prompt, tool_reply, {"local_tool_route": True})
        return _make_result(tool_reply, model=_model)

    model, tokenizer = _load_model_bundle(args.model, module_specs=selected_modules or None)

    # `selected_modules` above (knowledge_modules.py) and scoped_knowledge below are
    # intentionally two different mechanisms, not redundant: knowledge_modules picks
    # LoRA adapters by topic and permanently merges them into the loaded model, while
    # scoped_knowledge picks deltas by router score, applies them as reversible gates,
    # and contributes the memory_context fact block used a few lines down. See
    # ARCHITECTURE.md's module map for the full distinction before "simplifying" this.
    scoped_knowledge_result = None
    try:
        from src.scoped_knowledge import get_scoped_manager
        scoped_mgr = get_scoped_manager()
        scoped_mgr.set_model(model, tokenizer)
        scoped_knowledge_result = scoped_mgr.activate_for_query(str(args.prompt or ""))
    except Exception as exc:  # noqa: BLE001 -- optional context layer, chat must still proceed without it
        print(f"scoped_knowledge activation failed, continuing without it: {exc}", file=sys.stderr)

    web_data: dict[str, Any] | None = None
    topic_web_data: dict[str, Any] | None = None
    web_context = ""
    url = detect_web_request(args.prompt) if web_enabled else None
    if url:
        web_data = read_url_dynamic(url, timeout_ms=25000, max_chars=15000)
        if (not web_data.get("success")) or int(web_data.get("word_count", 0)) <= 0:
            wiki_fallback = wikipedia_payload_from_url(url, max_chars=12000, timeout_ms=20000)
            if wiki_fallback and wiki_fallback.get("success"):
                web_data = wiki_fallback
        if not web_data.get("success"):
            response = f"I could not read {url}: {web_data.get('error', 'unknown error')}."
            if memory:
                memory.remember_conversation(args.prompt, response, {"web_request": True, "web_success": False})
            return _make_result(response, model=_model)

        key_facts = _extract_key_facts_from_web(web_data, max_items=6)
        if key_facts and memory:
            topic = str(web_data.get("title", "")).split(" - ")[0].strip()
            topic = topic if topic else None
            fact_meta: dict[str, dict[str, Any]] = {}
            for item in list(web_data.get("evidence_items", []))[:12]:
                claim = str(item.get("claim", "")).strip()
                if not claim:
                    continue
                fact_meta[claim] = {
                    "score": float(item.get("score", 0.0)),
                    "confidence": float(item.get("confidence", 0.75)),
                    "corroboration_count": int(item.get("corroboration_count", 0)),
                }
            memory.add_web_learning(
                url=url,
                title=str(web_data.get("title", "")),
                key_facts=key_facts,
                topic=topic,
                fact_metadata=fact_meta,
            )
        web_context = format_web_content_for_ilm(web_data)
    elif web_enabled and auto_web_knowledge and _looks_like_knowledge_request(args.prompt):
        topic_hint = _topic_hint_from_prompt(args.prompt) or str(args.prompt or "").strip()
        try:
            topic_web_data = collect_topic_web_evidence(
                topic=topic_hint,
                max_sources=web_knowledge_max_sources,
                timeout_ms=web_knowledge_timeout_ms,
                max_chars=12000,
            )
        except Exception:  # noqa: BLE001
            topic_web_data = None
        if topic_web_data and topic_web_data.get("success"):
            web_context = format_topic_web_context_for_ilm(topic_web_data)
            if memory:
                for source in list(topic_web_data.get("sources", []))[: max(1, web_knowledge_max_sources)]:
                    source_url = str(source.get("url", "")).strip()
                    source_title = str(source.get("title", "")).strip()
                    facts = [str(item).strip() for item in list(source.get("facts", [])) if str(item).strip()]
                    if not source_url or not facts:
                        continue
                    fact_meta: dict[str, dict[str, Any]] = {}
                    for item in list(source.get("evidence_items", []))[:12]:
                        claim = str((item or {}).get("claim", "")).strip()
                        if not claim:
                            continue
                        fact_meta[claim] = {
                            "score": float((item or {}).get("score", 0.0)),
                            "confidence": float((item or {}).get("confidence", 0.75)),
                            "corroboration_count": int((item or {}).get("corroboration_count", 0)),
                        }
                    memory.add_web_learning(
                        url=source_url,
                        title=source_title or source_url,
                        key_facts=facts[:8],
                        topic=topic_hint,
                        fact_metadata=fact_meta,
                    )

    prompt_sections = [
        SYSTEM_PROMPT,
        (
            "Evidence policy:\n"
            "- Use memory/web evidence when available.\n"
            "- Do not invent citations or unsupported facts.\n"
            "- If evidence is weak, answer with uncertainty and propose verification."
        ),
    ]

    tone_guidance = _AUTONOMY_TONE_GUIDANCE.get(get_mode(autonomy_mode).tone)
    if tone_guidance:
        prompt_sections.append(tone_guidance)

    if args.skill:
        registry = SkillRegistry()
        activation = registry.activation_prompt(args.skill)
        prompt_sections.append(f"Skill activation:\n{activation}")

    epistemic_context = str(getattr(args, "epistemic_context", "") or "").strip()
    if epistemic_context:
        prompt_sections.append(epistemic_context)

    if selected_modules:
        active = ", ".join(item.module_id for item in selected_modules)
        prompt_sections.append(f"Active knowledge modules: {active}")

    memory_context = _build_memory_context(memory, args.prompt) if memory else ""
    if memory_context:
        prompt_sections.append("Memory context:\n" + memory_context)

    if scoped_knowledge_result and scoped_knowledge_result.get("memory_context"):
        prompt_sections.append("Relevant domain knowledge:\n" + scoped_knowledge_result["memory_context"])

    if web_context:
        prompt_sections.append("Web context:\n" + web_context)

    image_path = getattr(args, "image_path", None)
    if image_path:
        image_context = _build_image_context(image_path)
        if image_context:
            prompt_sections.append("Image context:\n" + image_context)

    prompt_sections.append(f"User: {args.prompt}")
    prompt_sections.append("Ickle:")
    prompt_text = "\n\n".join(prompt_sections)

    response = _generate_model_response(model, tokenizer, prompt_text, args, limits)

    reasoning = ""
    if thinking_mode:
        gen_result = _generate_with_reasoning(model, tokenizer, prompt_text, args, limits)
        response = gen_result.text
        reasoning = gen_result.reasoning

    if agent_mode:
        agent_result = agent_loop(model, tokenizer, SYSTEM_PROMPT, args.prompt, args, limits)
        response = agent_result.response
        reasoning = _format_agent_trace(agent_result)

    # Quality/relevance signals drive a non-text confidence indicator only --
    # they never cause the model's own words to be swapped for canned text
    # (persona.toml templates) or synthesized web-snippet prose. Ickle's real
    # output, first pass or repaired, is always what gets shown; see
    # low_confidence below and ARCHITECTURE.md's honesty-gate history.
    raw_output = bool(getattr(args, "raw_output", False))
    low_quality = _looks_low_quality_response(response)
    topic_overlap = _topic_overlap_count(args.prompt, response)
    required_overlap = _required_topic_overlap(args.prompt)
    low_relevance = _prompt_relevance_score(args.prompt, response) < 0.18
    if _looks_like_knowledge_request(args.prompt) and topic_overlap < required_overlap:
        low_relevance = True
    if _looks_like_knowledge_request(args.prompt) and time_sensitive and topic_overlap == 0:
        low_relevance = True

    if not raw_output and (low_quality or low_relevance):
        repaired = _attempt_repair_response(
            model=model,
            tokenizer=tokenizer,
            prompt_sections=prompt_sections,
            args=args,
            limits=limits,
        )
        if repaired:
            repaired_low_quality = _looks_low_quality_response(repaired)
            repaired_relevance = _prompt_relevance_score(args.prompt, repaired)
            repaired_topic_overlap = _topic_overlap_count(args.prompt, repaired)
            repaired_on_topic = repaired_relevance >= 0.20
            if _looks_like_knowledge_request(args.prompt) and repaired_topic_overlap < required_overlap:
                repaired_on_topic = False
            # Only switch to the repaired attempt if it's a real improvement --
            # otherwise keep showing the first pass rather than discarding it.
            if not repaired_low_quality and repaired_on_topic:
                response = repaired
                low_quality = repaired_low_quality
                low_relevance = not repaired_on_topic

    low_confidence = bool(low_quality or low_relevance)

    # A genuinely empty generation (zero tokens) is left as an empty string --
    # the UI renders that as a distinct empty-state, never composed text.

    if memory:
        memory.remember_conversation(
            args.prompt,
            response,
            {
                "web_request": bool(url) or bool(topic_web_data),
                "web_success": bool((web_data and web_data.get("success")) or (topic_web_data and topic_web_data.get("success"))),
            },
        )
    if scoped_knowledge_result:
        try:
            scoped_mgr.deactivate_all()
        except Exception as exc:  # noqa: BLE001 -- don't crash the response over cleanup, but don't hide it either
            print(f"scoped_knowledge deactivate_all failed, gate state may leak into next request: {exc}", file=sys.stderr)

    think_budget = max(0, int(getattr(args, "think_budget", 0) or 0))
    if think_budget != 0 or thinking_mode or agent_mode:
        confidence = score_confidence(response, str(args.prompt or ""))
        assessment = assess_self_knowledge(response, str(args.prompt or ""))
        if assessment["needs_research"] and not web_context and not web_data:
            try:
                topic_hint = str(args.prompt or "")[:200]
                research_data = collect_topic_web_evidence(
                    topic=topic_hint,
                    max_sources=3,
                    timeout_ms=20000,
                    max_chars=8000,
                )
                if research_data and research_data.get("success"):
                    research_context = format_topic_web_context_for_ilm(research_data)
                    research_prompt_sections = list(prompt_sections)
                    research_prompt_sections.insert(-2, "Updated research:\n" + research_context)
                    research_prompt_text = "\n\n".join(research_prompt_sections)
                    response = _generate_model_response(model, tokenizer, research_prompt_text, args, limits)
                    if memory:
                        memory.remember_conversation(
                            args.prompt, response,
                            {"think_loop_research": True, "web_success": True},
                        )
            except Exception:
                pass

        final_confidence = score_confidence(response, str(args.prompt or ""))
        result = _make_result(response, reasoning=reasoning, model=args.model, low_confidence=low_confidence)
        result["confidence"] = final_confidence["confidence"]
        result["think_assessment"] = assessment
        result["evidence_items"] = _answer_evidence_items(web_data, topic_web_data)
        return result

    result = _make_result(response, reasoning=reasoning, model=args.model, low_confidence=low_confidence)
    result["evidence_items"] = _answer_evidence_items(web_data, topic_web_data)
    return result


def interactive_mode(
    model_path: str,
    *,
    speculative: bool = False,
    speculative_gamma: int = 3,
    enable_memory: bool = True,
    enable_web_tools: bool = True,
):
    """Interactive chat loop."""
    print("=== ICKLE ILM CHAT ===")
    print("Type messages in plain English.")
    print("Type 'quit', 'exit', or press Ctrl+C to stop.")
    print("Web reading: include URLs like https://example.com")
    print("-" * 50)

    class Args:
        def __init__(self, prompt: str):
            self.model = model_path
            self.prompt = prompt
            self.max_new = 300
            self.max_new_limit = 500
            self.temperature = 0.55
            self.top_k = 30
            self.torch_threads = 4
            self.skill = ""
            self.speculative = bool(speculative)
            self.speculative_gamma = max(1, int(speculative_gamma))
            self.enable_memory = bool(enable_memory)
            self.enable_web_tools = bool(enable_web_tools)
            self.knowledge_registry = default_registry_path()
            self.knowledge_modules = ""
            self.knowledge_max_modules = 2
            self.knowledge_auto = True
            self.auto_web_knowledge = True
            self.web_knowledge_max_sources = 3
            self.web_knowledge_timeout_ms = 20000
            self.thinking = False
            self.thinking_mode = False
            self.agent = False
            self.agent_mode = False
            self.think_budget = 0

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in {"quit", "exit", "q"}:
                print("Goodbye!")
                break
            if not user_input:
                continue
            result = generate_response(Args(user_input))
            reasoning = result.get("reasoning", "")
            if reasoning:
                print(f"\n[Ickle is thinking...]\n{reasoning}\n")
            response_text = result.get("response", result)
            confidence = result.get("confidence")
            if confidence is not None:
                pct = int(confidence * 100)
                bar = "█" * (pct // 10) + "░" * (10 - pct // 10)
                print(f"Ickle: {response_text}")
                print(f"  confidence: {bar} {pct}%")
            else:
                print(f"Ickle: {response_text}")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as exc:
            print(f"Error: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Ickle ILM Chat")
    parser.add_argument("--model", required=False, help="Path to trained ILM model")
    parser.add_argument("--prompt", default="Hello", help="Input prompt")
    parser.add_argument("--max-new", type=int, default=300, help="Maximum new tokens to generate")
    parser.add_argument("--max-new-limit", type=int, default=500, help="Maximum allowed new tokens")
    parser.add_argument("--temperature", type=float, default=0.55, help="Generation temperature")
    parser.add_argument("--top-k", type=int, default=30, help="Top-k sampling")
    parser.add_argument("--torch-threads", type=int, default=4, help="PyTorch threads")
    parser.add_argument("--skill", default="", help="Optional registered skill to activate")
    parser.add_argument("--speculative", action="store_true", help="Enable speculative decoding (single-model mode).")
    parser.add_argument("--speculative-gamma", type=int, default=3, help="Draft length for speculative decoding.")
    parser.add_argument("--enable-memory", dest="enable_memory", action="store_true", help="Enable memory integration.")
    parser.add_argument("--disable-memory", dest="enable_memory", action="store_false", help="Disable memory integration.")
    parser.add_argument(
        "--enable-web-tools",
        dest="enable_web_tools",
        action="store_true",
        help="Enable web-tool context routing.",
    )
    parser.add_argument(
        "--disable-web-tools",
        dest="enable_web_tools",
        action="store_false",
        help="Disable web-tool context routing.",
    )
    parser.set_defaults(enable_memory=True, enable_web_tools=True)
    parser.add_argument("--interactive", action="store_true", help="Start interactive chat mode")
    parser.add_argument("--thinking", action="store_true", help="Enable D-CoT two-pass reasoning mode")
    parser.add_argument("--agent", action="store_true", help="Enable multi-step agent loop with tool use")
    parser.add_argument(
        "--raw",
        dest="raw_output",
        action="store_true",
        help=(
            "Show the model's actual output verbatim, skipping the quality/relevance gate that "
            "normally replaces garbled or off-topic responses with an honest uncertainty message. "
            "Useful for evaluating a freshly trained or experimental model's real capability -- "
            "the gate is designed for end-user chat, where it's more helpful than misleading "
            "output, but it makes a weak model's output indistinguishable from a canned response."
        ),
    )
    parser.add_argument("--think-budget", type=int, default=0, help="Max thinking tokens to spend on research/iteration (0 = unlimited)")
    parser.add_argument("--draft-model", default="", help="Optional speculative draft model path")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    parser.add_argument("--amp", default="", help="AMP dtype for CUDA (bf16, fp16)")
    parser.add_argument("--knowledge-registry", default=default_registry_path())
    parser.add_argument("--knowledge-modules", default="", help="Comma-separated module ids to force-enable")
    parser.add_argument("--knowledge-max-modules", type=int, default=2, help="Max auto-selected modules")
    parser.add_argument("--enable-knowledge-auto", dest="knowledge_auto", action="store_true")
    parser.add_argument("--disable-knowledge-auto", dest="knowledge_auto", action="store_false")
    parser.add_argument("--enable-auto-web-knowledge", dest="auto_web_knowledge", action="store_true")
    parser.add_argument("--disable-auto-web-knowledge", dest="auto_web_knowledge", action="store_false")
    parser.add_argument("--web-knowledge-max-sources", type=int, default=3)
    parser.add_argument("--web-knowledge-timeout-ms", type=int, default=20000)
    parser.add_argument(
        "--autonomy-mode",
        dest="autonomy_mode",
        default=None,
        help="Autonomy mode (balanced/direct/power-user); defaults to policy's default_mode.",
    )
    parser.set_defaults(knowledge_auto=True)
    parser.set_defaults(auto_web_knowledge=True)
    args = parser.parse_args()

    if not args.model:
        args.model = _resolve_default_model()

    if args.interactive:
        interactive_mode(
            args.model,
            speculative=bool(args.speculative),
            speculative_gamma=max(1, int(args.speculative_gamma)),
            enable_memory=bool(args.enable_memory),
            enable_web_tools=bool(args.enable_web_tools),
        )
        return

    result = generate_response(args)
    text = str(result.get("response", result))
    try:
        print(text)
    except UnicodeEncodeError:
        fallback = text.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(
            sys.stdout.encoding or "utf-8",
            errors="replace",
        )
        print(fallback)


if __name__ == "__main__":
    main()
