"""Persona configuration loader for Ickle.

Loads config/persona.toml and provides all Ickle's personality,
response text, and detection patterns. Falls back to hardcoded
defaults if the file is missing.

Override path via ICKLE_PERSONA_CONFIG env var.
"""

from __future__ import annotations

import os
import platform
import tomllib
from pathlib import Path
from typing import Any


_PERSONA_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config" / "persona.toml"


_PERSONA_DEFAULTS: dict[str, Any] = {
    "persona": {
        "name": "Ickle",
        "system_prompt": (
            "You are Ickle, a respectful and intellectually rigorous assistant. "
            "Be direct, accurate, and practical. "
            "Prefer grounded evidence over assumptions. "
            "If evidence is weak or missing, state uncertainty clearly and propose a concrete verification step. "
            "Answer humans with clear reasoning and professional courtesy."
        ),
    },
    "dialog": {
        "user_label": "User",
        "assistant_label": "Ickle",
        "stop_markers": ["\nUser:", "\nIckle:", "\nILM:", "\nAssistant:"],
    },
    "behavior": {
        "guidance_uncertainty": (
            "If I am unsure, I should say that clearly, avoid guessing, "
            "and verify with a reliable current source."
        ),
        "guidance_ambiguous": (
            "I should ask one focused clarification question "
            "about the goal, constraints, and success criteria first."
        ),
        "guidance_live_data": (
            "I should not claim live facts from memory alone; "
            "I should check a current source before answering."
        ),
        "guidance_general": (
            "I should be explicit about uncertainty, ask clarifying questions "
            "when needed, and ground claims in evidence."
        ),
    },
    "responses": {
        "greeting": "Hi. What would you like to work on?",
        "memory_saved": "Understood. I saved that to memory.",
        "memory_no_recall": "I do not have a stored memory for that yet.",
        "memory_name_saved": "Understood. I will remember your name is {name}.",
        "memory_timezone_followup": "From our previous message, that is {offset}.",
        "goodbye": "Goodbye!",
    },
    "evidence_policy": {
        "text": (
            "Evidence policy:\n"
            "- Use memory/web evidence when available.\n"
            "- Do not invent citations or unsupported facts.\n"
            "- If evidence is weak, answer with uncertainty and propose verification."
        ),
    },
    "repair_mode": {
        "enabled": False,
        "text": (
            "Response repair mode:\n"
            "- Answer the user's latest question directly in 1-4 sentences.\n"
            "- Stay on topic and include concrete content.\n"
            "- Do not output meta commentary about being an AI or about instructions."
        ),
    },
    "autonomy": {
        "default_mode": "balanced",
        "modes": {
            "balanced": {
                "description": "Direct but cautious; asks for clarification on ambiguity.",
                "tone": "neutral",
            },
            "direct": {
                "description": "Minimal guardrail friction for low/medium-risk tasks.",
                "tone": "brief",
            },
            "power-user": {
                "description": "Assumes competent adult operator; avoids patronizing language.",
                "tone": "technical",
            },
        },
    },
    "capabilities": {
        "unknown_summary": "Unknown capability request.",
        "unknown_suggestion": "Use '/policy' and '/help', or add a new tool integration and register it.",
        "known": {
            "web_read": "Can read webpages via a local headless browser (Playwright Firefox/Chromium, falling back to system Edge/Chrome if needed).",
            "news_research": "Can fetch market/news headlines via RSS research tool.",
            "write_notepad": "Can write text files and open them in Windows Notepad.",
            "minecraft_guide": "Can fetch beginner Minecraft wiki topics.",
            "timer": "Can set, check, pause, resume, and cancel wall-clock timers. Timers persist across sessions.",
            "desktop_control": "Can control mouse, keyboard, and take screenshots via pyautogui with strict allow-list.",
        },
        "suggestions": {
            "timer": "Use 'set a timer for 5 minutes' or 'check my timers'.",
            "desktop_control": "Requires: pip install pyautogui. Allowed: screenshot, click, type_text, key_press, move_mouse, get_mouse_position.",
        },
    },
    "detection": {
        "vague_patterns": [
            "do something", "handle it", "figure it out",
            "you decide", "make a plan",
            "help with marketing", "social media marketing", "help with minecraft",
        ],
        "low_quality_evasive_markers": [
            "i can help with that. share the exact outcome",
            "share the exact outcome you want",
            "i may not have enough reliable local knowledge",
            "if you want, i can research",
            "as an ai",
        ],
        "uncertainty_triggers": [
            "if you do not know", "if you don't know",
            "if you are unsure", "uncertain",
        ],
        "ambiguity_triggers": ["ambiguous", "missing details", "unclear request", "vague"],
        "live_data_triggers": ["without checking", "today's market data", "live data", "latest law update"],
        "general_guidance_triggers": ["how should you", "what should you do first"],
        "time_sensitive_triggers": [
            "latest", "today", "right now", "current", "breaking",
            "news", "stock", "market", "exchange rate",
            "traffic", "outage", "law update", "regulation",
        ],
        "knowledge_prefixes": [
            "what causes ", "what is ", "what are ", "where is ", "where are ",
            "where do ", "where does ", "who is ", "who was ",
            "tell me about", "explain ", "explaining ", "explanation of ", "define ", "describe ",
            "how are ", "how does ", "how do ", "how should ",
            "what should ", "if you are unsure", "summarize ",
        ],
        "recall_triggers": [
            "remember", "what do you know", "what did i tell you",
            "from earlier", "from before", "previous conversation", "recall",
        ],
        "context_followup_starters": [
            "and ", "also ", "then ", "so ",
            "what about", "how about", "ok ", "okay ",
        ],
        "context_reference_tokens": ["that", "it", "those", "them", "same", "again"],
        "needs_context_phrases": ["earlier", "before", "previous", "last message", "that topic", "continue"],
        "live_time_hints": [
            "what time", "what is the time", "tell me the time",
            "right now", "current time", "currently", "now", "time in ",
        ],
    },
    "clarification": {
        "vague_prompt_question": (
            "I might guess wrong from this prompt. "
            "Could you clarify objective, audience, output format, and constraints before I continue?"
        ),
        "social_prefix": "I should pause to avoid guessing. Please clarify: ",
        "minecraft_prefix": "I should pause so I do not give the wrong Minecraft advice. Please clarify: ",
        "social": {
            "platform": "Which platform should I target first (e.g., X, LinkedIn, Instagram, TikTok)?",
            "audience": "Who is the exact audience persona (role, industry, pain point)?",
            "goal": "What is the primary goal (awareness, leads, clicks, signups, sales)?",
            "offer": "What product/service or offer should the post promote?",
        },
        "minecraft": {
            "edition": "Are you on Java Edition or Bedrock Edition?",
            "platform": "What device are you using (Windows PC, console, mobile)?",
            "experience": "Are you brand new, intermediate, or returning?",
            "goal": "What do you want first (survive day 1, build a house, automate farms, beat the Ender Dragon)?",
        },
    },
    "cloud_assist": {
        "configured_text": "Cloud assist is configured (ILM_CLOUD_API_KEY present).",
        "not_configured_text": "Cloud assist is NOT configured. Set ILM_CLOUD_API_KEY to enable optional cloud help.",
    },
    "models": {
        "default_search_paths": [
            "models/ickle_best.pt",
            "models/ickle_language_first.pt",
            "models/ickle_clean.pt",
            "models/ickle_web_conversation.pt",
            "models/ickle_final.pt",
        ],
        "fallback_model": "models/ickle_best.pt",
    },
    "training": {
        "bootstrap_text": (
            "You are a helpful local assistant.\n"
            "You understand and generate English text clearly.\n"
            "When you are uncertain, ask a short clarifying question.\n"
            "Give concise, practical steps first, then optional detail.\n\n"
            "The user may ask for web research, writing help, coding help,\n"
            "and summaries. Always be polite and direct.\n\n"
            "This is starter English data only. Real quality comes from\n"
            "continued fine-tuning on your own curated dataset."
        ),
    },
    "partner_loop": {
        "underspecified_reason": "Prompt is underspecified; proceeding risks hallucination.",
        "underspecified_next_step": "Ask user for objective, constraints, and expected output format.",
        "unsupported_suggestion_format": "Add a local tool and register it.",
        "default_next_step": "Execute via hub/agent tool path and verify outcome.",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    result = {}
    keys = set(base) | set(override)
    for k in keys:
        if k in base and k in override:
            if isinstance(base[k], dict) and isinstance(override[k], dict):
                result[k] = _deep_merge(base[k], override[k])
            else:
                result[k] = override[k]
        elif k in override:
            result[k] = override[k]
        else:
            result[k] = base[k]
    return result


class Icklization:
    """Central persona and response configuration for Ickle.

    Usage:
        from src.icklization import ick

        print(ick.system_prompt())
        print(ick.response("greeting"))
        print(ick.dialog_label("user"))
    """

    _instance: Icklization | None = None

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(
            path or os.getenv("ICKLE_PERSONA_CONFIG", str(_PERSONA_DEFAULT_PATH))
        )
        self._data: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if self.path and self.path.exists():
            raw: dict[str, Any] = tomllib.loads(self.path.read_text(encoding="utf-8"))
        else:
            raw = {}
        self._data = _deep_merge(_PERSONA_DEFAULTS, raw)

    # â”€â”€ persona â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def persona_name(self) -> str:
        return str(self._data.get("persona", {}).get("name", "Ickle"))

    def system_prompt(self) -> str:
        return str(self._data.get("persona", {}).get("system_prompt", ""))

    # â”€â”€ dialog â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def user_label(self) -> str:
        return str(self._data.get("dialog", {}).get("user_label", "User"))

    def assistant_label(self) -> str:
        return str(self._data.get("dialog", {}).get("assistant_label", "Ickle"))

    def stop_markers(self) -> list[str]:
        return list(self._data.get("dialog", {}).get("stop_markers", []))

    # â”€â”€ behavior â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def guidance_response(self, kind: str) -> str:
        return str(self._data.get("behavior", {}).get(f"guidance_{kind}", ""))

    # â”€â”€ responses â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def response(self, key: str, **fmt: Any) -> str:
        template = str(self._data.get("responses", {}).get(key, ""))
        if fmt:
            return template.format(**fmt)
        return template

    # â”€â”€ evidence_policy â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def evidence_policy_text(self) -> str:
        return str(self._data.get("evidence_policy", {}).get("text", ""))

    # â”€â”€ repair_mode â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def repair_enabled(self) -> bool:
        return bool(self._data.get("repair_mode", {}).get("enabled", False))

    def repair_text(self) -> str:
        return str(self._data.get("repair_mode", {}).get("text", ""))

    # â”€â”€ autonomy â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def default_autonomy_mode(self) -> str:
        return str(self._data.get("autonomy", {}).get("default_mode", "balanced"))

    def autonomy_modes(self) -> dict[str, dict[str, Any]]:
        return dict(self._data.get("autonomy", {}).get("modes", {}))

    # â”€â”€ capabilities â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def capability_descriptions(self) -> dict[str, str]:
        return dict(self._data.get("capabilities", {}).get("known", {}))

    def capability_suggestions(self) -> dict[str, str]:
        return dict(self._data.get("capabilities", {}).get("suggestions", {}))

    def unknown_capability_summary(self) -> str:
        return str(self._data.get("capabilities", {}).get("unknown_summary", ""))

    def unknown_capability_suggestion(self) -> str:
        return str(self._data.get("capabilities", {}).get("unknown_suggestion", ""))

    # â”€â”€ detection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def detection_list(self, key: str) -> list[str]:
        return list(self._data.get("detection", {}).get(key, []))

    # â”€â”€ clarification â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def vague_prompt_question(self) -> str:
        return str(self._data.get("clarification", {}).get("vague_prompt_question", ""))

    def clarification_prefix(self, domain: str) -> str:
        return str(self._data.get("clarification", {}).get(f"{domain}_prefix", ""))

    def clarification_question(self, domain: str, field: str) -> str:
        return str(self._data.get("clarification", {}).get(domain, {}).get(field, ""))

    # â”€â”€ cloud_assist â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def cloud_configured_text(self) -> str:
        return str(self._data.get("cloud_assist", {}).get("configured_text", ""))

    def cloud_not_configured_text(self) -> str:
        return str(self._data.get("cloud_assist", {}).get("not_configured_text", ""))

    # â”€â”€ models â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def default_model_search_paths(self) -> list[str]:
        return list(self._data.get("models", {}).get("default_search_paths", []))

    def fallback_model(self) -> str:
        return str(self._data.get("models", {}).get("fallback_model", ""))

    # â”€â”€ training â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def bootstrap_text(self) -> str:
        return str(self._data.get("training", {}).get("bootstrap_text", ""))

    # â”€â”€ partner_loop â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def partner_loop_text(self, key: str) -> str:
        return str(self._data.get("partner_loop", {}).get(key, ""))


# Module-level singleton for easy import across the codebase.
ick = Icklization()
