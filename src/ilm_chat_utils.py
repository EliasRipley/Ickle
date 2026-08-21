"""Low-level text processing utilities for ILM chat."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from src.icklization import ick

_TOK_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can",
    "between", "could", "did", "does", "for", "from",
    "give", "help", "how", "ickle", "i", "in", "is", "it",
    "just", "like", "may", "me", "might", "my", "major", "now",
    "of", "on", "or", "please", "quick", "quickly",
    "simple", "simply", "right", "should", "show", "summary",
    "that", "tell", "term", "terms", "the", "this",
    "to", "was", "were", "what", "when", "where", "who", "why",
    "would", "with", "you", "your",
}
_TOPIC_GENERIC_TOKENS = {
    "answer", "brief", "briefly", "cause", "clear", "difference",
    "explain", "explaining", "explanation", "give", "help", "happen",
    "line", "one", "sentence", "sentences", "sentenc", "two", "write",
    "show", "simple", "summary", "tell", "terms",
}


def _fold_text(text: str) -> str:
    value = unicodedata.normalize("NFKD", str(text or ""))
    return "".join(ch for ch in value if not unicodedata.combining(ch))


def _token_set(text: str) -> set[str]:
    out: set[str] = set()
    for token in re.findall(r"[a-z0-9']+", _fold_text(text).lower()):
        if len(token) < 3:
            continue
        if token.endswith("ies") and len(token) > 4:
            token = token[:-3] + "y"
        elif token.endswith("es") and len(token) > 4 and not token.endswith("ses"):
            token = token[:-2]
        elif token.endswith("s") and len(token) > 4 and not token.endswith("ss"):
            token = token[:-1]
        if token in _TOK_STOPWORDS:
            continue
        out.add(token)
    return out


def _extract_response_text(raw_generated_text: str) -> str:
    text = raw_generated_text.strip()
    text = re.sub(r"^(?:Ickle|ILM|Assistant)\s*:\s*", "", text, flags=re.IGNORECASE)

    # Stop at any new turn marker even when it appears inline, not only on new
    # lines -- catches a leaked "User:"/"Ickle:" mid-sentence, which a plain
    # `stop_markers`-list find() (only anchored at "\n") would miss.
    turn_marker = re.search(r"(?:^|\s)(?:User|Ickle|ILM|Assistant)\s*:", text, flags=re.IGNORECASE)
    if turn_marker:
        text = text[: turn_marker.start()].strip()

    if not text:
        return ""

    cleaned_lines: list[str] = []
    for line in text.splitlines():
        chunk = line.strip()
        if not chunk:
            continue
        lower = chunk.lower()
        if lower.startswith((f"{ick.user_label().lower()}:", "system:", "instruction:")):
            continue
        if re.fullmatch(r"[A-Z][A-Z0-9 _-]{6,}", chunk):
            continue
        cleaned_lines.append(chunk)

    final = " ".join(cleaned_lines).strip()
    final = re.sub(r"\s+", " ", final)
    return final


def _looks_low_quality_response(text: str) -> bool:
    lower = text.lower()
    evasive_markers = tuple(ick.detection_list("low_quality_evasive_markers"))
    if any(marker in lower for marker in evasive_markers):
        return True
    if re.search(r"([^\s]{2,8})\1{2,}", text, flags=re.IGNORECASE):
        return True
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if any(len(word) > 32 for word in words):
        return True
    visible = [ch for ch in text if not ch.isspace()]
    if visible and (sum(ord(ch) > 127 for ch in visible) / len(visible)) > 0.18:
        return True
    if len(words) < 10:
        return False
    if re.search(r"\b(\w+)(?:\s+\1){2,}\b", text.lower()):
        return True
    unique_ratio = len(set(words)) / max(1, len(words))
    return unique_ratio < 0.35


def _prompt_relevance_score(prompt: str, response: str) -> float:
    p_tokens = _token_set(prompt)
    if not p_tokens:
        return 1.0
    r_tokens = _token_set(response)
    if not r_tokens:
        return 0.0
    return len(p_tokens.intersection(r_tokens)) / max(1, len(p_tokens))


def _topic_overlap_count(prompt: str, response: str) -> int:
    p_tokens = {t for t in _token_set(prompt) if t not in _TOPIC_GENERIC_TOKENS}
    if not p_tokens:
        return 0
    r_tokens = _token_set(response)
    return len(p_tokens.intersection(r_tokens))


def _topic_hint_from_prompt(prompt: str) -> str | None:
    ordered: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"[a-z0-9']+", _fold_text(prompt).lower()):
        t = token
        if len(t) < 3:
            continue
        if t.endswith("ies") and len(t) > 4:
            t = t[:-3] + "y"
        elif t.endswith("es") and len(t) > 4 and not t.endswith("ses"):
            t = t[:-2]
        elif t.endswith("s") and len(t) > 4 and not t.endswith("ss"):
            t = t[:-1]
        if t in _TOK_STOPWORDS or t in _TOPIC_GENERIC_TOKENS:
            continue
        if t in seen:
            continue
        seen.add(t)
        ordered.append(t)
        if len(ordered) >= 3:
            break
    if not ordered:
        return None
    return " ".join(ordered)


def _is_noisy_memory_fact(text: str) -> bool:
    value = str(text or "")
    lower = value.lower()
    markers = (
        "wikipedia does not have an article",
        "look for",
        "article wizard",
        "sister projects",
        "other reasons this message may be displayed",
        "wiktionary(",
        "wikibooks(",
        "wikiquote(",
        "wikisource(",
        "wikiversity(",
        "wikivoyage(",
        "wikinews(",
        "wikidata(",
        "wikispecies(",
        "from wikipedia, the free encyclopedia",
        "languagesafrikaans",
    )
    if any(marker in lower for marker in markers):
        return True
    if re.search(r"\b\d+\s+languages\b", lower):
        return True
    if len(value) > 160:
        non_ascii_ratio = sum(1 for ch in value if ord(ch) > 127) / max(1, len(value))
        if non_ascii_ratio > 0.12:
            return True
    return False


def _looks_like_recall_request(prompt: str) -> bool:
    lower = prompt.lower()
    triggers = ick.detection_list("recall_triggers")
    return any(t in lower for t in triggers)


def _looks_like_knowledge_request(prompt: str) -> bool:
    lower = prompt.lower().strip()
    patterns = ick.detection_list("knowledge_prefixes")
    return any(lower.startswith(p) or p in lower for p in patterns)


def _needs_recent_context(prompt: str) -> bool:
    lower = prompt.lower()
    phrases = ick.detection_list("needs_context_phrases")
    if any(p in lower for p in phrases):
        return True
    return _is_contextual_followup(prompt)


def _is_contextual_followup(prompt: str) -> bool:
    lower = prompt.lower().strip()
    if not lower:
        return False
    starters = tuple(ick.detection_list("context_followup_starters"))
    if lower.startswith(starters):
        return True
    if "relative to" in lower:
        return True
    tokens = set(re.findall(r"[a-z0-9']+", lower))
    refs = set(ick.detection_list("context_reference_tokens"))
    return bool(tokens.intersection(refs) and len(tokens) <= 14)


def _is_live_time_request(prompt: str) -> bool:
    lower = prompt.lower()
    if "time" not in lower:
        return False
    hints = ick.detection_list("live_time_hints")
    return any(h in lower for h in hints)


def _is_time_sensitive_request(prompt: str) -> bool:
    lower = str(prompt or "").lower()
    if _is_live_time_request(prompt):
        return True
    triggers = ick.detection_list("time_sensitive_triggers")
    return any(token in lower for token in triggers)


def _extract_utc_offset(text: str) -> str | None:
    cleaned = str(text or "")
    if not cleaned:
        return None
    match = re.search(r"\butc\s*(plus|minus|\+|-)?\s*(\d{1,2})(?::(\d{2}))?\b", cleaned, flags=re.IGNORECASE)
    if not match:
        return None
    sign_token = (match.group(1) or "").strip().lower()
    hours = int(match.group(2))
    minutes = int(match.group(3) or 0)
    if hours > 14 or minutes > 59:
        return None
    sign = "+"
    if sign_token in {"minus", "-"}:
        sign = "-"
    return f"UTC{sign}{hours:02d}:{minutes:02d}"


