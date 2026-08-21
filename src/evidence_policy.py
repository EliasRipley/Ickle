from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from urllib.parse import urlparse

from src.ilm_chat_utils import _fold_text


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "how",
    "i",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
}

_DOMAIN_REPUTATION = {
    "wikipedia.org": 0.78,
    "arxiv.org": 0.88,
    "nature.com": 0.90,
    "bbc.com": 0.82,
    "reuters.com": 0.88,
    "nasa.gov": 0.92,
    "gov.uk": 0.90,
    "nih.gov": 0.92,
    "who.int": 0.92,
    "huggingface.co": 0.76,
}


def content_tokens(text: str) -> set[str]:
    out: set[str] = set()
    for raw in re.findall(r"[a-z0-9']+", _fold_text(str(text or "")).lower()):
        token = raw
        if token.endswith("'s") and len(token) > 4:
            token = token[:-2]
        if token.endswith("ies") and len(token) > 4:
            token = token[:-3] + "y"
        elif token.endswith("es") and len(token) > 4 and not token.endswith("ses"):
            token = token[:-2]
        elif token.endswith("s") and len(token) > 4 and not token.endswith("ss"):
            token = token[:-1]
        if len(token) < 3 or token in _STOPWORDS:
            continue
        out.add(token)
    return out


def jaccard_similarity(a: str, b: str) -> float:
    ta = content_tokens(a)
    tb = content_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta.intersection(tb)) / max(1, len(ta.union(tb)))


def claim_signature(text: str, max_tokens: int = 10) -> str:
    tokens = sorted(content_tokens(text))
    if not tokens:
        return ""
    return " ".join(tokens[: max(2, max_tokens)])


def topic_relevance(topic: str, text: str) -> float:
    t = content_tokens(topic)
    if not t:
        return 0.0
    v = content_tokens(text)
    if not v:
        return 0.0
    overlap = len(t.intersection(v))
    return overlap / max(1, len(t))


def domain_reputation(url: str) -> float:
    host = (urlparse(str(url or "")).hostname or "").lower()
    if not host:
        return 0.55
    for suffix, value in _DOMAIN_REPUTATION.items():
        if host == suffix or host.endswith("." + suffix):
            return value
    if host.endswith(".gov") or host.endswith(".edu"):
        return 0.84
    if host.endswith(".org"):
        return 0.70
    return 0.60


def freshness_score(date_text: str, *, now_utc: datetime | None = None) -> float:
    raw = str(date_text or "").strip()
    if not raw:
        return 0.55
    now = now_utc or datetime.now(timezone.utc)
    parsed: datetime | None = None
    for fmt in (
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d",
        "%a, %d %b %Y %H:%M:%S %Z",
    ):
        try:
            parsed = datetime.strptime(raw, fmt)
            break
        except ValueError:
            continue
    if parsed is None:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
            try:
                parsed = datetime.strptime(raw, "%Y-%m-%d")
            except ValueError:
                parsed = None
    if parsed is None:
        return 0.55
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    age_days = max(0.0, (now - parsed.astimezone(timezone.utc)).total_seconds() / 86400.0)
    if age_days <= 1:
        return 0.95
    if age_days <= 7:
        return 0.88
    if age_days <= 30:
        return 0.74
    if age_days <= 180:
        return 0.62
    if age_days <= 365:
        return 0.52
    return 0.42


def corroboration_score(corroboration_count: int) -> float:
    n = max(0, int(corroboration_count))
    if n <= 0:
        return 0.0
    # Smoothly saturates near 1 after ~4 confirmations.
    return max(0.0, min(1.0, 1.0 - math.exp(-0.75 * n)))


def evidence_score(
    *,
    relevance: float,
    source_quality: float,
    confidence: float,
    corroboration_count: int,
) -> float:
    rel = max(0.0, min(1.0, float(relevance)))
    qual = max(0.0, min(1.0, float(source_quality)))
    conf = max(0.0, min(1.0, float(confidence)))
    corr = corroboration_score(corroboration_count)
    score = (0.36 * rel) + (0.28 * qual) + (0.20 * conf) + (0.16 * corr)
    return max(0.0, min(1.0, score))

