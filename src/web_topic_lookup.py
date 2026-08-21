from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote, quote_plus, unquote, urlparse
from urllib.request import Request, urlopen

from src.dynamic_web_reader import read_url_dynamic
from src.evidence_policy import topic_relevance


def _request_json(url: str, timeout_sec: int = 15) -> dict[str, Any]:
    req = Request(
        url,
        headers={
            "User-Agent": "IckleWebLookup/1.0 (+https://github.com/EliasRipley/Ickle)",
            "Accept": "application/json",
        },
    )
    with urlopen(req, timeout=timeout_sec) as response:
        import json

        payload = json.loads(response.read().decode("utf-8", errors="replace"))
    if not isinstance(payload, dict):
        return {}
    return payload


def _search_wikipedia(topic: str, max_pages: int = 4, timeout_sec: int = 15) -> list[dict[str, str]]:
    query = quote_plus(str(topic or "").strip())
    if not query:
        return []
    url = (
        "https://en.wikipedia.org/w/api.php"
        f"?action=query&list=search&format=json&utf8=1&srlimit={max(1, max_pages)}&srsearch={query}"
    )
    try:
        payload = _request_json(url, timeout_sec=timeout_sec)
    except Exception:  # noqa: BLE001
        return []
    rows = payload.get("query", {}).get("search", [])
    if not isinstance(rows, list):
        return []
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        title = str((row or {}).get("title", "")).strip()
        if not title:
            continue
        wiki_url = f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'), safe='()')}"
        if wiki_url in seen:
            continue
        seen.add(wiki_url)
        out.append({"url": wiki_url, "title": title, "source": "wikipedia"})
        if len(out) >= max(1, max_pages):
            break
    return out


def _wikipedia_title_from_url(url: str) -> str | None:
    parsed = urlparse(str(url or "").strip())
    host = (parsed.hostname or "").lower()
    if "wikipedia.org" not in host:
        return None
    path = str(parsed.path or "")
    if not path.startswith("/wiki/"):
        return None
    slug = path.split("/wiki/", 1)[1].strip()
    if not slug:
        return None
    return unquote(slug).replace("_", " ").strip()


def _fetch_wikipedia_extract(title: str, max_chars: int = 7000, timeout_sec: int = 15) -> str:
    title_q = quote_plus(str(title or "").strip())
    if not title_q:
        return ""
    url = (
        "https://en.wikipedia.org/w/api.php"
        "?action=query&prop=extracts&explaintext=1&format=json"
        f"&exchars={max(1400, max_chars)}&titles={title_q}"
    )
    try:
        payload = _request_json(url, timeout_sec=timeout_sec)
    except Exception:  # noqa: BLE001
        return ""
    pages = payload.get("query", {}).get("pages", {})
    if not isinstance(pages, dict):
        return ""
    for page in pages.values():
        extract = str((page or {}).get("extract", "")).strip()
        if extract:
            return extract
    return ""


def _split_extract_facts(text: str, max_facts: int = 8) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for chunk in re.split(r"(?<=[.!?])\s+", str(text or "")):
        fact = re.sub(r"\s+", " ", chunk).strip()
        if len(fact) < 40 or len(fact) > 260:
            continue
        key = fact.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(fact)
        if len(out) >= max_facts:
            break
    return out


def wikipedia_payload_from_url(url: str, *, max_chars: int = 9000, timeout_ms: int = 15000) -> dict[str, Any] | None:
    title = _wikipedia_title_from_url(url)
    if not title:
        return None
    extract = _fetch_wikipedia_extract(title, max_chars=max_chars, timeout_sec=max(5, timeout_ms // 1000))
    if not extract:
        return None
    facts = _split_extract_facts(extract, max_facts=12)
    return {
        "url": url,
        "title": title,
        "description": "",
        "structure": {"has_main": True, "has_header": True, "has_nav": True, "has_aside": False, "has_footer": True},
        "headlines": [{"text": title, "level": 1, "selector": "wiki_extract", "element": "h1"}],
        "content": extract[:max_chars],
        "word_count": len(extract.split()),
        "source_quality": {"score": 0.74, "domain_score": 0.78, "freshness_score": 0.5, "structure_score": 0.7},
        "evidence_items": [{"claim": fact, "kind": "content", "score": 0.74, "confidence": 0.78, "corroboration_count": 0} for fact in facts],
        "success": True,
        "reader_mode": "wikipedia_api_extract",
    }


def _search_duckduckgo(topic: str, max_results: int = 8, timeout_sec: int = 20) -> list[dict[str, str]]:
    query = quote_plus(str(topic or "").strip())
    if not query:
        return []
    url = f"https://html.duckduckgo.com/html/?q={query}"
    req = Request(
        url,
        headers={
            "User-Agent": "IckleWebLookup/1.0 (+https://github.com/EliasRipley/Ickle)",
            "Accept": "text/html",
        },
    )
    try:
        with urlopen(req, timeout=timeout_sec) as response:
            html = response.read().decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return []

    def _decode_url(raw_href: str) -> str:
        href = str(raw_href or "").replace("&amp;", "&").strip()
        if not href:
            return ""
        if "uddg=" in href:
            m = re.search(r"[?&]uddg=([^&]+)", href)
            if m:
                href = unquote(str(m.group(1) or "").strip())
        if href.startswith("//"):
            href = "https:" + href
        return href

    pattern = re.compile(
        r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
        flags=re.IGNORECASE | re.S,
    )

    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for match in pattern.finditer(html):
        href = _decode_url(match.group(1))
        if not href.startswith("http"):
            continue
        host = (urlparse(href).hostname or "").lower()
        if not host or "duckduckgo.com" in host:
            continue
        if href in seen:
            continue
        seen.add(href)
        raw_title = re.sub(r"<[^>]+>", " ", str(match.group(2) or ""))
        title = re.sub(r"\s+", " ", raw_title).strip() or href
        out.append({"url": href, "title": title, "source": "web"})
        if len(out) >= max(1, max_results):
            break
    return out


def _score_candidate(topic: str, row: dict[str, str]) -> float:
    title = str(row.get("title", "")).strip()
    link = str(row.get("url", "")).strip()
    source = str(row.get("source", "")).strip().lower()
    text = " ".join([title, link]).strip()
    score = float(topic_relevance(topic, text))
    if source == "wikipedia":
        score += 0.12
    return score


def _extract_facts_from_payload(payload: dict[str, Any], max_facts: int = 8) -> list[str]:
    facts: list[str] = []
    seen: set[str] = set()

    for row in list(payload.get("evidence_items", []))[: max_facts * 2]:
        claim = str((row or {}).get("claim", "")).strip()
        if not claim:
            continue
        key = claim.lower()
        if key in seen:
            continue
        seen.add(key)
        facts.append(claim)
        if len(facts) >= max_facts:
            return facts

    for row in list(payload.get("headlines", []))[: max_facts * 2]:
        text = str((row or {}).get("text", "")).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        facts.append(text)
        if len(facts) >= max_facts:
            return facts
    return facts


def collect_topic_web_evidence(
    *,
    topic: str,
    max_sources: int = 3,
    timeout_ms: int = 20000,
    max_chars: int = 10000,
) -> dict[str, Any]:
    clean_topic = str(topic or "").strip()
    if not clean_topic:
        return {"success": False, "topic": "", "sources": [], "facts": [], "error": "empty topic"}

    wiki = _search_wikipedia(clean_topic, max_pages=max(2, max_sources))
    web = _search_duckduckgo(clean_topic, max_results=max_sources * 3)
    candidates = wiki + web
    if not candidates:
        return {"success": False, "topic": clean_topic, "sources": [], "facts": [], "error": "no candidate urls"}

    deduped: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for row in candidates:
        url = str(row.get("url", "")).strip()
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        deduped.append(row)

    ranked = sorted(deduped, key=lambda row: _score_candidate(clean_topic, row), reverse=True)
    probe_rows = ranked[: max(max_sources * 2, max_sources)]

    sources: list[dict[str, Any]] = []
    merged_facts: list[str] = []
    seen_facts: set[str] = set()

    for row in probe_rows:
        url = str(row.get("url", "")).strip()
        if not url:
            continue
        payload: dict[str, Any] | None = None
        title = str(row.get("title") or url).strip()
        facts: list[str] = []

        wiki_title = _wikipedia_title_from_url(url)
        if wiki_title:
            title = wiki_title
            extract = _fetch_wikipedia_extract(wiki_title, max_chars=max_chars, timeout_sec=max(5, timeout_ms // 1000))
            facts = _split_extract_facts(extract, max_facts=8)
            payload = {
                "success": bool(facts),
                "title": wiki_title,
                "source_quality": {"score": 0.74, "domain_score": 0.78, "freshness_score": 0.50},
                "evidence_items": [{"claim": fact, "confidence": 0.78, "score": 0.74} for fact in facts],
            }
        else:
            payload = read_url_dynamic(url, timeout_ms=timeout_ms, max_chars=max_chars)
            if payload.get("success"):
                title = str(payload.get("title") or row.get("title") or url).strip()
                facts = _extract_facts_from_payload(payload, max_facts=8)

        if not payload or not payload.get("success"):
            continue
        if not facts:
            continue
        relevance = max(
            float(topic_relevance(clean_topic, title)),
            max(float(topic_relevance(clean_topic, fact)) for fact in facts[:6]),
        )
        if relevance <= 0.03:
            continue
        source_entry = {
            "url": url,
            "title": title,
            "source": str(row.get("source", "web")).strip(),
            "relevance": relevance,
            "facts": facts[:8],
            "evidence_items": list(payload.get("evidence_items", []))[:10],
            "quality": payload.get("source_quality", {}),
        }
        sources.append(source_entry)
        for fact in facts:
            key = fact.lower()
            if key in seen_facts:
                continue
            seen_facts.add(key)
            merged_facts.append(fact)

    sources.sort(key=lambda entry: float(entry.get("relevance", 0.0)), reverse=True)
    sources = sources[: max(1, max_sources)]

    if not sources:
        return {"success": False, "topic": clean_topic, "sources": [], "facts": [], "error": "no relevant sources"}

    return {
        "success": True,
        "topic": clean_topic,
        "sources": sources,
        "facts": merged_facts[:18],
    }
