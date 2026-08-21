from __future__ import annotations

import concurrent.futures
import json
import os
import re
import signal
import shutil
import subprocess
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
import urllib.error
from urllib.parse import quote, quote_plus, urlparse
from urllib.request import Request, urlopen
from urllib.parse import unquote

from src.build_clean_corpus import build_corpus
from src.build_smart_corpus import build_smart_corpus
from src.conversation_focus import build_focus_corpus_file, merge_dialog_corpora
from src.continual_guard import run_guarded_step
from src.verified_corrections import build_verified_corrections_corpus_file
from src.disagreement_curriculum import build_hedge_corpus_file
from src.dynamic_web_reader import read_url_dynamic
from src.evidence_policy import (
    claim_signature,
    domain_reputation,
    evidence_score,
    freshness_score,
    jaccard_similarity,
    topic_relevance,
)
from src.eval_harness import run_round as run_eval_round
from src.ilm_chat import extract_response_text, generate_response
from src.ilm_memory import get_memory
from src.model_accumulate import Accumulator
from src.resource_defaults import DEFAULT_CPU_PCT, DEFAULT_GPU_PCT, DEFAULT_RAM_PCT
from src.runtime_flags import RuntimeFlagsStore
from src.training_control import (
    clear_training_stop_request,
    get_training_stop_request_path,
    read_training_stop_request,
)
from src.tools.news_research import search_news
from src.workspace_paths import get_training_root


ProgressCb = Callable[[str], None]
ChatRunner = Callable[[dict[str, Any]], dict[str, Any]]


def _atomic_copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temp_path)
    os.replace(temp_path, destination)


def _record_active_model(path: Path) -> None:
    RuntimeFlagsStore().set_flag("current_model", str(path.resolve().as_posix()))

EVAL_STOPWORDS = {
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
    "in",
    "is",
    "it",
    "its",
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
}

TOPIC_MODIFIER_TOKENS = {
    "about",
    "basic",
    "basics",
    "behavior",
    "behaviour",
    "concept",
    "concepts",
    "detail",
    "details",
    "from",
    "group",
    "groups",
    "history",
    "internet",
    "introduction",
    "learn",
    "learning",
    "online",
    "social",
    "structure",
    "study",
    "web",
}

INFER_TOPIC_TRIM_PATTERNS = (
    r"\bfrom\s+(?:the\s+)?(?:internet|web|online)\b.*$",
    r"\busing\s+(?:the\s+)?(?:internet|web|online)\b.*$",
    r"\bfrom\s+wikipedia\b.*$",
    r"\bon\s+wikipedia\b.*$",
    r"\bfor\s+me\b$",
    r"\bplease\b$",
)


def _stop_process(proc: subprocess.Popen[str]):
    if proc.poll() is not None:
        return
    try:
        if hasattr(signal, "CTRL_BREAK_EVENT"):
            proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
            proc.wait(timeout=10)
            return
    except Exception:  # noqa: BLE001
        pass
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _request_json(
    url: str,
    timeout_sec: int = 15,
    *,
    max_retries: int = 5,
    base_backoff_sec: float = 1.0,
) -> dict[str, Any]:
    req = Request(url, headers={"User-Agent": "IckleTaskRunner/1.0 (+https://github.com/EliasRipley/Ickle)"})
    retries = max(0, int(max_retries))
    base_wait = max(0.1, float(base_backoff_sec))
    transient_http_codes = {429, 500, 502, 503, 504}
    last_error: Exception | None = None

    for attempt in range(retries + 1):
        try:
            with urlopen(req, timeout=timeout_sec) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw)
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in transient_http_codes or attempt >= retries:
                exc.close()
                raise
            retry_after = exc.headers.get("Retry-After") if exc.headers is not None else None
            wait = 0.0
            if retry_after:
                try:
                    wait = float(retry_after)
                except Exception:  # noqa: BLE001
                    wait = 0.0
            if wait <= 0.0:
                wait = min(30.0, base_wait * (2**attempt))
            time.sleep(wait)
        except urllib.error.URLError as exc:
            last_error = exc
            if attempt >= retries:
                raise
            wait = min(30.0, base_wait * (2**attempt))
            time.sleep(wait)

    if last_error is not None:
        raise last_error
    raise RuntimeError("Unexpected request_json failure")


def _fold_text(text: str) -> str:
    value = unicodedata.normalize("NFKD", str(text or ""))
    return "".join(ch for ch in value if not unicodedata.combining(ch))


def _topic_tokens_ordered(text: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    tokens = re.findall(r"[a-z0-9']+", _fold_text(text).lower())
    for token in tokens:
        t = token
        if t.endswith("'s") and len(t) > 4:
            t = t[:-2]
        if t.endswith("s") and len(t) > 5:
            t = t[:-1]
        if len(t) < 4:
            continue
        if t in EVAL_STOPWORDS:
            continue
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _topic_tokens(text: str) -> set[str]:
    return set(_topic_tokens_ordered(text))


def _topic_anchor_tokens(topic: str) -> set[str]:
    ordered = _topic_tokens_ordered(topic)
    anchors = [token for token in ordered if token not in TOPIC_MODIFIER_TOKENS]
    if anchors:
        return set(anchors[:3])
    return set(ordered[:2])


def _normalize_inferred_topic(raw_topic: str) -> str:
    topic = re.sub(r"\s+", " ", str(raw_topic or "")).strip(" .?!")
    topic = re.sub(r"^(?:as\s+much\s+as\s+you\s+can\s+about|about)\s+", "", topic, flags=re.IGNORECASE)
    topic = re.split(
        r"\b(?:,?\s*(?:and\s+)?then\s+train(?:\s+yourself)?|to\s+train(?:\s+yourself)?)\b",
        topic,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip(" ,.;:!?")
    for pattern in INFER_TOPIC_TRIM_PATTERNS:
        topic = re.sub(pattern, "", topic, flags=re.IGNORECASE).strip(" ,.;:!?")
    topic = re.sub(r"\b(?:as much as possible|as much as you can)\b", "", topic, flags=re.IGNORECASE).strip(
        " ,.;:!?"
    )
    return re.sub(r"\s+", " ", topic).strip(" ,.;:!?")


def _mentions_web_sources(text: str) -> bool:
    return bool(re.search(r"\b(internet|web|online|website|websites)\b", str(text or ""), flags=re.IGNORECASE))


def _mentions_wikipedia(text: str) -> bool:
    return bool(re.search(r"\bwikipedia\b", str(text or ""), flags=re.IGNORECASE))


def _wiki_result_score(
    *,
    topic_terms: set[str],
    anchor_terms: set[str],
    title: str,
    snippet: str,
    rank: int,
) -> dict[str, Any]:
    combined = f"{title} {snippet}"
    result_terms = _topic_tokens(combined)
    overlap = len(topic_terms.intersection(result_terms))
    anchor_overlap = len(anchor_terms.intersection(result_terms)) if anchor_terms else 0
    title_lower = title.lower()
    title_anchor_bonus = 2 if anchor_terms and any(t in title_lower for t in anchor_terms) else 0
    phrase_bonus = 2 if title_lower in " ".join(sorted(topic_terms)) else 0
    rank_bonus = max(0, 5 - rank)
    score = overlap * 12 + anchor_overlap * 10 + title_anchor_bonus + phrase_bonus + rank_bonus
    return {
        "score": score,
        "overlap": overlap,
        "anchor_overlap": anchor_overlap,
    }


def _filter_ranked_wikipedia_results(topic: str, search_rows: list[dict[str, Any]], max_pages: int) -> list[str]:
    titles = [str(row.get("title", "")).strip() for row in search_rows if str(row.get("title", "")).strip()]
    if not titles:
        return []
    topic_terms = _topic_tokens(topic)
    if not topic_terms:
        return titles[: max(1, max_pages)]

    anchor_terms = _topic_anchor_tokens(topic)
    required_overlap = 1 if len(topic_terms) <= 3 else 2
    scored: list[tuple[int, int, int, str]] = []
    for idx, row in enumerate(search_rows):
        title = str(row.get("title", "")).strip()
        if not title:
            continue
        snippet = str(row.get("snippet", "")).replace("<span class=\"searchmatch\">", "").replace("</span>", "")
        s = _wiki_result_score(
            topic_terms=topic_terms,
            anchor_terms=anchor_terms,
            title=title,
            snippet=snippet,
            rank=idx,
        )
        overlap = int(s["overlap"])
        anchor_overlap = int(s["anchor_overlap"])
        if overlap < required_overlap:
            continue
        if anchor_terms and anchor_overlap == 0 and overlap <= required_overlap:
            continue
        scored.append((int(s["score"]), overlap, anchor_overlap, title))

    scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    filtered = [title for _, _, _, title in scored]
    if not filtered:
        return titles[: max(1, max_pages)]
    return filtered[: max(1, max_pages)]


def _looks_useful_fact(chunk: str) -> bool:
    c = chunk.strip()
    if len(c) < 40:
        return False
    if "==" in c:
        return False
    lower = c.lower()
    if "external links" in lower or "references" in lower:
        return False
    if lower.startswith("see also"):
        return False
    noisy_markers = (
        "from wikipedia, the free encyclopedia",
        "wikipedia does not have an article",
        "look for",
        "article wizard",
        "other reasons this message may be displayed",
        "sister projects",
    )
    if any(marker in lower for marker in noisy_markers):
        return False
    words = re.findall(r"[a-zA-Z]{2,}", c)
    if len(words) < 8:
        return False
    letter_ratio = sum(1 for ch in c if ch.isalpha()) / max(1, len(c))
    if letter_ratio < 0.6:
        return False
    return True


def _split_into_facts(text: str, max_facts: int = 12) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    chunks = re.split(r"(?<=[.!?])\s+", cleaned)
    facts: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        c = chunk.strip()
        if not _looks_useful_fact(c):
            continue
        short = c[:320]
        key = short.lower()
        if key in seen:
            continue
        seen.add(key)
        facts.append(short)
        if len(facts) >= max_facts:
            break
    return facts


def _clean_web_reader_text(text: str) -> str:
    if not text:
        return ""
    # Drop extraction markers so facts are cleaner for memory/corpus.
    cleaned = re.sub(r"\b(?:HEADLINE|PARAGRAPH|TEXT|ITEM):\s*", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _facts_from_web_payload(web_data: dict[str, Any], max_facts: int = 10) -> list[str]:
    facts: list[str] = []
    seen: set[str] = set()

    for h in web_data.get("headlines", [])[:8]:
        text = str(h.get("text", "")).strip()
        if not _looks_useful_fact(text):
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        facts.append(text[:320])
        if len(facts) >= max_facts:
            return facts

    body = _clean_web_reader_text(str(web_data.get("content", "")))
    for fact in _split_into_facts(body, max_facts=max_facts):
        key = fact.lower()
        if key in seen:
            continue
        seen.add(key)
        facts.append(fact)
        if len(facts) >= max_facts:
            break
    return facts


def _source_quality_for_summary(summary_row: dict[str, Any]) -> float:
    source = str(summary_row.get("source", "")).strip().lower()
    url = str(summary_row.get("url", "")).strip()
    topic_match = summary_row.get("topic_match", {}) if isinstance(summary_row.get("topic_match"), dict) else {}
    ratio = max(0.0, min(1.0, float(topic_match.get("ratio", 0.0))))
    domain_score = domain_reputation(url)
    freshness = freshness_score(str(summary_row.get("published_date", "")))
    source_bias = 0.0
    if source == "wikipedia":
        source_bias = 0.08
    elif source == "news":
        source_bias = 0.04
    return max(0.0, min(1.0, (0.40 * domain_score) + (0.25 * freshness) + (0.25 * ratio) + source_bias))


def _score_fact_evidence(topic: str, summary_row: dict[str, Any], fact: str) -> dict[str, Any]:
    source_quality = _source_quality_for_summary(summary_row)
    relevance = max(
        topic_relevance(topic, fact),
        topic_relevance(str(summary_row.get("title", "")), fact) * 0.6,
    )
    confidence = 0.86 if str(summary_row.get("source", "")).lower() == "wikipedia" else 0.74
    return {
        "fact": fact,
        "url": str(summary_row.get("url", "")).strip(),
        "title": str(summary_row.get("title", "")).strip(),
        "source": str(summary_row.get("source", "")).strip(),
        "source_quality": round(source_quality, 4),
        "relevance": round(relevance, 4),
        "confidence": round(confidence, 4),
        "corroboration_count": 0,
        "signature": claim_signature(fact),
        "score": round(
            evidence_score(
                relevance=relevance,
                source_quality=source_quality,
                confidence=confidence,
                corroboration_count=0,
            ),
            4,
        ),
    }


def _attach_corroboration(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_signature: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        signature = str(row.get("signature", "")).strip()
        if not signature:
            continue
        by_signature.setdefault(signature, []).append(idx)

    for idx, row in enumerate(rows):
        corroborated_urls: set[str] = set()
        signature = str(row.get("signature", "")).strip()
        if signature and signature in by_signature:
            for other_idx in by_signature[signature]:
                if other_idx == idx:
                    continue
                other = rows[other_idx]
                sim = jaccard_similarity(str(row.get("fact", "")), str(other.get("fact", "")))
                if sim >= 0.70:
                    url = str(other.get("url", "")).strip().lower()
                    if url:
                        corroborated_urls.add(url)
        else:
            for other_idx, other in enumerate(rows):
                if other_idx == idx:
                    continue
                sim = jaccard_similarity(str(row.get("fact", "")), str(other.get("fact", "")))
                if sim >= 0.76:
                    url = str(other.get("url", "")).strip().lower()
                    if url:
                        corroborated_urls.add(url)
        row["corroboration_count"] = len(corroborated_urls)
        row["score"] = round(
            evidence_score(
                relevance=float(row.get("relevance", 0.0)),
                source_quality=float(row.get("source_quality", 0.0)),
                confidence=float(row.get("confidence", 0.0)),
                corroboration_count=int(row.get("corroboration_count", 0)),
            ),
            4,
        )
    return rows


def _apply_web_evidence_policy(
    topic: str,
    summaries: list[dict[str, Any]],
    *,
    min_score: float = 0.58,
    min_corroborated_score: float = 0.50,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fact_rows: list[dict[str, Any]] = []
    for summary in summaries:
        for fact in list(summary.get("facts", [])):
            cleaned = str(fact or "").strip()
            if not cleaned:
                continue
            if not _looks_useful_fact(cleaned):
                continue
            fact_rows.append(_score_fact_evidence(topic, summary, cleaned))
    if not fact_rows:
        return [], {"input_facts": 0, "kept_facts": 0, "dropped_facts": 0}

    fact_rows = _attach_corroboration(fact_rows)
    keep_keys: set[tuple[str, str]] = set()
    kept_rows = 0
    for row in fact_rows:
        score = float(row.get("score", 0.0))
        corr = int(row.get("corroboration_count", 0))
        if score >= min_score or (corr >= 1 and score >= min_corroborated_score):
            key = (str(row.get("url", "")).strip(), str(row.get("fact", "")).strip().lower())
            keep_keys.add(key)
            kept_rows += 1

    approved: list[dict[str, Any]] = []
    for summary in summaries:
        url = str(summary.get("url", "")).strip()
        scored_facts: list[dict[str, Any]] = []
        filtered_facts: list[str] = []
        for row in fact_rows:
            if str(row.get("url", "")).strip() != url:
                continue
            fact_text = str(row.get("fact", "")).strip()
            key = (url, fact_text.lower())
            if key not in keep_keys:
                continue
            filtered_facts.append(fact_text)
            scored_facts.append(
                {
                    "fact": fact_text,
                    "score": float(row.get("score", 0.0)),
                    "confidence": float(row.get("confidence", 0.0)),
                    "corroboration_count": int(row.get("corroboration_count", 0)),
                }
            )
        if not filtered_facts:
            continue
        deduped_facts: list[str] = []
        seen_fact: set[str] = set()
        for fact in filtered_facts:
            low = fact.lower()
            if low in seen_fact:
                continue
            seen_fact.add(low)
            deduped_facts.append(fact)
        summary["facts"] = deduped_facts[:10]
        summary["scored_facts"] = sorted(scored_facts, key=lambda item: float(item.get("score", 0.0)), reverse=True)[:10]
        summary["evidence_gate"] = {
            "min_score": min_score,
            "min_corroborated_score": min_corroborated_score,
            "retained_fact_count": len(summary["facts"]),
        }
        approved.append(summary)

    return approved, {
        "input_facts": len(fact_rows),
        "kept_facts": kept_rows,
        "dropped_facts": max(0, len(fact_rows) - kept_rows),
        "kept_sources": len(approved),
    }


def _domain_allowed(url: str, *, unrestricted: bool, allowed_domains: set[str]) -> bool:
    if unrestricted:
        return True
    host = (urlparse(url).hostname or "").lower()
    if not host:
        return False
    if host in allowed_domains:
        return True
    return any(host.endswith("." + d) for d in allowed_domains)


def _search_general_web(topic: str, *, max_results: int = 8, timeout_sec: int = 20) -> list[dict[str, str]]:
    query = quote_plus(topic)
    # HTML endpoint is intentionally simple and works without JS.
    url = f"https://html.duckduckgo.com/html/?q={query}"
    req = Request(
        url,
        headers={
            "User-Agent": "IckleTaskRunner/1.0 (+https://github.com/EliasRipley/Ickle)",
            "Accept": "text/html",
        },
    )
    try:
        with urlopen(req, timeout=timeout_sec) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return []

    def _decode_candidate_url(raw_href: str) -> str:
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

    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    pattern = re.compile(
        r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
        flags=re.IGNORECASE | re.S,
    )
    for match in pattern.finditer(html):
        href = _decode_candidate_url(match.group(1))
        if not href.startswith("http"):
            continue
        host = (urlparse(href).hostname or "").lower()
        if not host or "duckduckgo.com" in host:
            continue
        if href in seen:
            continue
        seen.add(href)
        raw_title = re.sub(r"<[^>]+>", " ", str(match.group(2) or ""))
        title = re.sub(r"\s+", " ", raw_title).strip()
        if not title:
            title = href
        rows.append({"url": href, "title": title, "source": "web"})
        if len(rows) >= max(1, max_results):
            break

    return rows


def _search_google_html(topic: str, *, max_results: int = 8, timeout_sec: int = 20) -> list[dict[str, str]]:
    query = quote_plus(topic)
    url = f"https://www.google.com/search?q={query}&hl=en&num={max(max_results, 10)}"
    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    try:
        with urlopen(req, timeout=timeout_sec) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return []

    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for m in re.finditer(r'href="/url\?q=(https?://[^"&]+)', html):
        href = unquote(str(m.group(1) or "").strip())
        if not href.startswith("http"):
            continue
        host = (urlparse(href).hostname or "").lower()
        if not host or "google.com" in host:
            continue
        if href in seen:
            continue
        seen.add(href)
        rows.append({"url": href, "title": href, "source": "web"})
        if len(rows) >= max(1, max_results):
            break

    return rows


def _search_general_web_multi(
    topic: str, *, max_results: int = 10, timeout_sec: int = 20
) -> list[dict[str, str]]:
    engines = [
        (_search_general_web, "ddg_html"),
        (_search_google_html, "google_html"),
    ]
    all_rows: list[dict[str, str]] = []
    seen: set[str] = set()

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(engines)) as pool:
        futures = {
            pool.submit(fn, topic, max_results=max_results, timeout_sec=timeout_sec): name
            for fn, name in engines
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                rows = future.result()
            except Exception:
                continue
            for row in rows:
                key = row["url"]
                if key in seen:
                    continue
                seen.add(key)
                all_rows.append(row)

    if len(all_rows) < 2:
        fallback = _search_general_web(topic, max_results=max(max_results, 12), timeout_sec=timeout_sec)
        for row in fallback:
            key = row["url"]
            if key in seen:
                continue
            seen.add(key)
            all_rows.append(row)

    return all_rows[:max(max_results, len(all_rows))]


def _collect_web_learning_urls(
    *,
    topic: str,
    include_general_web: bool,
    include_wikipedia: bool,
    include_news: bool,
    max_web_results: int,
    max_wiki_pages: int,
    max_news_results: int,
) -> list[dict[str, str]]:
    urls: list[dict[str, str]] = []
    seen: set[str] = set()

    if include_general_web:
        for row in _search_general_web_multi(topic, max_results=max_web_results):
            link = str(row.get("url", "")).strip()
            if not link or link in seen:
                continue
            seen.add(link)
            urls.append(
                {
                    "url": link,
                    "title": str(row.get("title", "")).strip(),
                    "source": "web",
                }
            )

    if include_wikipedia:
        for title in _search_wikipedia(topic, max_pages=max_wiki_pages):
            wiki_url = _wiki_url_from_title(title)
            if wiki_url in seen:
                continue
            seen.add(wiki_url)
            urls.append({"url": wiki_url, "title": title, "source": "wikipedia"})

    if include_news:
        try:
            for item in search_news(query=topic, max_results=max_news_results):
                link = str(item.link or "").strip()
                if not link or link in seen:
                    continue
                seen.add(link)
                urls.append({"url": link, "title": str(item.title or "").strip(), "source": "news"})
        except Exception:  # noqa: BLE001
            pass

    return urls


def _candidate_topic_score(topic: str, row: dict[str, str]) -> dict[str, Any]:
    topic_terms = _topic_tokens(topic)
    if not topic_terms:
        return {"score": 1, "overlap": 0, "anchor_overlap": 0}
    anchor_terms = _topic_anchor_tokens(topic)
    text = " ".join(
        [
            str(row.get("title", "")).strip(),
            str(row.get("url", "")).strip(),
            str(row.get("source", "")).strip(),
        ]
    )
    terms = _topic_tokens(text)
    overlap = len(topic_terms.intersection(terms))
    anchor_overlap = len(anchor_terms.intersection(terms)) if anchor_terms else 0
    source = str(row.get("source", "")).strip().lower()
    source_bonus = 2 if source == "wikipedia" else 0
    score = overlap * 10 + anchor_overlap * 14 + source_bonus
    return {"score": score, "overlap": overlap, "anchor_overlap": anchor_overlap}


def _search_wikipedia(topic: str, max_pages: int = 8, timeout_sec: int = 15) -> list[str]:
    query = quote_plus(topic)
    url = (
        "https://en.wikipedia.org/w/api.php"
        f"?action=query&list=search&format=json&utf8=1&srlimit={max(1, max_pages)}&srsearch={query}"
    )
    payload = _request_json(url, timeout_sec=timeout_sec)
    search = payload.get("query", {}).get("search", [])
    if not isinstance(search, list) or not search:
        return []
    return _filter_ranked_wikipedia_results(topic, search, max_pages=max_pages)


def _fetch_wikipedia_extract(title: str, max_chars: int = 6000, timeout_sec: int = 15) -> str:
    title_q = quote_plus(title)
    url = (
        "https://en.wikipedia.org/w/api.php"
        "?action=query&prop=extracts&explaintext=1&format=json"
        f"&exchars={max(1200, max_chars)}&titles={title_q}"
    )
    payload = _request_json(url, timeout_sec=timeout_sec)
    pages = payload.get("query", {}).get("pages", {})
    if not isinstance(pages, dict):
        return ""
    for _, page in pages.items():
        extract = page.get("extract", "")
        if extract:
            return str(extract)
    return ""


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "topic"


def _wiki_url_from_title(title: str) -> str:
    normalized = str(title or "").strip().replace(" ", "_")
    return f"https://en.wikipedia.org/wiki/{quote(normalized, safe='()')}"


def _wikipedia_title_from_url(url: str) -> str | None:
    parsed = urlparse(str(url or ""))
    host = (parsed.hostname or "").lower()
    if "wikipedia.org" not in host:
        return None
    path = str(parsed.path or "")
    if "/wiki/" not in path:
        return None
    tail = path.split("/wiki/", 1)[1].strip("/")
    if not tail:
        return None
    return unquote(tail).replace("_", " ").strip()


def _topic_queue_path(training_root: Path, topic: str) -> Path:
    out_dir = training_root / "topic_queues"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{_slugify(topic)}.txt"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _keyword_list(title: str, fact: str, max_items: int = 8) -> list[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9'-]{2,}", f"{title} {fact}")
    out: list[str] = []
    seen: set[str] = set()
    for word in words:
        token = word.lower()
        if token in EVAL_STOPWORDS:
            continue
        if len(token) < 4:
            continue
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
        if len(out) >= max(2, max_items):
            break
    return out


def _score_answer(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    text = answer.lower()
    hits = sum(1 for k in keywords if k in text)
    return hits / len(keywords)


def _build_wikipedia_quiz(topic: str, max_pages: int = 6) -> list[dict[str, Any]]:
    titles = _search_wikipedia(topic, max_pages=max_pages)
    quiz: list[dict[str, Any]] = []
    for title in titles[: max(1, max_pages)]:
        extract = _fetch_wikipedia_extract(title, max_chars=5000)
        facts = _split_into_facts(extract, max_facts=8)
        if not facts:
            continue
        expected = facts[0]
        quiz.append(
            {
                "title": title,
                "question": f"In one or two sentences, what is {title}?",
                "expected_fact": expected,
                "keywords": _keyword_list(title, expected, max_items=8),
            }
        )
    return quiz


def _page_topic_match(topic: str, title: str, facts: list[str], extract: str) -> dict[str, Any]:
    topic_terms = _topic_tokens(topic)
    if not topic_terms:
        return {"overlap": 0, "anchor_overlap": 0, "ratio": 1.0, "topic_term_count": 0}
    anchor_terms = _topic_anchor_tokens(topic)
    body = " ".join([title] + list(facts[:6]) + [extract[:700]])
    body_terms = _topic_tokens(body)
    overlap = len(topic_terms.intersection(body_terms))
    anchor_overlap = len(anchor_terms.intersection(body_terms)) if anchor_terms else 0
    ratio = overlap / max(1, len(topic_terms))
    return {
        "overlap": overlap,
        "anchor_overlap": anchor_overlap,
        "ratio": ratio,
        "topic_term_count": len(topic_terms),
    }


def _is_relevant_page(match: dict[str, Any]) -> bool:
    overlap = int(match.get("overlap", 0))
    anchor_overlap = int(match.get("anchor_overlap", 0))
    ratio = float(match.get("ratio", 0.0))
    topic_term_count = int(match.get("topic_term_count", 0))
    required_overlap = 1 if topic_term_count <= 3 else 2
    if overlap < required_overlap:
        return False
    if anchor_overlap > 0:
        return True
    return ratio >= 0.55


def _chat_answer(
    chat_runner: ChatRunner | None,
    *,
    model: str,
    prompt: str,
    enable_memory: bool | None = None,
    enable_web_tools: bool | None = None,
) -> str:
    if chat_runner:
        payload = {
            "model": model,
            "prompt": prompt,
            "max_new": 220,
            "max_new_limit": 320,
            "temperature": 0.55,
            "top_k": 40,
        }
        if enable_memory is not None:
            payload["enable_memory"] = bool(enable_memory)
        if enable_web_tools is not None:
            payload["enable_web_tools"] = bool(enable_web_tools)
        out = chat_runner(payload)
        return str(out.get("response", "")).strip()

    args = SimpleNamespace(
        model=model,
        prompt=prompt,
        max_new=220,
        max_new_limit=320,
        temperature=0.55,
        top_k=40,
        torch_threads=4,
        skill="",
        enable_memory=True if enable_memory is None else bool(enable_memory),
        enable_web_tools=False if enable_web_tools is None else bool(enable_web_tools),
    )
    return extract_response_text(generate_response(args))


def _append_training_examples(
    training_root: Path,
    topic: str,
    page_summaries: list[dict[str, Any]],
) -> dict[str, str]:
    training_root.mkdir(parents=True, exist_ok=True)
    out_path = training_root / "queued_wikipedia_learning.txt"
    topic_path = _topic_queue_path(training_root, topic)
    lines: list[str] = []

    def _compose_answer(title: str, facts: list[str]) -> str:
        cleaned = [re.sub(r"\s+", " ", str(f or "").strip()) for f in facts if str(f or "").strip()]
        if not cleaned:
            return ""
        primary = cleaned[0][:240]
        secondary = cleaned[1][:220] if len(cleaned) > 1 else ""
        if secondary:
            return f"{title}: {primary} {secondary}"
        return f"{title}: {primary}"

    for page in page_summaries:
        title = page["title"]
        facts = page["facts"]
        if not facts:
            continue
        answer = _compose_answer(title, list(facts))
        if not answer:
            continue
        lines.append(f"User: Give me a concise overview of {title} with verified points only.")
        lines.append(f"Ickle: {answer}")
        lines.append("")
        lines.append(f"User: We are studying {topic}. What from {title} is most reliable?")
        lines.append(f"Ickle: {answer}")
        lines.append("")
        for fact in facts[:3]:
            lines.append(f"User: Keep this specific: one grounded point from {title}.")
            lines.append(f"Ickle: {fact}")
            lines.append("")
    if lines:
        text = "\n".join(lines) + "\n"
        with out_path.open("a", encoding="utf-8") as f:
            f.write(text)
        with topic_path.open("a", encoding="utf-8") as f:
            f.write(text)
    return {"global_queue_path": str(out_path), "topic_queue_path": str(topic_path)}


def run_learn_wikipedia_topic(
    payload: dict[str, Any],
    progress: ProgressCb,
    *,
    memory_enabled: bool,
    training_root_override: str | None = None,
) -> dict[str, Any]:
    topic = str(payload.get("topic", "")).strip()
    if not topic:
        raise ValueError("Missing required payload field: topic")

    max_pages = int(payload.get("max_pages", 6))
    progress(f"Searching Wikipedia for '{topic}'")
    titles = _search_wikipedia(topic, max_pages=max_pages)
    if not titles:
        raise RuntimeError(f"No Wikipedia pages found for topic '{topic}'.")

    candidate_summaries: list[dict[str, Any]] = []
    dropped_pages: list[dict[str, Any]] = []
    memory = get_memory() if memory_enabled else None
    research_session_id = f"wiki_{_slugify(topic)}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    for idx, title in enumerate(titles, start=1):
        progress(f"Reading page {idx}/{len(titles)}: {title}")
        extract = _fetch_wikipedia_extract(title)
        facts = _split_into_facts(extract, max_facts=10)
        if not facts:
            dropped_pages.append({"title": title, "reason": "no useful facts extracted"})
            continue
        match = _page_topic_match(topic=topic, title=title, facts=facts, extract=extract)
        row = {
            "url": _wiki_url_from_title(title),
            "source": "wikipedia",
            "title": title,
            "facts": facts,
            "extract_preview": extract[:1200],
            "topic_match": match,
        }
        candidate_summaries.append(row)

    if not candidate_summaries:
        raise RuntimeError(f"Could not extract useful Wikipedia facts for topic '{topic}'.")

    summaries = [row for row in candidate_summaries if _is_relevant_page(row.get("topic_match", {}))]
    if len(summaries) < max(2, min(4, len(candidate_summaries) // 2)):
        candidate_sorted = sorted(
            candidate_summaries,
            key=lambda row: (
                float(row.get("topic_match", {}).get("ratio", 0.0)),
                int(row.get("topic_match", {}).get("anchor_overlap", 0)),
                int(row.get("topic_match", {}).get("overlap", 0)),
            ),
            reverse=True,
        )
        summaries = candidate_sorted[: max(1, min(max_pages, 4))]

    chosen_titles = {str(row.get("title", "")) for row in summaries}
    for row in candidate_summaries:
        title = str(row.get("title", ""))
        if title in chosen_titles:
            continue
        dropped_pages.append(
            {
                "title": title,
                "reason": "low topic relevance",
                "topic_match": row.get("topic_match", {}),
            }
        )

    summaries, wiki_evidence_stats = _apply_web_evidence_policy(
        topic=topic,
        summaries=summaries,
        min_score=0.56,
        min_corroborated_score=0.48,
    )
    if not summaries:
        raise RuntimeError(f"Could not find corroborated evidence for topic '{topic}'.")

    for row in summaries:
        title = row["title"]
        facts = row["facts"]
        if memory and facts:
            page_url = str(row.get("url", "")).strip() or _wiki_url_from_title(title)
            scored_facts = list(row.get("scored_facts", []))
            meta_map: dict[str, dict[str, Any]] = {}
            for item in scored_facts:
                fact_text = str(item.get("fact", "")).strip()
                if not fact_text:
                    continue
                meta_map[fact_text] = {
                    "score": float(item.get("score", 0.0)),
                    "confidence": float(item.get("confidence", 0.82)),
                    "corroboration_count": int(item.get("corroboration_count", 0)),
                }
            memory.add_web_learning(
                url=page_url,
                title=title,
                key_facts=facts[:5],
                topic=topic,
                fact_metadata=meta_map,
            )
            for fact in facts[:6]:
                meta = meta_map.get(fact, {})
                memory.add_research_note(
                    topic=topic,
                    question=f"What is important about {title}?",
                    finding=fact,
                    source_url=page_url,
                    source_title=title,
                    tags=["wikipedia", _slugify(topic)],
                    confidence=float(meta.get("confidence", 0.8)),
                    session_id=research_session_id,
                )

    report_dir = Path("data/tasks")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"wikipedia_{re.sub(r'[^a-zA-Z0-9]+', '_', topic).strip('_').lower() or 'topic'}.json"
    report = {
        "topic": topic,
        "research_session_id": research_session_id,
        "candidate_page_count": len(candidate_summaries),
        "page_count": len(summaries),
        "evidence_policy": {
            "min_score": 0.56,
            "min_corroborated_score": 0.48,
            "stats": wiki_evidence_stats,
        },
        "dropped_pages": dropped_pages,
        "pages": summaries,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    training_root = Path(training_root_override).resolve() if training_root_override else get_training_root()
    training_output_path = ""
    topic_training_output_path = ""
    fallback_path = Path("data/training_queue/queued_wikipedia_learning.txt")
    fallback_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        queue_paths = _append_training_examples(training_root, topic, summaries)
    except Exception:  # noqa: BLE001
        queue_paths = _append_training_examples(fallback_path.parent, topic, summaries)
    training_output_path = str(queue_paths.get("global_queue_path", ""))
    topic_training_output_path = str(queue_paths.get("topic_queue_path", ""))

    progress("Topic learning completed")
    return {
        "topic": topic,
        "pages_processed": len(summaries),
        "report_path": str(report_path),
        "training_queue_path": training_output_path,
        "topic_training_queue_path": topic_training_output_path,
        "memory_enabled": memory_enabled,
        "research_session_id": research_session_id if memory_enabled else None,
        "evidence_policy": report["evidence_policy"],
    }


def run_learn_web_topic(
    payload: dict[str, Any],
    progress: ProgressCb,
    *,
    memory_enabled: bool,
    training_root_override: str | None = None,
) -> dict[str, Any]:
    topic = str(payload.get("topic", "")).strip()
    if not topic:
        raise ValueError("Missing required payload field: topic")

    max_urls = min(24, max(1, int(payload.get("max_urls", 8))))
    max_chars = min(18000, max(1500, int(payload.get("max_chars", 7000))))
    max_facts_per_url = min(14, max(3, int(payload.get("max_facts_per_url", 8))))
    include_general_web = bool(payload.get("include_general_web", True))
    include_wikipedia = bool(payload.get("include_wikipedia", True))
    include_news = bool(payload.get("include_news", True))
    unrestricted = bool(payload.get("unrestricted", True))
    max_web_results = min(24, max(1, int(payload.get("max_web_results", max_urls))))
    max_wiki_pages = min(16, max(1, int(payload.get("max_wiki_pages", 6))))
    max_news_results = min(16, max(1, int(payload.get("max_news_results", 6))))

    default_allowed = {
        "en.wikipedia.org",
        "arxiv.org",
        "www.nature.com",
        "nature.com",
        "www.britannica.com",
        "britannica.com",
        "www.bbc.com",
        "bbc.com",
        "www.reuters.com",
        "reuters.com",
        "www.nasa.gov",
        "nasa.gov",
        ".edu",
        ".gov",
        ".org",
        "sciencedirect.com",
        "www.sciencedirect.com",
        "pubmed.ncbi.nlm.nih.gov",
        "www.ncbi.nlm.nih.gov",
        "plos.org",
        "journals.plos.org",
        "wiley.com",
        "onlinelibrary.wiley.com",
        "springer.com",
        "link.springer.com",
        "ieee.org",
        "ieeexplore.ieee.org",
        "acm.org",
        "dl.acm.org",
        "mit.edu",
        "news.mit.edu",
        "stanford.edu",
        "harvard.edu",
        "ox.ac.uk",
        "cam.ac.uk",
        "pnas.org",
        "www.pnas.org",
        "who.int",
        "www.who.int",
        "nih.gov",
        "www.nih.gov",
        "noaa.gov",
        "www.noaa.gov",
        "usgs.gov",
        "www.usgs.gov",
        "medium.com",
        "github.com",
        "stackoverflow.com",
        "stackexchange.com",
        "wikipedia.org",
    }
    raw_allowed = payload.get("allowed_domains", default_allowed)
    if not isinstance(raw_allowed, (list, tuple, set)):
        raw_allowed = default_allowed
    allowed_domains = {str(d).strip().lower() for d in raw_allowed if str(d).strip()}

    progress(f"Collecting web sources for '{topic}'")
    candidates = _collect_web_learning_urls(
        topic=topic,
        include_general_web=include_general_web,
        include_wikipedia=include_wikipedia,
        include_news=include_news,
        max_web_results=max_web_results,
        max_wiki_pages=max_wiki_pages,
        max_news_results=max_news_results,
    )
    if not candidates:
        raise RuntimeError(f"No web sources found for topic '{topic}'.")

    selected = [row for row in candidates if _domain_allowed(row["url"], unrestricted=unrestricted, allowed_domains=allowed_domains)]
    if not selected:
        raise RuntimeError("No candidate URLs matched the domain policy. Adjust allowed_domains or set unrestricted=true.")
    scored_rows: list[tuple[int, int, int, dict[str, str]]] = []
    required_overlap = 1 if len(_topic_tokens(topic)) <= 3 else 2
    for row in selected:
        metrics = _candidate_topic_score(topic, row)
        overlap = int(metrics.get("overlap", 0))
        anchor_overlap = int(metrics.get("anchor_overlap", 0))
        if overlap < required_overlap and anchor_overlap <= 0:
            continue
        scored_rows.append((int(metrics.get("score", 0)), overlap, anchor_overlap, row))
    if scored_rows:
        scored_rows.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        selected = [row for _, _, _, row in scored_rows[:max_urls]]
    else:
        selected = selected[:max_urls]

    summaries: list[dict[str, Any]] = []
    candidate_summaries: list[dict[str, Any]] = []
    dropped_sources: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    memory = get_memory() if memory_enabled else None
    research_session_id = f"web_{_slugify(topic)}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    def _read_one(idx: int, total: int, row: dict[str, str]) -> dict[str, Any] | None:
        url = row["url"]
        src = row.get("source", "web")
        title_hint = row.get("title", "")
        wiki_title = _wikipedia_title_from_url(url)
        content_preview = ""
        title = str(title_hint or url).strip()
        facts: list[str] = []
        if wiki_title:
            title = wiki_title
            extract = _fetch_wikipedia_extract(wiki_title, max_chars=max_chars)
            facts = _split_into_facts(extract, max_facts=max_facts_per_url)
            content_preview = extract[:1200]
        else:
            web_data = read_url_dynamic(url, timeout_ms=25000, max_chars=max_chars)
            if not web_data.get("success"):
                failures.append({"url": url, "error": str(web_data.get("error", "unknown error"))})
                return None
            title = str(web_data.get("title") or title_hint or url).strip()
            facts = _facts_from_web_payload(web_data, max_facts=max_facts_per_url)
            content_preview = _clean_web_reader_text(str(web_data.get("content", "")))[:1200]
        if not facts:
            failures.append({"url": url, "error": "no useful facts extracted"})
            return None
        match = _page_topic_match(topic=topic, title=title, facts=facts, extract=content_preview)
        return {
            "url": url,
            "source": src,
            "title": title,
            "facts": facts,
            "content_preview": content_preview,
            "topic_match": match,
            "_idx": idx,
        }

    max_workers = min(len(selected), 6)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures_map: dict[concurrent.futures.Future, int] = {}
        for idx, row in enumerate(selected, start=1):
            progress(f"Reading source {idx}/{len(selected)}: {row['url']}")
            f = pool.submit(_read_one, idx, len(selected), row)
            futures_map[f] = idx
        for future in concurrent.futures.as_completed(futures_map):
            result = future.result()
            if result is None:
                continue
            idx = result.pop("_idx", 0)
            summary_row = dict(result)
            candidate_summaries.append(summary_row)
            if not _is_relevant_page(summary_row["topic_match"]):
                dropped_sources.append(
                    {
                        "url": summary_row["url"],
                        "title": summary_row["title"],
                        "reason": "low topic relevance",
                        "topic_match": summary_row["topic_match"],
                    }
                )
                continue
            summaries.append(summary_row)

    if not summaries and candidate_summaries:
        candidate_sorted = sorted(
            candidate_summaries,
            key=lambda row: (
                float(row.get("topic_match", {}).get("ratio", 0.0)),
                int(row.get("topic_match", {}).get("anchor_overlap", 0)),
                int(row.get("topic_match", {}).get("overlap", 0)),
            ),
            reverse=True,
        )
        summaries = candidate_sorted[: max(1, min(3, len(candidate_sorted)))]

    evidence_gate_stats: dict[str, Any] = {
        "input_facts": 0,
        "kept_facts": 0,
        "dropped_facts": 0,
        "kept_sources": len(summaries),
    }
    if summaries:
        summaries, evidence_gate_stats = _apply_web_evidence_policy(
            topic=topic,
            summaries=summaries,
            min_score=max(0.40, float(payload.get("evidence_min_score", 0.58))),
            min_corroborated_score=max(0.35, float(payload.get("evidence_min_corroborated_score", 0.50))),
        )

    if memory:
        for row in summaries:
            url = str(row.get("url", "")).strip()
            title = str(row.get("title", "")).strip()
            src = str(row.get("source", "web")).strip()
            facts = [str(x).strip() for x in list(row.get("facts", [])) if str(x).strip()]
            if not url or not title or not facts:
                continue
            scored_facts = list(row.get("scored_facts", []))
            meta_map: dict[str, dict[str, Any]] = {}
            for item in scored_facts:
                fact_text = str(item.get("fact", "")).strip()
                if not fact_text:
                    continue
                meta_map[fact_text] = {
                    "score": float(item.get("score", 0.0)),
                    "confidence": float(item.get("confidence", 0.0)),
                    "corroboration_count": int(item.get("corroboration_count", 0)),
                }
            memory.add_web_learning(
                url=url,
                title=title,
                key_facts=facts[:6],
                topic=topic,
                fact_metadata=meta_map,
            )
            for fact in facts[:6]:
                meta = meta_map.get(fact, {})
                confidence = float(meta.get("confidence", 0.75 if src == "news" else 0.85))
                memory.add_research_note(
                    topic=topic,
                    question=f"What is important about {title}?",
                    finding=fact,
                    source_url=url,
                    source_title=title,
                    tags=["web", _slugify(topic), src],
                    confidence=confidence,
                    session_id=research_session_id,
                )

    if not summaries:
        excerpt = failures[0]["error"] if failures else "unknown extraction failure"
        raise RuntimeError(f"Web learning failed for topic '{topic}': {excerpt}")

    report_dir = Path("data/tasks")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"web_learning_{_slugify(topic)}.json"
    report = {
        "topic": topic,
        "research_session_id": research_session_id,
        "candidate_source_count": len(candidate_summaries),
        "source_count": len(summaries),
        "failed_sources": failures,
        "dropped_sources": dropped_sources,
        "sources": summaries,
        "domain_policy": {
            "unrestricted": unrestricted,
            "allowed_domains": sorted(allowed_domains),
            "include_general_web": include_general_web,
            "max_urls": max_urls,
            "max_chars": max_chars,
        },
        "evidence_policy": {
            "min_score": max(0.40, float(payload.get("evidence_min_score", 0.58))),
            "min_corroborated_score": max(0.35, float(payload.get("evidence_min_corroborated_score", 0.50))),
            "stats": evidence_gate_stats,
        },
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    training_root = Path(training_root_override).resolve() if training_root_override else get_training_root()
    fallback_path = Path("data/training_queue/queued_wikipedia_learning.txt")
    fallback_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        queue_paths = _append_training_examples(training_root, topic, summaries)
    except Exception:  # noqa: BLE001
        queue_paths = _append_training_examples(fallback_path.parent, topic, summaries)
    training_output_path = str(queue_paths.get("global_queue_path", ""))
    topic_training_output_path = str(queue_paths.get("topic_queue_path", ""))

    progress("Web topic learning completed")
    return {
        "topic": topic,
        "sources_processed": len(summaries),
        "sources_failed": len(failures),
        "report_path": str(report_path),
        "training_queue_path": training_output_path,
        "topic_training_queue_path": topic_training_output_path,
        "memory_enabled": memory_enabled,
        "research_session_id": research_session_id if memory_enabled else None,
        "domain_policy": report["domain_policy"],
        "evidence_policy": report["evidence_policy"],
    }


def run_build_clean_corpus_task(payload: dict[str, Any], progress: ProgressCb) -> dict[str, Any]:
    training_root = Path(str(payload.get("training_root") or get_training_root()))
    out_path = Path(str(payload.get("out_path") or "data/ickle_clean_corpus.txt"))
    max_lines = int(payload.get("max_lines", 20000))
    dictionary_items = int(payload.get("dictionary_items", 0))
    include_open_stream = bool(payload.get("include_open_stream", False))
    progress("Building cleaned corpus")
    lines, stats = build_corpus(
        training_root=training_root,
        max_lines=max_lines,
        dictionary_items=max(0, dictionary_items),
        include_open_stream=include_open_stream,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    progress("Cleaned corpus ready")
    return {"out_path": str(out_path), "stats": stats}


def run_research_notes_query_task(
    payload: dict[str, Any],
    progress: ProgressCb,
    *,
    memory_enabled: bool,
) -> dict[str, Any]:
    if not memory_enabled:
        raise PermissionError("Research note query requires memory_enabled runtime flag.")
    query = str(payload.get("query") or payload.get("topic") or "").strip()
    if not query:
        raise ValueError("research_notes_query requires query")
    topic_hint = str(payload.get("topic_hint", "")).strip() or None
    limit = max(1, int(payload.get("limit", 8)))
    progress(f"Searching research notes for '{query}'")
    memory = get_memory()
    matches = memory.search_research_notes(query, limit=limit, topic_hint=topic_hint)
    follow_up: list[str] = []
    if not matches:
        follow_up.append(f"No notes found yet for '{query}'. Collect new sources first.")
    else:
        low_conf = [row for row in matches if float(row.get("confidence", 0.0)) < 0.6]
        for row in low_conf[:3]:
            src = str(row.get("source_title") or row.get("source_url") or "unknown source")
            follow_up.append(f"Verify low-confidence claim from {src}.")
    sessions = memory.list_research_sessions(limit=10)
    return {
        "query": query,
        "topic_hint": topic_hint,
        "match_count": len(matches),
        "matches": matches,
        "follow_up_actions": follow_up,
        "recent_sessions": sessions,
    }


def run_lora_train_task(payload: dict[str, Any], progress: ProgressCb) -> dict[str, Any]:
    out_model = str(payload.get("out_model", "")).strip()
    if not out_model:
        raise ValueError("lora_train task requires out_model")
    from src.model_resolver import resolve_default_model
    base_model = str(payload.get("base_model", "")).strip()
    if not base_model:
        base_model = resolve_default_model()
    topic = str(payload.get("topic", "")).strip()
    corpus_path = out_model.replace(".pt", "_lora_corpus.txt")
    if topic:
        progress(f"Researching topic: {topic}")
        _build_topic_corpus(topic, corpus_path, payload)
    elif not Path(corpus_path).exists():
        raise ValueError(f"Corpus not found: {corpus_path}")
    steps = int(payload.get("steps", 500))
    lr = float(payload.get("lr", 1e-3))
    rank = int(payload.get("lora_rank", 8))
    alpha = int(payload.get("lora_alpha", 16))
    optimizer = str(payload.get("optimizer", "adamw"))
    cpu_pct = int(payload.get("cpu_pct", DEFAULT_CPU_PCT))
    ram_pct = int(payload.get("ram_pct", DEFAULT_RAM_PCT))
    gpu_pct = int(payload.get("gpu_pct", DEFAULT_GPU_PCT))
    cmd = ["python", "-u", "-m", "src.lora_train", "--base-model", base_model,
           "--data", corpus_path, "--out", out_model, "--steps", str(steps),
           "--lr", str(lr), "--rank", str(rank), "--alpha", str(alpha),
           "--optimizer", optimizer, "--cpu-pct", str(cpu_pct),
           "--ram-pct", str(ram_pct), "--gpu-pct", str(gpu_pct)]
    progress("Starting LoRA training")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    tail: list[str] = []
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            tail.append(line)
            if len(tail) > 120:
                tail = tail[-120:]
            progress(line[:400])
    proc.wait()
    if proc.returncode != 0:
        excerpt = "\n".join(tail[-20:])
        raise RuntimeError(f"LoRA training failed (exit {proc.returncode}).\n{excerpt}")
    return {"status": "completed", "out_model": out_model}


def _build_topic_corpus(topic: str, out_path: str, payload: dict[str, Any]):
    max_wiki = int(payload.get("max_wiki_pages", 8))
    max_urls = int(payload.get("max_urls", 8))
    include_wiki = bool(payload.get("include_wikipedia", True))
    include_web = bool(payload.get("include_web", True))
    text_parts = []
    if include_wiki:
        try:
            from src.tools.news_research import wikipedia_search
            from src.dynamic_web_reader import read_url_dynamic
            pages = wikipedia_search(topic, max_pages=max_wiki)
            for page in pages[:max_wiki]:
                content = read_url_dynamic(page["url"], max_chars=5000)
                if content.get("success"):
                    text_parts.append(str(content.get("text", "")))
        except Exception:
            pass
    if include_web:
        try:
            from src.web_topic_lookup import collect_topic_web_evidence
            data = collect_topic_web_evidence(topic=topic, max_sources=max_urls, max_chars=8000, timeout_ms=15000)
            if data and data.get("success"):
                for source in data.get("sources", [])[:max_urls]:
                    facts = source.get("facts", [])
                    text_parts.append("\n".join(str(f) for f in facts))
        except Exception:
            pass
    if text_parts:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text("\n\n".join(text_parts), encoding="utf-8")


def _resolve_existing_model_path(path_text: str) -> str:
    path = str(path_text or "").strip()
    if not path:
        return ""
    p = Path(path).resolve()
    if p.exists() and p.is_file():
        return str(p)
    return ""


def _select_candidate_model_for_review(
    *,
    out_model: str,
    checkpoint_path: str,
    best_model_path: str,
) -> str:
    for candidate in (best_model_path, out_model, checkpoint_path):
        resolved = _resolve_existing_model_path(candidate)
        if resolved:
            return resolved
    return ""


def _run_promotion_check_after_graceful_stop(
    payload: dict[str, Any],
    *,
    candidate_model: str,
    checkpoint_path: str,
    progress: ProgressCb,
) -> dict[str, Any]:
    candidate = _resolve_existing_model_path(candidate_model)
    if not candidate:
        return {
            "skipped": True,
            "reason": "candidate_model_missing",
            "candidate_model": candidate_model,
            "checkpoint_path": checkpoint_path,
        }

    baseline_model = _resolve_existing_model_path(str(payload.get("baseline_model", "")).strip())
    if not baseline_model:
        baseline_model = _resolve_existing_model_path(str(payload.get("init_model", "")).strip())
    if not baseline_model:
        baseline_model = _resolve_existing_model_path(str(payload.get("promote_to_model", "")).strip())
    if not baseline_model:
        try:
            from src.model_resolver import resolve_default_model

            baseline_model = _resolve_existing_model_path(resolve_default_model())
        except Exception:  # noqa: BLE001
            baseline_model = ""
    if not baseline_model:
        return {
            "skipped": True,
            "reason": "baseline_model_missing",
            "candidate_model": candidate,
            "checkpoint_path": checkpoint_path,
        }

    progress("Running promotion gate benchmark (baseline vs interrupted candidate)")
    try:
        from src.promotion_gate import PromotionGateConfig, run_promotion_cycle

        gate = PromotionGateConfig(
            min_avg_score=float(payload.get("promotion_min_avg_score", 0.45)),
            min_per_case_score=float(payload.get("promotion_min_case_score", 0.20)),
            max_regression=float(payload.get("promotion_max_regression", 0.02)),
            min_quality=float(payload.get("promotion_min_quality", 0.30)),
            require_all_passing=bool(payload.get("promotion_require_all_passing", True)),
        )

        report_path = str(payload.get("promotion_report_path", "")).strip()
        if not report_path:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            report_path = f"data/tasks/promotion_gate_stop_{stamp}.json"

        def _chat_fn(args):
            return extract_response_text(generate_response(args))

        report = run_promotion_cycle(
            _chat_fn,
            candidate_model=candidate,
            baseline_model=baseline_model,
            gate=gate,
            report_path=report_path,
        )
        gate_report = report.get("promotion_gate", {}) if isinstance(report, dict) else {}
        passed = bool(gate_report.get("passed", False))
    except Exception as exc:  # noqa: BLE001
        return {
            "skipped": False,
            "failed": True,
            "error": str(exc),
            "candidate_model": candidate,
            "baseline_model": baseline_model,
        }

    promote_if_pass = bool(payload.get("promote_if_pass", False))
    promote_to_model = str(payload.get("promote_to_model") or baseline_model).strip()
    promoted = False
    promotion_error = ""
    accumulated = False
    accumulation_method = "none"

    if passed and promote_if_pass and promote_to_model:
        try:
            dst = Path(promote_to_model)
            dst.parent.mkdir(parents=True, exist_ok=True)
            accumulator = Accumulator()
            candidate_avg = float(report.get("candidate", {}).get("avg_score", 0.0))
            acc_report = accumulator.try_accumulate(
                master_path=str(dst),
                candidate_path=str(candidate),
                candidate_score=candidate_avg,
            )
            accumulated = bool(acc_report.get("merged", False))
            accumulation_method = str(acc_report.get("method", "none"))
            if not accumulated:
                _atomic_copy_file(Path(candidate), dst)
            src_meta = Path(str(candidate) + ".meta.json")
            dst_meta = Path(str(dst) + ".meta.json")
            if src_meta.exists():
                _atomic_copy_file(src_meta, dst_meta)
            _record_active_model(dst)
            promoted = True
            progress(f"Promotion gate passed. Promoted candidate to {dst}")
        except Exception as exc:  # noqa: BLE001
            promotion_error = str(exc)

    return {
        "skipped": False,
        "failed": False,
        "passed": passed,
        "promoted": promoted,
        "promote_if_pass": promote_if_pass,
        "promote_to_model": promote_to_model if promote_to_model else None,
        "promotion_error": promotion_error or None,
        "candidate_model": candidate,
        "baseline_model": baseline_model,
        "report_path": report_path,
        "accumulated": accumulated,
        "accumulation_method": accumulation_method,
    }


_STREAM_CHARS_PER_TOKEN_ESTIMATE = 4
_STREAM_WASTE_FACTOR = 1.8  # rows skipped by the <80-char filter (or --stream-filter) are read but discarded
_STREAM_MAX_CHARS_FLOOR = 2_000_000  # never below the previous flat default
_STREAM_MAX_CHARS_CEILING = 120_000_000  # bounds how much a very large step count implies downloading


def _default_stream_max_chars(steps: int, batch_size: int, block_size: int) -> int:
    """A flat 2,000,000-char cap regardless of run size meant every streamed
    run -- including the plain-language "General knowledge"/"Helpful
    conversation examples" presets, not just custom Hugging Face datasets --
    saw the same ~2MB of text no matter how many steps it trained for. That
    is little enough to start repeating itself well before a real run
    finishes. This scales the cap with how much data the run can plausibly
    use (steps * batch * block, in estimated characters, with slack for rows
    a length/relevance filter throws away), instead of a number disconnected
    from the run's actual size."""
    effective_batch = batch_size if batch_size > 0 else 22
    effective_block = block_size if block_size > 0 else 512
    raw = int(steps) * effective_batch * effective_block * _STREAM_CHARS_PER_TOKEN_ESTIMATE * _STREAM_WASTE_FACTOR
    return max(_STREAM_MAX_CHARS_FLOOR, min(_STREAM_MAX_CHARS_CEILING, int(raw)))


def _detect_no_steps_executed(log_lines: list[str]) -> bool:
    """True when train.py's subprocess output shows it resumed a checkpoint
    already at or past the requested step count and exited without training
    -- which happens silently whenever out_model collides with a stale
    checkpoint from an unrelated earlier run, even if this run explicitly
    asked for a fresh start via init_model."""
    return any("no training steps executed" in line.lower() for line in log_lines)


def run_train_model_task(payload: dict[str, Any], progress: ProgressCb) -> dict[str, Any]:
    data_path = str(payload.get("data_path", "")).strip()
    out_model = str(payload.get("out_model", "")).strip()
    stream_dataset = str(payload.get("stream_dataset", "")).strip()
    if not out_model:
        raise ValueError("train_model task requires out_model")
    if not data_path and not stream_dataset:
        raise ValueError("train_model task requires data_path or stream_dataset")

    steps = int(payload.get("steps", 1200))
    resource_budget = payload.get("resource_budget", {})
    cpu_pct = int(resource_budget.get("cpu_percent", DEFAULT_CPU_PCT) if isinstance(resource_budget, dict) else DEFAULT_CPU_PCT)
    ram_pct = int(resource_budget.get("ram_percent", DEFAULT_RAM_PCT) if isinstance(resource_budget, dict) else DEFAULT_RAM_PCT)
    gpu_pct = int(resource_budget.get("gpu_percent", DEFAULT_GPU_PCT) if isinstance(resource_budget, dict) else DEFAULT_GPU_PCT)
    bootstrap = bool(payload.get("bootstrap_english", True))
    checkpoint_every_steps = int(payload.get("checkpoint_every_steps", 40))
    eval_every = max(0, int(payload.get("eval_every", 0)))
    eval_iters = max(0, int(payload.get("eval_iters", 0)))
    checkpoint_path = str(payload.get("checkpoint_path") or f"{out_model}.checkpoint.pt").strip()
    best_model_path = str(payload.get("best_model_path") or f"{out_model}.best.pt").strip()
    save_best_on_interrupt = bool(payload.get("save_best_on_interrupt", True))
    resume_from_checkpoint = str(payload.get("resume_from_checkpoint", "")).strip()
    resume_if_possible = bool(payload.get("resume_if_possible", True))
    init_model = str(payload.get("init_model", "")).strip()
    torch_threads = max(0, int(payload.get("torch_threads", 0)))
    batch_size = max(0, int(payload.get("batch_size", 0)))
    grad_accum_steps = max(1, int(payload.get("grad_accum_steps", 1)))
    block_size = max(0, int(payload.get("block_size", 0)))
    n_embd = max(0, int(payload.get("n_embd", 0)))
    n_head = max(0, int(payload.get("n_head", 0)))
    n_layer = max(0, int(payload.get("n_layer", 0)))
    tokenizer_kind = str(payload.get("tokenizer", "sentencepiece")).strip()
    spm_vocab_size = int(payload.get("spm_vocab_size", 0))
    stream_dataset_2 = str(payload.get("stream_dataset_2", "")).strip()
    stream_field_2 = str(payload.get("stream_field_2", "text")).strip()
    stream_template_2 = str(payload.get("stream_template_2", "")).strip()
    stream_max_2 = int(payload.get("stream_max_chars_2", 0))
    status_file = str(payload.get("status_file", "")).strip()
    if not status_file:
        training_root = Path(str(payload.get("training_root") or get_training_root()))
        status_file = str((training_root / "training_live.json").resolve())
    if status_file:
        try:
            Path(status_file).parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

    if not resume_from_checkpoint and resume_if_possible and checkpoint_path and Path(checkpoint_path).exists():
        resume_from_checkpoint = checkpoint_path

    stop_request_path = get_training_stop_request_path(
        out_model=out_model,
        checkpoint_path=checkpoint_path,
    )
    clear_training_stop_request(stop_request_path)

    cmd = [
        "python",
        "-u",
        "-m",
        "src.train",
        "--out",
        out_model,
        "--steps",
        str(steps),
        "--cpu-pct",
        str(cpu_pct),
        "--ram-pct",
        str(ram_pct),
        "--gpu-pct",
        str(gpu_pct),
    ]
    if stream_dataset:
        cmd.extend(["--stream-dataset", stream_dataset])
        stream_field = str(payload.get("stream_field", "text")).strip()
        cmd.extend(["--stream-field", stream_field])
        stream_config = str(payload.get("stream_config", "")).strip()
        if stream_config:
            cmd.extend(["--stream-config", stream_config])
        stream_filter = str(payload.get("stream_filter", "")).strip()
        if stream_filter:
            cmd.extend(["--stream-filter", stream_filter])
        stream_template = str(payload.get("stream_template", "")).strip()
        if stream_template:
            cmd.extend(["--stream-template", stream_template])
        stream_max = int(payload.get("stream_max_chars") or _default_stream_max_chars(steps, batch_size, block_size))
        cmd.extend(["--stream-max-chars", str(stream_max)])
        if "stream_shuffle_buffer" in payload:
            cmd.extend(["--stream-shuffle-buffer", str(int(payload.get("stream_shuffle_buffer", 0)))])
        if "stream_shuffle_seed" in payload:
            cmd.extend(["--stream-shuffle-seed", str(int(payload.get("stream_shuffle_seed", -1)))])
        if "stream_role_map" in payload:
            cmd.extend(["--stream-role-map", str(payload.get("stream_role_map", ""))])
        cmd.extend(["--tokenizer", tokenizer_kind])
        if tokenizer_kind == "sentencepiece" and spm_vocab_size > 0:
            cmd.extend(["--spm-vocab-size", str(spm_vocab_size)])
        if stream_dataset_2:
            cmd.extend(["--stream-dataset-2", stream_dataset_2])
            cmd.extend(["--stream-field-2", stream_field_2])
            if stream_template_2:
                cmd.extend(["--stream-template-2", stream_template_2])
            if stream_max_2 > 0:
                cmd.extend(["--stream-max-chars-2", str(stream_max_2)])
    else:
        cmd.extend(["--data", data_path])
        cmd.extend(["--tokenizer", tokenizer_kind])
        if tokenizer_kind == "sentencepiece" and spm_vocab_size > 0:
            cmd.extend(["--spm-vocab-size", str(spm_vocab_size)])
    if bootstrap:
        cmd.append("--bootstrap-english")
    if checkpoint_every_steps > 0:
        cmd.extend(["--checkpoint-every", str(checkpoint_every_steps)])
    if eval_every > 0:
        cmd.extend(["--eval-every", str(eval_every)])
    if eval_iters > 0:
        cmd.extend(["--eval-iters", str(eval_iters)])
    if checkpoint_path:
        cmd.extend(["--checkpoint-path", checkpoint_path])
    if best_model_path:
        cmd.extend(["--best-model-path", best_model_path])
        if save_best_on_interrupt:
            cmd.append("--save-best-on-interrupt")
    if resume_from_checkpoint:
        cmd.extend(["--resume-from-checkpoint", resume_from_checkpoint])
    elif init_model:
        cmd.extend(["--init-model", init_model])
    if torch_threads > 0:
        cmd.extend(["--torch-threads", str(torch_threads)])
    if batch_size > 0:
        cmd.extend(["--batch-size", str(batch_size)])
    if grad_accum_steps > 1:
        cmd.extend(["--grad-accum-steps", str(grad_accum_steps)])
    if block_size > 0:
        cmd.extend(["--block-size", str(block_size)])
    if n_embd > 0:
        cmd.extend(["--n-embd", str(n_embd)])
    if n_head > 0:
        cmd.extend(["--n-head", str(n_head)])
    if n_layer > 0:
        cmd.extend(["--n-layer", str(n_layer)])
    if status_file:
        cmd.extend(["--status-file", status_file])

    optimizer = str(payload.get("optimizer", "")).strip()
    if optimizer and optimizer in ("adamw", "muon"):
        cmd.extend(["--optimizer", optimizer])
    lr_schedule = str(payload.get("lr_schedule", "")).strip()
    if lr_schedule and lr_schedule in ("cosine", "linear", "constant"):
        cmd.extend(["--lr-schedule", lr_schedule])
    contrastive = float(payload.get("contrastive_coeff", -1))
    if contrastive >= 0:
        cmd.extend(["--contrastive-coeff", str(contrastive)])
    if bool(payload.get("embed_norm", False)):
        cmd.append("--embed-norm")
    if bool(payload.get("curriculum_sort", False)):
        cmd.append("--curriculum-sort")
    target_loss = float(payload.get("target_loss", 0))
    if target_loss > 0:
        cmd.extend(["--target-loss", str(target_loss)])

    if status_file:
        try:
            Path(status_file).write_text(
                json.dumps(
                    {
                        "step": 0,
                        "total_steps": steps,
                        "status": "running",
                        "train_loss": None,
                        "val_loss": None,
                        "perplexity": None,
                        "acc_top1": None,
                        "acc_top5": None,
                        "lr": None,
                        "best_val_loss": None,
                        "elapsed_seconds": 0.0,
                        "timestamp_utc": _utc_now(),
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
        except OSError:
            pass

    progress("Starting local training")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
    )
    tail: list[str] = []
    graceful_stop_requested = False
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            if (not graceful_stop_requested) and stop_request_path.exists():
                _ = read_training_stop_request(stop_request_path)
                graceful_stop_requested = True
                progress("Graceful stop requested. Ending training after saving interrupt checkpoint.")
                _stop_process(proc)

            msg = line.strip()
            if not msg:
                continue
            tail.append(msg)
            if len(tail) > 120:
                tail = tail[-120:]
            lower = msg.lower()
            if (
                msg.startswith("step=")
                or "checkpoint saved" in lower
                or "resumed from checkpoint" in lower
                or "saved model:" in lower
                or "training interrupted" in lower
            ):
                progress(msg[:400])
        proc.wait()
    except Exception:
        _stop_process(proc)
        raise
    finally:
        proc.stdout.close()
        clear_training_stop_request(stop_request_path)

    interrupted_exit_codes = {130, -1073741510, 3221225786}
    interrupted = proc.returncode in interrupted_exit_codes and graceful_stop_requested
    promotion_gate_result: dict[str, Any] | None = None
    candidate_model_for_review = _select_candidate_model_for_review(
        out_model=out_model,
        checkpoint_path=checkpoint_path,
        best_model_path=best_model_path,
    )

    if proc.returncode != 0 and not interrupted:
        excerpt = "\n".join(tail[-20:])
        raise RuntimeError(f"Training failed (exit {proc.returncode}).\n{excerpt}")
    if interrupted:
        progress("Training interrupted cleanly. Running promotion gate check on saved checkpoint/best model.")
        if bool(payload.get("promotion_gate", True)):
            promotion_gate_result = _run_promotion_check_after_graceful_stop(
                payload,
                candidate_model=candidate_model_for_review,
                checkpoint_path=checkpoint_path,
                progress=progress,
            )
        else:
            promotion_gate_result = {
                "skipped": True,
                "reason": "promotion_gate_disabled",
                "candidate_model": candidate_model_for_review,
            }
        if isinstance(promotion_gate_result, dict):
            if promotion_gate_result.get("failed"):
                progress(f"Promotion gate failed: {promotion_gate_result.get('error', 'unknown error')}")
            elif promotion_gate_result.get("skipped"):
                progress(f"Promotion gate skipped: {promotion_gate_result.get('reason', 'n/a')}")
            else:
                progress(
                    "Promotion gate complete: "
                    f"passed={bool(promotion_gate_result.get('passed', False))} "
                    f"promoted={bool(promotion_gate_result.get('promoted', False))}"
                )

    # A stale checkpoint left over from an earlier run at the same out_model
    # path can silently outrank an explicitly-requested init_model/fresh
    # start: resume_if_possible finds it, the process loads it, sees it's
    # already at or past the requested step count, and exits having done
    # nothing -- while still reporting task status "completed", identical to
    # a real successful run. Confirmed live: a "retrain" meant to pick up a
    # bug fix silently no-op'd this way and the bad model went untouched.
    no_steps_executed = _detect_no_steps_executed(tail)
    if no_steps_executed:
        progress(
            "Warning: this run executed zero new training steps -- it resumed a checkpoint "
            "already at or past the requested step count instead of training. If a fresh "
            "start was intended, delete the stale checkpoint_path first or pass "
            "resume_if_possible=false."
        )
    elif interrupted:
        progress("Training finalized after graceful early stop.")
    else:
        progress("Training completed")
    return {
        "out_model": out_model,
        "interrupted": interrupted,
        "no_steps_executed": no_steps_executed,
        "candidate_model_for_review": candidate_model_for_review or None,
        "promotion_gate": promotion_gate_result,
        "steps": steps,
        "cpu_pct": cpu_pct,
        "ram_pct": ram_pct,
        "checkpoint_path": checkpoint_path,
        "best_model_path": best_model_path,
        "resumed_from_checkpoint": bool(resume_from_checkpoint),
        "torch_threads": torch_threads,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "status_file": status_file or None,
        "log_tail": tail[-10:],
    }


def _generate_learned_summary(*, new_corpus_path: str, model: str, torch_threads: int = 0) -> str:
    corpus = Path(new_corpus_path)
    if not corpus.exists():
        return ""
    pairs = []
    pending_user = ""
    for raw in corpus.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = re.sub(r"\s+", " ", str(raw).strip()).strip()
        if not line:
            continue
        lower = line.lower()
        if lower.startswith("user:"):
            pending_user = line.split(":", 1)[1].strip()
        elif lower.startswith("ickle:") or lower.startswith("assistant:"):
            if not pending_user:
                continue
            assistant = line.split(":", 1)[1].strip()
            if len(pending_user) < 4 or len(assistant) < 4:
                continue
            pairs.append((pending_user, assistant))
            pending_user = ""
            if len(pairs) >= 6:
                break

    if len(pairs) < 2:
        return ""

    preview = "\n".join(f"Q: {u}\nA: {a}\n" for u, a in pairs)
    facts_summary = ". ".join(a[:130] for _, a in pairs)[:600]
    prompt = f"I just learned these facts: {facts_summary}. Summarize what I now know in one sentence."

    gen_args = SimpleNamespace(
        model=model,
        prompt=prompt,
        max_new=120,
        max_new_limit=180,
        temperature=0.5,
        top_k=30,
        torch_threads=max(1, torch_threads),
        skill="",
        enable_memory=True,
        enable_web_tools=False,
    )
    return extract_response_text(generate_response(gen_args))


def run_continual_guard_task(payload: dict[str, Any], progress: ProgressCb) -> dict[str, Any]:
    baseline_model = str(payload.get("baseline_model") or "models/ickle_clean.pt").strip()
    out_model = str(payload.get("out_model") or "models/ickle_continual_candidate.pt").strip()
    if not baseline_model or not out_model:
        raise ValueError("continual_guard_step requires baseline_model and out_model")

    core_corpus = str(payload.get("core_corpus") or "data/ickle_curated_only.txt").strip()
    new_corpus = str(payload.get("new_corpus") or payload.get("data_path") or "data/ickle_clean_corpus.txt").strip()
    training_root = Path(str(payload.get("training_root") or get_training_root())).resolve()
    auto_include_smart_corpus = bool(payload.get("auto_include_smart_corpus", True))
    rebuild_smart_corpus = bool(payload.get("rebuild_smart_corpus", True))
    auto_build_focus_corpus = bool(payload.get("auto_build_focus_corpus", True))
    smart_corpus_path = str(payload.get("smart_corpus_path") or "data/ickle_smart_corpus.txt").strip()
    focus_corpus_path = str(payload.get("focus_corpus_path") or "data/continual/conversation_focus.txt").strip()
    benchmark_file = str(payload.get("user_benchmark_file", "data/maintenance/user_chat_benchmark.json")).strip()

    corpora: list[str] = []
    raw_new_corpora = payload.get("new_corpora")
    if isinstance(raw_new_corpora, list):
        for item in raw_new_corpora:
            path = str(item or "").strip()
            if path:
                corpora.append(path)
    if new_corpus:
        corpora.append(new_corpus)
    if auto_include_smart_corpus and smart_corpus_path:
        smart_path_obj = Path(smart_corpus_path)
        if rebuild_smart_corpus or (not smart_path_obj.exists()):
            progress("Refreshing smart conversation corpus")
            smart_pairs, _smart_stats = build_smart_corpus(
                training_root=training_root,
                max_pairs=max(2000, int(payload.get("smart_max_pairs", 14000))),
                seed=int(payload.get("seed", 1337)),
                include_legacy_dictionary=False,
                include_topic_queues=False,
            )
            smart_path_obj.parent.mkdir(parents=True, exist_ok=True)
            smart_lines: list[str] = []
            for pair in smart_pairs:
                smart_lines.append(f"User: {pair.user}")
                smart_lines.append(f"Ickle: {pair.assistant}")
                smart_lines.append("")
            smart_path_obj.write_text("\n".join(smart_lines) + "\n", encoding="utf-8")
        if smart_path_obj.exists():
            corpora.append(str(smart_path_obj))

    focus_stats: dict[str, Any] | None = None
    if auto_build_focus_corpus and focus_corpus_path and benchmark_file:
        progress("Building focused conversation corpus")
        focus_candidates = list(corpora)
        queued_global = training_root / "queued_wikipedia_learning.txt"
        if queued_global.exists():
            focus_candidates.append(str(queued_global))
        focus_stats = build_focus_corpus_file(
            out_path=focus_corpus_path,
            benchmark_file=benchmark_file,
            candidate_paths=focus_candidates,
            max_pairs=max(80, int(payload.get("focus_max_pairs", 220))),
            seed=int(payload.get("seed", 1337)),
        )
        if int(focus_stats.get("written_pairs", 0)) > 0:
            corpora.append(str(focus_stats.get("out_path", focus_corpus_path)))

    auto_include_verified_corrections = bool(payload.get("auto_include_verified_corrections", True))
    verified_corrections_path = str(
        payload.get("verified_corrections_path") or "data/continual/verified_corrections.txt"
    ).strip()
    verified_stats: dict[str, Any] | None = None
    if auto_include_verified_corrections and verified_corrections_path:
        progress("Consolidating verified Epistemic Commons corrections")
        verified_stats = build_verified_corrections_corpus_file(
            out_path=verified_corrections_path,
            oversample=max(1, int(payload.get("verified_correction_oversample", 3))),
            max_pairs=max(0, int(payload.get("verified_corrections_max_pairs", 900))),
        )
        if int(verified_stats.get("written_pairs", 0)) > 0:
            corpora.append(str(verified_stats.get("out_path", verified_corrections_path)))

    auto_include_disagreement_hedges = bool(payload.get("auto_include_disagreement_hedges", True))
    disagreement_hedges_path = str(
        payload.get("disagreement_hedges_path") or "data/continual/disagreement_hedges.txt"
    ).strip()
    hedge_stats: dict[str, Any] | None = None
    if auto_include_disagreement_hedges and disagreement_hedges_path:
        progress("Building hedge corpus for unresolved swarm disagreements")
        hedge_stats = build_hedge_corpus_file(
            out_path=disagreement_hedges_path,
            oversample=max(1, int(payload.get("disagreement_hedge_oversample", 2))),
            max_pairs=max(0, int(payload.get("disagreement_hedges_max_pairs", 300))),
        )
        if int(hedge_stats.get("written_pairs", 0)) > 0:
            corpora.append(str(hedge_stats.get("out_path", disagreement_hedges_path)))

    normalized_corpora: list[str] = []
    seen_corpora: set[str] = set()
    for path in corpora:
        norm = str(path or "").strip()
        if not norm:
            continue
        key = str(Path(norm))
        if key in seen_corpora:
            continue
        seen_corpora.add(key)
        normalized_corpora.append(norm)

    merge_stats: dict[str, Any] | None = None
    if len(normalized_corpora) > 1:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        merged_path = str(payload.get("merged_new_corpus_out") or f"data/continual/new_mix_{stamp}.txt").strip()
        progress("Merging focused training corpora")
        merge_stats = merge_dialog_corpora(
            corpus_paths=normalized_corpora,
            out_path=merged_path,
            max_pairs_per_source=max(500, int(payload.get("max_pairs_per_source", 5000))),
            max_total_pairs=max(600, int(payload.get("max_total_pairs", 14000))),
            seed=int(payload.get("seed", 1337)),
        )
        if int(merge_stats.get("written_pairs", 0)) > 0:
            new_corpus = str(merge_stats.get("out_path", merged_path))

    replay_buffer = str(payload.get("replay_buffer") or "data/continual/replay_buffer.jsonl").strip()
    mixed_corpus_out = str(payload.get("mixed_corpus_out") or "data/continual/continual_mix.txt").strip()
    checkpoint_path = str(payload.get("checkpoint_path") or f"{out_model}.checkpoint.pt").strip()

    promote_if_pass = bool(payload.get("promote_if_pass", True))
    promote_to_model = ""
    if promote_if_pass:
        promote_to_model = str(payload.get("promote_to_model") or baseline_model).strip()

    report_path = str(payload.get("report_path") or "").strip()
    if not report_path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        report_path = f"data/continual/guard_step_{stamp}.json"

    resource_budget = payload.get("resource_budget", {})
    cpu_pct = int(resource_budget.get("cpu_percent", DEFAULT_CPU_PCT) if isinstance(resource_budget, dict) else DEFAULT_CPU_PCT)
    ram_pct = int(resource_budget.get("ram_percent", DEFAULT_RAM_PCT) if isinstance(resource_budget, dict) else DEFAULT_RAM_PCT)
    gpu_pct = int(resource_budget.get("gpu_percent", DEFAULT_GPU_PCT) if isinstance(resource_budget, dict) else DEFAULT_GPU_PCT)

    args = SimpleNamespace(
        core_corpus=core_corpus,
        new_corpus=new_corpus,
        replay_buffer=replay_buffer,
        mixed_corpus_out=mixed_corpus_out,
        baseline_model=baseline_model,
        out_model=out_model,
        checkpoint_path=checkpoint_path,
        promote_to=promote_to_model,
        report_path=report_path,
        steps=int(payload.get("steps", 1200)),
        lr=float(payload.get("lr", 8e-6)),
        warmup_steps=int(payload.get("warmup_steps", 80)),
        cpu_pct=cpu_pct,
        ram_pct=ram_pct,
        gpu_pct=gpu_pct,
        torch_threads=int(payload.get("torch_threads", 0)),
        batch_size=int(payload.get("batch_size", 0)),
        grad_accum_steps=int(payload.get("grad_accum_steps", 1)),
        replay_max_size=int(payload.get("replay_max_size", 20000)),
        total_pairs=int(payload.get("total_pairs", 12000)),
        max_core_pairs=int(payload.get("max_core_pairs", 6000)),
        max_new_pairs=int(payload.get("max_new_pairs", 10000)),
        core_ratio=float(payload.get("core_ratio", 0.45)),
        replay_ratio=float(payload.get("replay_ratio", 0.35)),
        new_ratio=float(payload.get("new_ratio", 0.20)),
        eval_core_prompts=int(payload.get("eval_core_prompts", 16)),
        eval_new_prompts=int(payload.get("eval_new_prompts", 16)),
        max_core_drop=float(payload.get("max_core_drop", 0.03)),
        min_new_gain=float(payload.get("min_new_gain", 0.00)),
        min_core_score=float(payload.get("min_core_score", 0.38)),
        min_core_quality=float(payload.get("min_core_quality", 0.35)),
        user_benchmark_file=benchmark_file,
        min_user_delta=float(payload.get("min_user_delta", 0.00)),
        min_user_score=float(payload.get("min_user_score", -1.0)),
        min_user_case_score=float(payload.get("min_user_case_score", -1.0)),
        max_user_evasive=int(payload.get("max_user_evasive", -1)),
        seed=int(payload.get("seed", 1337)),
        resume_if_possible=bool(payload.get("resume_if_possible", True)),
    )

    progress("Starting continual guarded training step")
    report = run_guarded_step(args, progress_cb=progress)
    progress(
        f"Continual guard done: passed={report['passed']} promoted={report['promoted']} "
        f"core_drop={report['scores']['core_drop']} new_gain={report['scores']['new_gain']} "
        f"user_delta={report['scores'].get('user_delta', 0.0)}"
    )

    learned_summary = ""
    if report["passed"] and report["promoted"]:
        try:
            progress("Generating what-I-learnt summary")
            learned_summary = _generate_learned_summary(
                new_corpus_path=new_corpus,
                model=str(report.get("promote_to") or args.baseline_model),
                torch_threads=int(payload.get("torch_threads", 0)),
            )
            if learned_summary:
                progress(f"Learnt summary: {learned_summary[:200]}")
        except Exception as exc:  # noqa: BLE001
            progress(f"Summary generation skipped: {exc}")

    return {
        "passed": bool(report["passed"]),
        "promoted": bool(report["promoted"]),
        "candidate_model": str(report["candidate_model"]),
        "baseline_model": str(report["baseline_model"]),
        "promote_to_model": report.get("promote_to"),
        "core_drop": float(report["scores"]["core_drop"]),
        "new_gain": float(report["scores"]["new_gain"]),
        "user_delta": float(report["scores"].get("user_delta", 0.0)),
        "user_score": float(report["scores"].get("user_score", 0.0)),
        "user_min_case_score": float(report["scores"].get("user_min_case_score", 0.0)),
        "user_evasive_count": int(report["scores"].get("user_evasive_count", 0)),
        "report_path": report_path,
        "mixing": report.get("mixing", {}),
        "new_corpus_used": new_corpus,
        "focus_corpus": focus_stats or {},
        "verified_corrections": verified_stats or {},
        "disagreement_hedges": hedge_stats or {},
        "merged_new_corpus": merge_stats or {},
        "learned_summary": learned_summary,
    }


def run_evaluate_model_task(
    payload: dict[str, Any],
    progress: ProgressCb,
    *,
    chat_runner: ChatRunner | None = None,
) -> dict[str, Any]:
    topic = str(payload.get("topic", "")).strip()
    candidate_model = str(payload.get("candidate_model", "")).strip()
    baseline_model = str(payload.get("baseline_model", "")).strip()
    if not topic:
        raise ValueError("evaluate_model requires topic")
    if not candidate_model:
        raise ValueError("evaluate_model requires candidate_model")
    if not baseline_model:
        raise ValueError("evaluate_model requires baseline_model")

    quiz_size = max(3, int(payload.get("quiz_size", 6)))
    max_pages = max(3, int(payload.get("max_pages", quiz_size)))
    min_delta = float(payload.get("min_delta", -0.02))
    min_candidate_avg = float(payload.get("min_candidate_avg", 0.18))
    require_model_only_pass = bool(payload.get("require_model_only_pass", False))
    model_only_min_delta = float(payload.get("model_only_min_delta", 0.0))
    model_only_min_candidate_avg = float(payload.get("model_only_min_candidate_avg", 0.05))
    promote_if_pass = bool(payload.get("promote_if_pass", True))
    promote_to_model = str(payload.get("promote_to_model", "")).strip()
    eval_enable_memory = payload.get("eval_enable_memory")
    eval_enable_web_tools = payload.get("eval_enable_web_tools")
    extended_eval = bool(payload.get("extended_eval", False))

    inline_quiz = payload.get("quiz_items")
    if isinstance(inline_quiz, list) and inline_quiz:
        quiz = inline_quiz[:quiz_size]
    else:
        progress(f"Building evaluation quiz for topic '{topic}'")
        quiz = _build_wikipedia_quiz(topic, max_pages=max_pages)[:quiz_size]
    if len(quiz) < 3:
        raise RuntimeError(f"Could not build enough quiz items for topic '{topic}'")

    ui_enable_memory = eval_enable_memory if isinstance(eval_enable_memory, bool) else None
    ui_enable_web_tools = eval_enable_web_tools if isinstance(eval_enable_web_tools, bool) else None

    def _bridge_chat_runner(payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "response": _chat_answer(
                chat_runner,
                model=str(payload.get("model", "")),
                prompt=str(payload.get("prompt", "")),
                enable_memory=payload.get("enable_memory"),
                enable_web_tools=payload.get("enable_web_tools"),
            )
        }

    ui_round = run_eval_round(
        quiz_items=quiz,
        candidate_model=candidate_model,
        baseline_model=baseline_model,
        chat_runner=_bridge_chat_runner,
        round_name="ui_eval",
        enable_memory=ui_enable_memory,
        enable_web_tools=ui_enable_web_tools,
        include_extended_suites=extended_eval,
    )
    baseline_avg = float(ui_round["baseline_avg_score"])
    candidate_avg = float(ui_round["candidate_avg_score"])
    delta = float(ui_round["delta"])
    ui_pass = candidate_avg >= min_candidate_avg and delta >= min_delta

    model_only = None
    model_only_pass = True
    if require_model_only_pass:
        model_only_round = run_eval_round(
            quiz_items=quiz,
            candidate_model=candidate_model,
            baseline_model=baseline_model,
            chat_runner=_bridge_chat_runner,
            round_name="model_only_eval",
            enable_memory=False,
            enable_web_tools=False,
            include_extended_suites=extended_eval,
        )
        mo_baseline_avg = float(model_only_round["baseline_avg_score"])
        mo_candidate_avg = float(model_only_round["candidate_avg_score"])
        mo_delta = float(model_only_round["delta"])
        model_only_pass = mo_candidate_avg >= model_only_min_candidate_avg and mo_delta >= model_only_min_delta
        model_only = {
            "baseline_avg_score": round(mo_baseline_avg, 4),
            "candidate_avg_score": round(mo_candidate_avg, 4),
            "delta": round(mo_delta, 4),
            "thresholds": {
                "model_only_min_delta": model_only_min_delta,
                "model_only_min_candidate_avg": model_only_min_candidate_avg,
            },
            "passed": model_only_pass,
            "round": model_only_round,
        }

    passed = ui_pass and model_only_pass

    promoted = False
    if passed and promote_if_pass and promote_to_model:
        src = Path(candidate_model)
        dst = Path(promote_to_model)
        if not src.exists():
            raise RuntimeError(f"Candidate model does not exist for promotion: {candidate_model}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        accumulator = Accumulator()
        acc_report = accumulator.try_accumulate(
            master_path=str(dst),
            candidate_path=str(src),
            candidate_score=candidate_avg,
        )
        if not acc_report.get("merged"):
            _atomic_copy_file(src, dst)
        src_meta = Path(str(src) + ".meta.json")
        dst_meta = Path(str(dst) + ".meta.json")
        if src_meta.exists():
            _atomic_copy_file(src_meta, dst_meta)
        _record_active_model(dst)
        promoted = True

    report = {
        "timestamp_utc": _utc_now(),
        "topic": topic,
        "candidate_model": candidate_model,
        "baseline_model": baseline_model,
        "quiz_size": len(quiz),
        "baseline_avg_score": round(baseline_avg, 4),
        "candidate_avg_score": round(candidate_avg, 4),
        "delta": round(delta, 4),
        "thresholds": {
            "min_delta": min_delta,
            "min_candidate_avg": min_candidate_avg,
        },
        "ui_pass": ui_pass,
        "require_model_only_pass": require_model_only_pass,
        "extended_eval": extended_eval,
        "model_only": model_only,
        "passed": passed,
        "promoted": promoted,
        "promote_to_model": promote_to_model if promote_to_model else None,
        "ui_round": ui_round,
        "baseline_answers": [
            {
                "title": row.get("title"),
                "question": row.get("prompt"),
                "expected_fact": row.get("expected_fact"),
                "keywords": row.get("keywords"),
                "answer": row.get("baseline_answer"),
                "score": row.get("baseline_score"),
            }
            for row in ui_round.get("rows", [])
        ],
        "candidate_answers": [
            {
                "title": row.get("title"),
                "question": row.get("prompt"),
                "expected_fact": row.get("expected_fact"),
                "keywords": row.get("keywords"),
                "answer": row.get("candidate_answer"),
                "score": row.get("candidate_score"),
            }
            for row in ui_round.get("rows", [])
        ],
    }

    out_dir = Path("data/tasks")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"evaluation_{_slugify(topic)}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    progress("Model evaluation completed")
    return {
        "passed": passed,
        "ui_pass": ui_pass,
        "model_only_pass": model_only_pass,
        "promoted": promoted,
        "baseline_avg_score": round(baseline_avg, 4),
        "candidate_avg_score": round(candidate_avg, 4),
        "delta": round(delta, 4),
        "report_path": str(out_path),
    }


def run_task(
    task_type: str,
    payload: dict[str, Any],
    progress: ProgressCb,
    *,
    memory_enabled: bool,
    allow_auto_training_tasks: bool,
    training_root_override: str | None = None,
    chat_runner: ChatRunner | None = None,
) -> dict[str, Any]:
    kind = task_type.strip().lower()
    if kind == "learn_wikipedia_topic":
        return run_learn_wikipedia_topic(
            payload,
            progress,
            memory_enabled=memory_enabled,
            training_root_override=training_root_override,
        )
    if kind == "learn_web_topic":
        return run_learn_web_topic(
            payload,
            progress,
            memory_enabled=memory_enabled,
            training_root_override=training_root_override,
        )
    if kind == "build_clean_corpus":
        return run_build_clean_corpus_task(payload, progress)
    if kind == "research_notes_query":
        return run_research_notes_query_task(payload, progress, memory_enabled=memory_enabled)
    if kind == "train_model":
        if not allow_auto_training_tasks:
            raise PermissionError("Auto-training tasks are disabled by runtime flags.")
        return run_train_model_task(payload, progress)
    if kind == "continual_guard_step":
        if not allow_auto_training_tasks:
            raise PermissionError("Auto-training tasks are disabled by runtime flags.")
        return run_continual_guard_task(payload, progress)
    if kind == "evaluate_model":
        return run_evaluate_model_task(payload, progress, chat_runner=chat_runner)
    if kind == "build_teacher_corpus":
        return run_build_teacher_corpus_task(payload, progress)
    if kind == "build_teacher_preferences":
        return run_build_teacher_prefs_task(payload, progress)
    if kind == "generate_teacher_data":
        return run_generate_teacher_data_task(payload, progress)
    if kind == "train_from_teacher":
        if not allow_auto_training_tasks:
            raise PermissionError("Auto-training tasks are disabled by runtime flags.")
        return run_train_from_teacher_task(payload, progress, chat_runner=chat_runner)
    if kind == "lora_train":
        if not allow_auto_training_tasks:
            raise PermissionError("Auto-training tasks are disabled by runtime flags.")
        return run_lora_train_task(payload, progress)
    if kind == "build_dpo_preferences":
        return run_build_dpo_prefs_task(payload, progress)
    if kind == "dpo_train":
        if not allow_auto_training_tasks:
            raise PermissionError("Auto-training tasks are disabled by runtime flags.")
        return run_dpo_train_task(payload, progress)
    if kind == "codistill_round":
        return run_codistill_round_task(payload, progress)
    raise ValueError(f"Unsupported task_type '{task_type}'")


def run_build_teacher_corpus_task(payload: dict[str, Any], progress: ProgressCb) -> dict[str, Any]:
    from src.teacher_ingest import TeacherStore
    out_path = str(payload.get("out_path", "data/teacher/teacher_sft_corpus.txt")).strip()
    progress(f"Building teacher SFT corpus -> {out_path}")
    result = TeacherStore().build_sft_corpus(out_path=out_path)
    progress(f"Teacher SFT corpus: {result['pairs']} pairs written")
    return result


def run_build_teacher_prefs_task(payload: dict[str, Any], progress: ProgressCb) -> dict[str, Any]:
    from src.teacher_ingest import TeacherStore
    out_path = str(payload.get("out_path", "data/teacher/teacher_prefs.jsonl")).strip()
    progress(f"Building teacher preference pairs -> {out_path}")
    result = TeacherStore().build_preference_pairs(out_path=out_path)
    progress(f"Teacher DPO pairs: {result['pairs']} pairs written")
    return result


def run_generate_teacher_data_task(payload: dict[str, Any], progress: ProgressCb) -> dict[str, Any]:
    """Ask an AI teacher (Claude directly, or any provider registered via
    `trainer-provider`) for training examples on a topic and store them --
    the same two calls the `anthropic-teach batch-sft --store`/
    `registry-teach batch-sft --store` CLIs already make
    (TeacherBase.batch_sft + TeacherBase.submit_to_store), just reachable
    from the web UI's Teach tab instead of only a terminal."""
    from src.teacher_anthropic import AnthropicTeacher
    from src.teacher_registry import RegistryTeacher

    topic = str(payload.get("topic", "")).strip()
    if not topic:
        raise ValueError("generate_teacher_data task requires topic")
    count = max(1, min(50, int(payload.get("count", 8))))
    provider = str(payload.get("provider", "anthropic")).strip() or "anthropic"

    if provider == "anthropic":
        teacher = AnthropicTeacher()
    elif provider.startswith("registry:"):
        provider_key = provider.split(":", 1)[1].strip()
        if not provider_key:
            raise ValueError("Missing registered provider key (expected 'registry:<key>')")
        teacher = RegistryTeacher(provider_key)
    else:
        raise ValueError(f"Unknown teacher provider '{provider}'")

    connection = teacher.check_connection()
    if not connection.get("ok"):
        raise RuntimeError(connection.get("error") or f"Teacher provider '{provider}' is not configured")

    progress(f"Asking {teacher.teacher_name} for {count} training examples about '{topic}'")
    pairs = teacher.batch_sft(topic, count=count)
    result = teacher.submit_to_store(pairs, topic=topic)
    progress(f"Stored {result['turns_submitted']} teaching turns from {teacher.teacher_name}")
    return result


def run_codistill_round_task(payload: dict[str, Any], progress: ProgressCb) -> dict[str, Any]:
    """Run one cross-architecture co-distillation round (src/federated/codistill.py):
    ask trust-ranked peers to answer the shared probe set, keep only answers
    that agree with the group, and write survivors as an ordinary distillation
    corpus. Reachable from the web UI's Network tab instead of only the
    `python -m src.federated.codistill round` CLI. Bootstraps from the same
    data/torickle/known_peers.json list the control server's own swarm node
    uses, so "add a peer" in the Network tab already covers who this reaches."""
    from src.federated.codistill import (
        DEFAULT_OUT_CORPUS_PATH,
        DEFAULT_PROBE_SET_PATH,
        DEFAULT_TRUST_STORE_PATH,
        PeerTrustStore,
        load_probes,
        run_codistillation_round,
        select_probes_for_round,
        write_distillation_corpus,
    )
    from src.federated.contribution_ledger import LedgerStore
    from src.federated.inference_swarm import DEFAULT_IDENTITY_PATH, _join_via_bootstrap, _parse_addr
    from src.federated.keys import ensure_ed_identity
    from src.federated.peer_discovery import PeerDiscovery

    known_peers_path = Path("data/torickle/known_peers.json")
    bootstrap: list[str] = []
    if known_peers_path.exists():
        try:
            data = json.loads(known_peers_path.read_text(encoding="utf-8"))
            bootstrap = [str(a).strip() for a in data.get("peers", []) if str(a).strip()]
        except (json.JSONDecodeError, OSError):
            bootstrap = []

    identity = ensure_ed_identity(Path(DEFAULT_IDENTITY_PATH))
    ledger = LedgerStore()
    trust_store = PeerTrustStore(DEFAULT_TRUST_STORE_PATH)
    all_probes = load_probes(DEFAULT_PROBE_SET_PATH)
    # Default to a rotating subset rather than the full fixed set every time
    # (see select_probes_for_round docstring): the same date-seeded subset
    # everywhere, so independent peers running today still agree on which
    # probes are "live" without any coordinator.
    sample_size = int(payload.get("sample_size", 14) or 0)
    probes = select_probes_for_round(all_probes, sample_size)

    progress(f"Contacting known peers to teach {len(probes)}/{len(all_probes)} probe(s)...")
    peer_discovery = PeerDiscovery()
    for addr in bootstrap:
        host, port = _parse_addr(addr)
        peer_discovery.add_bootstrap(host, port)
    _join_via_bootstrap(peer_discovery, bootstrap)

    report = run_codistillation_round(
        identity=identity,
        peer_discovery=peer_discovery,
        ledger=ledger,
        trust_store=trust_store,
        probes=probes,
    )
    write_distillation_corpus(DEFAULT_OUT_CORPUS_PATH, report["rows"])
    report["out_corpus"] = DEFAULT_OUT_CORPUS_PATH
    report["ran_at_utc"] = datetime.now(timezone.utc).isoformat()
    progress(f"Co-distillation round: taught on {report['probes_taught']}/{report['probes_total']} probe(s)")

    from src.disagreement_curriculum import record_conflicts

    conflict_total = 0
    for probe_report in report.get("probe_reports", []):
        conflicts = (probe_report.get("deliberation") or {}).get("possible_conflicts") or []
        if conflicts:
            record_conflicts(conflicts, source="codistill_round")
            conflict_total += len(conflicts)
    if conflict_total:
        progress(f"Recorded {conflict_total} peer disagreement(s) into the open-questions queue")

    last_report_path = Path("data/codistill/last_report.json")
    last_report_path.parent.mkdir(parents=True, exist_ok=True)
    last_report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def run_build_dpo_prefs_task(payload: dict[str, Any], progress: ProgressCb) -> dict[str, Any]:
    """Build DPO preference pairs from the chat feedback the app already
    collects (thumbs up/down on responses, src/feedback_store.py). Needs at
    least two differently-rated responses to the same prompt to form a pair
    -- a single rating isn't enough (src/build_preference_pairs.py)."""
    from src.build_preference_pairs import build_preference_pairs
    from src.feedback_store import DEFAULT_FEEDBACK_PATH

    feedback_path = str(payload.get("feedback_path", DEFAULT_FEEDBACK_PATH)).strip()
    out_path = str(payload.get("out_path", "data/dpo_prefs.jsonl")).strip()
    if not Path(feedback_path).exists():
        raise ValueError(
            "No feedback recorded yet -- rate a few chat responses (thumbs up/down) first."
        )
    progress(f"Building preference pairs from {feedback_path} -> {out_path}")
    result = build_preference_pairs(feedback_path, out_path)
    progress(f"{result['pairs_written']} preference pair(s) written from {result['prompts_used']} prompt(s)")
    return result


def run_dpo_train_task(payload: dict[str, Any], progress: ProgressCb) -> dict[str, Any]:
    """Align a model to the user's own feedback via DPO, using whatever
    preference pairs run_build_dpo_prefs_task already wrote. Calls run_dpo()
    directly (it's a plain importable function, not CLI-only) rather than
    spawning a subprocess like run_train_model_task does -- DPO runs are
    short enough not to need a separate live-status file."""
    from src.dpo_train import run_dpo

    prefs_path = str(payload.get("prefs_path", "data/dpo_prefs.jsonl")).strip()
    if not Path(prefs_path).exists():
        raise ValueError("No preference pairs found -- build them first (build_dpo_preferences).")

    model_path = str(payload.get("model", "")).strip()
    if not model_path:
        raise ValueError("dpo_train task requires model")
    out_path = str(payload.get("out_model", "models/candidates/ickle_dpo_aligned.pt")).strip()
    steps = max(1, int(payload.get("steps", 300)))

    progress(f"Aligning {model_path} to your feedback ({steps} steps) -> {out_path}")
    report = run_dpo(
        model_path=model_path,
        preference_data_path=prefs_path,
        out_path=out_path,
        steps=steps,
    )
    progress(f"DPO alignment complete: {report.get('sample_count', 0)} preference sample(s) used")
    return {**report, "out_model": out_path}


def run_train_from_teacher_task(
    payload: dict[str, Any],
    progress: ProgressCb,
    *,
    chat_runner: ChatRunner | None = None,
) -> dict[str, Any]:
    from src.teacher_ingest import TeacherStore
    store = TeacherStore()
    min_score = float(payload.get("min_score", 0.3))

    sft_path = str(payload.get("sft_out", "data/teacher/teacher_sft_corpus.txt")).strip()
    prefs_path = str(payload.get("prefs_out", "data/teacher/teacher_prefs.jsonl")).strip()

    progress(f"Building teacher corpora (min_score={min_score})")
    sft_result = store.build_sft_corpus(out_path=sft_path, min_score=min_score)
    prefs_result = store.build_preference_pairs(out_path=prefs_path, min_score=min_score)
    progress(f"Teacher SFT: {sft_result['pairs']} pairs, DPO: {prefs_result['pairs']} pairs")

    if sft_result["pairs"] == 0 and prefs_result["pairs"] == 0:
        raise ValueError("No teacher training pairs available. Ensure teaching turns have improved_answer with score >= min_score.")

    steps = int(payload.get("steps", 800))
    resource_budget = payload.get("resource_budget", {"cpu_percent": 80, "ram_percent": 80, "gpu_percent": 80})
    lr = float(payload.get("lr", 8e-6))
    warmup_steps = int(payload.get("warmup_steps", 80))

    baseline_model = str(payload.get("baseline_model", "models/ickle_clean.pt"))
    out_model = str(payload.get("out_model", "models/ickle_teacher_candidate.pt"))
    checkpoint_path = str(payload.get("checkpoint_path", f"{out_model}.checkpoint.pt"))

    progress(f"Training from teacher data ({steps} steps)")
    train_result = run_train_model_task({
        "data_path": sft_path,
        "out_model": out_model,
        "steps": steps,
        "resource_budget": resource_budget,
        "lr": lr,
        "warmup_steps": warmup_steps,
        "init_model": baseline_model,
        "resume_if_possible": True,
        "checkpoint_path": checkpoint_path,
        "bootstrap_english": True,
    }, progress)

    progress("Evaluating teacher-trained model")
    eval_result = run_evaluate_model_task(
        {
            "candidate_model": out_model,
            "baseline_model": baseline_model,
            "min_delta": float(payload.get("min_delta", 0.0)),
            "min_candidate_avg": float(payload.get("min_candidate_avg", 0.35)),
            "promote_if_pass": bool(payload.get("promote_if_pass", True)),
            "promote_to_model": str(payload.get("promote_to_model") or baseline_model),
            "quiz_size": int(payload.get("quiz_size", 8)),
            "extended_eval": bool(payload.get("extended_eval", True)),
        },
        progress,
        chat_runner=chat_runner,
    )

    return {
        "sft_corpus": sft_result,
        "dpo_prefs": prefs_result,
        "training": train_result,
        "evaluation": eval_result,
    }


def infer_task_from_instruction(instruction: str) -> dict[str, Any] | None:
    text = instruction.strip()
    if not text:
        return None
    lower = text.lower()
    wants_training = bool(re.search(r"\b(train|retrain|self[- ]?improv|self[- ]?train)\b", text, flags=re.IGNORECASE))
    wants_web = _mentions_web_sources(text)
    wants_wikipedia = _mentions_wikipedia(text)

    m = re.search(
        r"(?:revisit|review|recall|search)\s+(?:research|notes)\s+(?:about|on)\s+(.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        topic = m.group(1).strip(" .?!")
        if topic:
            return {
                "task_type": "research_notes_query",
                "payload": {"query": topic, "limit": 8},
            }

    m = re.search(
        r"(?:learn|study|research)\s+(?:as much as you can(?:\s+about)?\s+|about\s+)?(.+?)\s+"
        r"(?:from|on|across|using)\s+(?:the\s+)?(?:internet|web|online)\b",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        topic = _normalize_inferred_topic(m.group(1))
        if topic:
            return {
                "task_type": "learn_web_topic",
                "payload": {
                    "topic": topic,
                    "max_urls": 8,
                    "include_wikipedia": True,
                    "include_news": True,
                    "auto_pipeline": wants_training,
                    "unrestricted": True,
                },
            }

    m = re.search(
        r"(?:learn|study|research)\s+(?:from|on|across)\s+(?:the\s+)?(?:internet|web)\s+(?:about\s+)?(.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        topic = _normalize_inferred_topic(m.group(1))
        if topic:
            return {
                "task_type": "learn_web_topic",
                "payload": {
                    "topic": topic,
                    "max_urls": 8,
                    "include_wikipedia": True,
                    "include_news": True,
                    "auto_pipeline": wants_training,
                    "unrestricted": True,
                },
            }

    m = re.search(
        r"(?:learn|study|research)\s+(?:as much as you can about\s+|about\s+)?(.+?)(?:\s+from\s+wikipedia)?$",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        topic = _normalize_inferred_topic(m.group(1))
        if topic:
            if wants_web and not wants_wikipedia:
                return {
                    "task_type": "learn_web_topic",
                    "payload": {
                        "topic": topic,
                        "max_urls": 8,
                        "include_wikipedia": True,
                        "include_news": True,
                        "auto_pipeline": wants_training,
                        "unrestricted": True,
                    },
                }
            if wants_web and wants_wikipedia:
                return {
                    "task_type": "learn_web_topic",
                    "payload": {
                        "topic": topic,
                        "max_urls": 8,
                        "include_wikipedia": True,
                        "include_news": False,
                        "auto_pipeline": wants_training,
                        "unrestricted": True,
                    },
                }
            return {
                "task_type": "learn_wikipedia_topic",
                "payload": {"topic": topic, "max_pages": 8, "auto_pipeline": wants_training},
            }

    if "build clean corpus" in lower:
        return {"task_type": "build_clean_corpus", "payload": {}}

    if "train model" in lower or "retrain" in lower:
        return {
            "task_type": "train_model",
            "payload": {
                "data_path": "data/ickle_clean_corpus.txt",
                "out_model": "models/ickle_clean.pt",
                "steps": 1200,
                "bootstrap_english": True,
            },
        }

    return None
