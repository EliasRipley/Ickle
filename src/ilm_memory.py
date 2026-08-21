#!/usr/bin/env python3
"""Persistent memory for Ickle with basic hygiene against noisy entries."""

from __future__ import annotations

import copy
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MAX_FACTS = 5000
MAX_VISITED_URLS = 2000
MAX_EXTRACTED_FACTS = 5000
MAX_TOPIC_FACTS = 200
MAX_TOPIC_URLS = 100


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "does",
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
    "major",
    "between",
    "that",
    "the",
    "their",
    "them",
    "this",
    "to",
    "was",
    "were",
    "we",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "was",
    "were",
    "using",
    "used",
    "include",
    "includes",
    "including",
    "involved",
    "result",
    "results",
    "you",
    "your",
}

NOISE_MARKERS = (
    "wikipedia does not have an article",
    "look for ",
    "article wizard",
    "sister projects",
    "other reasons this message may be displayed",
    "from wikipedia, the free encyclopedia",
    "wiktionary(",
    "wikisource(",
    "wikidata(",
    "languagesafrikaans",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _owner_defaults() -> dict[str, Any]:
    now = _utc_now()
    return {
        "name": None,
        "creator": None,
        "preferences": {},
        "first_interaction": now,
        "last_interaction": now,
        "interaction_count": 0,
    }


def _facts_defaults() -> dict[str, Any]:
    return {
        "facts": [],
        "categories": {},
        "last_updated": _utc_now(),
    }


def _conversations_defaults() -> dict[str, Any]:
    return {
        "sessions": [],
        "total_conversations": 0,
        "last_conversation": None,
    }


def _web_defaults() -> dict[str, Any]:
    return {
        "visited_urls": [],
        "learned_topics": {},
        "extracted_facts": [],
        "last_learning": None,
    }


def _research_defaults() -> dict[str, Any]:
    return {
        "sessions": {},
        "notes": [],
        "last_updated": _utc_now(),
    }


def _deep_defaults(value: Any, defaults: Any) -> Any:
    if isinstance(defaults, dict):
        source = value if isinstance(value, dict) else {}
        result = {k: _deep_defaults(source.get(k), d) for k, d in defaults.items()}
        for k, v in source.items():
            if k not in result:
                result[k] = v
        return result
    if isinstance(defaults, list):
        return value if isinstance(value, list) else copy.deepcopy(defaults)
    if value is None:
        return copy.deepcopy(defaults)
    return value


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _fold_text(text: str) -> str:
    value = unicodedata.normalize("NFKD", str(text or ""))
    return "".join(ch for ch in value if not unicodedata.combining(ch))


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", _fold_text(text).lower()))


def _content_tokens(text: str) -> set[str]:
    raw = _tokenize(text)
    out: set[str] = set()
    for token in raw:
        t = token
        if t.endswith("'s") and len(t) > 4:
            t = t[:-2]
        if t.endswith("s") and len(t) > 5:
            t = t[:-1]
        if len(t) < 3 or t in STOPWORDS:
            continue
        out.add(t)
    return out


def _looks_noisy_text(text: str) -> bool:
    value = _normalize_ws(str(text or ""))
    if len(value) < 4:
        return True
    lower = value.lower()
    if any(marker in lower for marker in NOISE_MARKERS):
        return True
    if re.search(r"\b\d+\s+languages\b", lower):
        return True
    if re.search(r"\b(begininput|endinput|system prompt)\b", lower):
        return True
    if re.search(r"\b(\w+)(?:\s+\1){2,}\b", lower):
        return True
    return False


def _dedupe_rows_by_key(rows: list[dict[str, Any]], key_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for row in rows:
        key = tuple(_normalize_ws(str(row.get(field, ""))).lower() for field in key_fields)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


class ILMMemory:
    """Persistent memory system for Ickle ILM."""

    def __init__(self, memory_dir: str = "data/memory"):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.owner_file = self.memory_dir / "owner.json"
        self.facts_file = self.memory_dir / "facts.json"
        self.conversations_file = self.memory_dir / "conversations.json"
        self.web_learning_file = self.memory_dir / "web_learning.json"
        self.research_file = self.memory_dir / "research_notes.json"
        self._initialize_memory()

    def _default_for_file(self, file_path: Path) -> dict[str, Any]:
        if file_path == self.owner_file:
            return _owner_defaults()
        if file_path == self.facts_file:
            return _facts_defaults()
        if file_path == self.conversations_file:
            return _conversations_defaults()
        if file_path == self.web_learning_file:
            return _web_defaults()
        if file_path == self.research_file:
            return _research_defaults()
        return {}

    def _initialize_memory(self):
        for file_path in [
            self.owner_file,
            self.facts_file,
            self.conversations_file,
            self.web_learning_file,
            self.research_file,
        ]:
            if not file_path.exists():
                self._save_json(file_path, self._default_for_file(file_path))

    def _save_json(self, file_path: Path, data: dict[str, Any]):
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load_json(self, file_path: Path) -> dict[str, Any]:
        defaults = self._default_for_file(file_path)
        if not file_path.exists():
            return copy.deepcopy(defaults)
        try:
            with file_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError):
            return copy.deepcopy(defaults)
        return _deep_defaults(raw, defaults)

    def _rebuild_categories(self, facts: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        categories: dict[str, list[dict[str, Any]]] = {}
        for fact in facts:
            category = fact.get("category", "general")
            categories.setdefault(category, []).append(fact)
        return categories

    def set_owner_info(
        self,
        name: str | None = None,
        creator: str | None = None,
        preferences: dict[str, Any] | None = None,
    ):
        owner_data = self._load_json(self.owner_file)

        if name:
            owner_data["name"] = _normalize_ws(name)[:80]
        if creator:
            owner_data["creator"] = _normalize_ws(creator)[:120]
        if preferences:
            for k, v in preferences.items():
                owner_data["preferences"][str(k)] = v

        owner_data["last_interaction"] = _utc_now()
        owner_data["interaction_count"] = int(owner_data.get("interaction_count", 0)) + 1
        self._save_json(self.owner_file, owner_data)

    def get_owner_info(self) -> dict[str, Any]:
        return self._load_json(self.owner_file)

    def add_fact(
        self,
        fact: str,
        category: str = "general",
        source: str | None = None,
        confidence: float = 1.0,
    ):
        cleaned_fact = _normalize_ws(fact)
        if len(cleaned_fact) < 4:
            return

        cleaned_category = _normalize_ws(category) if category else "general"
        cleaned_source = _normalize_ws(source) if source else None

        facts_data = self._load_json(self.facts_file)
        facts = facts_data["facts"]

        normalized = cleaned_fact.lower()
        for existing in facts:
            existing_fact = str(existing.get("fact", "")).strip().lower()
            if (
                existing_fact == normalized
                and existing.get("category", "general") == cleaned_category
                and existing.get("source") == cleaned_source
            ):
                return

        entry = {
            "fact": cleaned_fact[:800],
            "category": cleaned_category[:80],
            "source": cleaned_source[:400] if cleaned_source else None,
            "confidence": max(0.0, min(1.0, float(confidence))),
            "learned_at": _utc_now(),
            "retrieved_count": 0,
        }
        facts.append(entry)
        if len(facts) > MAX_FACTS:
            facts.sort(key=lambda f: f.get("learned_at", ""), reverse=True)
            facts[:] = facts[:MAX_FACTS]
        facts_data["categories"] = self._rebuild_categories(facts)
        facts_data["last_updated"] = _utc_now()
        self._save_json(self.facts_file, facts_data)

    def get_facts(self, category: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        facts_data = self._load_json(self.facts_file)
        facts = facts_data["facts"]
        if category:
            filtered = [f for f in facts if f.get("category") == category]
            return filtered[:limit]
        sorted_facts = sorted(facts, key=lambda x: x.get("learned_at", ""), reverse=True)
        return sorted_facts[:limit]

    def search_facts(
        self,
        query: str,
        limit: int = 10,
        topic_hint: str | None = None,
    ) -> list[dict[str, Any]]:
        facts_data = self._load_json(self.facts_file)
        facts = facts_data["facts"]

        cleaned_query = _normalize_ws(query).lower()
        if not cleaned_query:
            return []

        query_tokens = _content_tokens(cleaned_query)
        if not query_tokens:
            return []
        required_overlap = 1 if len(query_tokens) <= 3 else 2
        hint_tokens = _content_tokens(_normalize_ws(topic_hint or "").lower()) if topic_hint else set()
        if hint_tokens:
            required_overlap = 1
        scored: list[tuple[int, int, dict[str, Any]]] = []
        for idx, fact in enumerate(facts):
            fact_text = str(fact.get("fact", "")).strip()
            if not fact_text:
                continue
            fact_lower = fact_text.lower()
            fact_tokens = _content_tokens(fact_text)
            overlap = len(query_tokens.intersection(fact_tokens))
            contains = 1 if cleaned_query in fact_lower else 0
            if overlap < required_overlap and contains == 0:
                continue
            hint_bonus = 0
            if hint_tokens:
                category = str(fact.get("category", ""))
                source = str(fact.get("source", ""))
                hint_space = _content_tokens(f"{category} {source}")
                hint_overlap = len(hint_tokens.intersection(hint_space.union(fact_tokens)))
                if hint_overlap == 0 and overlap <= required_overlap:
                    continue
                hint_bonus = hint_overlap * 5
            confidence = int(float(fact.get("confidence", 0.0)) * 10)
            retrieved_penalty = min(6, int(fact.get("retrieved_count", 0)) // 20)
            score = overlap * 12 + contains * 8 + confidence + hint_bonus - retrieved_penalty
            scored.append((score, idx, fact))

        scored.sort(key=lambda item: item[0], reverse=True)
        selected = [item[2] for item in scored[: max(1, limit)]]
        if not selected:
            return []

        selected_ids = {id(item) for item in selected}
        changed = False
        for fact in facts:
            if id(fact) in selected_ids:
                fact["retrieved_count"] = int(fact.get("retrieved_count", 0)) + 1
                changed = True

        if changed:
            facts_data["last_updated"] = _utc_now()
            facts_data["categories"] = self._rebuild_categories(facts)
            self._save_json(self.facts_file, facts_data)

        return selected

    def search_web_facts(
        self,
        query: str,
        limit: int = 8,
        topic_hint: str | None = None,
    ) -> list[dict[str, Any]]:
        web_data = self._load_json(self.web_learning_file)
        extracted = web_data.get("extracted_facts", [])
        if not isinstance(extracted, list):
            return []

        cleaned_query = _normalize_ws(query).lower()
        query_tokens = _content_tokens(cleaned_query)
        if not query_tokens:
            return []
        required_overlap = 1 if len(query_tokens) <= 3 else 2

        hint = _normalize_ws(topic_hint or "").lower()
        hint_tokens = _content_tokens(hint) if hint else set()
        if hint_tokens:
            required_overlap = 1
        scored: list[tuple[int, int, dict[str, Any]]] = []
        total = max(1, len(extracted))
        for idx, row in enumerate(extracted):
            if not isinstance(row, dict):
                continue
            fact = _normalize_ws(str(row.get("fact", "")))
            if not fact:
                continue
            if _looks_noisy_text(fact):
                continue
            fact_lower = fact.lower()
            source_title = _normalize_ws(str(row.get("source_title", "")))
            topic = _normalize_ws(str(row.get("topic", "")))
            combined = " ".join([fact, source_title, topic]).strip()
            combined_tokens = _content_tokens(combined)
            overlap = len(query_tokens.intersection(combined_tokens))
            contains = 1 if cleaned_query in fact_lower else 0
            if overlap < required_overlap and contains == 0:
                continue

            hint_bonus = 0
            if hint and hint == topic.lower():
                hint_bonus = 8
            elif hint and hint in topic.lower():
                hint_bonus = 4
            if hint_tokens:
                hint_overlap = len(hint_tokens.intersection(combined_tokens))
                if hint_overlap == 0 and overlap <= required_overlap:
                    continue
                hint_bonus += hint_overlap * 4

            recency_bonus = int(((idx + 1) / total) * 4)
            confidence_bonus = int(float(row.get("confidence", row.get("score", 0.0))) * 6)
            corroboration_bonus = min(6, int(row.get("corroboration_count", 0)) * 2)
            score = overlap * 12 + contains * 8 + hint_bonus + recency_bonus + confidence_bonus + corroboration_bonus
            scored.append((score, idx, row))

        scored.sort(key=lambda item: item[0], reverse=True)
        selected: list[dict[str, Any]] = []
        seen_facts: set[str] = set()
        for _, _, row in scored:
            fact_key = _normalize_ws(str(row.get("fact", ""))).lower()
            if not fact_key or fact_key in seen_facts:
                continue
            seen_facts.add(fact_key)
            selected.append(row)
            if len(selected) >= max(1, limit):
                break
        return selected

    def add_web_learning(
        self,
        url: str,
        title: str,
        key_facts: list[str],
        topic: str | None = None,
        fact_metadata: dict[str, dict[str, Any]] | None = None,
    ):
        web_data = self._load_json(self.web_learning_file)
        url_clean = _normalize_ws(url)
        title_clean = _normalize_ws(title)[:300]
        topic_clean = _normalize_ws(topic)[:120] if topic else None

        visited_urls = web_data["visited_urls"]
        if not any(str(entry.get("url")) == url_clean for entry in visited_urls):
            visited_urls.append(
                {
                    "url": url_clean,
                    "title": title_clean,
                    "visited_at": _utc_now(),
                    "topic": topic_clean,
                }
            )
            if len(visited_urls) > MAX_VISITED_URLS:
                visited_urls.sort(key=lambda e: e.get("visited_at", ""), reverse=True)
                visited_urls[:] = visited_urls[:MAX_VISITED_URLS]

        existing_pairs = {
            (str(item.get("fact", "")).lower(), str(item.get("source_url", "")).lower())
            for item in web_data["extracted_facts"]
        }
        accepted_facts: list[str] = []
        for fact in key_facts:
            cleaned = _normalize_ws(fact)
            if len(cleaned) < 4:
                continue
            if _looks_noisy_text(cleaned):
                continue
            pair = (cleaned.lower(), url_clean.lower())
            if pair in existing_pairs:
                continue
            meta = {}
            if fact_metadata:
                meta = fact_metadata.get(cleaned) or fact_metadata.get(cleaned.lower()) or {}
            confidence = max(0.0, min(1.0, float(meta.get("confidence", 0.8))))
            score = max(0.0, min(1.0, float(meta.get("score", confidence))))
            corroboration_count = max(0, int(meta.get("corroboration_count", 0)))
            web_data["extracted_facts"].append(
                {
                    "fact": cleaned[:500],
                    "source_url": url_clean,
                    "source_title": title_clean,
                    "topic": topic_clean,
                    "learned_at": _utc_now(),
                    "confidence": confidence,
                    "score": score,
                    "corroboration_count": corroboration_count,
                }
            )
            existing_pairs.add(pair)
            accepted_facts.append(cleaned)

            category = f"web_{topic_clean}" if topic_clean else "web_general"
            self.add_fact(fact=cleaned, category=category, source=url_clean, confidence=confidence)

        if len(web_data["extracted_facts"]) > MAX_EXTRACTED_FACTS:
            web_data["extracted_facts"].sort(key=lambda f: f.get("learned_at", ""), reverse=True)
            web_data["extracted_facts"][:] = web_data["extracted_facts"][:MAX_EXTRACTED_FACTS]

        if topic_clean:
            topic_data = web_data["learned_topics"].setdefault(topic_clean, {"facts": [], "urls": []})
            for cleaned in accepted_facts:
                if cleaned not in topic_data["facts"]:
                    topic_data["facts"].append(cleaned)
            if url_clean not in topic_data["urls"]:
                topic_data["urls"].append(url_clean)
            topic_data["facts"] = topic_data["facts"][-MAX_TOPIC_FACTS:]
            topic_data["urls"] = topic_data["urls"][-MAX_TOPIC_URLS:]

        web_data["last_learning"] = _utc_now()
        self._save_json(self.web_learning_file, web_data)

    def get_web_learning(self, topic: str | None = None) -> dict[str, Any]:
        web_data = self._load_json(self.web_learning_file)
        if topic:
            return web_data["learned_topics"].get(topic, {})
        return web_data

    def add_research_note(
        self,
        *,
        topic: str,
        question: str,
        finding: str,
        source_url: str = "",
        source_title: str = "",
        tags: list[str] | None = None,
        confidence: float = 0.7,
        session_id: str = "",
    ) -> str:
        clean_topic = _normalize_ws(topic)[:120] or "general"
        clean_question = _normalize_ws(question)[:260] or "research"
        clean_finding = _normalize_ws(finding)[:900]
        if len(clean_finding) < 8:
            return session_id or ""
        clean_source_url = _normalize_ws(source_url)[:500]
        clean_source_title = _normalize_ws(source_title)[:260]
        clean_tags = [t for t in [_normalize_ws(str(x)).lower()[:40] for x in (tags or [])] if t]
        sid = _normalize_ws(session_id) or f"s_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

        research = self._load_json(self.research_file)
        notes = research.get("notes", [])
        if not isinstance(notes, list):
            notes = []
        dedupe_key = (
            clean_topic.lower(),
            clean_question.lower(),
            clean_finding.lower(),
            clean_source_url.lower(),
        )
        for row in notes:
            key = (
                str(row.get("topic", "")).lower(),
                str(row.get("question", "")).lower(),
                str(row.get("finding", "")).lower(),
                str(row.get("source_url", "")).lower(),
            )
            if key == dedupe_key:
                return sid

        note = {
            "session_id": sid,
            "topic": clean_topic,
            "question": clean_question,
            "finding": clean_finding,
            "source_url": clean_source_url or None,
            "source_title": clean_source_title or None,
            "tags": clean_tags,
            "confidence": max(0.0, min(1.0, float(confidence))),
            "created_at_utc": _utc_now(),
        }
        notes.append(note)
        notes = notes[-8000:]

        sessions = research.get("sessions", {})
        if not isinstance(sessions, dict):
            sessions = {}
        session = sessions.get(sid, {})
        if not isinstance(session, dict):
            session = {}
        if not session.get("created_at_utc"):
            session["created_at_utc"] = _utc_now()
        session["updated_at_utc"] = _utc_now()
        session["topic"] = clean_topic
        session["note_count"] = int(session.get("note_count", 0)) + 1
        sessions[sid] = session

        research["notes"] = notes
        research["sessions"] = sessions
        research["last_updated"] = _utc_now()
        self._save_json(self.research_file, research)
        return sid

    def search_research_notes(
        self,
        query: str,
        *,
        limit: int = 6,
        topic_hint: str | None = None,
    ) -> list[dict[str, Any]]:
        research = self._load_json(self.research_file)
        notes = research.get("notes", [])
        if not isinstance(notes, list):
            return []
        cleaned_query = _normalize_ws(query).lower()
        if not cleaned_query:
            return []
        query_tokens = _content_tokens(cleaned_query)
        if not query_tokens:
            return []
        required_overlap = 1 if len(query_tokens) <= 3 else 2
        hint = _normalize_ws(topic_hint or "").lower()
        hint_tokens = _content_tokens(hint) if hint else set()
        if hint_tokens:
            required_overlap = 1
        scored: list[tuple[int, int, dict[str, Any]]] = []
        total = max(1, len(notes))
        for idx, row in enumerate(notes):
            if not isinstance(row, dict):
                continue
            finding = _normalize_ws(str(row.get("finding", "")))
            if not finding:
                continue
            topic = _normalize_ws(str(row.get("topic", "")))
            question = _normalize_ws(str(row.get("question", "")))
            combined = " ".join([finding, topic, question]).strip()
            combined_tokens = _content_tokens(combined)
            overlap = len(query_tokens.intersection(combined_tokens))
            contains = 1 if cleaned_query in finding.lower() else 0
            if overlap < required_overlap and contains == 0:
                continue
            hint_bonus = 0
            if hint and hint == topic.lower():
                hint_bonus = 8
            elif hint and hint in topic.lower():
                hint_bonus = 4
            if hint_tokens:
                hint_overlap = len(hint_tokens.intersection(combined_tokens))
                if hint_overlap == 0 and overlap <= required_overlap:
                    continue
                hint_bonus += hint_overlap * 4
            confidence_bonus = int(float(row.get("confidence", 0.0)) * 6)
            recency_bonus = int(((idx + 1) / total) * 4)
            score = overlap * 12 + contains * 8 + hint_bonus + confidence_bonus + recency_bonus
            scored.append((score, idx, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        selected: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for _, _, row in scored:
            key = (
                str(row.get("finding", "")).strip().lower(),
                str(row.get("source_url", "")).strip().lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            selected.append(row)
            if len(selected) >= max(1, limit):
                break
        return selected

    def list_research_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        research = self._load_json(self.research_file)
        sessions = research.get("sessions", {})
        if not isinstance(sessions, dict):
            return []
        out: list[dict[str, Any]] = []
        for sid, row in sessions.items():
            if isinstance(row, dict):
                out.append({"session_id": sid, **row})
        out.sort(key=lambda item: str(item.get("updated_at_utc", "")), reverse=True)
        return out[: max(1, limit)]

    def remember_conversation(
        self,
        user_input: str,
        ickle_response: str,
        context: dict[str, Any] | None = None,
    ):
        clean_user = _normalize_ws(user_input)[:1200]
        clean_response = _normalize_ws(ickle_response)[:1800]
        if not clean_user or not clean_response:
            return

        conv_data = self._load_json(self.conversations_file)
        entry = {
            "timestamp": _utc_now(),
            "user_input": clean_user,
            "ickle_response": clean_response,
            "context": context or {},
        }

        sessions = conv_data["sessions"]
        if not sessions or sessions[-1].get("completed", False):
            sessions.append(
                {
                    "start_time": _utc_now(),
                    "conversations": [entry],
                    "completed": False,
                }
            )
        else:
            sessions[-1]["conversations"].append(entry)
            sessions[-1]["conversations"] = sessions[-1]["conversations"][-200:]

        conv_data["sessions"] = sessions[-30:]
        conv_data["last_conversation"] = _utc_now()
        conv_data["total_conversations"] = int(conv_data.get("total_conversations", 0)) + 1
        self._save_json(self.conversations_file, conv_data)

    def get_recent_context(self, limit: int = 5) -> list[dict[str, Any]]:
        conv_data = self._load_json(self.conversations_file)
        sessions = conv_data["sessions"]
        if not sessions:
            return []

        items: list[dict[str, Any]] = []
        for session in reversed(sessions):
            conversations = session.get("conversations", [])
            for entry in reversed(conversations):
                items.append(entry)
                if len(items) >= max(1, limit):
                    return list(reversed(items))
        return list(reversed(items))

    def get_memory_summary(self) -> dict[str, Any]:
        owner = self.get_owner_info()
        facts = self._load_json(self.facts_file)
        web = self.get_web_learning()
        conv = self._load_json(self.conversations_file)
        sessions = conv["sessions"]
        current_active = bool(sessions and not sessions[-1].get("completed", False))
        return {
            "owner": owner,
            "fact_count": len(facts["facts"]),
            "categories": sorted(facts["categories"].keys()),
            "web_urls_visited": len(web["visited_urls"]),
            "web_extracted_fact_count": len(web.get("extracted_facts", [])),
            "learned_topics": sorted(web["learned_topics"].keys()),
            "research_note_count": len(self._load_json(self.research_file).get("notes", [])),
            "research_session_count": len(self._load_json(self.research_file).get("sessions", {})),
            "total_conversations": conv["total_conversations"],
            "current_session_active": current_active,
        }

    def clear_memory(self, memory_type: str | None = None):
        targets = {
            "owner": self.owner_file,
            "facts": self.facts_file,
            "web": self.web_learning_file,
            "conversations": self.conversations_file,
            "research": self.research_file,
        }
        if memory_type is None:
            for file_path in targets.values():
                if file_path.exists():
                    file_path.unlink()
            self._initialize_memory()
            return

        file_path = targets.get(memory_type)
        if file_path and file_path.exists():
            file_path.unlink()
            self._save_json(file_path, self._default_for_file(file_path))

    def clear_short_term_memory(self):
        self.clear_memory("conversations")

    def prune_nonsense(
        self,
        *,
        clear_short_term: bool = True,
        min_fact_confidence: float = 0.45,
        min_research_confidence: float = 0.5,
    ) -> dict[str, Any]:
        stats: dict[str, Any] = {}
        if clear_short_term:
            conv_data = self._load_json(self.conversations_file)
            stats["conversation_sessions_before"] = len(conv_data.get("sessions", []))
            conv_data["sessions"] = []
            conv_data["total_conversations"] = 0
            conv_data["last_conversation"] = None
            self._save_json(self.conversations_file, conv_data)
            stats["conversation_sessions_after"] = 0

        facts_data = self._load_json(self.facts_file)
        before_facts = len(facts_data.get("facts", []))
        cleaned_facts: list[dict[str, Any]] = []
        for row in list(facts_data.get("facts", [])):
            fact = _normalize_ws(str(row.get("fact", "")))
            confidence = float(row.get("confidence", 0.0))
            if _looks_noisy_text(fact):
                continue
            if confidence < float(min_fact_confidence):
                continue
            row["fact"] = fact[:800]
            cleaned_facts.append(row)
        cleaned_facts = _dedupe_rows_by_key(cleaned_facts, ("fact", "category", "source"))
        facts_data["facts"] = cleaned_facts
        facts_data["categories"] = self._rebuild_categories(cleaned_facts)
        facts_data["last_updated"] = _utc_now()
        self._save_json(self.facts_file, facts_data)
        stats["facts_removed"] = max(0, before_facts - len(cleaned_facts))
        stats["facts_remaining"] = len(cleaned_facts)

        web_data = self._load_json(self.web_learning_file)
        before_web = len(web_data.get("extracted_facts", []))
        cleaned_web: list[dict[str, Any]] = []
        for row in list(web_data.get("extracted_facts", [])):
            fact = _normalize_ws(str(row.get("fact", "")))
            score = float(row.get("score", row.get("confidence", 0.0)))
            if _looks_noisy_text(fact):
                continue
            if score < float(min_fact_confidence):
                continue
            row["fact"] = fact[:500]
            cleaned_web.append(row)
        cleaned_web = _dedupe_rows_by_key(cleaned_web, ("fact", "source_url"))
        web_data["extracted_facts"] = cleaned_web
        web_data["last_learning"] = _utc_now()
        self._save_json(self.web_learning_file, web_data)
        stats["web_facts_removed"] = max(0, before_web - len(cleaned_web))
        stats["web_facts_remaining"] = len(cleaned_web)

        research_data = self._load_json(self.research_file)
        before_notes = len(research_data.get("notes", []))
        cleaned_notes: list[dict[str, Any]] = []
        for row in list(research_data.get("notes", [])):
            finding = _normalize_ws(str(row.get("finding", "")))
            confidence = float(row.get("confidence", 0.0))
            if _looks_noisy_text(finding):
                continue
            if confidence < float(min_research_confidence):
                continue
            row["finding"] = finding[:900]
            cleaned_notes.append(row)
        cleaned_notes = _dedupe_rows_by_key(cleaned_notes, ("finding", "source_url"))
        research_data["notes"] = cleaned_notes
        research_data["last_updated"] = _utc_now()
        self._save_json(self.research_file, research_data)
        stats["research_notes_removed"] = max(0, before_notes - len(cleaned_notes))
        stats["research_notes_remaining"] = len(cleaned_notes)
        return stats


_memory_instance: ILMMemory | None = None


def get_memory() -> ILMMemory:
    """Get singleton memory instance."""
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = ILMMemory()
    return _memory_instance
