"""Topic-related utilities for the web runtime."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.task_actions import _request_json

AUTONOMOUS_TOPIC_FALLBACKS = [
    "cell biology basics",
    "Roman Empire society",
    "plate tectonics and earthquakes",
    "probability theory fundamentals",
    "electromagnetism basics",
    "photosynthesis and plant respiration",
    "operating systems process scheduling",
    "world geography and climate zones",
]

TOPIC_TOKEN_STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "for",
    "from",
    "in",
    "of",
    "on",
    "the",
    "to",
    "with",
}

DEFAULT_SUBJECT_CATALOG: dict[str, Any] = {
    "version": "0.2",
    "domains": [
        {
            "name": "Natural science",
            "subjects": [
                {"name": "Biology", "topics": ["cell biology", "genetics", "ecology", "evolution"]},
                {"name": "Chemistry", "topics": ["chemical bonding", "thermodynamics", "reaction kinetics"]},
                {"name": "Physics", "topics": ["classical mechanics", "electromagnetism", "wave optics"]},
            ],
        },
        {
            "name": "Formal science",
            "subjects": [
                {"name": "Mathematics", "topics": ["algebra", "calculus", "linear algebra", "statistics"]},
                {"name": "Computer science", "topics": ["algorithms", "data structures", "distributed systems"]},
            ],
        },
        {
            "name": "Social science",
            "subjects": [
                {"name": "Economics", "topics": ["microeconomics", "macroeconomics", "econometrics"]},
                {"name": "Psychology", "topics": ["cognitive psychology", "developmental psychology"]},
            ],
        },
        {
            "name": "Humanities",
            "subjects": [
                {"name": "History", "topics": ["ancient history", "modern history", "history of science"]},
                {"name": "Philosophy", "topics": ["ethics", "epistemology", "logic and argumentation"]},
            ],
        },
    ],
    "subjects": [
        {"name": "Natural science: Biology", "domain": "Natural science", "topics": ["cell biology", "genetics", "ecology", "evolution"]},
        {"name": "Natural science: Chemistry", "domain": "Natural science", "topics": ["chemical bonding", "thermodynamics", "reaction kinetics"]},
        {"name": "Natural science: Physics", "domain": "Natural science", "topics": ["classical mechanics", "electromagnetism", "wave optics"]},
        {"name": "Formal science: Mathematics", "domain": "Formal science", "topics": ["algebra", "calculus", "linear algebra", "statistics"]},
        {"name": "Formal science: Computer science", "domain": "Formal science", "topics": ["algorithms", "data structures", "distributed systems"]},
        {"name": "Social science: Economics", "domain": "Social science", "topics": ["microeconomics", "macroeconomics", "econometrics"]},
        {"name": "Social science: Psychology", "domain": "Social science", "topics": ["cognitive psychology", "developmental psychology"]},
        {"name": "Humanities: History", "domain": "Humanities", "topics": ["ancient history", "modern history", "history of science"]},
        {"name": "Humanities: Philosophy", "domain": "Humanities", "topics": ["ethics", "epistemology", "logic and argumentation"]},
    ],
}


def _slugify_for_model_name(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(text or "").lower()).strip("_")
    if not slug:
        return "topic"
    return slug[:48].strip("_") or "topic"


def _default_auto_pipeline_model_path(task_id: str, topic: str) -> str:
    slug = _slugify_for_model_name(topic)
    short_id = _slugify_for_model_name(task_id)[:12] or "task"
    return f"models/ickle_auto_{slug}_{short_id}.pt"


def _default_topic_queue_path(training_root: str, topic: str) -> str:
    root = Path(training_root).resolve()
    queue_dir = root / "topic_queues"
    queue_dir.mkdir(parents=True, exist_ok=True)
    return str(queue_dir / f"{_slugify_for_model_name(topic)}.txt")


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = re.sub(r"\s+", " ", str(raw or "")).strip(" .,:;!?")
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _topic_tokens(text: str) -> set[str]:
    out: set[str] = set()
    tokens = re.findall(r"[a-z0-9']+", str(text or "").lower())
    for token in tokens:
        t = token
        if t.endswith("'s") and len(t) > 4:
            t = t[:-2]
        if t.endswith("s") and len(t) > 5:
            t = t[:-1]
        if t in TOPIC_TOKEN_STOPWORDS:
            continue
        if len(t) < 3 and t not in {"ai", "uk", "us"}:
            continue
        out.add(t)
    return out


def _topic_similarity(a: str, b: str) -> float:
    norm_a = re.sub(r"\s+", " ", str(a or "").strip().lower())
    norm_b = re.sub(r"\s+", " ", str(b or "").strip().lower())
    if not norm_a or not norm_b:
        return 0.0
    if norm_a == norm_b:
        return 1.0
    ta = _topic_tokens(norm_a)
    tb = _topic_tokens(norm_b)
    if not ta or not tb:
        return 0.0
    intersection = len(ta.intersection(tb))
    if intersection == 0:
        return 0.0
    union = len(ta.union(tb))
    jaccard = intersection / max(1, union)
    overlap = intersection / max(1, min(len(ta), len(tb)))
    if len(ta) >= 2 and len(tb) >= 2 and (norm_a in norm_b or norm_b in norm_a):
        return max(jaccard, overlap, 0.95)
    return max(jaccard, overlap * 0.85)


def _topic_is_covered(topic: str, known_topics: list[str], *, similarity_threshold: float = 0.56) -> bool:
    if not str(topic or "").strip():
        return False
    for known in known_topics:
        if _topic_similarity(topic, known) >= similarity_threshold:
            return True
    return False


def _topic_from_benchmark_row(row: dict[str, Any]) -> str:
    name = str(row.get("name", "")).strip().lower()
    keywords = [str(k).strip() for k in list(row.get("keywords") or []) if str(k).strip()]
    if name.startswith("memory_followup"):
        return "Japan Standard Time and UTC offset"
    if "constraint" in name:
        return "decision analysis and prioritization methods"
    if "uncertainty" in name:
        return "scientific uncertainty and evidence quality"
    if "probability" in name:
        return "probability theory and statistics"
    if "debug" in name:
        return "software debugging"
    if keywords:
        cleaned = [k for k in keywords if len(k) >= 3]
        if cleaned:
            return " ".join(cleaned[:3])
    prompt = str(row.get("prompt", "")).strip()
    prompt = re.sub(
        r"^(?:what|where|when|why|how|who|explain|tell me|give me|if)\s+",
        "",
        prompt,
        flags=re.IGNORECASE,
    )
    prompt = re.sub(r"[?.!]+$", "", prompt).strip()
    return prompt or "general knowledge"


def _extract_topics_from_queue_text(path: Path) -> list[str]:
    if not path.exists() or not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:  # noqa: BLE001
        return []
    patterns = [
        r"^User:\s+Tell me about (.+?)\.$",
        r"^User:\s+We are studying (.+?)\.",
        r"^User:\s+Give me a concise overview of (.+?) with verified points only\.$",
    ]
    out: list[str] = []
    for line in lines:
        text = str(line or "").strip()
        for pattern in patterns:
            m = re.match(pattern, text, flags=re.IGNORECASE)
            if not m:
                continue
            topic = re.sub(r"\s+", " ", str(m.group(1) or "").strip(" .,:;!?"))
            topic = re.sub(r"\s+from the internet$", "", topic, flags=re.IGNORECASE).strip(" .,:;!?")
            if topic:
                out.append(topic)
    return _dedupe_preserve_order(out)


def _collect_known_training_topics(training_root: str) -> list[str]:
    root = Path(training_root).resolve()
    topic_dir = root / "topic_queues"
    known: list[str] = []
    if topic_dir.exists() and topic_dir.is_dir():
        for path in sorted(topic_dir.glob("*.txt")):
            slug = re.sub(r"[_\-]+", " ", path.stem).strip()
            if slug:
                known.append(slug)
            known.extend(_extract_topics_from_queue_text(path))
    known.extend(_extract_topics_from_queue_text(root / "queued_wikipedia_learning.txt"))
    return _dedupe_preserve_order(known)


def _clean_topics(values: Any) -> list[str]:
    if not isinstance(values, (list, tuple, set)):
        return []
    return _dedupe_preserve_order([str(x).strip() for x in list(values) if str(x).strip()])


def _upsert_subject(
    rows: dict[tuple[str, str], dict[str, Any]],
    order: list[tuple[str, str]],
    *,
    domain: str,
    name: str,
    topics: list[str],
):
    domain_clean = re.sub(r"\s+", " ", str(domain or "").strip())
    name_clean = re.sub(r"\s+", " ", str(name or "").strip())
    if not name_clean or not topics:
        return
    key = (domain_clean.lower(), name_clean.lower())
    if key not in rows:
        rows[key] = {"domain": domain_clean, "name": name_clean, "topics": []}
        order.append(key)
    rows[key]["topics"] = _dedupe_preserve_order(list(rows[key]["topics"]) + list(topics))


def _load_subject_catalog(training_root: str) -> dict[str, Any]:
    root = Path(training_root).resolve()
    catalog_path = root / "subject_catalog.json"
    if not catalog_path.exists() or not catalog_path.is_file():
        return dict(DEFAULT_SUBJECT_CATALOG)
    try:
        raw = json.loads(catalog_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return dict(DEFAULT_SUBJECT_CATALOG)
    if not isinstance(raw, dict):
        return dict(DEFAULT_SUBJECT_CATALOG)

    rows: dict[tuple[str, str], dict[str, Any]] = {}
    row_order: list[tuple[str, str]] = []

    domains_raw = raw.get("domains", [])
    if isinstance(domains_raw, list):
        for domain_row in domains_raw:
            if not isinstance(domain_row, dict):
                continue
            domain_name = str(domain_row.get("name", "")).strip()
            subject_items = domain_row.get("subjects", domain_row.get("disciplines", []))
            if not isinstance(subject_items, list):
                continue
            for subject in subject_items:
                if not isinstance(subject, dict):
                    continue
                subject_name = str(subject.get("name", "")).strip()
                subject_topics = _clean_topics(subject.get("topics", []))
                _upsert_subject(rows, row_order, domain=domain_name, name=subject_name, topics=subject_topics)

    subjects = raw.get("subjects", [])
    if isinstance(subjects, list):
        for row in subjects:
            if not isinstance(row, dict):
                continue
            raw_name = str(row.get("name", "")).strip()
            raw_domain = str(row.get("domain", "")).strip()
            if ":" in raw_name and not raw_domain:
                maybe_domain, maybe_name = [x.strip() for x in raw_name.split(":", 1)]
                raw_domain = maybe_domain
                raw_name = maybe_name
            subject_topics = _clean_topics(row.get("topics", []))
            _upsert_subject(rows, row_order, domain=raw_domain, name=raw_name, topics=subject_topics)

    cleaned_subjects: list[dict[str, Any]] = []
    for key in row_order:
        row = rows.get(key)
        if not row:
            continue
        name = str(row.get("name", "")).strip()
        domain = str(row.get("domain", "")).strip()
        topics = _clean_topics(row.get("topics", []))
        if not name or not topics:
            continue
        full_name = f"{domain}: {name}" if domain else name
        cleaned_subjects.append({"name": full_name, "domain": domain, "topics": topics})

    if not cleaned_subjects:
        return dict(DEFAULT_SUBJECT_CATALOG)

    domains_map: dict[str, list[dict[str, Any]]] = {}
    for row in cleaned_subjects:
        domain = str(row.get("domain", "")).strip() or "General"
        base_name = str(row.get("name", "")).split(":", 1)[-1].strip()
        domains_map.setdefault(domain, []).append({"name": base_name, "topics": list(row.get("topics", []))})

    cleaned_domains = [
        {"name": domain, "subject_count": len(subject_rows), "subjects": subject_rows}
        for domain, subject_rows in sorted(domains_map.items())
    ]
    return {
        "version": str(raw.get("version", "0.2")),
        "domains": cleaned_domains,
        "subjects": cleaned_subjects,
    }


def _catalog_subject_rows(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in list(catalog.get("subjects") or []):
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", "")).strip()
        domain = str(row.get("domain", "")).strip()
        topics = _dedupe_preserve_order([str(x).strip() for x in list(row.get("topics") or []) if str(x).strip()])
        if ":" in name and not domain:
            maybe_domain, maybe_name = [x.strip() for x in name.split(":", 1)]
            domain = maybe_domain
            name = maybe_name
        if not name or not topics:
            continue
        key = (domain.lower(), name.lower())
        if key in seen:
            continue
        seen.add(key)
        rows.append({"domain": domain, "name": name, "topics": topics})
    return rows


def _pick_subject_seed_topics(
    *,
    catalog: dict[str, Any],
    known_topics: list[str],
    max_topics: int,
    similarity_threshold: float = 0.56,
) -> tuple[list[str], dict[str, Any]]:
    selected: list[str] = []
    coverage_rows: list[dict[str, Any]] = []
    subject_rows = _catalog_subject_rows(catalog)
    by_domain: dict[str, list[dict[str, Any]]] = {}
    for row in subject_rows:
        domain = str(row.get("domain", "")).strip() or "General"
        topics = [str(x).strip() for x in list(row.get("topics") or []) if str(x).strip()]
        uncovered = [
            t for t in topics if not _topic_is_covered(t, known_topics, similarity_threshold=similarity_threshold)
        ]
        enriched = {**row, "uncovered_topics": uncovered, "domain": domain}
        by_domain.setdefault(domain, []).append(enriched)
        coverage_rows.append(
            {
                "name": str(row.get("name", "")).strip(),
                "domain": domain,
                "topic_count": len(topics),
                "uncovered_count": len(uncovered),
                "coverage_ratio": round((len(topics) - len(uncovered)) / max(1, len(topics)), 4),
            }
        )

    domains_sorted = sorted(
        by_domain.keys(),
        key=lambda d: (
            -sum(len(list(r.get("uncovered_topics") or [])) for r in by_domain[d]),
            d.lower(),
        ),
    )
    for domain in domains_sorted:
        rows = sorted(
            by_domain[domain],
            key=lambda r: (-len(list(r.get("uncovered_topics") or [])), str(r.get("name", "")).lower()),
        )
        for row in rows:
            uncovered = [str(x).strip() for x in list(row.get("uncovered_topics") or []) if str(x).strip()]
            if not uncovered:
                continue
            selected.append(uncovered[0])
            break
        if len(selected) >= max_topics:
            break

    if len(selected) < max_topics:
        ordered_rows = sorted(
            subject_rows,
            key=lambda r: (
                str(r.get("domain", "")).lower(),
                str(r.get("name", "")).lower(),
            ),
        )
        per_subject_idx: dict[tuple[str, str], int] = {}
        while len(selected) < max_topics:
            added_any = False
            for row in ordered_rows:
                if len(selected) >= max_topics:
                    break
                domain = str(row.get("domain", "")).strip()
                name = str(row.get("name", "")).strip()
                topics = [str(x).strip() for x in list(row.get("topics") or []) if str(x).strip()]
                key = (domain.lower(), name.lower())
                start_idx = int(per_subject_idx.get(key, 0))
                found = None
                for idx in range(start_idx, len(topics)):
                    candidate = topics[idx]
                    if _topic_is_covered(candidate, known_topics, similarity_threshold=similarity_threshold):
                        continue
                    if candidate in selected:
                        continue
                    found = candidate
                    per_subject_idx[key] = idx + 1
                    break
                if found:
                    selected.append(found)
                    added_any = True
            if not added_any:
                break

    domain_coverage: list[dict[str, Any]] = []
    for domain in sorted(by_domain.keys()):
        domain_rows = by_domain[domain]
        topic_total = sum(len(list(r.get("topics") or [])) for r in domain_rows)
        uncovered_total = sum(len(list(r.get("uncovered_topics") or [])) for r in domain_rows)
        domain_coverage.append(
            {
                "domain": domain,
                "subject_count": len(domain_rows),
                "topic_count": topic_total,
                "uncovered_count": uncovered_total,
                "coverage_ratio": round((topic_total - uncovered_total) / max(1, topic_total), 4),
            }
        )

    return _dedupe_preserve_order(selected)[:max_topics], {
        "subjects": coverage_rows,
        "domains": domain_coverage,
    }


def _fetch_random_wikipedia_titles(limit: int, timeout_sec: int = 15) -> list[str]:
    count = max(1, min(20, int(limit)))
    url = (
        "https://en.wikipedia.org/w/api.php"
        f"?action=query&list=random&format=json&rnnamespace=0&rnlimit={count}"
    )
    payload = _request_json(url, timeout_sec=timeout_sec, max_retries=5, base_backoff_sec=1.0)
    rows = payload.get("query", {}).get("random", [])
    if not isinstance(rows, list):
        return []
    out: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        title = re.sub(r"\s+", " ", str(row.get("title", "")).strip())
        if not title:
            continue
        out.append(title)
    return out
