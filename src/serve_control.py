from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from src.federated.identity import ensure_identity
from src.federated.swarm import (
    SwarmNode,
    DEFAULT_SWARM_PORT,
    DEFAULT_DATA_DIR,
    DEFAULT_IDENTITY_PATH,
    _join_via_bootstrap,
    _parse_addr,
)
from src.chat_benchmark import _load_cases as _load_benchmark_cases
from src.chat_benchmark import run_benchmark
from src.friendly_errors import friendly_error_message
from src.http_handler_base import IckleHTTPHandler
from src.ilm_chat import _resolve_default_model
from src.ilm_profile import detect_resources, ResourceConfig
from src.ilm_memory import get_memory
from src.model_maintain import run_data_maintenance, run_model_maintenance
from src.reality_check import collect_checks
from src.runtime_flags import RuntimeFlagsStore
from src.serve_web import ChatRuntime
from src.task_actions import _request_json, infer_task_from_instruction, run_task
from src.task_queue import TaskQueue, _is_fatal_error
from src.teacher_ingest import TeacherStore
from src.training_maintain import run_training_maintenance
from src.workspace_check import collect_workspace_checks
from src.workspace_paths import get_training_corpus_path, get_training_root


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except Exception:  # noqa: BLE001
        return int(default)


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


def _load_subject_catalog(training_root: str) -> dict[str, Any]:
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

    # Pass 1: breadth-first by domain to keep a diverse plan.
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

    # Pass 2: round-robin across all subjects to fill remaining slots.
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


def _detect_total_ram_bytes() -> int:
    try:
        if os.name == "nt":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullTotalPhys)
    except Exception:  # noqa: BLE001
        pass
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        if page_size > 0 and pages > 0:
            return page_size * pages
    except Exception:  # noqa: BLE001
        pass
    return 0


def _detect_gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {"available": False, "count": 0, "names": [], "backend": "cpu"}
    try:
        from src.device_bridge import get_gpu_info
        gpu = get_gpu_info()
        info.update(gpu)
    except Exception:  # noqa: BLE001
        return info
    return info


def _normalize_resource_budget(
    raw: dict[str, Any] | None,
    *,
    defaults: dict[str, Any],
    gpu_available: bool,
) -> dict[str, int]:
    max_percent = 90
    cpu_default = _as_int(defaults.get("cpu_percent", 70), 70)
    ram_default = _as_int(defaults.get("ram_percent", 70), 70)
    gpu_default = _as_int(defaults.get("gpu_percent", 70), 70) if gpu_available else 0
    source = raw if isinstance(raw, dict) else {}
    cpu_percent = _as_int(source.get("cpu_percent", cpu_default), cpu_default)
    ram_percent = _as_int(source.get("ram_percent", ram_default), ram_default)
    gpu_percent = _as_int(source.get("gpu_percent", gpu_default), gpu_default)
    cpu_percent = max(10, min(max_percent, cpu_percent))
    ram_percent = max(10, min(max_percent, ram_percent))
    if gpu_available:
        gpu_percent = max(10, min(max_percent, gpu_percent))
    else:
        gpu_percent = 0
    return {
        "cpu_percent": cpu_percent,
        "ram_percent": ram_percent,
        "gpu_percent": gpu_percent,
        "max_percent": max_percent,
    }


class ControlRuntime:
    def __init__(self, training_root: str):
        self.flags = RuntimeFlagsStore()
        self.tasks = TaskQueue()
        self._chat_runtime = ChatRuntime()
        self.training_root = str(Path(training_root).resolve())
        # Must go through self._resolve_default_model() (catches
        # FileNotFoundError), not the bare module-level function -- a fresh
        # install/checkout with no models/ yet used to crash the constructor
        # (and therefore the whole control server) before it could even
        # start and report "no model yet" gracefully, the same bug
        # ChatRuntime in serve_web.py was already fixed for.
        self.default_model = ""
        self.default_model = self._resolve_default_model()
        self.hardware_info = self._collect_hardware_info()
        self.resource_budget_path = Path("data/training_resource_budget.json")
        self.resource_budget_path.parent.mkdir(parents=True, exist_ok=True)
        self.telemetry_log_path = Path("data/runtime/training_live.log")
        self.telemetry_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.training_resource_budget = self._load_training_resource_budget()
        # _sync_worker() previously only ran reactively (task creation, flag
        # updates) -- a server restart with tasks already sitting in the
        # persisted queue left them stuck at "queued" forever, since nothing
        # ever triggered the background worker thread to start. Confirmed
        # live: a freshly restarted server with a queued task did not touch
        # it until an unrelated flags update happened to kick the worker.
        self._sync_worker()
        self._init_swarm()

    def _resolve_default_model(self) -> str:
        try:
            self.default_model = _resolve_default_model()
        except FileNotFoundError:
            pass
        return self.default_model

    def _training_corpus_path(self, filename: str) -> str:
        return str(get_training_corpus_path(filename, self.training_root))

    KNOWN_PEERS_PATH = Path("data/torickle/known_peers.json")

    def _init_swarm(self):
        try:
            identity = ensure_identity(DEFAULT_IDENTITY_PATH, label="control-api")
            # federated_enabled (src/runtime_flags.py) is the real on/off switch
            # for whether this node actually participates in the peer network:
            # off (the default) keeps the swarm loopback-only, so nothing it
            # does is reachable from another machine -- matching the "nothing
            # sent anywhere unless you turn it on" promise in the UI. On binds
            # externally, starts trackerless Mainline-DHT discovery, and also
            # joins any explicitly configured direct/private peers.
            network_enabled = bool(self.flags.get_flags().get("federated_enabled", False))
            env_host = str(os.getenv("ICKLE_SWARM_HOST", "")).strip()
            swarm_host = env_host or ("0.0.0.0" if network_enabled else "127.0.0.1")
            env_external = str(os.getenv("ICKLE_SWARM_EXTERNAL_HOST", "")).strip()
            self.swarm = SwarmNode(
                identity=identity,
                data_dir=DEFAULT_DATA_DIR,
                host=swarm_host,
                external_host=env_external or None,
            )
            self.swarm.start(
                attempt_nat_traversal=network_enabled,
                public_discovery=network_enabled,
            )
            if network_enabled:
                _join_via_bootstrap(self.swarm.peer_discovery, self._load_known_peers())
            print(f"Swarm node active: {identity.peer_id} (network_enabled={network_enabled}, host={swarm_host})")
        except Exception as exc:
            print(f"Swarm node not available: {exc}")
            self.swarm = None

    def _load_known_peers(self) -> list[str]:
        try:
            data = json.loads(self.KNOWN_PEERS_PATH.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return []
        return [str(a).strip() for a in data.get("peers", []) if str(a).strip()]

    def _save_known_peers(self, peers: list[str]):
        self.KNOWN_PEERS_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.KNOWN_PEERS_PATH.write_text(json.dumps({"peers": peers}, indent=2), encoding="utf-8")

    def list_known_peers(self) -> list[str]:
        return self._load_known_peers()

    def add_known_peer(self, address: str) -> dict[str, Any]:
        host, port = _parse_addr(str(address or "").strip())
        if not host:
            raise ValueError("Invalid peer address -- expected host:port")
        normalized = f"{host}:{port}"
        peers = self._load_known_peers()
        if normalized not in peers:
            peers.append(normalized)
            self._save_known_peers(peers)
        if self.swarm is not None:
            _join_via_bootstrap(self.swarm.peer_discovery, [normalized])
        return {"peers": peers}

    def remove_known_peer(self, address: str) -> dict[str, Any]:
        peers = [p for p in self._load_known_peers() if p != str(address or "").strip()]
        self._save_known_peers(peers)
        return {"peers": peers}

    def set_network_enabled(self, enabled: bool) -> dict[str, Any]:
        self.flags.set_flag("federated_enabled", bool(enabled))
        # Rebinding the interface (loopback <-> external) requires a fresh
        # socket, not just flipping a variable -- stop the current node and
        # bring up a new one under the new flag value.
        old_swarm = getattr(self, "swarm", None)
        if old_swarm is not None:
            try:
                old_swarm.stop()
            except Exception:
                pass
        self._init_swarm()
        return self.get_swarm_status()

    def get_swarm_status(self) -> dict[str, Any]:
        network_enabled = bool(self.flags.get_flags().get("federated_enabled", False))
        if not self.swarm:
            return {
                "active": False,
                "enabled": network_enabled,
                "known_peers": self.list_known_peers(),
                "public_discovery": {"enabled": False, "phase": "off"},
            }
        remote_peers = [
            peer
            for peer in self.swarm.peer_discovery.store.all_peers()
            if peer.peer_id != self.swarm.identity.peer_id_bytes
        ]
        if self.swarm._external_host_explicit or self.swarm._port_mapped:
            reachability = "reachable"
        elif network_enabled:
            reachability = "outbound-only"
        else:
            reachability = "local"
        return {
            "active": True,
            "enabled": network_enabled,
            "peer_id": self.swarm.identity.peer_id,
            "bundles_served": len(self.swarm.bundles),
            "peers_known": len(remote_peers),
            "host": self.swarm.host,
            "port": self.swarm.port,
            "external_host": self.swarm.external_host,
            "reachability": reachability,
            "port_mapped": bool(self.swarm._port_mapped),
            "known_peers": self.list_known_peers(),
            "public_discovery": self.swarm.public_discovery_status(),
            "commons": self.swarm.commons.summary(),
        }

    def refresh_public_swarm(self) -> dict[str, Any]:
        if not self.swarm:
            return {"error": "Swarm node not available"}
        self.swarm.refresh_public_discovery()
        return self.get_swarm_status()

    def get_torickle_bundles(self) -> list[dict[str, Any]]:
        if not self.swarm:
            return []
        self.swarm._scan_local_bundles()
        entries = []
        for bundle_id, info in self.swarm.bundles.items():
            m = info.manifest
            entries.append({
                "bundle_id": bundle_id,
                "piece_count": m.get("piece_count", 0),
                "total_bytes": m.get("total_bytes", 0),
                "payload_sha256": m.get("payload_sha256", ""),
                "merkle_root": m.get("merkle_root", ""),
                "created_at": m.get("created_at_utc", ""),
            })
        return entries

    def import_torickle_bundle(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.swarm:
            return {"error": "Swarm node not available"}
        bundle_path = str(payload.get("path", "")).strip()
        if not bundle_path:
            return {"error": "Missing 'path'"}
        bundle_id = self.swarm.import_bundle(bundle_path)
        if not bundle_id:
            return {"error": f"Failed to import bundle from {bundle_path}"}
        return {"bundle_id": bundle_id, "status": "imported"}

    def announce_torickle_bundle(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.swarm:
            return {"error": "Swarm node not available"}
        bundle_id = str(payload.get("bundle_id", "")).strip()
        model_hash = str(payload.get("model_hash", "")).strip()
        if not bundle_id:
            return {"error": "Missing 'bundle_id'"}
        ann = self.swarm.announce_bundle(bundle_id, model_hash=model_hash)
        if not ann:
            return {"error": f"Bundle {bundle_id} not found"}
        return ann.to_dict()

    def find_torickle_bundles(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        if not self.swarm:
            return []
        model_hash = str(payload.get("model_hash", "")).strip()
        results = self.swarm.find_bundles(model_hash=model_hash)
        return [ann.to_dict() for ann in results]

    def pull_torickle_bundle(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.swarm:
            return {"error": "Swarm node not available"}
        host = str(payload.get("host", "")).strip()
        port = int(payload.get("port", DEFAULT_SWARM_PORT))
        bundle_id = str(payload.get("bundle_id", "")).strip()
        if not host or not bundle_id:
            return {"error": "Missing 'host' or 'bundle_id'"}
        result = self.swarm.download_bundle(host, port, bundle_id, verify=True)
        if not result:
            return {"error": f"Failed to download bundle {bundle_id} from {host}:{port}"}
        return {"bundle_id": bundle_id, "path": str(result), "status": "downloaded"}

    def reassemble_torickle_bundle(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.swarm:
            return {"error": "Swarm node not available"}
        bundle_id = str(payload.get("bundle_id", "")).strip()
        out_path = str(payload.get("out_path", f"data/torickle/{bundle_id}_delta.pt")).strip()
        if not bundle_id:
            return {"error": "Missing 'bundle_id'"}
        result = self.swarm.reassemble_delta(bundle_id, out_path=out_path)
        if not result:
            return {"error": f"Failed to reassemble bundle {bundle_id}"}
        return dict(result)

    def _collect_hardware_info(self) -> dict[str, Any]:
        rc = detect_resources()
        return {
            "cpu": {"logical_cores": max(1, int(os.cpu_count() or 1))},
            "memory": {"total_bytes": int(_detect_total_ram_bytes())},
            "gpu": _detect_gpu_info(),
            "resource_config": rc.summary(),
        }

    def _default_training_resource_budget(self) -> dict[str, int]:
        gpu_available = bool(self.hardware_info.get("gpu", {}).get("available", False))
        return _normalize_resource_budget(
            {"cpu_percent": 70, "ram_percent": 70, "gpu_percent": 70 if gpu_available else 0},
            defaults={"cpu_percent": 70, "ram_percent": 70, "gpu_percent": 70 if gpu_available else 0},
            gpu_available=gpu_available,
        )

    def _load_training_resource_budget(self) -> dict[str, int]:
        defaults = self._default_training_resource_budget()
        if not self.resource_budget_path.exists():
            self.resource_budget_path.write_text(json.dumps(defaults, indent=2), encoding="utf-8")
            return defaults
        try:
            raw = json.loads(self.resource_budget_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            raw = {}
        budget = _normalize_resource_budget(
            raw if isinstance(raw, dict) else {},
            defaults=defaults,
            gpu_available=bool(self.hardware_info.get("gpu", {}).get("available", False)),
        )
        self.resource_budget_path.write_text(json.dumps(budget, indent=2), encoding="utf-8")
        return budget

    def _save_training_resource_budget(self, budget: dict[str, int]):
        self.resource_budget_path.write_text(json.dumps(budget, indent=2), encoding="utf-8")
        self.training_resource_budget = dict(budget)

    def _resolve_resource_budget(self, payload: dict[str, Any] | None) -> dict[str, int]:
        raw = {}
        if isinstance(payload, dict):
            raw = payload.get("resource_budget", {})
        return _normalize_resource_budget(
            raw if isinstance(raw, dict) else {},
            defaults=self.training_resource_budget,
            gpu_available=bool(self.hardware_info.get("gpu", {}).get("available", False)),
        )

    def _derive_training_runtime_overrides(
        self,
        *,
        resource_budget: dict[str, int],
    ) -> dict[str, Any]:
        rc = detect_resources()
        cpu_pct = max(10, min(90, int(resource_budget.get("cpu_percent", 70))))
        ram_pct = max(10, min(90, int(resource_budget.get("ram_percent", 70))))
        gpu_available = bool(self.hardware_info.get("gpu", {}).get("available", False))
        gpu_pct = max(10, min(90, int(resource_budget.get("gpu_percent", 70)))) if gpu_available else 0
        rc.cpu_percent = cpu_pct
        rc.ram_percent = ram_pct
        rc.gpu_percent = gpu_pct
        cpu_total = max(1, int(self.hardware_info.get("cpu", {}).get("logical_cores", 1)))
        cpu_scale = max(0.1, min(0.9, float(cpu_pct) / 100.0))
        ram_scale = max(0.1, min(0.9, float(ram_pct) / 100.0))
        if gpu_available:
            gpu_scale = max(0.1, min(0.9, float(gpu_pct) / 100.0))
        else:
            gpu_scale = ram_scale
        scale = min(cpu_scale, ram_scale, gpu_scale)
        threads = max(1, min(16, int(round(cpu_total * cpu_scale))))
        base_batch = max(2, int(rc.batch_size))
        batch_size = max(2, int(round(base_batch * max(0.25, scale))))
        grad_accum_steps = max(1, int(round(base_batch / max(1, batch_size))))
        return {
            "torch_threads": threads,
            "batch_size": batch_size,
            "grad_accum_steps": grad_accum_steps,
            "resource_budget": dict(resource_budget),
        }

    def get_system_resources(self) -> dict[str, Any]:
        return {
            "detected": self.hardware_info,
            "resource_budget": self.training_resource_budget,
            "updated_at_utc": _utc_now(),
        }

    def update_system_resources(self, payload: dict[str, Any]) -> dict[str, Any]:
        budget = _normalize_resource_budget(
            payload if isinstance(payload, dict) else {},
            defaults=self.training_resource_budget,
            gpu_available=bool(self.hardware_info.get("gpu", {}).get("available", False)),
        )
        self._save_training_resource_budget(budget)
        return {
            "resource_budget": budget,
            "derived_training_overrides": self._derive_training_runtime_overrides(
                resource_budget=budget,
            ),
        }

    def _sync_worker(self):
        flags = self.flags.get_flags()
        if flags.get("background_task_worker_enabled", True):
            parallel_enabled = bool(flags.get("parallel_training_enabled", True))
            configured_parallel = _as_int(flags.get("max_parallel_training_tasks", 2), 2)
            max_parallel = configured_parallel if parallel_enabled else 1
            cpu_cores = max(1, int(self.hardware_info.get("cpu", {}).get("logical_cores", 1)))
            hard_cap = 3 if cpu_cores >= 12 else 2
            max_parallel = max(1, min(hard_cap, max_parallel))
            self.tasks.start_worker(
                self._run_task,
                max_parallel_tasks=max_parallel,
                max_running_resource_percent=int(self.training_resource_budget.get("max_percent", 90)),
            )
        else:
            self.tasks.stop_worker()

    def _task_console_label(self, task_type: str, payload: dict[str, Any]) -> str:
        topic = re.sub(r"\s+", " ", str((payload or {}).get("topic", "")).strip())
        if topic:
            return f"{task_type} | topic={topic}"
        out_model = str((payload or {}).get("out_model", "")).strip()
        if out_model:
            return f"{task_type} | out={Path(out_model).name}"
        return task_type

    def _console_log(self, message: str):
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        line = f"[{stamp}Z] {message}"
        print(line, flush=True)
        try:
            with self.telemetry_log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:  # noqa: BLE001
            pass

    def _run_task(self, task_type: str, payload: dict, progress):
        label = self._task_console_label(task_type, payload if isinstance(payload, dict) else {})
        self._console_log(f"task-start: {label}")

        def _progress_cb(message: str):
            clean = re.sub(r"\s+", " ", str(message or "").strip())
            if clean:
                self._console_log(f"task-progress: {label} | {clean}")
            progress(message)

        flags = self.flags.get_flags()
        try:
            result = run_task(
                task_type=task_type,
                payload=payload,
                progress=_progress_cb,
                memory_enabled=bool(flags.get("memory_enabled", True)),
                allow_auto_training_tasks=bool(flags.get("allow_auto_training_tasks", False)),
                training_root_override=self.training_root,
                chat_runner=self.run_chat,
            )
            self._console_log(f"task-done: {label}")
            return result
        except Exception as exc:  # noqa: BLE001
            self._console_log(f"task-fail: {label} | {exc}")
            raise

    def get_status(self) -> dict:
        flags = self.flags.get_flags()
        tasks = self.tasks.list_tasks(limit=200)
        mastery = self._build_topic_mastery_map(limit=300)
        default_model = self._resolve_default_model()
        counts = {
            "queued": sum(1 for t in tasks if t.get("status") == "queued"),
            "running": sum(1 for t in tasks if t.get("status") == "running"),
            "completed": sum(1 for t in tasks if t.get("status") == "completed"),
            "failed": sum(1 for t in tasks if t.get("status") == "failed"),
            "cancelled": sum(1 for t in tasks if t.get("status") == "cancelled"),
        }
        return {
            "flags": flags,
            "chat_model": default_model,
            "task_counts": counts,
            "training_root": self.training_root,
            "resource_budget": dict(self.training_resource_budget),
            "task_worker": {
                "parallel_enabled": bool(flags.get("parallel_training_enabled", True)),
                "max_parallel_training_tasks": _as_int(flags.get("max_parallel_training_tasks", 2), 2),
                "max_running_resource_percent": int(self.training_resource_budget.get("max_percent", 90)),
            },
            "hardware": self.hardware_info,
            "memory_summary": get_memory().get_memory_summary(),
            "topic_mastery": {
                "topic_count": int(mastery.get("topic_count", 0)),
                "weak_topics": list(mastery.get("weak_topics") or [])[:10],
                "strong_topics": list(mastery.get("strong_topics") or [])[:10],
            },
            "reality_checks": [asdict(c) for c in collect_checks()],
            "workspace_checks": [asdict(c) for c in collect_workspace_checks()],
        }

    def list_models(
        self,
        *,
        limit: int = 80,
        include_checkpoints: bool = False,
        policy_only: bool = True,
    ) -> list[dict]:
        # Delegates to serve_web.py's ChatRuntime rather than maintaining a
        # byte-for-byte duplicate -- this used to be an independent copy
        # that carried the same bug as the original (model_root.glob("*.pt")
        # never looking in models/candidates/, where every training task
        # actually writes its output) until it was fixed in exactly one
        # place instead of two.
        return self._chat_runtime.list_models(
            limit=limit, include_checkpoints=include_checkpoints, policy_only=policy_only
        )

    def _is_training_task(self, row: dict[str, Any]) -> bool:
        kind = str(row.get("task_type", "")).strip().lower()
        return kind in {
            "learn_wikipedia_topic",
            "learn_web_topic",
            "build_clean_corpus",
            "train_model",
            "continual_guard_step",
            "evaluate_model",
        }

    def get_training_progress(self, limit: int = 25) -> dict[str, Any]:
        tasks = self.tasks.list_tasks(limit=500)
        training_tasks = [t for t in tasks if self._is_training_task(t)]
        mastery = self._build_topic_mastery_map(limit=600)
        running_all = [t for t in training_tasks if str(t.get("status", "")) in {"queued", "running"}]
        running = running_all[: max(1, limit)]
        recent = training_tasks[: max(1, limit)]

        all_models = self.list_models(limit=240, include_checkpoints=True, policy_only=False)
        checkpoints = [m for m in all_models if str(m.get("name", "")).endswith(".checkpoint.pt")]
        regular_models = [m for m in all_models if not str(m.get("name", "")).endswith(".checkpoint.pt")]
        latest_checkpoint = checkpoints[0] if checkpoints else None
        latest_model = regular_models[0] if regular_models else None

        queue_path = Path(self.training_root) / "queued_wikipedia_learning.txt"
        queue_stats = {
            "path": str(queue_path),
            "exists": queue_path.exists(),
            "size_bytes": int(queue_path.stat().st_size) if queue_path.exists() else 0,
            "updated_at": queue_path.stat().st_mtime if queue_path.exists() else 0,
        }
        return {
            "updated_at_utc": _utc_now(),
            "resource_budget": dict(self.training_resource_budget),
            "running_or_queued": running,
            "recent_training_tasks": recent,
            "latest_checkpoint": latest_checkpoint,
            "latest_model": latest_model,
            "training_queue_file": queue_stats,
            "topic_mastery": mastery,
            "counts": {
                "training_task_total": len(training_tasks),
                "running_or_queued": len(running_all),
            },
        }

    def get_memory_summary(self) -> dict[str, Any]:
        return get_memory().get_memory_summary()

    def get_memory_facts(self, limit: int = 40, category: str | None = None) -> list[dict[str, Any]]:
        memory = get_memory()
        clean_category = str(category or "").strip() or None
        return memory.get_facts(category=clean_category, limit=max(1, limit))

    def get_memory_context(self, limit: int = 30) -> list[dict[str, Any]]:
        return get_memory().get_recent_context(limit=max(1, limit))

    def search_memory(self, query: str, limit: int = 10) -> dict[str, Any]:
        clean_query = str(query or "").strip()
        if not clean_query:
            return {"query": "", "facts": [], "research_notes": [], "web_facts": []}
        memory = get_memory()
        return {
            "query": clean_query,
            "facts": memory.search_facts(clean_query, limit=max(1, limit)),
            "research_notes": memory.search_research_notes(clean_query, limit=max(1, limit)),
            "web_facts": memory.search_web_facts(clean_query, limit=max(1, limit)),
        }

    def add_memory_fact(self, payload: dict[str, Any]) -> dict[str, Any]:
        fact = str(payload.get("fact", "")).strip()
        if not fact:
            raise ValueError("Missing fact")
        category = str(payload.get("category", "general")).strip() or "general"
        source = str(payload.get("source", "")).strip() or None
        confidence = float(payload.get("confidence", 0.8))
        get_memory().add_fact(fact=fact, category=category, source=source, confidence=confidence)
        return {"ok": True, "summary": self.get_memory_summary()}

    def clear_memory(self, payload: dict[str, Any]) -> dict[str, Any]:
        memory_type_raw = str(payload.get("memory_type", "")).strip().lower()
        memory_type = memory_type_raw or None
        valid = {None, "owner", "facts", "web", "conversations", "research"}
        if memory_type not in valid:
            raise ValueError("memory_type must be one of owner|facts|web|conversations|research or omitted")
        get_memory().clear_memory(memory_type=memory_type)
        return {"ok": True, "cleared": memory_type or "all", "summary": self.get_memory_summary()}

    def export_memory(self) -> dict[str, Any]:
        memory = get_memory()
        blob: dict[str, Any] = {"summary": memory.get_memory_summary()}
        files = {
            "owner": memory.owner_file,
            "facts": memory.facts_file,
            "conversations": memory.conversations_file,
            "web_learning": memory.web_learning_file,
            "research_notes": memory.research_file,
        }
        for key, file_path in files.items():
            try:
                blob[key] = json.loads(file_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                blob[key] = {}
        return blob

    def save_memory_export(self) -> dict[str, Any]:
        """Write export_memory()'s blob to a timestamped file under data/ and
        return its path, instead of handing raw JSON to the browser -- the
        image-attach bug this session showed pywebview's WebView2 backend
        has real quirks with blob/data URIs, and a file save avoids that
        class of problem entirely for a local-first app that already has
        full filesystem access."""
        blob = self.export_memory()
        out_dir = Path("data/memory/exports")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"memory_export_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        out_path.write_text(json.dumps(blob, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"path": str(out_path.resolve())}

    def get_live_system_stats(self) -> dict[str, Any]:
        """Live CPU/RAM/disk usage for the Dashboard panel -- distinct from
        get_system_resources(), which reports static hardware detection and
        the configured training budget, not current usage."""
        cpu_percent = None
        ram_percent = None
        ram_used_bytes = None
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0.1)
            vm = psutil.virtual_memory()
            ram_percent = vm.percent
            ram_used_bytes = int(vm.total - vm.available)
        except Exception:  # noqa: BLE001
            pass
        disk_free_bytes = None
        disk_total_bytes = None
        try:
            import shutil as _shutil
            usage = _shutil.disk_usage(".")
            disk_free_bytes = int(usage.free)
            disk_total_bytes = int(usage.total)
        except Exception:  # noqa: BLE001
            pass
        return {
            "cpu_percent": cpu_percent,
            "ram_percent": ram_percent,
            "ram_used_bytes": ram_used_bytes,
            "ram_total_bytes": int(self.hardware_info.get("memory", {}).get("total_bytes", 0)),
            "disk_free_bytes": disk_free_bytes,
            "disk_total_bytes": disk_total_bytes,
            "gpu": self.hardware_info.get("gpu", {}),
        }

    def get_federated_status(self) -> dict[str, Any]:
        """Plain-shape summary of the federated coordinator, if one has run
        on this machine: how many contributors have registered, and what the
        current/most recent training round looks like. Reads
        data/federated/coordinator_state.json directly -- there is no
        running coordinator process required for this to report something,
        since the state file persists across restarts."""
        state_path = Path("data/federated/coordinator_state.json")
        if not state_path.exists():
            return {"coordinator_running": False, "clients": [], "active_round": None, "completed_rounds": 0}
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {"coordinator_running": False, "clients": [], "active_round": None, "completed_rounds": 0}

        clients_raw = state.get("clients", {}) if isinstance(state.get("clients"), dict) else {}
        clients = [
            {
                "client_id": cid,
                "platform": c.get("platform", ""),
                "device_name": c.get("device_name", ""),
                "last_seen_utc": c.get("last_seen_utc", ""),
                "revoked": bool(c.get("revoked", False)),
            }
            for cid, c in clients_raw.items()
            if isinstance(c, dict)
        ]
        active_round = state.get("active_round")
        return {
            "coordinator_running": True,
            "clients": clients,
            "active_client_count": sum(1 for c in clients if not c["revoked"]),
            "active_round": active_round,
            "round_id": state.get("round_id"),
            "completed_rounds": len(state.get("completed_rounds", []) or []),
            "min_clients": state.get("min_clients"),
        }

    def get_codistill_status(self) -> dict[str, Any]:
        """Plain-shape summary of cross-architecture co-distillation
        (src/federated/codistill.py): locally-observed peer teaching trust
        and the most recent round's results, if any. Reads the trust/report
        files directly -- no round needs to be running for this to report
        something, matching get_federated_status()'s pattern."""
        from src.federated.codistill import DEFAULT_TRUST_STORE_PATH, PeerTrustStore

        trust_store = PeerTrustStore(DEFAULT_TRUST_STORE_PATH)
        trust = [
            {"peer_id": peer_id, "trust": trust_store.overall(peer_id), "domains": trust_store.domains_for(peer_id)}
            for peer_id in trust_store.ranked_peer_ids()
        ]

        report_path = Path("data/codistill/last_report.json")
        last_report = None
        if report_path.exists():
            try:
                last_report = json.loads(report_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                last_report = None

        corpus_pairs = 0
        corpus_path = Path("data/codistill/distilled_corpus.txt")
        if corpus_path.exists():
            try:
                corpus_pairs = corpus_path.read_text(encoding="utf-8").count("User: ")
            except OSError:
                corpus_pairs = 0

        return {"trust": trust, "last_report": last_report, "corpus_pairs": corpus_pairs}

    def ask_swarm(self, prompt: str) -> dict[str, Any]:
        """Live, in-chat counterpart to the codistill_round task: ask
        trust-ranked peers the user's actual question right now instead of
        waiting for a training round. Reuses the same bootstrap peer list
        and identity file as run_codistill_round_task. Dedicated inference
        serving remains a separate, explicit privacy boundary, so this uses
        directly configured inference peer addresses rather than silently
        sending a prompt to every model/training-swarm participant."""
        from src.federated.codistill import DEFAULT_TRUST_STORE_PATH, PeerTrustStore, ask_swarm as _ask_swarm
        from src.federated.contribution_ledger import LedgerStore
        from src.federated.inference_swarm import DEFAULT_IDENTITY_PATH
        from src.federated.keys import ensure_ed_identity
        from src.federated.peer_discovery import PeerDiscovery

        prompt = str(prompt or "").strip()
        if not prompt:
            raise ValueError("Missing prompt")

        bootstrap = self._load_known_peers()
        identity = ensure_ed_identity(Path(DEFAULT_IDENTITY_PATH))
        ledger = LedgerStore()
        trust_store = PeerTrustStore(DEFAULT_TRUST_STORE_PATH)

        peer_discovery = PeerDiscovery()
        for addr in bootstrap:
            host, port = _parse_addr(addr)
            peer_discovery.add_bootstrap(host, port)
        _join_via_bootstrap(peer_discovery, bootstrap)

        result = _ask_swarm(
            prompt,
            self_peer_id=identity.peer_id,
            peer_discovery=peer_discovery,
            ledger=ledger,
            trust_store=trust_store,
        )
        conflicts = (result.get("deliberation") or {}).get("possible_conflicts") or []
        if conflicts:
            from src.disagreement_curriculum import record_conflicts

            record_conflicts(conflicts, source="live_ask")
        return result

    def rate_swarm_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Let the owner, rather than peer conformity, govern local trust."""
        from src.federated.codistill import DEFAULT_TRUST_STORE_PATH, PeerTrustStore, classify_domain

        peer_id = str(payload.get("peer_id", "")).strip()
        prompt = str(payload.get("prompt", "")).strip()
        if not peer_id or not prompt:
            raise ValueError("Missing peer_id or prompt")
        if not re.fullmatch(r"[a-fA-F0-9]{40}", peer_id):
            raise ValueError("Invalid peer_id")
        helpful_raw = payload.get("helpful")
        if not isinstance(helpful_raw, bool):
            raise ValueError("helpful must be true or false")
        domain = classify_domain(prompt)
        trust_store = PeerTrustStore(DEFAULT_TRUST_STORE_PATH)
        trust = trust_store.update(peer_id, 1.0 if helpful_raw else 0.0, domain)
        trust_store.save()
        return {"saved": True, "peer_id": peer_id, "domain": domain, "trust": trust, "signal": "human_review"}

    def get_commons_status(self) -> dict[str, Any]:
        ledger = self._chat_runtime.epistemic_ledger()
        return {
            **ledger.summary(),
            "events_recent": ledger.public_events(limit=80),
            "conflict_policy": "preserve",
            "prompt_policy": "peer events are inert until locally adopted",
        }

    def sync_commons(self) -> dict[str, Any]:
        if not bool(self.flags.get_flags().get("federated_enabled", False)):
            raise PermissionError("Join the peer network before syncing shared knowledge.")
        from src.federated.knowledge_commons import sync_with_peers

        peers = self._load_known_peers()
        report = sync_with_peers(self._chat_runtime.epistemic_ledger(), peers)
        return {**report, "commons": self._chat_runtime.epistemic_ledger().summary()}

    def adopt_commons_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        event_id = str(payload.get("event_id", "")).strip()
        if not event_id:
            raise ValueError("Missing event_id")
        event = self._chat_runtime.epistemic_ledger().adopt_event(
            event_id,
            shared=bool(payload.get("shared", False)),
        )
        return {"adopted": True, "event": event, "commons": self._chat_runtime.epistemic_ledger().summary()}

    def get_consolidation_status(self) -> dict[str, Any]:
        """How many of the owner's own adopted Epistemic Commons corrections
        are eligible to be folded into the next continual-guard training
        step (see src/verified_corrections.py). This is always-on by default
        -- every continual_guard_step task run already includes them -- this
        endpoint just makes that otherwise-invisible pipeline stage visible,
        the same way get_codistill_status() surfaces trust without a round
        needing to be running."""
        from src.verified_corrections import DEFAULT_CORRECTIONS_CORPUS, verified_corrections_status

        status = verified_corrections_status()
        status["corpus_path"] = DEFAULT_CORRECTIONS_CORPUS
        return status

    def get_disagreement_status(self) -> dict[str, Any]:
        """The network's own ranked list of what it's currently least sure
        about: claim clusters where independently-trained peers gave
        conflicting answers, accumulated by both the live 'ask the swarm
        too' path and periodic co-distillation rounds (see
        src/disagreement_curriculum.py). An entry drops off the moment the
        owner corrects/adopts that exact claim in the Epistemic Commons."""
        from src.disagreement_curriculum import disagreement_status

        return disagreement_status()

    def get_contribution_status(self) -> dict[str, Any]:
        """Seed:peer contribution ledger summary (torrent-ratio style: how much
        this device has given the network -- training rounds completed,
        torickle pieces served, inference answered for peers -- vs. how much
        it has consumed). Same "read the local state file directly, no
        running process required" pattern as get_federated_status() above.
        Backs the Sharing tab's contribution numbers in the web UI; the CLI
        equivalent is `python -m src.app infer report`."""
        from src.federated.contribution_ledger import DEFAULT_LEDGER_PATH, LedgerStore

        store = LedgerStore(DEFAULT_LEDGER_PATH)
        summary = store.summary()
        summary["has_history"] = bool(
            summary.get("seed_pieces_served")
            or summary.get("seed_training_rounds")
            or summary.get("peer_requests_served")
            or summary.get("peer_requests_consumed")
        )
        return summary

    def list_knowledge_deltas(self) -> list[dict[str, Any]]:
        from src.scoped_knowledge import get_scoped_manager
        return get_scoped_manager().list_deltas()

    def set_knowledge_delta_enabled(self, delta_id: str, enabled: bool) -> dict[str, Any]:
        from src.scoped_knowledge import get_scoped_manager
        mgr = get_scoped_manager()
        ok = mgr.enable_delta(delta_id) if enabled else mgr.disable_delta(delta_id)
        return {"ok": ok, "delta_id": delta_id, "enabled": enabled}

    def remove_knowledge_delta(self, delta_id: str) -> dict[str, Any]:
        """ScopedKnowledgeManager.remove_delta() already existed at the
        registry layer (DeltaRegistry.remove()) -- disable/rollback had
        routes wired but delete never did, so a bad or junk-labeled add-on
        (see e.g. train.py's --auto-register fallback labeling) had no way
        to actually go away, only be turned off forever."""
        from src.scoped_knowledge import get_scoped_manager

        mgr = get_scoped_manager()
        ok = mgr.remove_delta(delta_id)
        return {"ok": ok, "delta_id": delta_id}

    def rollback_knowledge_delta(self, delta_id: str) -> dict[str, Any]:
        from src.scoped_knowledge import get_scoped_manager
        from src.delta_router import KnowledgeDelta
        mgr = get_scoped_manager()
        result = mgr.registry.rollback(delta_id)
        if not result:
            return {"ok": False, "delta_id": delta_id, "detail": "No earlier version available"}
        mgr.router.remove(delta_id)
        mgr.router.register(KnowledgeDelta.from_dict(result))
        return {"ok": True, "delta_id": delta_id, "version": result.get("version", "?")}

    def set_knowledge_delta_threshold(self, delta_id: str, threshold: float) -> dict[str, Any]:
        """DeltaRegistry.update_threshold() (src/delta_registry.py) already
        implemented this -- it just had no HTTP route wiring it to anything,
        so the web UI had no way to call it."""
        from src.scoped_knowledge import get_scoped_manager
        mgr = get_scoped_manager()
        clamped = max(0.0, min(1.0, float(threshold)))
        entry = mgr.registry.get(delta_id)
        if entry is None:
            return {"ok": False, "delta_id": delta_id, "detail": "Delta not found"}
        mgr.registry.update_threshold(delta_id, clamped)
        return {"ok": True, "delta_id": delta_id, "activation_threshold": clamped}

    def get_research_sessions(self, limit: int = 20) -> list[dict]:
        memory = get_memory()
        return memory.list_research_sessions(limit=max(1, limit))

    def find_research_notes(self, query: str, limit: int = 10, topic_hint: str | None = None) -> list[dict]:
        memory = get_memory()
        return memory.search_research_notes(query, limit=max(1, limit), topic_hint=topic_hint)

    def run_model_maintenance(self, payload: dict) -> dict:
        return run_model_maintenance(
            models_root=str(payload.get("models_root", "models")),
            archive_dir=str(payload.get("archive_dir", "data/model_archive")),
            keep_recent=max(0, int(payload.get("keep_recent", 2))),
            keep_names_csv=str(payload.get("keep_names", "")),
            checkpoint_keep_recent=max(0, int(payload.get("checkpoint_keep_recent", 4))),
            checkpoint_ttl_days=max(0.0, float(payload.get("checkpoint_ttl_days", 14.0))),
            compress_level=max(1, min(9, int(payload.get("compress_level", 6)))),
            apply=bool(payload.get("apply", False)),
            include_candidates=bool(payload.get("include_candidates", True)),
            candidates_dirname=str(payload.get("candidates_dirname", "candidates")),
            candidates_keep_recent=max(0, int(payload.get("candidates_keep_recent", 1))),
        )

    def run_training_maintenance(self, payload: dict) -> dict:
        return run_training_maintenance(
            training_root=str(payload.get("training_root", self.training_root)),
            archive_dir=str(payload.get("archive_dir", "")) or None,
            min_age_days=max(0.0, float(payload.get("min_age_days", 7.0))),
            min_size_bytes=max(1, int(payload.get("min_size_bytes", 5_000_000))),
            compress_level=max(1, min(9, int(payload.get("compress_level", 6)))),
            max_queue_lines=max(0, int(payload.get("max_queue_lines", 20000))),
            apply=bool(payload.get("apply", False)),
        )

    def run_data_maintenance(self, payload: dict) -> dict:
        return run_data_maintenance(
            data_root=str(payload.get("data_root", "data")),
            continual_dir=str(payload.get("continual_dir", "data/continual")),
            tasks_dir=str(payload.get("tasks_dir", "data/tasks")),
            runtime_dir=str(payload.get("runtime_dir", "data/runtime")),
            maintenance_dir=str(payload.get("maintenance_dir", "data/maintenance")),
            continual_keep_recent=max(0, int(payload.get("continual_keep_recent", 5))),
            tasks_keep_recent=max(0, int(payload.get("tasks_keep_recent", 20))),
            runtime_ttl_days=max(0.0, float(payload.get("runtime_ttl_days", 7.0))),
            maintenance_ttl_days=max(0.0, float(payload.get("maintenance_ttl_days", 30.0))),
            compress_level=max(1, min(9, int(payload.get("compress_level", 6)))),
            apply=bool(payload.get("apply", False)),
        )

    def _extract_task_topic(self, row: dict[str, Any]) -> str:
        payload = row.get("payload", {}) if isinstance(row.get("payload"), dict) else {}
        topic = re.sub(r"\s+", " ", str(payload.get("topic", "")).strip())
        if topic:
            topic = re.sub(r"\s+from the internet$", "", topic, flags=re.IGNORECASE).strip(" .,:;!?")
            return topic
        task_type = str(row.get("task_type", "")).strip().lower()
        if task_type != "continual_guard_step":
            return ""
        new_corpus = str(payload.get("new_corpus", "")).strip()
        if not new_corpus:
            return ""
        stem = Path(new_corpus).stem
        stem = re.sub(r"[_\-]+", " ", stem).strip()
        if stem.lower() in {"queued wikipedia learning", "continual mix"}:
            return ""
        return stem

    def _topic_outcome_score(self, row: dict[str, Any]) -> float | None:
        status = str(row.get("status", "")).strip().lower()
        task_type = str(row.get("task_type", "")).strip().lower()
        result = row.get("result", {}) if isinstance(row.get("result"), dict) else {}
        if status in {"queued", "running"}:
            return None

        error_text = str(row.get("error", "")).strip()
        is_fatal = bool(error_text and _is_fatal_error(RuntimeError(error_text)))

        base_by_status = {
            "completed": 0.55,
            "failed": 0.2,
            "cancelled": 0.25,
        }
        if status == "failed" and is_fatal:
            score = 0.5
        else:
            score = base_by_status.get(status, 0.4)

        if task_type == "evaluate_model":
            candidate_avg = float(result.get("candidate_avg_score", 0.0))
            delta = float(result.get("delta", 0.0))
            ui_pass = bool(result.get("ui_pass", False))
            model_only_pass = bool(result.get("model_only_pass", False))
            score = 0.35 + (candidate_avg * 0.45) + (max(-0.2, min(0.2, delta)) + 0.2) * 0.3
            if ui_pass:
                score += 0.08
            if model_only_pass:
                score += 0.05
        elif task_type == "continual_guard_step":
            if is_fatal and status == "failed":
                score = 0.5
            else:
                passed = bool(result.get("passed", False))
                new_gain = float(result.get("new_gain", 0.0))
                core_drop = float(result.get("core_drop", 0.0))
                score = 0.5 + (new_gain * 1.3) - max(0.0, core_drop) * 1.2
                score += 0.12 if passed else -0.08
        elif task_type in {"learn_wikipedia_topic", "learn_web_topic"}:
            evidence = result.get("evidence_policy", {}) if isinstance(result.get("evidence_policy"), dict) else {}
            stats = evidence.get("stats", {}) if isinstance(evidence.get("stats"), dict) else {}
            kept = float(stats.get("kept_facts", 0.0))
            total = float(stats.get("input_facts", 0.0))
            ratio = (kept / total) if total > 0 else 0.0
            score = 0.45 + min(0.45, ratio * 0.45)

        return max(0.0, min(1.0, float(score)))

    def _build_topic_mastery_map(self, limit: int = 500) -> dict[str, Any]:
        tasks = self.tasks.list_tasks(limit=max(50, int(limit)))
        rows: dict[str, dict[str, Any]] = {}
        for task in tasks:
            if not self._is_training_task(task):
                continue
            topic = self._extract_task_topic(task)
            if not topic:
                continue
            score = self._topic_outcome_score(task)
            if score is None:
                continue
            key = topic.lower()
            bucket = rows.setdefault(
                key,
                {
                    "topic": topic,
                    "scores": [],
                    "task_count": 0,
                    "completed": 0,
                    "failed": 0,
                    "cancelled": 0,
                    "recent_task_type": "",
                    "recent_status": "",
                    "updated_at_utc": "",
                },
            )
            bucket["scores"].append(float(score))
            bucket["task_count"] = int(bucket["task_count"]) + 1
            status = str(task.get("status", "")).strip().lower()
            if status in {"completed", "failed", "cancelled"}:
                bucket[status] = int(bucket.get(status, 0)) + 1
            if not bucket["updated_at_utc"]:
                bucket["updated_at_utc"] = str(task.get("updated_at_utc", "")).strip()
                bucket["recent_task_type"] = str(task.get("task_type", "")).strip()
                bucket["recent_status"] = status

        topic_rows: list[dict[str, Any]] = []
        for _, bucket in rows.items():
            scores = [float(x) for x in list(bucket.get("scores") or [])]
            if not scores:
                continue
            avg_score = sum(scores) / max(1, len(scores))
            row = {
                "topic": bucket["topic"],
                "task_count": int(bucket["task_count"]),
                "avg_score": round(avg_score, 4),
                "completed": int(bucket["completed"]),
                "failed": int(bucket["failed"]),
                "cancelled": int(bucket["cancelled"]),
                "recent_task_type": str(bucket.get("recent_task_type", "")),
                "recent_status": str(bucket.get("recent_status", "")),
                "updated_at_utc": str(bucket.get("updated_at_utc", "")),
            }
            topic_rows.append(row)
        topic_rows.sort(key=lambda row: (float(row.get("avg_score", 0.0)), -int(row.get("task_count", 0))))
        weak = [row["topic"] for row in topic_rows if float(row.get("avg_score", 0.0)) < 0.5][:18]
        strong = [row["topic"] for row in topic_rows if float(row.get("avg_score", 0.0)) >= 0.7][:18]
        return {
            "topic_count": len(topic_rows),
            "topics": topic_rows,
            "weak_topics": weak,
            "strong_topics": strong,
        }

    def _plan_autonomous_topics(
        self,
        *,
        model: str,
        desired_count: int,
        benchmark_file: str,
        weakness_share: float,
        include_random_exploration: bool,
        avoid_known_topics: bool,
    ) -> tuple[list[str], dict[str, Any]]:
        target = max(1, int(desired_count))
        weakness_target = max(1, min(target, int(round(target * max(0.0, min(1.0, weakness_share))))))
        mastery = self._build_topic_mastery_map(limit=600)
        known_topics = _collect_known_training_topics(self.training_root) if avoid_known_topics else []
        if avoid_known_topics:
            historical = [str(r.get("topic", "")).strip() for r in list(mastery.get("topics") or []) if str(r.get("topic", "")).strip()]
            known_topics = _dedupe_preserve_order(known_topics + historical)
        subject_catalog = _load_subject_catalog(self.training_root)
        collected: list[str] = []
        plan_meta: dict[str, Any] = {
            "strategy": "autonomous_hybrid",
            "model": model,
            "desired_count": target,
            "weakness_target": weakness_target,
            "benchmark_file": benchmark_file,
            "avoid_known_topics": bool(avoid_known_topics),
            "known_topic_count": len(known_topics),
            "known_topics_preview": known_topics[:12],
            "benchmark_eval": {},
            "subject_catalog_version": str(subject_catalog.get("version", "0.1")),
            "subject_seed_planning": {},
            "random_exploration": {"enabled": bool(include_random_exploration), "selected_titles": []},
            "topic_mastery": {
                "topic_count": int(mastery.get("topic_count", 0)),
                "weak_topics": list(mastery.get("weak_topics") or [])[:12],
                "strong_topics": list(mastery.get("strong_topics") or [])[:12],
            },
        }

        mastery_weak_topics = [
            str(t).strip()
            for t in list(mastery.get("weak_topics") or [])
            if str(t).strip()
        ]
        for topic in mastery_weak_topics[:weakness_target]:
            if avoid_known_topics and _topic_is_covered(topic, known_topics):
                continue
            collected.append(topic)

        bench_path = Path(benchmark_file)
        if bench_path.exists():
            cases = _load_benchmark_cases(bench_path)
            if cases:
                benchmark = run_benchmark(
                    model=model,
                    cases=cases,
                    enable_memory=False,
                    enable_web_tools=False,
                )
                rows = list(benchmark.get("rows", []))
                rows.sort(key=lambda row: float(row.get("score", 0.0)))
                selected_weak: list[dict[str, Any]] = []
                skipped_covered = 0
                for row in rows:
                    topic = _topic_from_benchmark_row(row if isinstance(row, dict) else {})
                    if not topic:
                        continue
                    if avoid_known_topics and _topic_is_covered(topic, known_topics):
                        skipped_covered += 1
                        continue
                    collected.append(topic)
                    selected_weak.append(
                        {
                            "name": str(row.get("name", "")),
                            "score": float(row.get("score", 0.0)),
                            "topic": topic,
                        }
                    )
                    if len(selected_weak) >= weakness_target:
                        break
                plan_meta["benchmark_eval"] = {
                    "avg_score": float(benchmark.get("avg_score", 0.0)),
                    "case_count": int(benchmark.get("case_count", 0)),
                    "selected_weak_cases": selected_weak,
                    "skipped_as_already_covered": skipped_covered,
                }
        else:
            plan_meta["benchmark_eval"] = {"warning": f"benchmark file not found: {bench_path}"}

        subject_seed_count = max(0, target - len(_dedupe_preserve_order(collected)))
        if subject_seed_count > 0:
            subject_seed_topics, subject_plan = _pick_subject_seed_topics(
                catalog=subject_catalog,
                known_topics=known_topics,
                max_topics=subject_seed_count,
            )
            collected.extend(subject_seed_topics)
            plan_meta["subject_seed_planning"] = {
                **subject_plan,
                "selected_topics": subject_seed_topics,
            }

        if include_random_exploration and len(collected) < target:
            need = max(1, target - len(_dedupe_preserve_order(collected)))
            random_titles: list[str] = []
            try:
                random_titles = _fetch_random_wikipedia_titles(limit=max(need * 2, 4), timeout_sec=15)
            except Exception as exc:  # noqa: BLE001
                plan_meta["random_exploration"]["warning"] = f"random title fetch failed: {exc}"
            random_filtered = [
                t
                for t in random_titles
                if not avoid_known_topics or not _topic_is_covered(t, known_topics)
            ]
            collected.extend(random_filtered)
            plan_meta["random_exploration"]["selected_titles"] = random_filtered
            plan_meta["random_exploration"]["skipped_as_already_covered"] = max(
                0, len(random_titles) - len(random_filtered)
            )

        fallback_pool = [
            t
            for t in AUTONOMOUS_TOPIC_FALLBACKS
            if not avoid_known_topics or not _topic_is_covered(t, known_topics)
        ]
        collected.extend(fallback_pool)
        selected_topics = _dedupe_preserve_order(collected)[:target]
        plan_meta["selected_topics"] = selected_topics
        return selected_topics, plan_meta

    def queue_research_training_program(self, payload: dict) -> dict:
        flags = self.flags.get_flags()
        if not bool(flags.get("allow_auto_training_tasks", False)):
            raise PermissionError(
                "Training program requires allow_auto_training_tasks=true in runtime flags."
            )
        default_model = self._resolve_default_model()
        baseline_model = str(payload.get("baseline_model") or default_model)
        topics = [str(x).strip() for x in payload.get("topics", []) if str(x).strip()]

        source_mode = str(payload.get("source_mode", "mixed")).strip().lower()
        if source_mode not in {"mixed", "wikipedia", "web"}:
            source_mode = "mixed"

        requested_passes = int(payload.get("passes", min(2, len(topics) or 2)))
        passes = max(1, min(8, requested_passes))
        auto_topics_meta: dict[str, Any] | None = None
        if not topics:
            benchmark_file = str(payload.get("benchmark_file") or "data/maintenance/user_chat_benchmark.json").strip()
            weakness_share = float(payload.get("weakness_share", 0.30))
            include_random_exploration = bool(payload.get("include_random_exploration", True))
            avoid_known_topics = bool(payload.get("avoid_known_topics", True))
            topics, auto_topics_meta = self._plan_autonomous_topics(
                model=baseline_model,
                desired_count=passes,
                benchmark_file=benchmark_file,
                weakness_share=weakness_share,
                include_random_exploration=include_random_exploration,
                avoid_known_topics=avoid_known_topics,
            )

        selected_topics = topics[:passes]
        max_attempts = max(1, int(payload.get("max_attempts", 3)))
        out_prefix = str(payload.get("out_prefix", "models/ickle_program")).strip() or "models/ickle_program"
        promote_to_model = str(payload.get("promote_to_model") or default_model)
        steps = max(100, int(payload.get("steps", 1000)))
        resource_budget = self._resolve_resource_budget(payload)
        parallel_slots = max(1, min(4, int(payload.get("parallel_slots", 1))))
        per_task_budget = dict(resource_budget)
        if parallel_slots > 1:
            per_task_budget = _normalize_resource_budget(
                {
                    "cpu_percent": max(10, int(resource_budget.get("cpu_percent", 70) / parallel_slots)),
                    "ram_percent": max(10, int(resource_budget.get("ram_percent", 70) / parallel_slots)),
                    "gpu_percent": (
                        max(10, int(resource_budget.get("gpu_percent", 0) / parallel_slots))
                        if bool(self.hardware_info.get("gpu", {}).get("available", False))
                        else 0
                    ),
                },
                defaults=resource_budget,
                gpu_available=bool(self.hardware_info.get("gpu", {}).get("available", False)),
            )
        unrestricted = bool(payload.get("unrestricted", source_mode in {"web", "mixed"}))
        max_urls = max(3, min(24, int(payload.get("max_urls", 10))))
        max_pages = max(3, min(18, int(payload.get("max_pages", 10))))
        quiz_size = max(4, min(12, int(payload.get("quiz_size", 8))))

        queued: list[dict] = []
        for idx, topic in enumerate(selected_topics, start=1):
            model_out = f"{out_prefix}_{idx}_{topic.replace(' ', '_').replace('/', '_')}.pt"
            root_task_type = "learn_wikipedia_topic"
            learning_payload: dict = {
                "topic": topic,
                "auto_pipeline": True,
                "steps": steps,
                "resource_budget": per_task_budget,
                "checkpoint_every_steps": int(payload.get("checkpoint_every_steps", 40)),
                "quiz_size": quiz_size,
                "eval_max_pages": quiz_size,
                "min_delta": float(payload.get("min_delta", -0.01)),
                "min_candidate_avg": float(payload.get("min_candidate_avg", 0.2)),
                "require_model_only_pass": True,
                "model_only_min_delta": float(payload.get("model_only_min_delta", 0.0)),
                "model_only_min_candidate_avg": float(payload.get("model_only_min_candidate_avg", 0.08)),
                "promote_if_pass": True,
                "promote_to_model": promote_to_model,
                "baseline_model": baseline_model,
                "out_model": model_out,
                "extended_eval": bool(payload.get("extended_eval", True)),
                "use_continual_guard": bool(payload.get("use_continual_guard", True)),
            }
            if source_mode in {"web", "mixed"}:
                root_task_type = "learn_web_topic"
                learning_payload.update(
                    {
                        "include_general_web": bool(payload.get("include_general_web", True)),
                        "max_urls": max_urls,
                        "include_wikipedia": True,
                        "include_news": True,
                        "max_web_results": max_urls,
                        "max_wiki_pages": max_pages,
                        "max_news_results": max_pages,
                        "unrestricted": unrestricted,
                    }
                )
            else:
                learning_payload["max_pages"] = max_pages

            task_req = {
                "task_type": root_task_type,
                "payload": learning_payload,
                "idempotency_key": f"program:{source_mode}:{idx}:{topic}",
                "max_attempts": max_attempts,
            }
            queued.append(self.create_task(task_req))
        return {
            "program_topics": selected_topics,
            "source_mode": source_mode,
            "autonomous_topic_plan": auto_topics_meta or {},
            "resource_budget": resource_budget,
            "per_task_resource_budget": per_task_budget,
            "parallel_slots": parallel_slots,
            "queued_roots": queued,
        }

    def run_open_dataset_ingest(self, payload: dict) -> dict:
        preset = str(payload.get("preset", "fineweb_edu")).strip() or "fineweb_edu"
        max_records = max(1, int(payload.get("max_records", 500)))
        out_path = str(payload.get("out", "")).strip()
        cleanup = bool(payload.get("cleanup_temp_cache", True))

        cmd = [
            "python",
            "-m",
            "src.app",
            "open-dataset-ingest",
            "--preset",
            preset,
            "--max-records",
            str(max_records),
            "--json",
        ]
        if out_path:
            cmd.extend(["--out", out_path])
        if cleanup:
            cmd.append("--cleanup-temp-cache")

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        output = str(proc.stdout or "")
        if proc.returncode != 0:
            raise RuntimeError(f"open-dataset-ingest failed (exit {proc.returncode}):\n{output[-1200:]}")
        return {
            "success": True,
            "command": cmd,
            "output_tail": output[-2200:],
        }

    def run_chat(self, payload: dict) -> dict:
        # Delegates to serve_web.py's ChatRuntime rather than maintaining a
        # second copy of the args-building logic. This used to be its own,
        # independently-drifted implementation missing thinking_mode,
        # agent_mode, allow_code_execution, and image_base64 support --
        # everything added to the chat/agent path this session -- because
        # nothing kept the two in sync. RuntimeFlagsStore is file-backed
        # (data/runtime_flags.json), so a second instance stays consistent
        # with this class's own self.flags rather than diverging in memory.
        return self._chat_runtime.run_chat(payload)

    def create_task(self, payload: dict) -> dict:
        task_type = str(payload.get("task_type", "")).strip()
        task_payload = payload.get("payload", {})
        idempotency_key = str(payload.get("idempotency_key", "")).strip()
        max_attempts = int(payload.get("max_attempts", 3))
        if not task_type:
            raise ValueError("Missing task_type")
        if not isinstance(task_payload, dict):
            raise ValueError("payload must be an object")
        task_payload = dict(task_payload)

        if task_type == "train_model":
            resource_budget = self._resolve_resource_budget(task_payload)
            derived = self._derive_training_runtime_overrides(
                resource_budget=resource_budget,
            )
            task_payload.setdefault("torch_threads", int(derived["torch_threads"]))
            task_payload.setdefault("batch_size", int(derived["batch_size"]))
            task_payload.setdefault("grad_accum_steps", int(derived["grad_accum_steps"]))
            task_payload.setdefault("resource_budget", resource_budget)

        pipeline_requested = task_type in {"learn_wikipedia_topic", "learn_web_topic"} and bool(
            task_payload.get("auto_pipeline", False)
        )
        if pipeline_requested:
            flags = self.flags.get_flags()
            if not bool(flags.get("allow_auto_training_tasks", False)):
                raise PermissionError(
                    "Auto pipeline requested, but allow_auto_training_tasks is disabled by runtime flags."
                )

        task = self.tasks.add_task(
            task_type=task_type,
            payload=task_payload,
            idempotency_key=idempotency_key,
            max_attempts=max_attempts,
        )
        task_out = dict(task)

        if pipeline_requested:
            followups: list[dict] = []
            clean_corpus_path = self._training_corpus_path("ickle_clean_corpus.txt")
            curated_corpus_path = self._training_corpus_path("ickle_curated_only.txt")
            build_payload = {
                "training_root": self.training_root,
                "out_path": clean_corpus_path,
                "max_lines": int(task_payload.get("build_max_lines", 22000)),
                "dictionary_items": int(task_payload.get("dictionary_items", 0)),
                "include_open_stream": bool(task_payload.get("include_open_stream", False)),
            }
            use_continual_guard = bool(task_payload.get("use_continual_guard", True))
            pipeline_data_mode = str(task_payload.get("pipeline_data_mode", "topic_queue")).strip().lower()
            if pipeline_data_mode not in {"topic_queue", "clean_corpus"}:
                pipeline_data_mode = "topic_queue"
            default_out_model = _default_auto_pipeline_model_path(
                task_id=str(task_out["task_id"]),
                topic=str(task_payload.get("topic", "")),
            )
            out_model = str(task_payload.get("out_model") or default_out_model)
            default_model = self._resolve_default_model()
            baseline_model = str(task_payload.get("baseline_model") or default_model)
            promote_to_model = str(task_payload.get("promote_to_model") or default_model)
            resource_budget = self._resolve_resource_budget(task_payload)
            training_overrides = self._derive_training_runtime_overrides(
                resource_budget=resource_budget,
            )
            topic_queue_path = str(
                task_payload.get("topic_queue_path")
                or _default_topic_queue_path(self.training_root, str(task_payload.get("topic", "")))
            )
            rebuild_clean_corpus = bool(
                task_payload.get("rebuild_clean_corpus", pipeline_data_mode == "clean_corpus")
            )

            dependency_task_id = str(task_out["task_id"])
            if rebuild_clean_corpus:
                build_task = self.tasks.add_task(
                    task_type="build_clean_corpus",
                    payload=build_payload,
                    depends_on=[task_out["task_id"]],
                    idempotency_key=f"{task_out['task_id']}:build_clean_corpus",
                    max_attempts=max_attempts,
                )
                followups.append(build_task)
                dependency_task_id = str(build_task["task_id"])

            if use_continual_guard:
                default_new_corpus = topic_queue_path if pipeline_data_mode == "topic_queue" else build_payload["out_path"]
                guard_payload = {
                    "core_corpus": str(task_payload.get("core_corpus") or curated_corpus_path),
                    "new_corpus": str(task_payload.get("new_corpus") or default_new_corpus),
                    "replay_buffer": str(task_payload.get("replay_buffer") or "data/continual/replay_buffer.jsonl"),
                    "mixed_corpus_out": str(task_payload.get("mixed_corpus_out") or "data/continual/continual_mix.txt"),
                    "baseline_model": baseline_model,
                    "out_model": out_model,
                    "checkpoint_path": str(task_payload.get("checkpoint_path") or f"{out_model}.checkpoint.pt"),
                    "promote_if_pass": bool(task_payload.get("promote_if_pass", True)),
                    "promote_to_model": promote_to_model,
                    "report_path": str(task_payload.get("guard_report_path") or ""),
                    "steps": int(task_payload.get("steps", 450)),
                    "lr": float(task_payload.get("lr", 3e-6)),
                    "warmup_steps": int(task_payload.get("warmup_steps", 60)),
                    "resume_if_possible": bool(task_payload.get("resume_if_possible", True)),
                    "torch_threads": int(task_payload.get("torch_threads", training_overrides["torch_threads"])),
                    "batch_size": int(task_payload.get("batch_size", training_overrides["batch_size"])),
                    "grad_accum_steps": int(
                        task_payload.get("grad_accum_steps", training_overrides["grad_accum_steps"])
                    ),
                    "resource_budget": resource_budget,
                    "replay_max_size": int(task_payload.get("replay_max_size", 20000)),
                    "total_pairs": int(task_payload.get("total_pairs", 12000)),
                    "max_core_pairs": int(task_payload.get("max_core_pairs", 6000)),
                    "max_new_pairs": int(task_payload.get("max_new_pairs", 10000)),
                    "core_ratio": float(task_payload.get("core_ratio", 0.50)),
                    "replay_ratio": float(task_payload.get("replay_ratio", 0.30)),
                    "new_ratio": float(task_payload.get("new_ratio", 0.20)),
                    "eval_core_prompts": int(task_payload.get("eval_core_prompts", 16)),
                    "eval_new_prompts": int(task_payload.get("eval_new_prompts", 16)),
                    "max_core_drop": float(task_payload.get("max_core_drop", 0.03)),
                    "min_new_gain": float(task_payload.get("min_new_gain", 0.0)),
                    "min_core_score": float(task_payload.get("min_core_score", 0.38)),
                    "min_core_quality": float(task_payload.get("min_core_quality", 0.35)),
                    "user_benchmark_file": str(
                        task_payload.get("user_benchmark_file", "data/maintenance/user_chat_benchmark.json")
                    ),
                    "min_user_delta": float(task_payload.get("min_user_delta", 0.0)),
                    "min_user_score": float(task_payload.get("min_user_score", -1.0)),
                    "min_user_case_score": float(task_payload.get("min_user_case_score", -1.0)),
                    "max_user_evasive": int(task_payload.get("max_user_evasive", -1)),
                    "auto_include_smart_corpus": bool(task_payload.get("auto_include_smart_corpus", True)),
                    "auto_build_focus_corpus": bool(task_payload.get("auto_build_focus_corpus", True)),
                    "focus_corpus_path": str(task_payload.get("focus_corpus_path") or "data/continual/conversation_focus.txt"),
                    "training_root": self.training_root,
                }
                guard_task = self.tasks.add_task(
                    task_type="continual_guard_step",
                    payload=guard_payload,
                    depends_on=[dependency_task_id],
                    idempotency_key=f"{task_out['task_id']}:continual_guard_step",
                    max_attempts=max_attempts,
                )
                followups.append(guard_task)
            else:
                default_data_path = topic_queue_path if pipeline_data_mode == "topic_queue" else clean_corpus_path
                train_payload = {
                    "data_path": str(task_payload.get("data_path") or default_data_path),
                    "out_model": out_model,
                    "steps": int(task_payload.get("steps", 900)),
                    "checkpoint_every_steps": int(task_payload.get("checkpoint_every_steps", 30)),
                    "init_model": str(task_payload.get("init_model") or baseline_model or default_model),
                    "resume_if_possible": True,
                    "torch_threads": int(task_payload.get("torch_threads", training_overrides["torch_threads"])),
                    "batch_size": int(task_payload.get("batch_size", training_overrides["batch_size"])),
                    "grad_accum_steps": int(
                        task_payload.get("grad_accum_steps", training_overrides["grad_accum_steps"])
                    ),
                    "resource_budget": resource_budget,
                }
                evaluate_payload = {
                    "topic": str(task_payload.get("topic", "")),
                    "candidate_model": train_payload["out_model"],
                    "baseline_model": baseline_model,
                    "quiz_size": int(task_payload.get("quiz_size", 6)),
                    "max_pages": int(task_payload.get("eval_max_pages", task_payload.get("quiz_size", 6))),
                    "min_delta": float(task_payload.get("min_delta", -0.02)),
                    "min_candidate_avg": float(task_payload.get("min_candidate_avg", 0.18)),
                    "require_model_only_pass": bool(task_payload.get("require_model_only_pass", True)),
                    "model_only_min_delta": float(task_payload.get("model_only_min_delta", 0.0)),
                    "model_only_min_candidate_avg": float(task_payload.get("model_only_min_candidate_avg", 0.05)),
                    "promote_if_pass": bool(task_payload.get("promote_if_pass", True)),
                    "promote_to_model": promote_to_model,
                    "eval_enable_memory": bool(task_payload.get("eval_enable_memory", True)),
                    "eval_enable_web_tools": bool(task_payload.get("eval_enable_web_tools", False)),
                    "extended_eval": bool(task_payload.get("extended_eval", True)),
                }
                train_task = self.tasks.add_task(
                    task_type="train_model",
                    payload=train_payload,
                    depends_on=[dependency_task_id],
                    idempotency_key=f"{task_out['task_id']}:train_model",
                    max_attempts=max_attempts,
                )
                eval_task = self.tasks.add_task(
                    task_type="evaluate_model",
                    payload=evaluate_payload,
                    depends_on=[train_task["task_id"]],
                    idempotency_key=f"{task_out['task_id']}:evaluate_model",
                    max_attempts=max_attempts,
                )
                followups.append(train_task)
                followups.append(eval_task)
            task_out["auto_pipeline_followups"] = followups
            task_out["training_resource_overrides"] = training_overrides
            task_out["pipeline_data_mode"] = pipeline_data_mode
            task_out["topic_queue_path"] = topic_queue_path

        self._sync_worker()
        return task_out

    def update_flags(self, updates: dict) -> dict:
        if not isinstance(updates, dict):
            raise ValueError("Flag updates must be an object")
        out = self.flags.update_flags(updates)
        self._sync_worker()
        return out

    def shutdown(self):
        if hasattr(self, "swarm") and self.swarm:
            self.swarm.stop()
        # No explicit save needed here: TaskQueue persists after every
        # mutation already (_save_tasks is called from add_task/cancel_task/
        # etc.), and there was never a public save() method -- this used to
        # call one anyway and crash on every graceful shutdown.
        self.tasks.stop_worker()

    def list_teaching_sessions(self) -> list[dict[str, Any]]:
        return TeacherStore().list_sessions()

    def get_teaching_session(self, session_id: str) -> dict[str, Any] | None:
        return TeacherStore().get_session(session_id)

    def get_teacher_corpus_stats(self) -> dict[str, Any]:
        sessions = TeacherStore().list_sessions()
        turn_count = sum(int(s.get("turn_count", 0)) for s in sessions)
        return {"session_count": len(sessions), "turn_count": turn_count}

    def get_dpo_prefs_stats(self, prefs_path: str = "data/dpo_prefs.jsonl") -> dict[str, Any]:
        p = Path(prefs_path)
        if not p.exists():
            return {"pair_count": 0, "exists": False}
        pair_count = 0
        try:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        pair_count += 1
        except OSError:
            return {"pair_count": 0, "exists": False}
        return {"pair_count": pair_count, "exists": True}

    def start_teaching_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        store = TeacherStore()
        session = store.start_session(
            topic=str(payload.get("topic", "")).strip(),
            source_model=str(payload.get("source_model", "")).strip(),
            tags=payload.get("tags") if isinstance(payload.get("tags"), list) else [],
        )
        return {
            "session_id": session.session_id,
            "topic": session.topic,
            "source_model": session.source_model,
            "tags": session.tags,
            "opened_at": session.opened_at,
            "status": session.status,
        }

    def add_teaching_turn(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return TeacherStore().add_turn(
            session_id=session_id,
            prompt=str(payload.get("prompt", "")).strip(),
            ickle_answer=str(payload.get("ickle_answer", "")).strip(),
            teacher_feedback=str(payload.get("teacher_feedback", "")).strip(),
            improved_answer=str(payload.get("improved_answer", "")).strip(),
            score=float(payload.get("score", 0.0)),
            tags=payload.get("tags") if isinstance(payload.get("tags"), list) else [],
            source_model=str(payload.get("source_model", "")).strip(),
        )

    def close_teaching_session(self, session_id: str) -> dict[str, Any]:
        return TeacherStore().close_session(session_id)

    def build_teacher_sft_corpus(self, payload: dict[str, Any]) -> dict[str, Any]:
        return TeacherStore().build_sft_corpus(
            out_path=str(payload.get("out_path", "data/teacher/teacher_sft_corpus.txt")).strip(),
            min_score=float(payload.get("min_score", 0.0)),
        )

    def build_teacher_prefs(self, payload: dict[str, Any]) -> dict[str, Any]:
        return TeacherStore().build_preference_pairs(
            out_path=str(payload.get("out_path", "data/teacher/teacher_prefs.jsonl")).strip(),
            min_score=float(payload.get("min_score", 0.0)),
        )

    def train_from_teacher(self, payload: dict[str, Any]) -> dict[str, Any]:
        store = TeacherStore()
        sft_path = str(payload.get("sft_out", "data/teacher/teacher_sft_corpus.txt")).strip()
        prefs_path = str(payload.get("prefs_out", "data/teacher/teacher_prefs.jsonl")).strip()
        min_score = float(payload.get("min_score", 0.3))

        sft_result = store.build_sft_corpus(out_path=sft_path, min_score=min_score)
        prefs_result = store.build_preference_pairs(out_path=prefs_path, min_score=min_score)

        if sft_result["pairs"] == 0 and prefs_result["pairs"] == 0:
            raise ValueError("No teacher training pairs available. Ensure teaching turns have improved_answer with score >= min_score.")

        baseline_model = str(payload.get("baseline_model") or self._resolve_default_model())
        out_model = str(payload.get("out_model") or "models/ickle_teacher_candidate.pt")

        guard_payload = {
            "core_corpus": str(payload.get("core_corpus") or self._training_corpus_path("ickle_curated_only.txt")),
            "new_corpus": sft_path,
            "replay_buffer": str(payload.get("replay_buffer") or "data/continual/replay_buffer.jsonl"),
            "mixed_corpus_out": str(payload.get("mixed_corpus_out") or "data/continual/teacher_mix.txt"),
            "baseline_model": baseline_model,
            "out_model": out_model,
            "checkpoint_path": str(payload.get("checkpoint_path") or f"{out_model}.checkpoint.pt"),
            "promote_to": str(payload.get("promote_to") or baseline_model),
            "promotion_gate": True,
            "promotion_report_path": str(payload.get("promotion_report_path") or "data/continual/teacher_gate_report.json"),
            "report_path": str(payload.get("report_path") or "data/continual/teacher_step_report.json"),
            "steps": int(payload.get("steps", 800)),
            "resource_budget": self._resolve_resource_budget(payload),
            "lr": float(payload.get("lr", 8e-6)),
            "warmup_steps": int(payload.get("warmup_steps", 80)),
            "torch_threads": int(payload.get("torch_threads", 0)),
            "replay_max_size": int(payload.get("replay_max_size", 20000)),
            "total_pairs": int(payload.get("total_pairs", 12000)),
            "max_core_pairs": int(payload.get("max_core_pairs", 6000)),
            "max_new_pairs": int(payload.get("max_new_pairs", 10000)),
            "core_ratio": float(payload.get("core_ratio", 0.45)),
            "replay_ratio": float(payload.get("replay_ratio", 0.25)),
            "new_ratio": float(payload.get("new_ratio", 0.30)),
            "eval_core_prompts": int(payload.get("eval_core_prompts", 16)),
            "eval_new_prompts": int(payload.get("eval_new_prompts", 16)),
            "max_core_drop": float(payload.get("max_core_drop", 0.03)),
            "min_new_gain": float(payload.get("min_new_gain", 0.0)),
            "min_core_score": float(payload.get("min_core_score", 0.38)),
            "min_core_quality": float(payload.get("min_core_quality", 0.35)),
            "user_benchmark_file": str(payload.get("user_benchmark_file", "data/maintenance/user_chat_benchmark.json")),
            "training_root": self.training_root,
        }

        task_type = "continual_guard_step"
        task = self.tasks.add_task(
            task_type=task_type,
            payload=guard_payload,
            idempotency_key=str(payload.get("idempotency_key", "")).strip(),
            max_attempts=int(payload.get("max_attempts", 3)),
        )
        self._sync_worker()
        result = dict(task)
        result["sft_corpus"] = sft_result
        result["dpo_prefs"] = prefs_result
        return result


class ControlHandler(IckleHTTPHandler):
    web_root = "."
    # The desktop app's chat page (served by serve_web.py, a different port)
    # calls this server's API for the Manage panel -- a different port is a
    # different origin under browser same-origin policy, so without CORS
    # headers every fetch() from that page is silently blocked by the
    # browser even though the server itself responds fine (curl/
    # Invoke-RestMethod don't enforce CORS, which is why this passed manual
    # endpoint checks but failed for real in a browser).
    enable_cors = True

    @property
    def runtime(self) -> ControlRuntime:
        return self.server.runtime  # type: ignore[attr-defined]

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if parsed.path == "/reality-check":
            self._send_json(200, [asdict(c) for c in collect_checks()])
            return
        if parsed.path == "/api/status":
            self._send_json(200, self.runtime.get_status())
            return
        if parsed.path == "/api/system/resources":
            self._send_json(200, self.runtime.get_system_resources())
            return
        if parsed.path == "/api/training/progress":
            limit = int((query.get("limit", ["25"])[0] or "25"))
            self._send_json(200, self.runtime.get_training_progress(limit=limit))
            return
        if parsed.path == "/api/models":
            limit = int((query.get("limit", ["80"])[0] or "80"))
            include_checkpoints = str((query.get("include_checkpoints", ["0"])[0] or "0")).strip() in {
                "1",
                "true",
                "yes",
            }
            all_models = str((query.get("all", ["0"])[0] or "0")).strip() in {"1", "true", "yes"}
            self._send_json(
                200,
                {
                    "models": self.runtime.list_models(
                        limit=limit,
                        include_checkpoints=include_checkpoints,
                        policy_only=not all_models,
                    )
                },
            )
            return
        if parsed.path == "/api/flags":
            self._send_json(200, self.runtime.flags.get_flags())
            return
        if parsed.path == "/api/tasks":
            self._send_json(200, {"tasks": self.runtime.tasks.list_tasks(limit=200)})
            return
        if parsed.path == "/api/memory/summary":
            self._send_json(200, self.runtime.get_memory_summary())
            return
        if parsed.path == "/api/memory/facts":
            limit = int((query.get("limit", ["40"])[0] or "40"))
            category = str((query.get("category", [""])[0] or "")).strip() or None
            self._send_json(200, {"facts": self.runtime.get_memory_facts(limit=limit, category=category)})
            return
        if parsed.path == "/api/memory/context":
            limit = int((query.get("limit", ["30"])[0] or "30"))
            self._send_json(200, {"context": self.runtime.get_memory_context(limit=limit)})
            return
        if parsed.path == "/api/memory/export":
            self._send_json(200, self.runtime.export_memory())
            return
        if parsed.path == "/api/research/sessions":
            limit = int((query.get("limit", ["20"])[0] or "20"))
            self._send_json(200, {"sessions": self.runtime.get_research_sessions(limit=limit)})
            return
        if parsed.path == "/api/research/find":
            q = str((query.get("query", [""])[0] or "")).strip()
            limit = int((query.get("limit", ["10"])[0] or "10"))
            topic_hint = str((query.get("topic_hint", [""])[0] or "")).strip() or None
            if not q:
                self._send_json(200, {"notes": []})
                return
            self._send_json(200, {"notes": self.runtime.find_research_notes(q, limit=limit, topic_hint=topic_hint)})
            return
        if parsed.path == "/api/workspace-check":
            self._send_json(200, {"checks": [asdict(c) for c in collect_workspace_checks()]})
            return
        if parsed.path == "/api/system/live":
            self._send_json(200, self.runtime.get_live_system_stats())
            return
        if parsed.path == "/api/federated/status":
            self._send_json(200, self.runtime.get_federated_status())
            return
        if parsed.path == "/api/codistill/status":
            self._send_json(200, self.runtime.get_codistill_status())
            return
        if parsed.path == "/api/commons/status":
            self._send_json(200, self.runtime.get_commons_status())
            return
        if parsed.path == "/api/consolidation/status":
            self._send_json(200, self.runtime.get_consolidation_status())
            return
        if parsed.path == "/api/disagreements/status":
            self._send_json(200, self.runtime.get_disagreement_status())
            return
        if parsed.path == "/api/contribution/status":
            self._send_json(200, self.runtime.get_contribution_status())
            return
        if parsed.path == "/api/deltas":
            self._send_json(200, {"deltas": self.runtime.list_knowledge_deltas()})
            return
        if parsed.path == "/api/torickle/bundles":
            self._send_json(200, {"bundles": self.runtime.get_torickle_bundles()})
            return
        if parsed.path == "/api/torickle/find":
            model_hash = str((query.get("model_hash", [""])[0] or "")).strip()
            results = self.runtime.find_torickle_bundles({"model_hash": model_hash})
            self._send_json(200, {"bundles": results})
            return
        if parsed.path == "/api/torickle/status":
            # Single source of truth for both the Network tab (join toggle +
            # peer count) and the Sharing tab (bundles) -- they're two views
            # onto the one real swarm, not two separate network features.
            self._send_json(200, self.runtime.get_swarm_status())
            return
        if parsed.path == "/api/teach/sessions":
            self._send_json(200, {"sessions": self.runtime.list_teaching_sessions()})
            return
        if parsed.path == "/api/teach/stats":
            self._send_json(200, self.runtime.get_teacher_corpus_stats())
            return
        if parsed.path == "/api/teach/dpo-stats":
            self._send_json(200, self.runtime.get_dpo_prefs_stats())
            return
        if parsed.path.startswith("/api/teach/sessions/"):
            parts = parsed.path.rstrip("/").split("/")
            if len(parts) == 5 and parts[-1] == "turns":
                session_id = parts[3]
                session = self.runtime.get_teaching_session(session_id)
                if session is None:
                    self._send_json(404, {"error": "Session not found"})
                else:
                    self._send_json(200, session)
                return
            else:
                session_id = parts[3]
                session = self.runtime.get_teaching_session(session_id)
                if session is None:
                    self._send_json(404, {"error": "Session not found"})
                else:
                    self._send_json(200, session)
                return
        return super().do_GET()

    def do_POST(self):  # noqa: N802
        parsed = urlparse(self.path)
        try:
            payload = self._read_json()
        except Exception as exc:  # noqa: BLE001
            self._send_json(400, {"error": f"Invalid JSON body: {exc}"})
            return

        try:
            if parsed.path == "/api/chat":
                self._send_json(200, self.runtime.run_chat(payload))
                return
            if parsed.path == "/api/flags":
                self._send_json(200, self.runtime.update_flags(payload))
                return
            if parsed.path == "/api/system/resources":
                self._send_json(200, self.runtime.update_system_resources(payload))
                return
            if parsed.path == "/api/tasks":
                self._send_json(200, self.runtime.create_task(payload))
                return
            if parsed.path == "/api/memory/export/save":
                self._send_json(200, self.runtime.save_memory_export())
                return
            if parsed.path == "/api/memory/search":
                query = str(payload.get("query", "")).strip()
                limit = int(payload.get("limit", 10))
                self._send_json(200, self.runtime.search_memory(query=query, limit=limit))
                return
            if parsed.path == "/api/memory/facts":
                self._send_json(200, self.runtime.add_memory_fact(payload))
                return
            if parsed.path == "/api/memory/clear":
                self._send_json(200, self.runtime.clear_memory(payload))
                return
            if parsed.path == "/api/research/find":
                query = str(payload.get("query", "")).strip()
                limit = int(payload.get("limit", 10))
                topic_hint = str(payload.get("topic_hint", "")).strip() or None
                if not query:
                    self._send_json(200, {"notes": []})
                    return
                self._send_json(
                    200,
                    {"notes": self.runtime.find_research_notes(query, limit=limit, topic_hint=topic_hint)},
                )
                return
            if parsed.path == "/api/tasks/infer":
                instruction = str(payload.get("instruction", "")).strip()
                inferred = infer_task_from_instruction(instruction)
                if not inferred:
                    self._send_json(200, {"inferred": None})
                    return
                if bool(payload.get("queue", False)):
                    task = self.runtime.create_task(inferred)
                    self._send_json(200, {"inferred": inferred, "queued_task": task})
                else:
                    self._send_json(200, {"inferred": inferred})
                return
            if parsed.path == "/api/programs/research-train":
                self._send_json(200, self.runtime.queue_research_training_program(payload))
                return
            if parsed.path == "/api/open-dataset-ingest":
                self._send_json(200, self.runtime.run_open_dataset_ingest(payload))
                return
            if parsed.path == "/api/maintenance/model":
                self._send_json(200, self.runtime.run_model_maintenance(payload))
                return
            if parsed.path == "/api/maintenance/training":
                self._send_json(200, self.runtime.run_training_maintenance(payload))
                return
            if parsed.path == "/api/maintenance/data":
                self._send_json(200, self.runtime.run_data_maintenance(payload))
                return
            if parsed.path == "/api/deltas/enable":
                self._send_json(200, self.runtime.set_knowledge_delta_enabled(str(payload.get("delta_id", "")).strip(), True))
                return
            if parsed.path == "/api/deltas/disable":
                self._send_json(200, self.runtime.set_knowledge_delta_enabled(str(payload.get("delta_id", "")).strip(), False))
                return
            if parsed.path == "/api/deltas/rollback":
                self._send_json(200, self.runtime.rollback_knowledge_delta(str(payload.get("delta_id", "")).strip()))
                return
            if parsed.path == "/api/deltas/remove":
                self._send_json(200, self.runtime.remove_knowledge_delta(str(payload.get("delta_id", "")).strip()))
                return
            if parsed.path == "/api/deltas/threshold":
                try:
                    threshold = float(payload.get("threshold", 0.6))
                except (TypeError, ValueError):
                    self._send_json(400, {"error": "threshold must be a number"})
                    return
                self._send_json(
                    200,
                    self.runtime.set_knowledge_delta_threshold(str(payload.get("delta_id", "")).strip(), threshold),
                )
                return
            if parsed.path == "/api/torickle/import":
                self._send_json(200, self.runtime.import_torickle_bundle(payload))
                return
            if parsed.path == "/api/torickle/announce":
                self._send_json(200, self.runtime.announce_torickle_bundle(payload))
                return
            if parsed.path == "/api/torickle/find":
                self._send_json(200, {"bundles": self.runtime.find_torickle_bundles(payload)})
                return
            if parsed.path == "/api/torickle/pull":
                self._send_json(200, self.runtime.pull_torickle_bundle(payload))
                return
            if parsed.path == "/api/torickle/reassemble":
                self._send_json(200, self.runtime.reassemble_torickle_bundle(payload))
                return
            if parsed.path == "/api/swarm/join":
                self._send_json(200, self.runtime.set_network_enabled(True))
                return
            if parsed.path == "/api/swarm/leave":
                self._send_json(200, self.runtime.set_network_enabled(False))
                return
            if parsed.path == "/api/swarm/refresh":
                self._send_json(200, self.runtime.refresh_public_swarm())
                return
            if parsed.path == "/api/swarm/peers/add":
                address = str(payload.get("address", "")).strip()
                if not address:
                    self._send_json(400, {"error": "Missing peer address (host:port)"})
                    return
                self._send_json(200, self.runtime.add_known_peer(address))
                return
            if parsed.path == "/api/swarm/peers/remove":
                address = str(payload.get("address", "")).strip()
                self._send_json(200, self.runtime.remove_known_peer(address))
                return
            if parsed.path == "/api/swarm/ask":
                self._send_json(200, self.runtime.ask_swarm(str(payload.get("prompt", ""))))
                return
            if parsed.path == "/api/swarm/feedback":
                self._send_json(200, self.runtime.rate_swarm_response(payload))
                return
            if parsed.path == "/api/commons/sync":
                self._send_json(200, self.runtime.sync_commons())
                return
            if parsed.path == "/api/commons/adopt":
                self._send_json(200, self.runtime.adopt_commons_event(payload))
                return
            if parsed.path == "/api/teach/sessions":
                self._send_json(201, self.runtime.start_teaching_session(payload))
                return
            if parsed.path.startswith("/api/teach/sessions/"):
                parts = parsed.path.rstrip("/").split("/")
                if len(parts) == 5 and parts[-1] == "turn":
                    session_id = parts[3]
                    self._send_json(201, self.runtime.add_teaching_turn(session_id, payload))
                    return
                if len(parts) == 5 and parts[-1] == "close":
                    session_id = parts[3]
                    self._send_json(200, self.runtime.close_teaching_session(session_id))
                    return
            if parsed.path == "/api/teach/corpus/build-sft":
                self._send_json(200, self.runtime.build_teacher_sft_corpus(payload))
                return
            if parsed.path == "/api/teach/corpus/build-prefs":
                self._send_json(200, self.runtime.build_teacher_prefs(payload))
                return
            if parsed.path == "/api/teach/train":
                self._send_json(200, self.runtime.train_from_teacher(payload))
                return
            if parsed.path.startswith("/api/tasks/") and parsed.path.endswith("/cancel"):
                task_id = parsed.path.split("/")[3]
                ok = self.runtime.tasks.cancel_task(task_id)
                self._send_json(200, {"cancelled": ok, "task_id": task_id})
                return
        except PermissionError as exc:
            self._send_json(403, {"error": str(exc)})
            return
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc(file=sys.stderr)
            self._send_json(400, {"error": friendly_error_message(exc)})
            return

        self._send_json(404, {"error": "Not found"})


def create_server(
    host: str = "127.0.0.1",
    port: int = 8788,
    web_root: str = "web",
    training_root: str = "",
) -> ThreadingHTTPServer:
    """Build (but don't start) a ready-to-serve control API server. Shared by
    the CLI entry point below and src/desktop_app.py, which runs this
    alongside the chat server in the same process so the desktop app has one
    window with both chat and management (training/tasks/models) instead of
    requiring a separate `mini` process for anything beyond chat."""
    from src.serve_web import resolve_web_root

    resolved_root = resolve_web_root(web_root)
    if not resolved_root.exists():
        raise SystemExit(f"Web root not found: {resolved_root}")

    runtime = ControlRuntime(training_root=training_root or str(get_training_root()))
    ControlHandler.web_root = str(resolved_root)
    server = ThreadingHTTPServer((host, port), ControlHandler)
    server.runtime = runtime  # type: ignore[attr-defined]
    return server


def shutdown_server(server: ThreadingHTTPServer):
    runtime = getattr(server, "runtime", None)
    if runtime is not None:
        runtime.shutdown()
    server.server_close()


def main():
    parser = argparse.ArgumentParser(description="Serve Ickle control API (training, tasks, maintenance, swarm)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument("--web-root", default="web")
    parser.add_argument("--training-root", default=str(get_training_root()))
    args = parser.parse_args()

    server = create_server(args.host, args.port, args.web_root, args.training_root)
    print(f"Ickle control API: http://{args.host}:{args.port}")
    print(
        "API: /api/status, /api/system/resources, /api/training/progress, /api/chat, /api/flags, /api/models, "
        "/api/tasks, /api/tasks/infer, /api/memory/summary, /api/memory/facts, /api/memory/context, "
        "/api/memory/export, /api/memory/export/save, /api/memory/search, /api/memory/clear, "
        "/api/swarm/join, /api/swarm/leave, /api/swarm/peers/add, /api/swarm/peers/remove, /api/swarm/ask, /api/swarm/feedback, "
        "/api/commons/status, /api/commons/sync, /api/commons/adopt, /api/consolidation/status, "
        "/api/disagreements/status, "
        "/api/research/sessions, /api/research/find, "
        "/api/programs/research-train, /api/open-dataset-ingest, /api/maintenance/model, "
        "/api/maintenance/training, /api/maintenance/data, /api/workspace-check, "
        "/api/torickle/bundles, /api/torickle/find, /api/torickle/status, "
        "/api/torickle/import, /api/torickle/announce, /api/torickle/pull, /api/torickle/reassemble, "
        "/api/federated/status, /api/contribution/status, "
        "/api/teach/sessions, /api/teach/sessions/<id>/turn, /api/teach/sessions/<id>/close, /api/teach/stats, "
        "/api/teach/corpus/build-sft, /api/teach/corpus/build-prefs, /api/teach/train"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping web console...")
    finally:
        shutdown_server(server)


if __name__ == "__main__":
    main()
