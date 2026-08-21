from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator
from urllib.request import Request, urlopen

from src.data_quality import dialogue_pair_fails_content_checks, dialogue_pair_has_structural_noise
from src.workspace_paths import get_training_root


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class DatasetPreset:
    dataset: str
    config: str | None
    split: str
    text_field: str
    format_kind: str
    license_hint: str
    note: str


PRESETS: dict[str, DatasetPreset] = {
    "fineweb": DatasetPreset(
        dataset="HuggingFaceFW/fineweb",
        config=None,
        split="train",
        text_field="text",
        format_kind="plain_text",
        license_hint="ODC-BY (dataset card)",
        note="Large Common Crawl-derived pretraining corpus.",
    ),
    "fineweb_edu": DatasetPreset(
        dataset="HuggingFaceFW/fineweb-edu",
        config=None,
        split="train",
        text_field="text",
        format_kind="plain_text",
        license_hint="ODC-BY (dataset card)",
        note="Education-focused filtered subset of FineWeb.",
    ),
    "oasst1": DatasetPreset(
        dataset="OpenAssistant/oasst1",
        config=None,
        split="train",
        text_field="text",
        format_kind="oasst_pairs",
        license_hint="Apache-2.0 (dataset card)",
        note="OpenAssistant human preference conversation trees.",
    ),
    "openhermes_2_5": DatasetPreset(
        dataset="teknium/OpenHermes-2.5",
        config=None,
        split="train",
        text_field="conversations",
        format_kind="openhermes_pairs",
        license_hint="Open compilation card (check source subsets before redistribution)",
        note="Large synthetic instruction/chat compilation.",
    ),
}


def check_training_internet(timeout_sec: int = 8) -> dict[str, Any]:
    targets = [
        ("huggingface", "https://huggingface.co"),
        ("wikipedia", "https://en.wikipedia.org/wiki/Main_Page"),
    ]
    checks: list[dict[str, Any]] = []
    for name, url in targets:
        ok = False
        error = ""
        try:
            req = Request(url, headers={"User-Agent": "IckleDatasetIngest/1.0"})
            with urlopen(req, timeout=max(2, int(timeout_sec))) as response:
                ok = int(getattr(response, "status", 200)) < 400
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
        checks.append({"name": name, "url": url, "ok": ok, "error": error or None})
    return {"ok": all(bool(item["ok"]) for item in checks), "checks": checks}


def _clean_text(text: str, max_chars: int) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip()
    if cleaned and cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def _extract_text_field(row: dict[str, Any], preferred_field: str, max_chars: int) -> str:
    if preferred_field in row and isinstance(row[preferred_field], str):
        return _clean_text(row[preferred_field], max_chars=max_chars)

    for value in row.values():
        if isinstance(value, str) and len(value) >= 80:
            return _clean_text(value, max_chars=max_chars)
    return ""


def _normalize_pair_text(text: str, *, max_chars: int) -> str:
    cleaned = _clean_text(text, max_chars=max_chars)
    if len(cleaned) < 20:
        return ""
    return cleaned


def _pair_is_low_quality(prompt: str, response: str) -> bool:
    p = str(prompt or "").strip()
    r = str(response or "").strip()
    if len(p) < 6 or len(r) < 20:
        return True
    if len(p) > 220 or len(r) > 280:
        return True
    if dialogue_pair_has_structural_noise(p, r):
        return True
    if p.lower() == r.lower():
        return True
    return dialogue_pair_fails_content_checks(p, r)


def _iter_plaintext_pairs(
    rows: Iterable[dict[str, Any]],
    *,
    text_field: str,
    max_chars_per_record: int,
) -> Iterator[tuple[str, str]]:
    prompt_templates = [
        "Share one useful factual chunk from your reading.",
        "Give one concise fact from this source.",
        "Provide a short, relevant takeaway from this text.",
    ]
    idx = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        text = _extract_text_field(row, preferred_field=text_field, max_chars=max_chars_per_record)
        if len(text) < 80:
            continue
        prompt = prompt_templates[idx % len(prompt_templates)]
        idx += 1
        yield prompt, text


def _iter_oasst_pairs(rows: Iterable[dict[str, Any]], *, max_chars_per_record: int) -> Iterator[tuple[str, str]]:
    seen_pairs: set[tuple[str, str]] = set()
    messages: dict[str, tuple[str, str]] = {}
    waiting_children: dict[str, list[str]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        role = str(row.get("role", "")).strip().lower()
        text = _normalize_pair_text(str(row.get("text", "")), max_chars=max_chars_per_record)
        if not text:
            continue

        message_id = str(row.get("message_id", "")).strip()
        parent_id = str(row.get("parent_id", "")).strip()
        if message_id:
            messages[message_id] = (role, text)

        if role in {"assistant", "gpt"} and parent_id:
            parent = messages.get(parent_id)
            if parent and parent[0] in {"prompter", "user", "human"}:
                pair = (parent[1], text)
                key = (pair[0].lower(), pair[1].lower())
                if key not in seen_pairs:
                    seen_pairs.add(key)
                    yield pair
            else:
                waiting_children.setdefault(parent_id, []).append(text)

        if role in {"prompter", "user", "human"} and message_id:
            pending = waiting_children.pop(message_id, [])
            for child_text in pending:
                pair = (text, child_text)
                key = (pair[0].lower(), pair[1].lower())
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                yield pair


def _iter_openhermes_pairs(
    rows: Iterable[dict[str, Any]],
    *,
    max_chars_per_record: int,
) -> Iterator[tuple[str, str]]:
    for row in rows:
        if not isinstance(row, dict):
            continue
        conv = row.get("conversations")
        if not isinstance(conv, list) or len(conv) < 2:
            continue
        pending_user = ""
        for msg in conv:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("from", "")).strip().lower()
            value = _normalize_pair_text(str(msg.get("value", "")), max_chars=max_chars_per_record)
            if not value:
                continue
            if role in {"human", "user", "prompter"}:
                pending_user = value
                continue
            if role in {"gpt", "assistant"} and pending_user:
                yield pending_user, value
                pending_user = ""


def _iter_pairs(
    rows: Iterable[dict[str, Any]],
    *,
    format_kind: str,
    text_field: str,
    max_chars_per_record: int,
) -> Iterator[tuple[str, str]]:
    kind = format_kind.strip().lower()
    if kind == "oasst_pairs":
        yield from _iter_oasst_pairs(rows, max_chars_per_record=max_chars_per_record)
        return
    if kind == "openhermes_pairs":
        yield from _iter_openhermes_pairs(rows, max_chars_per_record=max_chars_per_record)
        return
    yield from _iter_plaintext_pairs(rows, text_field=text_field, max_chars_per_record=max_chars_per_record)


def ingest_stream(
    *,
    dataset: str,
    config: str | None,
    split: str,
    text_field: str,
    format_kind: str,
    max_records: int,
    max_chars_per_record: int,
    out_path: Path,
) -> dict[str, Any]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "The 'datasets' package is required for open dataset streaming. "
            "Install with: pip install datasets"
        ) from exc

    stream = load_dataset(dataset, name=config, split=split, streaming=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written_records = 0
    lines_written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for prompt, response in _iter_pairs(
            stream,
            format_kind=format_kind,
            text_field=text_field,
            max_chars_per_record=max_chars_per_record,
        ):
            if _pair_is_low_quality(prompt, response):
                continue
            f.write(f"User: {prompt}\n")
            f.write(f"Ickle: {response}\n\n")
            written_records += 1
            lines_written += 3
            if written_records >= max_records:
                break

    return {
        "dataset": dataset,
        "config": config,
        "split": split,
        "text_field": text_field,
        "format_kind": format_kind,
        "max_records": max_records,
        "written_records": written_records,
        "lines_written": lines_written,
        "out_path": str(out_path.resolve()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Stream a bounded sample from an open LLM pretraining dataset into IckleTraining corpus format."
    )
    parser.add_argument("--preset", default="fineweb_edu", choices=sorted(PRESETS.keys()))
    parser.add_argument("--dataset", default="")
    parser.add_argument("--config", default="")
    parser.add_argument("--split", default="")
    parser.add_argument("--text-field", default="")
    parser.add_argument("--max-records", type=int, default=3000)
    parser.add_argument("--max-chars-per-record", type=int, default=260)
    parser.add_argument("--out", default="")
    parser.add_argument(
        "--cleanup-temp-cache",
        action="store_true",
        help="Use a temporary HF cache dir and delete it after ingestion.",
    )
    parser.add_argument("--check-internet", action="store_true", help="Run internet preflight checks before ingest.")
    parser.add_argument(
        "--require-internet",
        action="store_true",
        help="Fail immediately if internet preflight checks do not pass.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    preset = PRESETS[args.preset]
    dataset = args.dataset.strip() or preset.dataset
    config = args.config.strip() or preset.config
    split = args.split.strip() or preset.split
    text_field = args.text_field.strip() or preset.text_field

    training_root = get_training_root()
    default_out = training_root / f"open_{args.preset}_stream.txt"
    out_path = Path(args.out).resolve() if args.out else default_out.resolve()

    internet_status: dict[str, Any] | None = None
    if args.check_internet or args.require_internet:
        internet_status = check_training_internet()
        if args.require_internet and not internet_status.get("ok"):
            raise RuntimeError("Internet preflight failed. Training ingestion requires external access.")

    cache_root: Path | None = None
    old_env: dict[str, str | None] = {}
    if args.cleanup_temp_cache:
        tmp_parent = Path("data/.tmp")
        tmp_parent.mkdir(parents=True, exist_ok=True)
        cache_root = (tmp_parent / f"hf_stream_{uuid.uuid4().hex[:12]}").resolve()
        cache_root.mkdir(parents=True, exist_ok=False)
        for key in ("HF_HOME", "HF_DATASETS_CACHE", "HF_HUB_CACHE"):
            old_env[key] = os.environ.get(key)
            os.environ[key] = str(cache_root)

    try:
        report = ingest_stream(
            dataset=dataset,
            config=config,
            split=split,
            text_field=text_field,
            format_kind=preset.format_kind,
            max_records=max(1, int(args.max_records)),
            max_chars_per_record=max(120, int(args.max_chars_per_record)),
            out_path=out_path,
        )
        report["preset"] = args.preset
        report["license_hint"] = preset.license_hint
        report["note"] = preset.note
        report["timestamp_utc"] = _utc_now()
        if internet_status is not None:
            report["internet_status"] = internet_status

        meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
        meta_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        report["meta_path"] = str(meta_path.resolve())

        if args.json:
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return
        print(f"dataset: {report['dataset']}")
        print(f"config: {report['config']}")
        print(f"split: {report['split']}")
        print(f"text_field: {report['text_field']}")
        print(f"written_records: {report['written_records']}")
        print(f"lines_written: {report['lines_written']}")
        print(f"out_path: {report['out_path']}")
        print(f"meta_path: {report['meta_path']}")
        print(f"license_hint: {report['license_hint']}")
        if internet_status is not None:
            print(f"internet_ok: {internet_status.get('ok')}")
    finally:
        if cache_root and args.cleanup_temp_cache:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            shutil.rmtree(cache_root, ignore_errors=True)


if __name__ == "__main__":
    main()
