#!/usr/bin/env python3
"""Build a plain-text base language-model corpus from open datasets."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

from src.workspace_paths import get_training_root


def _clean_paragraph(text: str, max_chars: int) -> str:
    value = str(text or "")
    value = value.replace("\r", "\n")
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in value.split("\n")]
    lines = [ln for ln in lines if ln]
    if not lines:
        return ""

    merged = " ".join(lines)
    merged = re.sub(r"\s+", " ", merged).strip()
    if len(merged) > max_chars:
        merged = merged[:max_chars].rstrip()
    if len(merged) < 120:
        return ""
    if "user:" in merged.lower() and "ickle:" in merged.lower():
        return ""
    if re.search(r"\b(package main|def\s+\w+\(|class\s+\w+\(|import\s+\w+)\b", merged, flags=re.IGNORECASE):
        return ""
    alpha_ratio = sum(1 for ch in merged if ch.isalpha()) / max(1, len(merged))
    if alpha_ratio < 0.65:
        return ""
    return merged


def _iter_dataset_rows(
    *,
    dataset: str,
    config: str | None,
    split: str,
    text_field: str,
    max_records: int,
    row_filter: Any = None,
) -> Iterable[str]:
    from datasets import load_dataset  # imported lazily to keep optional dependency behavior

    stream = load_dataset(dataset, name=config, split=split, streaming=True)
    count = 0
    for row in stream:
        if not isinstance(row, dict):
            continue
        if row_filter is not None and not row_filter(row):
            continue
        value = row.get(text_field)
        if not isinstance(value, str):
            # Fallback: first long text field
            value = ""
            for v in row.values():
                if isinstance(v, str) and len(v) >= 120:
                    value = v
                    break
        if not value:
            continue
        yield value
        count += 1
        if count >= max_records:
            break


def _iter_local_assistant_rows(path: Path, max_records: int) -> Iterable[str]:
    if not path.exists() or max_records <= 0:
        return
    count = 0
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        lower = line.lower()
        if not (lower.startswith("ickle:") or lower.startswith("assistant:")):
            continue
        text = line.split(":", 1)[1].strip()
        if not text:
            continue
        yield text
        count += 1
        if count >= max_records:
            break


def build_base_corpus(
    *,
    out_path: Path,
    training_root: Path,
    max_local_records: int,
    max_wikitext: int,
    max_fineweb_edu: int,
    max_common_corpus: int,
    max_chars: int,
) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    written = 0
    source_counts: dict[str, int] = {
        "local_fineweb_edu_stream": 0,
        "local_fineweb_stream": 0,
        "local_oasst1_stream": 0,
        "local_openhermes_stream": 0,
        "wikitext": 0,
        "fineweb_edu": 0,
        "common_corpus": 0,
    }
    remote_errors: list[str] = []

    with out_path.open("w", encoding="utf-8") as f:
        local_sources = [
            ("local_fineweb_edu_stream", training_root / "open_fineweb_edu_stream.txt"),
            ("local_fineweb_stream", training_root / "open_fineweb_stream.txt"),
            ("local_oasst1_stream", training_root / "open_oasst1_stream.txt"),
            ("local_openhermes_stream", training_root / "open_openhermes_2_5_stream.txt"),
        ]
        for source_name, source_path in local_sources:
            for raw in _iter_local_assistant_rows(source_path, max_records=max_local_records):
                cleaned = _clean_paragraph(raw, max_chars=max_chars)
                if not cleaned:
                    continue
                key = cleaned.lower()
                if key in seen:
                    continue
                seen.add(key)
                f.write(cleaned + "\n\n")
                written += 1
                source_counts[source_name] += 1

        if max_wikitext > 0:
            try:
                for raw in _iter_dataset_rows(
                    dataset="wikitext",
                    config="wikitext-103-raw-v1",
                    split="train",
                    text_field="text",
                    max_records=max_wikitext,
                ):
                    cleaned = _clean_paragraph(raw, max_chars=max_chars)
                    if not cleaned:
                        continue
                    key = cleaned.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    f.write(cleaned + "\n\n")
                    written += 1
                    source_counts["wikitext"] += 1
            except Exception as exc:  # noqa: BLE001
                remote_errors.append(f"wikitext: {exc}")

        if max_fineweb_edu > 0:
            try:
                for raw in _iter_dataset_rows(
                    dataset="HuggingFaceFW/fineweb-edu",
                    config=None,
                    split="train",
                    text_field="text",
                    max_records=max_fineweb_edu,
                ):
                    cleaned = _clean_paragraph(raw, max_chars=max_chars)
                    if not cleaned:
                        continue
                    key = cleaned.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    f.write(cleaned + "\n\n")
                    written += 1
                    source_counts["fineweb_edu"] += 1
            except Exception as exc:  # noqa: BLE001
                remote_errors.append(f"fineweb_edu: {exc}")

        if max_common_corpus > 0:
            try:
                for raw in _iter_dataset_rows(
                    dataset="PleIAs/common_corpus",
                    config=None,
                    split="train",
                    text_field="text",
                    max_records=max_common_corpus,
                    row_filter=lambda row: str(row.get("language", "")).strip().lower() in ("english", "en"),
                ):
                    cleaned = _clean_paragraph(raw, max_chars=max_chars)
                    if not cleaned:
                        continue
                    key = cleaned.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    f.write(cleaned + "\n\n")
                    written += 1
                    source_counts["common_corpus"] += 1
            except Exception as exc:  # noqa: BLE001
                remote_errors.append(f"common_corpus: {exc}")

    report = {
        "out_path": str(out_path.resolve()),
        "paragraphs_written": written,
        "source_counts": source_counts,
    }
    if remote_errors:
        report["remote_errors"] = remote_errors
    return report


def main():
    parser = argparse.ArgumentParser(description="Build base plain-text corpus for stronger LM pretraining.")
    parser.add_argument("--out", default=str(get_training_root() / "base_lm_corpus.txt"))
    parser.add_argument("--training-root", default=str(get_training_root()))
    parser.add_argument("--max-local-records", type=int, default=25000)
    parser.add_argument("--max-wikitext", type=int, default=30000)
    parser.add_argument("--max-fineweb-edu", type=int, default=10000)
    parser.add_argument("--max-common-corpus", type=int, default=20000)
    parser.add_argument("--max-chars", type=int, default=900)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = build_base_corpus(
        out_path=Path(args.out),
        training_root=Path(args.training_root),
        max_local_records=max(0, int(args.max_local_records)),
        max_wikitext=max(0, int(args.max_wikitext)),
        max_fineweb_edu=max(0, int(args.max_fineweb_edu)),
        max_common_corpus=max(0, int(args.max_common_corpus)),
        max_chars=max(180, int(args.max_chars)),
    )

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(
            f"base corpus ready paragraphs={report['paragraphs_written']} "
            f"path={report['out_path']}"
        )


if __name__ == "__main__":
    main()
