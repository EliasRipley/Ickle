from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.maintain_utils import age_days as _age_days, gzip_file as _gzip_file, sha256_file as _sha256_file, utc_now as _utc_now
from src.workspace_paths import get_training_root


@dataclass
class TrainingAction:
    kind: str
    source: str
    target: str | None
    status: str
    detail: str


def _load_archive_index(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": "1.0", "updated_at_utc": _utc_now(), "entries": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        payload = {"schema_version": "1.0", "updated_at_utc": _utc_now(), "entries": []}
    if "entries" not in payload or not isinstance(payload["entries"], list):
        payload["entries"] = []
    return payload


def _save_archive_index(path: Path, payload: dict[str, Any]):
    payload["updated_at_utc"] = _utc_now()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _compact_queue_file(queue_path: Path, *, max_lines: int, apply: bool) -> tuple[TrainingAction, dict[str, Any]]:
    if not queue_path.exists():
        return (
            TrainingAction(
                kind="queue_compact",
                source=str(queue_path),
                target=None,
                status="skipped",
                detail="queue_file_missing",
            ),
            {"before_lines": 0, "after_lines": 0},
        )
    lines = queue_path.read_text(encoding="utf-8", errors="replace").splitlines()
    before = len(lines)

    deduped: list[str] = []
    seen: set[str] = set()
    for line in lines:
        key = line.strip()
        if not key:
            deduped.append("")
            continue
        if key in seen:
            continue
        seen.add(key)
        deduped.append(line)

    if max_lines > 0 and len(deduped) > max_lines:
        deduped = deduped[-max_lines:]

    after = len(deduped)
    changed = after != before or deduped != lines
    if not changed:
        return (
            TrainingAction(
                kind="queue_compact",
                source=str(queue_path),
                target=None,
                status="kept",
                detail="queue_already_compact",
            ),
            {"before_lines": before, "after_lines": after},
        )

    if not apply:
        return (
            TrainingAction(
                kind="queue_compact",
                source=str(queue_path),
                target=None,
                status="planned",
                detail=f"would_compact_lines({before}->{after})",
            ),
            {"before_lines": before, "after_lines": after},
        )

    queue_path.write_text("\n".join(deduped) + ("\n" if deduped else ""), encoding="utf-8")
    return (
        TrainingAction(
            kind="queue_compact",
            source=str(queue_path),
            target=None,
            status="done",
            detail=f"compacted_lines({before}->{after})",
        ),
        {"before_lines": before, "after_lines": after},
    )


def run_training_maintenance(
    *,
    training_root: str | None = None,
    archive_dir: str | None = None,
    min_age_days: float = 7.0,
    min_size_bytes: int = 5_000_000,
    compress_level: int = 6,
    max_queue_lines: int = 20000,
    apply: bool = False,
) -> dict[str, Any]:
    root = Path(training_root).resolve() if training_root else get_training_root()
    archive_root = Path(archive_dir).resolve() if archive_dir else (root / "archive")
    archive_files_root = archive_root / "files"
    archive_index_path = archive_root / "archive_index.json"
    actions: list[TrainingAction] = []
    archived_entries: list[dict[str, Any]] = []

    if not root.exists():
        return {
            "training_root": str(root),
            "archive_dir": str(archive_root),
            "apply": apply,
            "actions": [],
            "error": "training_root_not_found",
        }

    protected_names = {
        "combined_corpus.txt",
        "comprehensive_english.txt",
        "queued_wikipedia_learning.txt",
        "webster_dictionary.json",
        "advanced_mathematics_dataset.txt",
    }
    allowed_suffixes = {".txt", ".jsonl", ".json"}

    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        try:
            file_path.relative_to(archive_root)
            continue
        except ValueError:
            pass
        if file_path.name in protected_names:
            actions.append(
                TrainingAction(
                    kind="training_keep",
                    source=str(file_path),
                    target=None,
                    status="kept",
                    detail="protected_file",
                )
            )
            continue
        if file_path.suffix.lower() not in allowed_suffixes:
            continue
        size = int(file_path.stat().st_size)
        age = _age_days(file_path)
        if size < max(1, min_size_bytes):
            continue
        if age < max(0.0, min_age_days):
            continue

        rel = file_path.relative_to(root)
        target = archive_files_root / (str(rel).replace("\\", "/") + ".gz")
        if not apply:
            actions.append(
                TrainingAction(
                    kind="training_archive",
                    source=str(file_path),
                    target=str(target),
                    status="planned",
                    detail=f"would_archive(size={size},age_days={age:.1f})",
                )
            )
            continue
        try:
            if target.exists():
                target.unlink()
            _gzip_file(file_path, target, compress_level=compress_level)
            file_path.unlink()
            actions.append(
                TrainingAction(
                    kind="training_archive",
                    source=str(file_path),
                    target=str(target),
                    status="done",
                    detail=f"archived(size={size},age_days={age:.1f})",
                )
            )
            archived_entries.append(
                {
                    "archived_at_utc": _utc_now(),
                    "source": str(file_path),
                    "target": str(target),
                    "target_sha256": _sha256_file(target),
                    "target_size_bytes": int(target.stat().st_size),
                }
            )
        except Exception as exc:  # noqa: BLE001
            actions.append(
                TrainingAction(
                    kind="training_archive",
                    source=str(file_path),
                    target=str(target),
                    status="error",
                    detail=str(exc),
                )
            )

    queue_action, queue_stats = _compact_queue_file(
        root / "queued_wikipedia_learning.txt",
        max_lines=max(0, max_queue_lines),
        apply=apply,
    )
    actions.append(queue_action)

    if apply and archived_entries:
        index = _load_archive_index(archive_index_path)
        entries = index.get("entries", [])
        if not isinstance(entries, list):
            entries = []
        entries.extend(archived_entries)
        index["entries"] = entries[-5000:]
        _save_archive_index(archive_index_path, index)

    status_counts: dict[str, int] = {}
    for action in actions:
        status_counts[action.status] = status_counts.get(action.status, 0) + 1

    return {
        "training_root": str(root),
        "archive_dir": str(archive_root),
        "apply": apply,
        "status_counts": status_counts,
        "queue_stats": queue_stats,
        "actions": [action.__dict__ for action in actions],
    }


def main():
    parser = argparse.ArgumentParser(description="Archive old training artifacts and compact queue files.")
    parser.add_argument("--training-root", default=str(get_training_root()))
    parser.add_argument("--archive-dir", default="")
    parser.add_argument("--min-age-days", type=float, default=7.0)
    parser.add_argument("--min-size-bytes", type=int, default=5_000_000)
    parser.add_argument("--compress-level", type=int, default=6)
    parser.add_argument("--max-queue-lines", type=int, default=20000)
    parser.add_argument("--apply", action="store_true", help="Actually perform maintenance (default is dry-run preview).")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = run_training_maintenance(
        training_root=args.training_root,
        archive_dir=args.archive_dir or None,
        min_age_days=max(0.0, args.min_age_days),
        min_size_bytes=max(1, args.min_size_bytes),
        compress_level=max(1, min(9, args.compress_level)),
        max_queue_lines=max(0, args.max_queue_lines),
        apply=bool(args.apply),
    )
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    print(f"training_root: {report['training_root']}")
    print(f"archive_dir: {report['archive_dir']}")
    print(f"apply: {report['apply']}")
    print(f"status_counts: {report.get('status_counts', {})}")
    print(f"queue_stats: {report.get('queue_stats', {})}")
    for action in report.get("actions", [])[:40]:
        target = action.get("target")
        if target:
            print(f"[{action['status']}] {action['kind']}: {action['source']} -> {target} ({action['detail']})")
        else:
            print(f"[{action['status']}] {action['kind']}: {action['source']} ({action['detail']})")
    if len(report.get("actions", [])) > 40:
        print(f"... truncated {len(report['actions']) - 40} additional actions")


if __name__ == "__main__":
    main()

