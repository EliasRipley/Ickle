from __future__ import annotations

import argparse
import gzip
import json
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.maintain_utils import age_days as _age_days, gzip_file as _gzip_file, sha256_file as _sha256_file, utc_now as _utc_now


@dataclass
class MaintenanceAction:
    kind: str
    source: str
    target: str | None
    status: str
    detail: str


def _manifest_path() -> Path:
    out = Path("data/maintenance")
    out.mkdir(parents=True, exist_ok=True)
    return out / "model_archive_manifest.json"


def _load_manifest() -> dict[str, Any]:
    path = _manifest_path()
    if not path.exists():
        return {"schema_version": "1.0", "updated_at_utc": _utc_now(), "archives": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        corrupt_backup = Path(str(path) + f".corrupt.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.bak")
        try:
            shutil.copyfile(path, corrupt_backup)
        except Exception:
            pass
        payload = {"schema_version": "1.0", "updated_at_utc": _utc_now(), "archives": []}
    if "archives" not in payload or not isinstance(payload["archives"], list):
        payload["archives"] = []
    return payload


def _save_manifest(payload: dict[str, Any]):
    payload["updated_at_utc"] = _utc_now()
    _manifest_path().write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _archive_file(
    source: Path,
    target: Path,
    *,
    apply: bool,
    compress_level: int,
    kind: str,
) -> MaintenanceAction:
    if not source.exists():
        return MaintenanceAction(
            kind=kind,
            source=str(source),
            target=str(target),
            status="skipped",
            detail="source_missing",
        )
    if target.exists() and not apply:
        return MaintenanceAction(
            kind=kind,
            source=str(source),
            target=str(target),
            status="planned",
            detail="target_exists_would_overwrite",
        )
    if not apply:
        return MaintenanceAction(
            kind=kind,
            source=str(source),
            target=str(target),
            status="planned",
            detail="would_archive_and_compress",
        )
    try:
        source_hash = _sha256_file(source) if source.exists() else ""
        if target.exists():
            target.unlink()
        _gzip_file(source, target, compress_level=compress_level)
        if not target.exists() or target.stat().st_size == 0:
            return MaintenanceAction(
                kind=kind, source=str(source), target=str(target),
                status="error", detail="compression_produced_empty_or_missing_file",
            )
        target_hash = _sha256_file(target)
        if source_hash:
            source.unlink()
        return MaintenanceAction(
            kind=kind,
            source=str(source),
            target=str(target),
            status="done",
            detail=f"archived_and_compressed source_sha256={source_hash[:16]} target_sha256={target_hash[:16]}",
        )
    except Exception as exc:  # noqa: BLE001
        return MaintenanceAction(
            kind=kind,
            source=str(source),
            target=str(target),
            status="error",
            detail=str(exc),
        )


def _parse_keep_names(values: str) -> set[str]:
    return {x.strip().lower() for x in values.split(",") if x.strip()}


def _sweep_model_dir(
    *,
    dir_path: Path,
    archive_root: Path,
    archive_subdir: str,
    keep_recent: int,
    keep_names: set[str],
    checkpoint_keep_recent: int,
    checkpoint_ttl_days: float,
    compress_level: int,
    apply: bool,
    actions: list[MaintenanceAction],
    archived_records: list[dict[str, Any]],
):
    """Archive stale top-level *.pt files and prune stale *.checkpoint.pt files
    in a single directory. Shared by the main models_root sweep and the
    models/candidates/ sweep below — candidate checkpoints used to be
    invisible to maintenance entirely because Path.glob("*.pt") only looks at
    the given directory, never subdirectories, so a naive single-root sweep
    silently skipped everything training writes to models/candidates/."""
    model_files = [p for p in dir_path.glob("*.pt") if not p.name.endswith(".checkpoint.pt")]
    model_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    local_keep_names = set(keep_names)
    for p in model_files[: max(0, keep_recent)]:
        local_keep_names.add(p.name.lower())

    for model_file in model_files:
        if model_file.name.lower() in local_keep_names:
            actions.append(
                MaintenanceAction(
                    kind="model_keep",
                    source=str(model_file),
                    target=None,
                    status="kept",
                    detail="protected_or_recent",
                )
            )
            continue
        target = archive_root / archive_subdir / f"{model_file.name}.gz"
        action = _archive_file(
            model_file,
            target,
            apply=apply,
            compress_level=compress_level,
            kind="model_archive",
        )
        actions.append(action)
        if action.status == "done":
            archived_records.append(
                {
                    "archived_at_utc": _utc_now(),
                    "kind": "model",
                    "source": action.source,
                    "target": str(target),
                    "target_sha256": _sha256_file(target),
                    "target_size_bytes": int(target.stat().st_size),
                }
            )

        meta_file = Path(str(model_file) + ".meta.json")
        if meta_file.exists():
            meta_target = archive_root / archive_subdir / f"{meta_file.name}.gz"
            meta_action = _archive_file(
                meta_file,
                meta_target,
                apply=apply,
                compress_level=compress_level,
                kind="meta_archive",
            )
            actions.append(meta_action)
            if meta_action.status == "done":
                archived_records.append(
                    {
                        "archived_at_utc": _utc_now(),
                        "kind": "meta",
                        "source": meta_action.source,
                        "target": str(meta_target),
                        "target_sha256": _sha256_file(meta_target),
                        "target_size_bytes": int(meta_target.stat().st_size),
                    }
                )

    checkpoints = list(dir_path.glob("*.checkpoint.pt"))
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for idx, ckpt in enumerate(checkpoints):
        if idx < max(0, checkpoint_keep_recent):
            actions.append(
                MaintenanceAction(
                    kind="checkpoint_keep",
                    source=str(ckpt),
                    target=None,
                    status="kept",
                    detail="recent_checkpoint",
                )
            )
            continue
        age = _age_days(ckpt)
        if age < checkpoint_ttl_days:
            actions.append(
                MaintenanceAction(
                    kind="checkpoint_keep",
                    source=str(ckpt),
                    target=None,
                    status="kept",
                    detail=f"younger_than_ttl({age:.1f}d)",
                )
            )
            continue
        if not apply:
            actions.append(
                MaintenanceAction(
                    kind="checkpoint_delete",
                    source=str(ckpt),
                    target=None,
                    status="planned",
                    detail=f"would_delete_stale_checkpoint({age:.1f}d)",
                )
            )
        else:
            try:
                ckpt.unlink()
                actions.append(
                    MaintenanceAction(
                        kind="checkpoint_delete",
                        source=str(ckpt),
                        target=None,
                        status="done",
                        detail=f"deleted_stale_checkpoint({age:.1f}d)",
                    )
                )
            except Exception as exc:  # noqa: BLE001
                actions.append(
                    MaintenanceAction(
                        kind="checkpoint_delete",
                        source=str(ckpt),
                        target=None,
                        status="error",
                        detail=str(exc),
                    )
                )


def run_model_maintenance(
    *,
    models_root: str = "models",
    archive_dir: str = "models/archive",
    keep_recent: int = 2,
    keep_names_csv: str = "ickle_clean.pt,ickle_brain_candidate_v2.pt,ickle_brain_base_v2.pt",
    checkpoint_keep_recent: int = 4,
    checkpoint_ttl_days: float = 14.0,
    compress_level: int = 6,
    apply: bool = False,
    include_candidates: bool = True,
    candidates_dirname: str = "candidates",
    candidates_keep_recent: int = 1,
) -> dict[str, Any]:
    root = Path(models_root).resolve()
    archive_root = Path(archive_dir).resolve()
    keep_names = _parse_keep_names(keep_names_csv)
    actions: list[MaintenanceAction] = []
    archived_records: list[dict[str, Any]] = []

    if not root.exists():
        return {
            "models_root": str(root),
            "archive_dir": str(archive_root),
            "apply": apply,
            "actions": [],
            "error": "models_root_not_found",
        }

    _sweep_model_dir(
        dir_path=root,
        archive_root=archive_root,
        archive_subdir=".",
        keep_recent=keep_recent,
        keep_names=keep_names,
        checkpoint_keep_recent=checkpoint_keep_recent,
        checkpoint_ttl_days=checkpoint_ttl_days,
        compress_level=compress_level,
        apply=apply,
        actions=actions,
        archived_records=archived_records,
    )

    candidates_root = root / candidates_dirname
    swept_candidates = False
    if include_candidates and candidates_root.is_dir() and candidates_root.resolve() != root.resolve():
        swept_candidates = True
        _sweep_model_dir(
            dir_path=candidates_root,
            archive_root=archive_root,
            archive_subdir=candidates_dirname,
            keep_recent=candidates_keep_recent,
            keep_names=set(),
            checkpoint_keep_recent=checkpoint_keep_recent,
            checkpoint_ttl_days=checkpoint_ttl_days,
            compress_level=compress_level,
            apply=apply,
            actions=actions,
            archived_records=archived_records,
        )

    if apply and archived_records:
        manifest = _load_manifest()
        archives = manifest.get("archives", [])
        if not isinstance(archives, list):
            archives = []
        archives.extend(archived_records)
        manifest["archives"] = archives[-3000:]
        _save_manifest(manifest)

    status_counts: dict[str, int] = {}
    for action in actions:
        status_counts[action.status] = status_counts.get(action.status, 0) + 1

    return {
        "models_root": str(root),
        "archive_dir": str(archive_root),
        "apply": apply,
        "candidates_root": str(candidates_root) if swept_candidates else None,
        "status_counts": status_counts,
        "actions": [action.__dict__ for action in actions],
    }


def main():
    parser = argparse.ArgumentParser(description="Archive/compress older models and prune stale checkpoints.")
    parser.add_argument("--models-root", default="models")
    parser.add_argument("--archive-dir", default="models/archive")
    parser.add_argument("--keep-recent", type=int, default=2)
    parser.add_argument("--keep-names", default="ickle_clean.pt")
    parser.add_argument("--checkpoint-keep-recent", type=int, default=4)
    parser.add_argument("--checkpoint-ttl-days", type=float, default=14.0)
    parser.add_argument("--compress-level", type=int, default=6)
    parser.add_argument("--apply", action="store_true", help="Actually perform changes (default is dry-run preview).")
    parser.add_argument(
        "--no-candidates",
        action="store_true",
        help="Skip the models/candidates/ subfolder (swept by default alongside --models-root).",
    )
    parser.add_argument("--candidates-dirname", default="candidates", help="Subfolder name under --models-root to also sweep.")
    parser.add_argument("--candidates-keep-recent", type=int, default=1, help="How many recent candidate models to keep.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = run_model_maintenance(
        models_root=args.models_root,
        archive_dir=args.archive_dir,
        keep_recent=max(0, args.keep_recent),
        keep_names_csv=args.keep_names,
        checkpoint_keep_recent=max(0, args.checkpoint_keep_recent),
        checkpoint_ttl_days=max(0.0, args.checkpoint_ttl_days),
        compress_level=max(1, min(9, args.compress_level)),
        apply=bool(args.apply),
        include_candidates=not args.no_candidates,
        candidates_dirname=args.candidates_dirname,
        candidates_keep_recent=max(0, args.candidates_keep_recent),
    )
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    print(f"models_root: {report['models_root']}")
    print(f"archive_dir: {report['archive_dir']}")
    if report.get("candidates_root"):
        print(f"candidates_root: {report['candidates_root']}")
    print(f"apply: {report['apply']}")
    print(f"status_counts: {report.get('status_counts', {})}")
    for action in report.get("actions", [])[:40]:
        target = action.get("target")
        if target:
            print(f"[{action['status']}] {action['kind']}: {action['source']} -> {target} ({action['detail']})")
        else:
            print(f"[{action['status']}] {action['kind']}: {action['source']} ({action['detail']})")


def restore_archive(source_gz: str, target_path: str) -> dict[str, Any]:
    src = Path(source_gz)
    dst = Path(target_path)
    if not src.exists():
        return {"error": f"Archive file not found: {src}"}
    if not src.suffix.lower().endswith(".gz"):
        return {"error": f"Not a .gz file: {src}"}
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(src, "rb") as f_in, dst.open("wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        manifest = _load_manifest()
        for entry in manifest.get("archives", []):
            if entry.get("target") == str(src.resolve()):
                expected_hash = entry.get("target_sha256", "")
                actual_hash = _sha256_file(dst)
                if expected_hash and actual_hash != expected_hash:
                    dst.unlink()
                    return {"error": f"Integrity check failed: expected {expected_hash[:16]}, got {actual_hash[:16]}"}
                return {
                    "restored": True,
                    "source": str(src),
                    "target": str(dst),
                    "sha256": actual_hash,
                    "size_bytes": dst.stat().st_size,
                    "verified": bool(expected_hash),
                }
        actual_hash = _sha256_file(dst)
        return {
            "restored": True,
            "source": str(src),
            "target": str(dst),
            "sha256": actual_hash,
            "size_bytes": dst.stat().st_size,
            "verified": False,
            "warning": "No manifest entry found for integrity verification",
        }
    except Exception as exc:
        if dst.exists():
            dst.unlink()
        return {"error": str(exc)}


def run_data_maintenance(
    *,
    data_root: str = "data",
    continual_dir: str = "data/continual",
    tasks_dir: str = "data/tasks",
    runtime_dir: str = "data/runtime",
    maintenance_dir: str = "data/maintenance",
    continual_keep_recent: int = 5,
    tasks_keep_recent: int = 20,
    runtime_ttl_days: float = 7.0,
    maintenance_ttl_days: float = 30.0,
    compress_level: int = 6,
    apply: bool = False,
) -> dict[str, Any]:
    actions: list[MaintenanceAction] = []

    def _prune_dir(label: str, dir_path: str, keep_recent: int | None, ttl_days: float | None):
        p = Path(dir_path)
        if not p.exists():
            return
        files = sorted(
            [f for f in p.iterdir() if f.is_file()],
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        cutoff = time.time() - (ttl_days * 86400) if ttl_days is not None else None
        for i, f in enumerate(files):
            keep = True
            if keep_recent is not None and i >= keep_recent:
                keep = False
            if cutoff is not None and f.stat().st_mtime < cutoff:
                keep = False
            if keep:
                actions.append(MaintenanceAction(
                    kind=f"{label}_keep", source=str(f), target="", status="kept",
                    detail=f"within retention policy",
                ))
            elif apply:
                try:
                    # Size must be read before unlink() -- reading it after
                    # (the original order here) raises FileNotFoundError on
                    # the just-deleted path, which the broad except below
                    # caught and misreported as "status": "error" for every
                    # single successful deletion.
                    size_bytes = f.stat().st_size
                    f.unlink()
                    actions.append(MaintenanceAction(
                        kind=f"{label}_delete", source=str(f), target="", status="done",
                        detail=f"removed {size_bytes} bytes",
                    ))
                except Exception as exc:
                    actions.append(MaintenanceAction(
                        kind=f"{label}_delete", source=str(f), target="", status="error",
                        detail=str(exc),
                    ))
            else:
                actions.append(MaintenanceAction(
                    kind=f"{label}_delete", source=str(f), target="", status="planned",
                    detail=f"would remove {f.stat().st_size} bytes",
                ))

    _prune_dir("continual", continual_dir, continual_keep_recent, None)
    _prune_dir("tasks", tasks_dir, tasks_keep_recent, None)
    _prune_dir("runtime", runtime_dir, None, runtime_ttl_days)
    _prune_dir("maintenance", maintenance_dir, None, maintenance_ttl_days)

    status_counts: dict[str, int] = {}
    for a in actions:
        status_counts[a.status] = status_counts.get(a.status, 0) + 1

    return {
        "data_root": str(Path(data_root).resolve()),
        "apply": apply,
        "status_counts": status_counts,
        "actions": [asdict(a) for a in actions],
    }


if __name__ == "__main__":
    main()

