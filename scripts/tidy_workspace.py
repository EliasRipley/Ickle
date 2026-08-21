"""Ickle workspace tidy — automated cleanup of build/runtime/test artifacts.
Run anytime to keep the workspace lean. Safe to run at startup.
"""
import os
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

DIRS_TO_CLEAN = [
    "data/.tmp",
    "data/.tmp_tests",
    "data/.tmp_tooling",
    "data/test_task_queue",
    "data/tasks",
    "data/sessions",
    "data/runtime",
]

DIRS_TO_CLEAN_TREES = [
    "build",
    "dist",
]

FILES_TO_REMOVE = [
    "data/task_queue.json",
    "data/runtime_flags.json",
    "data/training_resource_budget.json",
    "data/ilm.db",
]

FILES_TO_REMOVE_GLOB = [
    "data/*.log",
    "data/web_server_*.log",
    "data/.tmp*",
]

CACHE_PATTERNS = [
    "__pycache__",
    ".pytest_cache",
    "*.egg-info",
]

STALE_CONTINUAL = [
    "data/continual/conversation_focus.txt",
    "data/continual/new_mix_*.txt",
]


def _rmtree(path: Path):
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
        print(f"  removed: {path}")


def _rmfile(path: Path):
    if path.exists() and path.is_file():
        path.unlink(missing_ok=True)
        print(f"  removed: {path}")


def _glob_rm(root: Path, pattern: str):
    for p in root.glob(pattern):
        if p.is_file():
            p.unlink(missing_ok=True)
            print(f"  removed: {p}")
        elif p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
            print(f"  removed: {p}")


def _clean_pycache(root: Path):
    for p in root.rglob("__pycache__"):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)


def tidy():
    print("=== Ickle workspace tidy ===")

    for d in DIRS_TO_CLEAN:
        _rmtree(ROOT / d)

    for d in DIRS_TO_CLEAN_TREES:
        _rmtree(ROOT / d)

    for f in FILES_TO_REMOVE:
        _rmfile(ROOT / f)

    for pat in FILES_TO_REMOVE_GLOB:
        _glob_rm(ROOT / pat)

    for pat in STALE_CONTINUAL:
        _glob_rm(ROOT / pat)

    _clean_pycache(ROOT)

    print("=== Done ===")


if __name__ == "__main__":
    tidy()
