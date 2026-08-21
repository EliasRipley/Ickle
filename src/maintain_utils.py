"""Shared file utilities for the maintenance sweepers (model_maintain.py,
training_maintain.py). Both previously carried byte-for-byte copies of these
four functions; kept here so they can't silently drift apart again."""

from __future__ import annotations

import gzip
import hashlib
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def age_days(path: Path) -> float:
    return max(0.0, (time.time() - path.stat().st_mtime) / 86400.0)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def gzip_file(source: Path, target: Path, compress_level: int = 6):
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as f_in, gzip.open(target, "wb", compresslevel=max(1, min(9, compress_level))) as f_out:
        shutil.copyfileobj(f_in, f_out)
