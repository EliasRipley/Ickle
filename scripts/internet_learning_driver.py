"""Continuous, time-boxed internet-learning driver.

Runs for a fixed wall-clock budget (default ~55 minutes). Each cycle:
  1. Fetches a fresh batch of random Wikipedia articles (genuinely random
     topic selection -- no fixed curated list, no topic restriction) and
     appends them to a growing corpus file.
  2. Runs a chunk of raw-text pretraining (src/train.py --pretrain-data) on
     the corpus accumulated so far, resuming from the previous cycle's
     output so the model keeps growing across the whole run.
  3. Writes live status to the same status file Ickle Mini already polls
     (IckleTraining/training_live.json), so progress is visible the whole
     time, not just at the end.

This reuses train.py unchanged (no new training-loop code) -- it is just
called repeatedly, chained via --init-model/--out, interleaved with real
internet fetches, so the model is continuously fed fresh internet content
for the whole time budget rather than a single upfront snapshot.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = PROJECT_ROOT / "IckleTraining" / "corpuses" / "open_internet_continuous.txt"
SFT_DATA_PATH = PROJECT_ROOT / "IckleTraining" / "corpuses" / "open_oasst1_20260812.txt"
STATUS_FILE = PROJECT_ROOT / "IckleTraining" / "training_live.json"
BASE_MODEL = PROJECT_ROOT / "models" / "ickle_v5_bpe.best.pt"
CANDIDATE_OUT = PROJECT_ROOT / "models" / "candidates" / "ickle_internet_live.pt"
CANDIDATE_BEST = PROJECT_ROOT / "models" / "candidates" / "ickle_internet_live.best.pt"
CANDIDATE_CKPT = PROJECT_ROOT / "models" / "candidates" / "ickle_internet_live.checkpoint.pt"
DRIVER_LOG = PROJECT_ROOT / "data" / "internet_learning_driver.log"

TIME_BUDGET_SECONDS = 40 * 60  # trimmed to account for setup/debugging time already spent this session
FETCH_BATCH_SIZE = 40
PRETRAIN_STEPS_PER_CYCLE = 150
WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
WIKI_RANDOM_PAGE_COUNT = 20  # per action=query&list=random call
UA_HEADERS = {"User-Agent": "IckleContinuousLearning/1.0 (local research build; contact: local-user)"}


def _wiki_api_get(params: dict) -> dict:
    query = "&".join(f"{k}={urllib.parse.quote(str(v))}" for k, v in params.items())
    req = urllib.request.Request(f"{WIKI_API_URL}?{query}", headers=UA_HEADERS)
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with DRIVER_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def fetch_batch(n: int) -> int:
    """Bulk-fetch ~n random article extracts using two batched API calls
    (list=random for titles, prop=extracts for text) instead of n separate
    per-article requests -- far fewer HTTP calls, and the correct way to use
    the MediaWiki API for bulk access rather than hammering a single-item
    endpoint repeatedly (which trips Wikimedia's rate limiter fast)."""
    written = 0
    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    rounds = max(1, (n + WIKI_RANDOM_PAGE_COUNT - 1) // WIKI_RANDOM_PAGE_COUNT)
    with CORPUS_PATH.open("a", encoding="utf-8") as f:
        for r in range(rounds):
            if r > 0:
                time.sleep(1.0)
            try:
                rand_data = _wiki_api_get({
                    "action": "query", "list": "random", "rnnamespace": 0,
                    "rnlimit": WIKI_RANDOM_PAGE_COUNT, "format": "json",
                })
                titles = [p["title"] for p in rand_data.get("query", {}).get("random", []) if p.get("title")]
                if not titles:
                    continue
                time.sleep(0.5)
                extract_data = _wiki_api_get({
                    "action": "query", "prop": "extracts", "exintro": 1, "explaintext": 1,
                    "titles": "|".join(titles), "format": "json",
                })
                pages = extract_data.get("query", {}).get("pages", {})
                for page in pages.values():
                    title = str(page.get("title", "")).strip()
                    extract = str(page.get("extract", "")).strip()
                    if title and extract and len(extract) > 40:
                        f.write(f"{title}\n\n{extract}\n\n")
                        written += 1
                f.flush()
            except Exception as exc:  # noqa: BLE001
                _log(f"fetch batch error (continuing): {exc}")
                time.sleep(3)
                continue
    return written


def run_train_cycle(init_model: Path, cycle: int) -> bool:
    # Deliberately using --data (not --pretrain-data/--pretrain-steps) for the
    # whole per-cycle step budget: train.py only writes the live status file
    # during the --data/--steps phase (tied to --eval-every), never during
    # --pretrain-data/--pretrain-steps, which stays silent until it fully
    # finishes. Routing the real work through --data is what makes progress
    # actually show up in Ickle Mini while a cycle is running, not just after.
    cmd = [
        sys.executable, "-m", "src.app", "train",
        "--data", str(CORPUS_PATH),
        "--steps", str(PRETRAIN_STEPS_PER_CYCLE),
        "--init-model", str(init_model),
        "--out", str(CANDIDATE_OUT),
        "--best-model-path", str(CANDIDATE_BEST),
        "--checkpoint-path", str(CANDIDATE_CKPT),
        "--status-file", str(STATUS_FILE),
        "--eval-every", "10",
        "--eval-iters", "8",
        "--no-auto-register",
    ]
    _log(f"cycle {cycle}: launching train.py chunk ({PRETRAIN_STEPS_PER_CYCLE} steps, status every 10)")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode == 0


def main():
    DRIVER_LOG.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    _log(f"=== internet learning driver started, budget={TIME_BUDGET_SECONDS}s ===")

    # Resume from wherever a previous run of this driver left off, rather
    # than always restarting from the original base model -- so tuning a
    # setting (batch size, eval cost) mid-run and relaunching doesn't throw
    # away progress already made. CANDIDATE_BEST is checked first: it is
    # written incrementally whenever validation improves, whereas
    # CANDIDATE_OUT is only written when a cycle finishes ALL of its steps --
    # an interrupted cycle only ever leaves CANDIDATE_BEST behind.
    if CANDIDATE_BEST.exists():
        current_model = CANDIDATE_BEST
        _log(f"resuming from existing best candidate: {CANDIDATE_BEST}")
    elif CANDIDATE_OUT.exists():
        current_model = CANDIDATE_OUT
        _log(f"resuming from existing candidate: {CANDIDATE_OUT}")
    else:
        current_model = BASE_MODEL
        _log(f"starting fresh from base model: {BASE_MODEL}")
    cycle = 0
    MIN_CORPUS_BYTES = 8000  # enough tokens for block_size=512 train+val split with margin

    while time.time() - started < TIME_BUDGET_SECONDS:
        cycle += 1
        remaining = TIME_BUDGET_SECONDS - (time.time() - started)
        _log(f"cycle {cycle}: fetching {FETCH_BATCH_SIZE} random Wikipedia articles ({remaining:.0f}s remaining)")
        got = fetch_batch(FETCH_BATCH_SIZE)
        corpus_bytes = CORPUS_PATH.stat().st_size if CORPUS_PATH.exists() else 0
        _log(f"cycle {cycle}: fetched {got} articles, corpus now {corpus_bytes} bytes")

        if corpus_bytes < MIN_CORPUS_BYTES:
            _log(f"cycle {cycle}: corpus still below {MIN_CORPUS_BYTES} bytes, fetching another round before training")
            continue

        ok = run_train_cycle(current_model, cycle)
        if not ok:
            _log(f"cycle {cycle}: train.py exited non-zero, stopping driver")
            break
        if CANDIDATE_OUT.exists():
            current_model = CANDIDATE_OUT

    _log(f"=== internet learning driver finished after {time.time() - started:.0f}s, {cycle} cycles ===")


if __name__ == "__main__":
    main()
