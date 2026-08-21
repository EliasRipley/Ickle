from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from typing import Any
from urllib.request import Request, urlopen


def _fetch_json(url: str) -> dict[str, Any]:
    req = Request(url, headers={"User-Agent": "IckleTrainingWatch/1.0"})
    with urlopen(req, timeout=8) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _topic_from_task(task: dict[str, Any]) -> str:
    payload = task.get("payload", {}) if isinstance(task.get("payload"), dict) else {}
    return str(payload.get("topic", "")).strip()


def _line_for_task(task: dict[str, Any]) -> str:
    task_id = str(task.get("task_id", ""))[:12]
    task_type = str(task.get("task_type", ""))
    status = str(task.get("status", ""))
    topic = _topic_from_task(task)
    progress = str(task.get("progress", "")).strip()
    if len(progress) > 140:
        progress = progress[:137] + "..."
    suffix = f" topic={topic}" if topic else ""
    return f"{task_id} | {task_type} | {status}{suffix} | {progress}"


def _guard_summary(task: dict[str, Any]) -> str:
    result = task.get("result", {}) if isinstance(task.get("result"), dict) else {}
    if not result:
        return ""
    passed = result.get("passed")
    promoted = result.get("promoted")
    core_drop = result.get("core_drop")
    new_gain = result.get("new_gain")
    user_delta = result.get("user_delta")
    learned = result.get("learned_summary", "").strip()
    base = (
        f"guard passed={passed} promoted={promoted} "
        f"core_drop={core_drop} new_gain={new_gain} user_delta={user_delta}"
    )
    if learned:
        return base + "\n    what i've learnt: " + learned
    return base


def main() -> None:
    parser = argparse.ArgumentParser(description="Live monitor for Ickle training progress API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8788, help="Control API port (serve-control, default 8788)")
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()

    base = f"http://{args.host}:{args.port}"
    progress_url = f"{base}/api/training/progress?limit=20"
    status_url = f"{base}/api/status"
    seen_signatures: set[str] = set()
    seen_completed: set[str] = set()
    last_counts = ""
    last_weak = ""
    first_cycle = True

    print(f"Watching {base} (Ctrl+C to stop)")
    while True:
        try:
            status = _fetch_json(status_url)
            prog = _fetch_json(progress_url)
        except Exception as exc:  # noqa: BLE001
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"[{ts}] monitor-error: {exc}")
            time.sleep(max(0.5, args.interval))
            continue

        ts = datetime.now().strftime("%H:%M:%S")
        counts = status.get("task_counts", {})
        counts_sig = f"q={counts.get('queued',0)} r={counts.get('running',0)} c={counts.get('completed',0)} f={counts.get('failed',0)}"
        if counts_sig != last_counts or first_cycle:
            print(
                f"\n[{ts}] queued={counts.get('queued', 0)} running={counts.get('running', 0)} "
                f"completed={counts.get('completed', 0)} failed={counts.get('failed', 0)}"
            )
            last_counts = counts_sig

        running_or_queued = list(prog.get("running_or_queued", []))
        if not running_or_queued:
            print("No active training tasks.")
        for task in running_or_queued:
            signature = f"{task.get('task_id')}|{task.get('status')}|{task.get('progress')}"
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            print("  " + _line_for_task(task))

        recent = list(prog.get("recent_training_tasks", []))
        for task in recent:
            if str(task.get("status", "")) != "completed":
                continue
            task_id = str(task.get("task_id", ""))
            if task_id in seen_completed:
                continue
            seen_completed.add(task_id)
            ttype = str(task.get("task_type", ""))
            topic = _topic_from_task(task)
            print(f"  completed: {task_id[:12]} | {ttype} | topic={topic}")
            if ttype == "continual_guard_step":
                summary = _guard_summary(task)
                if summary:
                    print("    " + summary)

        mastery = prog.get("topic_mastery", {}) if isinstance(prog.get("topic_mastery"), dict) else {}
        weak = list(mastery.get("weak_topics", []))[:5]
        weak_sig = "|".join(weak)
        if weak and weak_sig != last_weak:
            print("  weak-topics: " + " | ".join(weak))
            last_weak = weak_sig

        first_cycle = False
        time.sleep(max(0.5, args.interval))


if __name__ == "__main__":
    main()
