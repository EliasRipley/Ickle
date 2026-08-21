from __future__ import annotations

import argparse
import json

from src.ilm_memory import get_memory


def main():
    parser = argparse.ArgumentParser(description="Inspect Ickle research memory notes and sessions.")
    sub = parser.add_subparsers(dest="command", required=True)

    find = sub.add_parser("find", help="Search research notes by query.")
    find.add_argument("--query", required=True)
    find.add_argument("--limit", type=int, default=8)
    find.add_argument("--json", action="store_true")

    sessions = sub.add_parser("sessions", help="List recent research sessions.")
    sessions.add_argument("--limit", type=int, default=20)
    sessions.add_argument("--json", action="store_true")

    summary = sub.add_parser("summary", help="Show memory summary including research counts.")
    summary.add_argument("--json", action="store_true")

    args = parser.parse_args()
    memory = get_memory()

    if args.command == "find":
        rows = memory.search_research_notes(args.query, limit=max(1, int(args.limit)))
        if args.json:
            print(json.dumps(rows, indent=2, ensure_ascii=False))
            return
        if not rows:
            print("(no results)")
            return
        for idx, row in enumerate(rows, start=1):
            source = row.get("source_title") or row.get("source_url") or "unknown"
            print(f"{idx}. [{row.get('topic', 'general')}] {row.get('finding', '')} (source: {source})")
        return

    if args.command == "sessions":
        rows = memory.list_research_sessions(limit=max(1, int(args.limit)))
        if args.json:
            print(json.dumps(rows, indent=2, ensure_ascii=False))
            return
        if not rows:
            print("(no sessions)")
            return
        for idx, row in enumerate(rows, start=1):
            print(
                f"{idx}. {row.get('session_id')} topic={row.get('topic')} "
                f"notes={row.get('note_count', 0)} updated={row.get('updated_at_utc')}"
            )
        return

    if args.command == "summary":
        out = memory.get_memory_summary()
        if args.json:
            print(json.dumps(out, indent=2, ensure_ascii=False))
            return
        for key, value in out.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()

