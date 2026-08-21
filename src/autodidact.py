import argparse
import json
from pathlib import Path
from typing import Any


def build_python_autodidact_corpus(log_path: str, out_path: str) -> int:
    """Build a corpus from the ILM's own coding attempts.

    Expected JSONL fields per row:
    - prompt: str
    - response: str
    - tests_passed: bool
    - lint_passed: bool (optional)

    Only keeps rows that passed objective checks.
    """
    src = Path(log_path)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    with src.open("r", encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
        for line in fin:
            row = json.loads(line)
            if not row.get("tests_passed", False):
                continue
            if row.get("lint_passed") is False:
                continue
            prompt = str(row.get("prompt", "")).strip()
            response = str(row.get("response", "")).strip()
            if not prompt or not response:
                continue
            fout.write(f"User: {prompt}\nAssistant: {response}\n\n")
            kept += 1

    return kept


def build_general_autodidact_corpus(log_path: str, out_path: str) -> int:
    """Build a corpus from general web research and knowledge responses.

    Expected JSONL fields per row:
    - prompt: str
    - response: str
    - evidence_score: float (optional, threshold >= 0.5)
    - format_ok: bool (optional, checks for hallucinated citations)

    Keeps rows with strong evidence or good formatting.
    """
    src = Path(log_path)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    with src.open("r", encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
        for line in fin:
            row = json.loads(line)
            evidence = float(row.get("evidence_score", 0.0))
            format_ok = row.get("format_ok")
            if format_ok is False:
                continue
            if evidence < 0.5 and format_ok is not True:
                continue
            prompt = str(row.get("prompt", "")).strip()
            response = str(row.get("response", "")).strip()
            if not prompt or not response:
                continue
            fout.write(f"User: {prompt}\nIckle: {response}\n\n")
            kept += 1

    return kept


def build_conversation_autodidact_corpus(log_path: str, out_path: str, min_rating: int = 3) -> int:
    """Build a corpus from hub-rated conversations.

    Expected JSONL fields per row:
    - prompt: str
    - response: str
    - rating: int (1-5)

    Keeps rows rated >= min_rating.
    """
    src = Path(log_path)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    with src.open("r", encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
        for line in fin:
            row = json.loads(line)
            rating = int(row.get("rating", 0))
            if rating < min_rating:
                continue
            prompt = str(row.get("prompt", "")).strip()
            response = str(row.get("response", "")).strip()
            if not prompt or not response:
                continue
            fout.write(f"User: {prompt}\nIckle: {response}\n\n")
            kept += 1

    return kept


def build_unified_autodidact_corpus(
    python_log: str = "data/python_attempts.jsonl",
    general_log: str = "data/general_research.jsonl",
    conversation_log: str = "data/hub_feedback.jsonl",
    out_path: str = "data/autodidact_corpus.txt",
) -> int:
    total = 0
    parts: list[str] = []

    for src, builder, label in [
        (python_log, build_python_autodidact_corpus, "python"),
        (general_log, build_general_autodidact_corpus, "general"),
        (conversation_log, build_conversation_autodidact_corpus, "conversation"),
    ]:
        if Path(src).exists():
            tmp = f"{out_path}.{label}.tmp"
            n = builder(src, tmp)
            if n > 0:
                parts.append(tmp)
                total += n

    if not parts:
        return 0

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fout:
        for part_path in parts:
            with open(part_path, "r", encoding="utf-8") as fin:
                fout.write(fin.read())
            Path(part_path).unlink(missing_ok=True)

    return total


def main():
    parser = argparse.ArgumentParser(description="Build self-improvement corpus from ILM outcomes")
    parser.add_argument("--log", default="data/python_attempts.jsonl")
    parser.add_argument("--out", default="data/python_autodidact_corpus.txt")
    parser.add_argument("--mode", default="python", choices=["python", "general", "conversation", "unified"])
    parser.add_argument("--general-log", default="data/general_research.jsonl")
    parser.add_argument("--conversation-log", default="data/hub_feedback.jsonl")
    parser.add_argument("--min-rating", type=int, default=3)
    args = parser.parse_args()

    if args.mode == "unified":
        kept = build_unified_autodidact_corpus(
            python_log=args.log,
            general_log=args.general_log,
            conversation_log=args.conversation_log,
            out_path=args.out,
        )
    elif args.mode == "general":
        kept = build_general_autodidact_corpus(args.log, args.out)
    elif args.mode == "conversation":
        kept = build_conversation_autodidact_corpus(args.log, args.out, args.min_rating)
    else:
        kept = build_python_autodidact_corpus(args.log, args.out)

    print(f"wrote {kept} rows to {args.out}")


if __name__ == "__main__":
    main()

