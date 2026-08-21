import argparse
import json
from pathlib import Path


def build_corpus(feedback_path: str, out_path: str, min_rating: int = 4):
    src = Path(feedback_path)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    skipped = 0
    with src.open("r", encoding="utf-8") as f_in, out.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            try:
                rating = int(item.get("rating", 0))
            except (TypeError, ValueError):
                rating = 0
            if rating < min_rating:
                continue
            prompt = item.get("prompt", "").strip()
            response = item.get("response", "").strip()
            notes = item.get("notes", "").strip()
            if not prompt or not response:
                continue
            f_out.write(f"User: {prompt}\nAssistant: {response}\n")
            if notes:
                f_out.write(f"Feedback: {notes}\n")
            f_out.write("\n")
            kept += 1

    print(f"Wrote {kept} examples to {out}" + (f" ({skipped} malformed lines skipped)" if skipped else ""))


def main():
    parser = argparse.ArgumentParser(description="Build a training corpus from hub feedback")
    parser.add_argument("--feedback", default="data/hub_feedback.jsonl")
    parser.add_argument("--out", default="data/hub_feedback_corpus.txt")
    parser.add_argument("--min-rating", type=int, default=4)
    args = parser.parse_args()
    build_corpus(args.feedback, args.out, args.min_rating)


if __name__ == "__main__":
    main()
