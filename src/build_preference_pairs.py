from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FeedbackRow:
    prompt: str
    response: str
    rating: int


def _read_feedback_rows(path: Path) -> list[FeedbackRow]:
    rows: list[FeedbackRow] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError:
                continue
            prompt = str(item.get("prompt", "")).strip()
            response = str(item.get("response", "")).strip()
            if not prompt or not response:
                continue
            try:
                rating = int(item.get("rating", 0))
            except (TypeError, ValueError):
                rating = 0
            rows.append(FeedbackRow(prompt=prompt, response=response, rating=rating))
    return rows


def build_preference_pairs(
    feedback_path: str,
    out_path: str,
    *,
    min_rating_gap: int = 2,
    min_chosen_rating: int = 4,
) -> dict[str, int]:
    src = Path(feedback_path)
    out = Path(out_path)
    if not src.exists():
        raise FileNotFoundError(f"feedback file not found: {src}")

    rows = _read_feedback_rows(src)
    by_prompt: dict[str, list[FeedbackRow]] = defaultdict(list)
    for row in rows:
        by_prompt[row.prompt].append(row)

    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    used_prompts = 0

    with out.open("w", encoding="utf-8") as f:
        for prompt, variants in by_prompt.items():
            if len(variants) < 2:
                skipped += 1
                continue

            sorted_rows = sorted(variants, key=lambda r: r.rating, reverse=True)
            chosen = sorted_rows[0]
            rejected = sorted_rows[-1]

            if chosen.response == rejected.response:
                skipped += 1
                continue
            if chosen.rating < min_chosen_rating:
                skipped += 1
                continue
            if (chosen.rating - rejected.rating) < min_rating_gap:
                skipped += 1
                continue

            payload = {
                "prompt": prompt,
                "chosen": chosen.response,
                "rejected": rejected.response,
                "chosen_rating": chosen.rating,
                "rejected_rating": rejected.rating,
                "source": "hub_feedback",
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            written += 1
            used_prompts += 1

    return {
        "rows_total": len(rows),
        "prompts_total": len(by_prompt),
        "prompts_used": used_prompts,
        "pairs_written": written,
        "prompts_skipped": skipped,
    }


def main():
    parser = argparse.ArgumentParser(description="Build DPO-style preference pairs from hub feedback JSONL.")
    parser.add_argument("--feedback", default="data/hub_feedback.jsonl")
    parser.add_argument("--out", default="data/hub_feedback_pairs.jsonl")
    parser.add_argument("--min-rating-gap", type=int, default=2)
    parser.add_argument("--min-chosen-rating", type=int, default=4)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    stats = build_preference_pairs(
        args.feedback,
        args.out,
        min_rating_gap=max(1, int(args.min_rating_gap)),
        min_chosen_rating=max(1, int(args.min_chosen_rating)),
    )

    if args.json:
        print(json.dumps(stats, indent=2, ensure_ascii=False))
    else:
        print(f"feedback rows: {stats['rows_total']}")
        print(f"prompts: {stats['prompts_total']}")
        print(f"pairs written: {stats['pairs_written']}")
        print(f"prompts used: {stats['prompts_used']}")
        print(f"prompts skipped: {stats['prompts_skipped']}")
        print(f"saved preference pairs: {args.out}")


if __name__ == "__main__":
    main()

