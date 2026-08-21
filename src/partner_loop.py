import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.capabilities import check_capability
from src.clarify import detect_vague_prompt
from src.icklization import ick


@dataclass
class PartnerDecision:
    prompt: str
    supported: bool
    needs_clarification: bool
    decision: str
    reason: str
    next_step: str


class PartnerLoop:
    """Human-first ILM control loop.

    Cycle:
    1) Clarify (no guessing)
    2) Capability truth check
    3) Plan with explicit next step
    4) Journal decision for later learning
    """

    def __init__(self, journal_path: str = "data/partner_journal.jsonl"):
        self.journal_path = Path(journal_path)
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)

    def decide(self, prompt: str) -> PartnerDecision:
        if detect_vague_prompt(prompt):
            return PartnerDecision(
                prompt=prompt,
                supported=False,
                needs_clarification=True,
                decision="ask_clarification",
                reason=ick.partner_loop_text("underspecified_reason"),
                next_step=ick.partner_loop_text("underspecified_next_step"),
            )

        cap = check_capability(prompt)
        if not cap.supported:
            suggestion = cap.suggestion or ick.partner_loop_text("unsupported_suggestion_format")
            return PartnerDecision(
                prompt=prompt,
                supported=False,
                needs_clarification=False,
                decision="declare_limit",
                reason=cap.summary,
                next_step=suggestion,
            )

        return PartnerDecision(
            prompt=prompt,
            supported=True,
            needs_clarification=False,
            decision="proceed",
            reason=cap.summary,
            next_step=ick.partner_loop_text("default_next_step"),
        )

    def journal(self, decision: PartnerDecision):
        row = asdict(decision)
        row["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        with self.journal_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Human-first ILM partner loop")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--journal", default="data/partner_journal.jsonl")
    args = parser.parse_args()

    loop = PartnerLoop(journal_path=args.journal)
    decision = loop.decide(args.prompt)
    loop.journal(decision)
    print(json.dumps(asdict(decision), indent=2))


if __name__ == "__main__":
    main()
