from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from src.data_quality import is_contaminated as _is_contaminated_phrase
from src.data_quality import minhash_deduplicate
from src.workspace_paths import get_training_root

MINHASH_DEDUP_THRESHOLD = 0.85

TEMPLATE_QA_RE = re.compile(r"^(what|who|where|when|why|how)\b.+\?$", flags=re.IGNORECASE)
RECOVERY_PAIRS: list[tuple[str, str]] = [
    (
        "Use our previous context and remind me of my preferred response style.",
        "You prefer concise practical answers, so I will keep this focused and actionable.",
    ),
    (
        "I am unsure about a fact that might have changed this week. How should you respond?",
        "I should state uncertainty clearly, avoid guessing, and verify with a current reliable source.",
    ),
    (
        "Continue from our previous topic and give one relevant next step.",
        "I will continue using the latest relevant context and propose one concrete next step.",
    ),
    (
        "Handle conflicting memory notes with an evidence-first approach.",
        "I should prefer corroborated evidence, flag uncertainty, and ask a clarifying question when needed.",
    ),
    (
        "Store web findings with source attribution and confidence labels.",
        "I should keep source attribution, confidence level, and only retain useful non-noisy facts.",
    ),
    (
        "Keep responses respectful while remaining technically precise.",
        "I should be direct, courteous, and explicit about assumptions, risks, and evidence.",
    ),
]


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _is_contaminated(text: str) -> bool:
    return _is_contaminated_phrase(_clean(text))


def _is_low_quality_pair(user: str, assistant: str, *, drop_qa_templates: bool) -> bool:
    u = _clean(user)
    a = _clean(assistant)
    if len(u) < 6 or len(a) < 20:
        return True
    if len(u) > 280 or len(a) > 520:
        return True
    if _is_contaminated(u) or _is_contaminated(a):
        return True
    if re.search(r"\b(\w+)(?:\s+\1){2,}\b", a.lower()):
        return True
    if drop_qa_templates and TEMPLATE_QA_RE.match(u):
        return True
    return False


def _parse_pairs(lines: list[str], *, drop_qa_templates: bool) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    pending_user = ""
    for raw in lines:
        line = _clean(raw)
        if not line:
            continue
        lower = line.lower()
        if lower.startswith("user:"):
            pending_user = _clean(line.split(":", 1)[1])
            continue
        if lower.startswith("ickle:") or lower.startswith("assistant:"):
            if not pending_user:
                continue
            answer = _clean(line.split(":", 1)[1])
            key = (pending_user.lower(), answer.lower())
            if key in seen:
                pending_user = ""
                continue
            seen.add(key)
            if _is_low_quality_pair(pending_user, answer, drop_qa_templates=drop_qa_templates):
                pending_user = ""
                continue
            pairs.append((pending_user, answer))
            pending_user = ""
    # The exact-match `seen` set above only catches byte-identical pairs.
    # A rephrased near-duplicate ("What's the capital of France?" vs "What
    # is the capital of France?") sailed straight through -- reuse the same
    # MinHash near-dedup already relied on by build_quality_filtered_corpus
    # instead of a second bespoke exact-match implementation.
    return minhash_deduplicate(pairs, threshold=MINHASH_DEDUP_THRESHOLD)


def _dialog_ratio(lines: list[str]) -> float:
    non_empty = [ln for ln in lines if _clean(ln)]
    if not non_empty:
        return 0.0
    dialog_prefixed = 0
    for line in non_empty:
        lower = line.lower().strip()
        if lower.startswith("user:") or lower.startswith("ickle:") or lower.startswith("assistant:"):
            dialog_prefixed += 1
    return dialog_prefixed / max(1, len(non_empty))


def _duplicate_ratio(lines: list[str]) -> float:
    normalized = [_clean(ln).lower() for ln in lines if _clean(ln)]
    if not normalized:
        return 0.0
    unique = len(set(normalized))
    return max(0.0, 1.0 - (unique / max(1, len(normalized))))


def sanitize_training_file(path: Path, *, apply: bool, drop_qa_templates: bool) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    original_lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    pairs = _parse_pairs(original_lines, drop_qa_templates=drop_qa_templates)
    lower_name = path.name.lower()
    if len(pairs) < 6 and ("memory_aware" in lower_name or "override_conversation" in lower_name):
        seen = {(u.lower(), a.lower()) for u, a in pairs}
        for user, assistant in RECOVERY_PAIRS:
            key = (user.lower(), assistant.lower())
            if key in seen:
                continue
            seen.add(key)
            pairs.append((user, assistant))

    cleaned_lines: list[str] = []
    for user, assistant in pairs:
        cleaned_lines.append(f"User: {user}")
        cleaned_lines.append(f"Ickle: {assistant}")
        cleaned_lines.append("")

    before_dialog_ratio = _dialog_ratio(original_lines)
    before_dup_ratio = _duplicate_ratio(original_lines)
    after_dialog_ratio = _dialog_ratio(cleaned_lines)
    after_dup_ratio = _duplicate_ratio(cleaned_lines)

    changed = cleaned_lines != original_lines
    if apply and changed:
        payload = "\n".join(cleaned_lines).rstrip()
        if payload:
            payload += "\n"
        path.write_text(payload, encoding="utf-8")

    return {
        "path": str(path),
        "exists": True,
        "changed": changed,
        "before": {
            "line_count": len(original_lines),
            "dialog_ratio": round(before_dialog_ratio, 4),
            "duplicate_ratio": round(before_dup_ratio, 4),
        },
        "after": {
            "line_count": len(cleaned_lines),
            "dialog_ratio": round(after_dialog_ratio, 4),
            "duplicate_ratio": round(after_dup_ratio, 4),
            "pair_count": len(pairs),
        },
    }


def run_sanitize_training_data(
    *,
    training_root: Path,
    apply: bool,
    drop_qa_templates: bool,
    include_files: list[str] | None = None,
) -> dict[str, Any]:
    default_files = [
        "memory_aware_training.txt",
        "override_conversation_training.txt",
        "natural_conversation_training.txt",
        "queued_wikipedia_learning.txt",
        "open_oasst1_stream.txt",
        "open_openhermes_2_5_stream.txt",
    ]
    rel_files = include_files or default_files
    reports: list[dict[str, Any]] = []
    changed = 0
    for rel in rel_files:
        report = sanitize_training_file(
            training_root / rel,
            apply=apply,
            drop_qa_templates=drop_qa_templates,
        )
        reports.append(report)
        if report.get("changed"):
            changed += 1
    return {
        "training_root": str(training_root),
        "apply": bool(apply),
        "drop_qa_templates": bool(drop_qa_templates),
        "file_count": len(reports),
        "changed_files": changed,
        "files": reports,
    }


def main():
    parser = argparse.ArgumentParser(description="Sanitize training corpora: dedupe, decontaminate, and reduce QA template skew.")
    parser.add_argument("--training-root", default=str(get_training_root()))
    parser.add_argument("--apply", action="store_true", help="Write changes in place. Default is preview only.")
    parser.add_argument(
        "--keep-qa-templates",
        action="store_true",
        help="Keep short template-style QA pairs (default removes them).",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=[],
        help="Optional list of relative file names under training root.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = run_sanitize_training_data(
        training_root=Path(args.training_root).resolve(),
        apply=bool(args.apply),
        drop_qa_templates=not bool(args.keep_qa_templates),
        include_files=[str(x) for x in list(args.files or [])] or None,
    )
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return
    print(
        f"training_root={report['training_root']} apply={report['apply']} "
        f"changed_files={report['changed_files']}/{report['file_count']}"
    )
    for row in report["files"]:
        if not row.get("exists"):
            print(f"[skip] {row['path']} (missing)")
            continue
        before = row["before"]
        after = row["after"]
        status = "changed" if row.get("changed") else "kept"
        print(
            f"[{status}] {row['path']} "
            f"dialog_ratio {before['dialog_ratio']:.4f}->{after['dialog_ratio']:.4f} "
            f"dup_ratio {before['duplicate_ratio']:.4f}->{after['duplicate_ratio']:.4f}"
        )


if __name__ == "__main__":
    main()
