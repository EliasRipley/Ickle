from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class DiffExample:
    file_path: str
    old_content: str
    new_content: str
    description: str = ""
    tests_passed: bool = False
    lint_passed: bool = False
    commit_hash: str = ""
    commit_message: str = ""


def _run_git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git"] + list(args),
        cwd=str(repo),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def extract_bug_fix_pairs(repo_root: str, max_commits: int = 200) -> list[DiffExample]:
    """Extract bug-fix pairs from a repo by scanning commits with 'fix' or 'bug' in message."""
    repo = Path(repo_root)
    if not (repo / ".git").exists():
        raise FileNotFoundError(f"No .git directory found in {repo_root}")

    pairs: list[DiffExample] = []

    log = _run_git(repo, "log", "--oneline", f"-n{max_commits}", "--grep=fix", "--grep=bug", "--all-match", "-i", "--no-merges")
    if not log:
        return pairs

    for line in log.splitlines():
        line = line.strip()
        if not line:
            continue
        commit_hash = line.split()[0]
        msg = _run_git(repo, "log", "-1", "--format=%B", commit_hash)

        changed_files = _run_git(repo, "diff-tree", "--no-commit-id", "--name-only", "-r", commit_hash, "--diff-filter=AM")
        if not changed_files:
            continue

        parent = _run_git(repo, "rev-parse", f"{commit_hash}^")
        if not parent:
            continue

        for fname in changed_files.splitlines():
            fname = fname.strip()
            if not fname:
                continue

            old_content = _run_git(repo, "show", f"{parent}:{fname}")
            new_content = _run_git(repo, "show", f"{commit_hash}:{fname}")
            if old_content == "" and new_content == "":
                continue

            pairs.append(DiffExample(
                file_path=fname,
                old_content=old_content,
                new_content=new_content,
                description=msg[:200] if msg else "",
                commit_hash=commit_hash,
                commit_message=msg[:300] if msg else "",
            ))

            if len(pairs) >= max_commits * 3:
                break

        if len(pairs) >= max_commits * 3:
            break

    return pairs


def extract_test_driven_edits(repo_root: str, max_commits: int = 100) -> list[DiffExample]:
    """Extract edits where both test files and source files were changed in same commit."""
    repo = Path(repo_root)
    if not (repo / ".git").exists():
        raise FileNotFoundError(f"No .git directory found in {repo_root}")

    pairs: list[DiffExample] = []

    log = _run_git(repo, "log", "--oneline", f"-n{max_commits * 2}", "--no-merges")
    if not log:
        return pairs

    for line in log.splitlines():
        line = line.strip()
        if not line:
            continue
        commit_hash = line.split()[0]

        changed = _run_git(repo, "diff-tree", "--no-commit-id", "--name-only", "-r", commit_hash)
        if not changed:
            continue

        files = [f.strip() for f in changed.splitlines() if f.strip()]
        has_test = any("test" in f.lower() for f in files)
        has_src = any("test" not in f.lower() and f.endswith((".py", ".js", ".ts", ".rs", ".go", ".java")) for f in files)

        if not (has_test and has_src):
            continue

        msg = _run_git(repo, "log", "-1", "--format=%B", commit_hash)
        parent = _run_git(repo, "rev-parse", f"{commit_hash}^")
        if not parent:
            continue

        for fname in files:
            old_content = _run_git(repo, "show", f"{parent}:{fname}")
            new_content = _run_git(repo, "show", f"{commit_hash}:{fname}")
            if not old_content and not new_content:
                continue

            pairs.append(DiffExample(
                file_path=fname,
                old_content=old_content,
                new_content=new_content,
                description=f"test-driven: {msg[:150]}" if msg else "test-driven edit",
                commit_hash=commit_hash,
                commit_message=msg[:300] if msg else "",
            ))

            if len(pairs) >= max_commits * 3:
                break

        if len(pairs) >= max_commits * 3:
            break

    return pairs


def extract_multifile_changes(repo_root: str, max_commits: int = 50) -> list[dict[str, Any]]:
    """Extract commits that modify multiple files as a single logical change."""
    repo = Path(repo_root)
    if not (repo / ".git").exists():
        raise FileNotFoundError(f"No .git directory found in {repo_root}")

    groups: list[dict[str, Any]] = []

    log = _run_git(repo, "log", "--oneline", f"-n{max_commits * 2}", "--no-merges")
    if not log:
        return groups

    for line in log.splitlines():
        line = line.strip()
        if not line:
            continue
        commit_hash = line.split()[0]

        changed = _run_git(repo, "diff-tree", "--no-commit-id", "--name-only", "-r", commit_hash)
        if not changed:
            continue

        files = [f.strip() for f in changed.splitlines() if f.strip()]
        if len(files) < 2:
            continue

        msg = _run_git(repo, "log", "-1", "--format=%B", commit_hash)
        parent = _run_git(repo, "rev-parse", f"{commit_hash}^")
        if not parent:
            continue

        file_changes: list[dict[str, str]] = []
        for fname in files:
            old_content = _run_git(repo, "show", f"{parent}:{fname}")
            new_content = _run_git(repo, "show", f"{commit_hash}:{fname}")
            file_changes.append({
                "file": fname,
                "old_content": old_content[:5000] if old_content else "",
                "new_content": new_content[:5000] if new_content else "",
            })

        groups.append({
            "commit": commit_hash,
            "message": msg[:300] if msg else "",
            "files_count": len(files),
            "changes": file_changes,
        })

        if len(groups) >= max_commits:
            break

    return groups


def build_code_corpus_from_repo(repo_root: str, out_path: str, max_commits: int = 100) -> int:
    """Build a training corpus from a repo's history â€” bug-fixes + test-driven edits.

    Output format (JSONL):
    {prompt, response, tests_passed, lint_passed, source, commit, file}
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out.open("w", encoding="utf-8") as f:
        for pair in extract_bug_fix_pairs(repo_root, max_commits):
            if not pair.old_content or not pair.new_content:
                continue
            row = {
                "prompt": f"Fix the code in {pair.file_path}: {pair.description}",
                "response": pair.new_content[:4000],
                "pre_fix_code": pair.old_content[:4000],
                "tests_passed": pair.tests_passed,
                "lint_passed": pair.lint_passed,
                "source": "bug_fix",
                "commit": pair.commit_hash,
                "file": pair.file_path,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

        for pair in extract_test_driven_edits(repo_root, max_commits):
            if not pair.old_content or not pair.new_content:
                continue
            row = {
                "prompt": f"Implement this change: {pair.description}",
                "response": pair.new_content[:4000],
                "pre_change_code": pair.old_content[:4000],
                "tests_passed": pair.tests_passed,
                "lint_passed": pair.lint_passed,
                "source": "test_driven_edit",
                "commit": pair.commit_hash,
                "file": pair.file_path,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    return written


def build_patch_corpus_from_repo(repo_root: str, out_path: str, max_commits: int = 50) -> int:
    """Build a compact patch-fix corpus: (pre-code, post-code, instruction).

    Useful for training code-repair models.
    """
    repo = Path(repo_root)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with out.open("w", encoding="utf-8") as f:
        for pair in extract_bug_fix_pairs(repo_root, max_commits):
            if not pair.old_content and not pair.new_content:
                continue
            if pair.old_content == pair.new_content:
                continue
            row = {
                "instruction": pair.description or f"Fix bug in {pair.file_path}",
                "input": pair.old_content[:3000],
                "output": pair.new_content[:3000],
                "file": pair.file_path,
                "commit": pair.commit_hash,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    return written


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Build code-native training data from git repos")
    ap.add_argument("repo", help="Path to git repository")
    ap.add_argument("--out", default="data/code_corpus.jsonl")
    ap.add_argument("--mode", default="code", choices=["code", "patch", "bug-fix", "test-driven", "multifile"])
    ap.add_argument("--max-commits", type=int, default=100)
    ap.add_argument("--out-patch", default="data/patch_corpus.jsonl")
    args = ap.parse_args()

    if args.mode == "code":
        n = build_code_corpus_from_repo(args.repo, args.out, args.max_commits)
        print(f"Wrote {n} rows to {args.out}")
    elif args.mode == "patch":
        n = build_patch_corpus_from_repo(args.repo, args.out_patch, args.max_commits)
        print(f"Wrote {n} patch pairs to {args.out_patch}")
    elif args.mode == "bug-fix":
        pairs = extract_bug_fix_pairs(args.repo, args.max_commits)
        print(f"Found {len(pairs)} bug-fix pairs")
    elif args.mode == "test-driven":
        pairs = extract_test_driven_edits(args.repo, args.max_commits)
        print(f"Found {len(pairs)} test-driven edits")
    elif args.mode == "multifile":
        groups = extract_multifile_changes(args.repo, args.max_commits)
        print(f"Found {len(groups)} multifile change groups")


if __name__ == "__main__":
    main()
