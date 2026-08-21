from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from src.workspace_paths import detect_workspace_layout


@dataclass
class WorkspaceCheckItem:
    area: str
    status: str
    detail: str


def _find_large_corpus_files(root: Path, max_items: int = 20) -> list[tuple[str, int]]:
    corpus_ext = {".txt", ".jsonl", ".json"}
    out: list[tuple[str, int]] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in corpus_ext:
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size >= 5_000_000:
            out.append((str(path), size))
    out.sort(key=lambda item: item[1], reverse=True)
    return out[:max_items]


def collect_workspace_checks() -> list[WorkspaceCheckItem]:
    layout = detect_workspace_layout()
    checks: list[WorkspaceCheckItem] = []

    if not layout.separated:
        checks.append(
            WorkspaceCheckItem(
                area="workspace_separation",
                status="error",
                detail="Project and training roots resolve to the same path.",
            )
        )
    elif layout.training_inside_project:
        checks.append(
            WorkspaceCheckItem(
                area="workspace_separation",
                status="warning",
                detail=(
                    "Training root is inside project root. This can mix corpora and app runtime files."
                ),
            )
        )
    elif layout.project_inside_training:
        checks.append(
            WorkspaceCheckItem(
                area="workspace_separation",
                status="warning",
                detail=(
                    "Project root is inside training root. Keep code and training data in sibling directories."
                ),
            )
        )
    else:
        checks.append(
            WorkspaceCheckItem(
                area="workspace_separation",
                status="ok",
                detail=f"Separated roots detected: project={layout.project_root}, training={layout.training_root}",
            )
        )

    training_exists = layout.training_root.exists()
    checks.append(
        WorkspaceCheckItem(
            area="training_root_presence",
            status="ok" if training_exists else "warning",
            detail=(
                f"Training root {'exists' if training_exists else 'is missing'} at {layout.training_root}"
            ),
        )
    )

    project_large = _find_large_corpus_files(layout.project_root / "data")
    if project_large:
        preview = "; ".join([f"{Path(path).name} ({size // 1_000_000}MB)" for path, size in project_large[:4]])
        checks.append(
            WorkspaceCheckItem(
                area="large_local_corpora_in_project_data",
                status="warning",
                detail=(
                    "Large corpus-like files found under Ickle data/. Consider moving canonical corpora to "
                    f"IckleTraining. Examples: {preview}"
                ),
            )
        )
    else:
        checks.append(
            WorkspaceCheckItem(
                area="large_local_corpora_in_project_data",
                status="ok",
                detail="No large corpus-like files detected under project data/.",
            )
        )

    if training_exists:
        training_files = list(layout.training_root.glob("*.txt"))
        checks.append(
            WorkspaceCheckItem(
                area="training_dataset_files",
                status="ok" if training_files else "warning",
                detail=f"Detected {len(training_files)} top-level .txt files in training root.",
            )
        )

    return checks


def main():
    parser = argparse.ArgumentParser(description="Check workspace separation between Ickle and IckleTraining.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    checks = collect_workspace_checks()
    if args.json:
        print(json.dumps([asdict(c) for c in checks], indent=2))
        return
    for item in checks:
        print(f"[{item.status}] {item.area}: {item.detail}")


if __name__ == "__main__":
    main()

