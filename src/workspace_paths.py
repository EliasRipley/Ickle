from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def get_app_root() -> Path:
    """Where bundled non-Python assets (web/, config/, schemas/, sql/) actually
    live at runtime. In a normal checkout that's the project root. In a
    PyInstaller build, `datas=[...]` entries get unpacked under `sys._MEIPASS`
    (the `_internal/` folder for --onedir builds, a temp dir for --onefile) --
    NOT necessarily the process's current working directory, which is what a
    plain relative-path default like `Path("web").resolve()` assumes and gets
    wrong the moment the packaged .exe is invoked from anywhere else.
    """
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass)
    return get_project_root()


def get_training_root() -> Path:
    configured = os.getenv("ICKLE_TRAINING_ROOT")
    if not configured:
        project_root = get_project_root()
        in_repo = project_root / "IckleTraining"
        if in_repo.is_dir():
            return in_repo
        configured = str(in_repo)
    return Path(configured).resolve()


def get_training_corpus_dir(training_root: str | Path | None = None) -> Path:
    root = Path(training_root).resolve() if training_root is not None else get_training_root()
    return root / "corpuses"


def get_training_corpus_path(filename: str, training_root: str | Path | None = None) -> Path:
    return get_training_corpus_dir(training_root) / filename


@dataclass
class WorkspaceLayout:
    project_root: Path
    training_root: Path

    @property
    def separated(self) -> bool:
        return self.project_root != self.training_root

    @property
    def training_inside_project(self) -> bool:
        try:
            self.training_root.relative_to(self.project_root)
            return True
        except ValueError:
            return False

    @property
    def project_inside_training(self) -> bool:
        try:
            self.project_root.relative_to(self.training_root)
            return True
        except ValueError:
            return False


def detect_workspace_layout() -> WorkspaceLayout:
    return WorkspaceLayout(
        project_root=get_project_root(),
        training_root=get_training_root(),
    )
