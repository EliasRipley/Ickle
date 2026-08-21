from pathlib import Path

from src.runtime_flags import RuntimeFlagsStore
from src.state_store import ILMStateStore


def _valid_model_path(path: Path) -> bool:
    return (
        path.exists()
        and path.is_file()
        and path.suffix.lower() == ".pt"
        and not path.name.endswith(".checkpoint.pt")
    )


def _preferred_model_from_runtime() -> str | None:
    try:
        flags = RuntimeFlagsStore().get_flags()
    except Exception:  # noqa: BLE001
        flags = {}
    preferred = str(flags.get("current_model", "")).strip() if isinstance(flags, dict) else ""
    if preferred:
        p = Path(preferred).resolve()
        if _valid_model_path(p):
            return str(p.as_posix())

    try:
        preferred_state = ILMStateStore().get_preference("current_model")
    except Exception:  # noqa: BLE001
        preferred_state = ""
    if preferred_state:
        p = Path(str(preferred_state).strip()).resolve()
        if _valid_model_path(p):
            return str(p.as_posix())
    return None


def resolve_default_model() -> str:
    """Return the most recently modified .pt file in models/ (or
    models/candidates/), ignoring checkpoints.

    The AI Teacher / P2P network promotes models by copying candidate files
    into models/, which updates their modification time.  This function
    naturally selects the most-recently-promoted model. models/candidates/
    is scanned too -- that's where every training task actually writes its
    output, and a normally-completed run (as opposed to one stopped and
    promotion-gated mid-way) never gets copied into models/ at all, so
    without this a freshly-trained model was invisible to chat entirely.
    """
    preferred = _preferred_model_from_runtime()
    if preferred:
        return preferred

    model_root = Path("models")
    if not model_root.exists():
        raise FileNotFoundError(
            "No 'models/' directory found. Train or install a model first."
        )
    search_dirs = [model_root, model_root / "candidates"]
    candidates = sorted(
        [
            p
            for search_dir in search_dirs
            if search_dir.exists()
            for p in search_dir.glob("*.pt")
            if _valid_model_path(p)
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "No model files (*.pt) found in 'models/' or 'models/candidates/'. Train or install a model first."
        )
    return str(candidates[0].resolve().as_posix())
