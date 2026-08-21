import tomllib
from pathlib import Path


def load_policy(path: str = "config/ilm_policy.toml") -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Policy file not found: {path}")
    with p.open("rb") as f:
        return tomllib.load(f)


def load_policy_safe(path: str = "config/ilm_policy.toml") -> dict:
    try:
        policy = load_policy(path)
    except Exception:  # noqa: BLE001
        return {}
    return policy if isinstance(policy, dict) else {}
