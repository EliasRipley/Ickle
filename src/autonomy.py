from dataclasses import dataclass
from typing import Any

from src.icklization import ick
from src.policy_loader import load_policy_safe


@dataclass
class AutonomyMode:
    name: str
    description: str
    tone: str


def default_mode_name() -> str:
    policy = load_policy_safe() or {}
    return str(policy.get("autonomy", {}).get("default_mode", "balanced"))


def _build_modes() -> dict[str, AutonomyMode]:
    raw = ick.autonomy_modes()
    modes: dict[str, AutonomyMode] = {}
    for key, cfg in raw.items():
        modes[key] = AutonomyMode(
            name=key,
            description=str(cfg.get("description", "")),
            tone=str(cfg.get("tone", "neutral")),
        )
    return modes


def get_mode(name: str) -> AutonomyMode:
    modes = _build_modes()
    key = name.strip().lower()
    if key not in modes:
        valid = ", ".join(sorted(modes.keys()))
        raise ValueError(f"Unknown autonomy mode '{name}'. Valid: {valid}")
    return modes[key]


def policy_summary(mode_name: str) -> str:
    mode = get_mode(mode_name)
    return f"mode={mode.name}; tone={mode.tone}; notes={mode.description}"
