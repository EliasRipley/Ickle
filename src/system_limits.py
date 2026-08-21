from dataclasses import dataclass, field

from src.policy_loader import load_policy_safe


def _load_limit_defaults() -> dict[str, int | bool]:
    defaults: dict[str, int | bool] = {
        "max_tool_calls": 8,
        "max_web_chars": 5000,
        "web_timeout_ms": 15000,
        "torch_threads": 4,
        "require_clarification_on_vague": True,
    }
    policy = load_policy_safe()
    limits = policy.get("limits", {}) if isinstance(policy, dict) else {}
    if not isinstance(limits, dict):
        return defaults

    def _as_int(name: str, current: int) -> int:
        try:
            return max(1, int(limits.get(name, current)))
        except Exception:  # noqa: BLE001
            return current

    defaults["max_tool_calls"] = _as_int("max_tool_calls", int(defaults["max_tool_calls"]))
    defaults["max_web_chars"] = _as_int("max_web_chars", int(defaults["max_web_chars"]))
    defaults["web_timeout_ms"] = _as_int("web_timeout_ms", int(defaults["web_timeout_ms"]))
    defaults["torch_threads"] = _as_int("torch_threads", int(defaults["torch_threads"]))

    if "require_clarification_on_vague" in limits:
        defaults["require_clarification_on_vague"] = bool(limits.get("require_clarification_on_vague"))
    return defaults


@dataclass
class SystemLimits:
    """Runtime limits to keep local usage bounded.

    Policy-driven fields (everything but max_new_tokens) default to None and
    are resolved from config/ilm_policy.toml in __post_init__, called fresh
    on every instantiation -- not read once at module-import time. ilm_chat.py
    builds a new SystemLimits() on every chat turn, so a stale import-time
    snapshot would mean editing ilm_policy.toml while Ickle is already
    running (a long-lived desktop app) could never take effect without a
    full process restart. Explicit constructor args (e.g.
    SystemLimits(torch_threads=4)) still take precedence over policy.

    None (not a module-level sentinel object) is used deliberately: a custom
    `_UNSET = object()` sentinel breaks under importlib.reload() (used by
    tests/test_policy_integration.py) -- reload rebinds the module-level name
    to a *new* object, while a SystemLimits class imported before the reload
    still has dataclass field defaults baked in as the *old* object, so an
    `is _UNSET` check against the reloaded name silently never matches.
    None is a language-level singleton unaffected by module reloads, and
    none of these fields have a legitimate real value of None.
    """

    max_new_tokens: int = 256
    max_tool_calls: int | None = None
    max_web_chars: int | None = None
    web_timeout_ms: int | None = None
    torch_threads: int | None = None
    require_clarification_on_vague: bool | None = None

    def __post_init__(self):
        if (
            self.max_tool_calls is None or self.max_web_chars is None
            or self.web_timeout_ms is None or self.torch_threads is None
            or self.require_clarification_on_vague is None
        ):
            defaults = _load_limit_defaults()
            if self.max_tool_calls is None:
                self.max_tool_calls = int(defaults["max_tool_calls"])
            if self.max_web_chars is None:
                self.max_web_chars = int(defaults["max_web_chars"])
            if self.web_timeout_ms is None:
                self.web_timeout_ms = int(defaults["web_timeout_ms"])
            if self.torch_threads is None:
                self.torch_threads = int(defaults["torch_threads"])
            if self.require_clarification_on_vague is None:
                self.require_clarification_on_vague = bool(defaults["require_clarification_on_vague"])


def clamp_new_tokens(requested: int, limit: int) -> int:
    return max(1, min(requested, limit))
