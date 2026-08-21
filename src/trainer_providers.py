"""Trainer Integration Layer â€” Phase 1: Provider registry, budget ledger, adapter-style cloud providers.

Supports:
- Multi-provider API key config with hard budgets (requests/day, tokens/day, $/day, concurrency)
- Budget enforcement ledger (JSONL append-only)
- Shared `generate()` interface across providers

Advanced, CLI-only surface: managed via `python -m src.app trainer-provider` /
`trainer-budget` (see src/trainer_providers_cli.py). There is no web UI or
control-server HTTP endpoint for this -- it is not exposed anywhere under
/api/, unlike the main chat/training control API in serve_control.py.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _today_key() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


@dataclass
class ProviderConfig:
    provider: str
    model: str
    enabled: bool = True
    api_key_env: str = ""
    base_url: str = ""
    endpoint: str = "/chat/completions"
    max_requests_day: int = 100
    max_tokens_day: int = 100000
    max_usd_day: float = 5.0
    max_concurrent: int = 4
    extra_headers: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "enabled": self.enabled,
            "api_key_env": self.api_key_env,
            "base_url": self.base_url,
            "endpoint": self.endpoint,
            "max_requests_day": self.max_requests_day,
            "max_tokens_day": self.max_tokens_day,
            "max_usd_day": self.max_usd_day,
            "max_concurrent": self.max_concurrent,
            "extra_headers": self.extra_headers,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ProviderConfig":
        return ProviderConfig(
            provider=str(data.get("provider", "")),
            model=str(data.get("model", "")),
            enabled=bool(data.get("enabled", True)),
            api_key_env=str(data.get("api_key_env", "")),
            base_url=str(data.get("base_url", "")),
            endpoint=str(data.get("endpoint", "/chat/completions")),
            max_requests_day=int(data.get("max_requests_day", 100)),
            max_tokens_day=int(data.get("max_tokens_day", 100000)),
            max_usd_day=float(data.get("max_usd_day", 5.0)),
            max_concurrent=int(data.get("max_concurrent", 4)),
            extra_headers=data.get("extra_headers", {}),
        )


@dataclass
class UsageRecord:
    provider: str
    model: str
    timestamp_utc: str = field(default_factory=_utc_now)
    request_tokens: int = 0
    response_tokens: int = 0
    total_tokens: int = 0
    usd_cost: float = 0.0
    duration_ms: float = 0.0
    success: bool = True
    error: str = ""


PRICING_PER_1K_INPUT: dict[str, float] = {
    "gpt-4o": 0.00250,
    "gpt-4o-mini": 0.00015,
    "gpt-4-turbo": 0.01000,
    "gpt-3.5-turbo": 0.00050,
    "claude-3-opus": 0.01500,
    "claude-3-sonnet": 0.00300,
    "claude-3-haiku": 0.00025,
    "claude-3.5-sonnet": 0.00300,
}

PRICING_PER_1K_OUTPUT: dict[str, float] = {
    "gpt-4o": 0.01000,
    "gpt-4o-mini": 0.00060,
    "gpt-4-turbo": 0.03000,
    "gpt-3.5-turbo": 0.00150,
    "claude-3-opus": 0.07500,
    "claude-3-sonnet": 0.01500,
    "claude-3-haiku": 0.00125,
    "claude-3.5-sonnet": 0.01500,
}

DEFAULT_PRICING = {"input": 0.003, "output": 0.015}


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    in_price = PRICING_PER_1K_INPUT.get(model, DEFAULT_PRICING["input"])
    out_price = PRICING_PER_1K_OUTPUT.get(model, DEFAULT_PRICING["output"])
    return (input_tokens * in_price + output_tokens * out_price) / 1000.0


def _count_tokens_approx(text: str) -> int:
    return max(1, len(str(text).split()) * 4 // 3)


class BudgetLedger:
    """Append-only JSONL budget ledger with in-memory daily aggregation."""

    def __init__(self, ledger_path: str = "data/trainer_usage.jsonl"):
        self._path = Path(ledger_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._daily: dict[str, dict[str, int | float]] = {}
        self._today = _today_key()
        self._reload()

    def _reload(self):
        today = _today_key()
        self._today = today
        self._daily = {}
        if not self._path.exists():
            return
        with self._lock:
            with self._path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    ts = str(rec.get("timestamp_utc", ""))[:10]
                    if ts != today:
                        continue
                    provider = str(rec.get("provider", ""))
                    day = self._daily.setdefault(provider, {
                        "requests": 0, "tokens": 0, "usd": 0.0,
                    })
                    day["requests"] = int(day.get("requests", 0)) + 1
                    day["tokens"] = int(day.get("tokens", 0)) + int(rec.get("total_tokens", 0))
                    day["usd"] = float(day.get("usd", 0.0)) + float(rec.get("usd_cost", 0.0))

    def record(self, usage: UsageRecord):
        if self._today != _today_key():
            self._reload()
        provider = usage.provider
        with self._lock:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "provider": provider,
                    "model": usage.model,
                    "timestamp_utc": usage.timestamp_utc,
                    "request_tokens": usage.request_tokens,
                    "response_tokens": usage.response_tokens,
                    "total_tokens": usage.total_tokens,
                    "usd_cost": round(usage.usd_cost, 6),
                    "duration_ms": round(usage.duration_ms, 1),
                    "success": usage.success,
                    "error": usage.error,
                }, ensure_ascii=False) + "\n")
            day = self._daily.setdefault(provider, {
                "requests": 0, "tokens": 0, "usd": 0.0,
            })
            day["requests"] = int(day.get("requests", 0)) + 1
            day["tokens"] = int(day.get("tokens", 0)) + usage.total_tokens
            day["usd"] = float(day.get("usd", 0.0)) + usage.usd_cost

    def check_budget(self, provider: str, config: ProviderConfig) -> str | None:
        if self._today != _today_key():
            self._reload()
        day = self._daily.get(provider, {"requests": 0, "tokens": 0, "usd": 0.0})
        if int(day.get("requests", 0)) >= config.max_requests_day:
            return f"Daily request limit reached ({config.max_requests_day}) for {provider}"
        if int(day.get("tokens", 0)) >= config.max_tokens_day:
            return f"Daily token limit reached ({config.max_tokens_day}) for {provider}"
        if float(day.get("usd", 0.0)) >= config.max_usd_day:
            return f"Daily USD limit reached (${config.max_usd_day:.2f}) for {provider}"
        return None

    def daily_usage(self, provider: str = "") -> dict[str, Any]:
        if self._today != _today_key():
            self._reload()
        if provider:
            return {
                "date": self._today,
                "usage": self._daily.get(provider, {"requests": 0, "tokens": 0, "usd": 0.0}),
            }
        return {"date": self._today, "usage": dict(self._daily)}

    def reset_daily(self, provider: str = ""):
        with self._lock:
            if provider and provider in self._daily:
                self._daily[provider] = {"requests": 0, "tokens": 0, "usd": 0.0}
            elif not provider:
                self._daily = {}


class ProviderRegistry:
    """Multi-provider registry with budget enforcement and shared generate() interface."""

    def __init__(
        self,
        registry_path: str = "data/trainer_providers.json",
        ledger_path: str = "data/trainer_usage.jsonl",
    ):
        self._path = Path(registry_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # RLock, not Lock: add_provider()/remove_provider() hold this lock
        # while calling _save(), which also acquires it -- a plain Lock
        # self-deadlocks on every single add/remove call.
        self._lock = threading.RLock()
        self._ledger = BudgetLedger(ledger_path)
        self._semaphores: dict[str, threading.BoundedSemaphore] = {}
        self._providers: dict[str, ProviderConfig] = {}
        self._load()

    def _load(self):
        if self._path.exists():
            with self._path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data if isinstance(data, list) else data.get("providers", []):
                p = ProviderConfig.from_dict(item)
                key = f"{p.provider}:{p.model}"
                self._providers[key] = p
                self._semaphores[key] = threading.BoundedSemaphore(p.max_concurrent)

    def _save(self):
        with self._lock:
            self._path.write_text(
                json.dumps([p.as_dict() for p in self._providers.values()], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    @property
    def ledger(self) -> BudgetLedger:
        return self._ledger

    def add_provider(self, config: ProviderConfig) -> dict[str, Any]:
        key = f"{config.provider}:{config.model}"
        with self._lock:
            self._providers[key] = config
            self._semaphores[key] = threading.BoundedSemaphore(config.max_concurrent)
            self._save()
        return {"key": key, "status": "added"}

    def remove_provider(self, key: str) -> bool:
        with self._lock:
            removed = self._providers.pop(key, None) is not None
            self._semaphores.pop(key, None)
            if removed:
                self._save()
        return removed

    def list_providers(self) -> list[dict[str, Any]]:
        return [
            {"key": key, **cfg.as_dict()}
            for key, cfg in sorted(self._providers.items())
        ]

    def get_provider(self, key: str) -> ProviderConfig | None:
        return self._providers.get(key)

    def set_enabled(self, key: str, enabled: bool) -> bool:
        cfg = self._providers.get(key)
        if not cfg:
            return False
        cfg.enabled = enabled
        self._save()
        return True

    def generate(
        self,
        key: str,
        prompt: str,
        *,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> tuple[str, UsageRecord]:
        cfg = self._providers.get(key)
        if not cfg:
            raise ValueError(f"Unknown provider: {key}")
        if not cfg.enabled:
            raise PermissionError(f"Provider {key} is disabled")

        budget_err = self._ledger.check_budget(cfg.provider, cfg)
        if budget_err:
            raise PermissionError(budget_err)

        api_key = os.getenv(cfg.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(f"Missing API key: set {cfg.api_key_env}")

        sem = self._semaphores.get(key)
        if sem is None:
            raise RuntimeError(f"No semaphore for {key}")

        acquired = sem.acquire(timeout=60)
        if not acquired:
            raise RuntimeError(f"Concurrency limit ({cfg.max_concurrent}) reached for {key}")

        t0 = time.monotonic()
        try:
            input_tokens = _count_tokens_approx(prompt)
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            base_url = cfg.base_url or "https://api.openai.com/v1"
            url = base_url.rstrip("/") + cfg.endpoint

            payload = {
                "model": cfg.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                **cfg.extra_headers,
            }
            req = Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            try:
                with urlopen(req, timeout=120) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")[:500]
                exc.close()
                elapsed_ms = (time.monotonic() - t0) * 1000
                rec = UsageRecord(
                    provider=cfg.provider, model=cfg.model,
                    request_tokens=input_tokens, total_tokens=input_tokens,
                    duration_ms=elapsed_ms, success=False,
                    error=f"HTTP {exc.code}: {error_body}",
                )
                self._ledger.record(rec)
                raise RuntimeError(f"Provider {key} returned {exc.code}: {error_body}")

            response_text = ""
            choices = body.get("choices", [])
            if choices:
                msg = choices[0].get("message", {})
                response_text = str(msg.get("content", ""))
            elif isinstance(body.get("output_text"), str):
                response_text = body["output_text"]
            else:
                output = body.get("output", [])
                if output and isinstance(output, list):
                    for item in output:
                        content = item.get("content", []) if isinstance(item, dict) else []
                        for c in content:
                            text = c.get("text") if isinstance(c, dict) else None
                            if text:
                                response_text = text
                                break

            output_tokens = _count_tokens_approx(response_text)
            total_tokens = input_tokens + output_tokens
            cost = _estimate_cost(cfg.model, input_tokens, output_tokens)
            elapsed_ms = (time.monotonic() - t0) * 1000

            rec = UsageRecord(
                provider=cfg.provider, model=cfg.model,
                request_tokens=input_tokens, response_tokens=output_tokens,
                total_tokens=total_tokens, usd_cost=cost,
                duration_ms=elapsed_ms, success=True,
            )
            self._ledger.record(rec)
            return response_text, rec
        finally:
            sem.release()


_registry_singleton: ProviderRegistry | None = None


def get_provider_registry() -> ProviderRegistry:
    global _registry_singleton
    if _registry_singleton is None:
        _registry_singleton = ProviderRegistry()
    return _registry_singleton
