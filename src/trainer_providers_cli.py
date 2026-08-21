"""CLI for trainer provider management and budget commands."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.trainer_providers import ProviderConfig, ProviderRegistry, get_provider_registry


def _print_json(data):
    print(json.dumps(data, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="Trainer provider and budget management")
    sub = parser.add_subparsers(dest="action", required=True)

    p_add = sub.add_parser("add", help="Add a provider")
    p_add.add_argument("--provider", required=True, help="Provider name (openai, anthropic, etc.)")
    p_add.add_argument("--model", required=True, help="Model name (gpt-4o-mini, claude-3-haiku, etc.)")
    p_add.add_argument("--api-key-env", required=True, help="Env var name for API key")
    p_add.add_argument("--base-url", default="", help="API base URL (default: OpenAI)")
    p_add.add_argument("--endpoint", default="/chat/completions")
    p_add.add_argument("--max-requests-day", type=int, default=100)
    p_add.add_argument("--max-tokens-day", type=int, default=100000)
    p_add.add_argument("--max-usd-day", type=float, default=5.0)
    p_add.add_argument("--max-concurrent", type=int, default=4)
    p_add.add_argument("--json", action="store_true")

    p_list = sub.add_parser("list", help="List all providers")
    p_list.add_argument("--json", action="store_true")

    p_disable = sub.add_parser("disable", help="Disable a provider")
    p_disable.add_argument("--key", required=True, help="Provider key (provider:model)")
    p_disable.add_argument("--json", action="store_true")

    p_budget = sub.add_parser("budget", help="Show budget usage")
    p_budget.add_argument("--provider", default="", help="Filter by provider name")
    p_budget.add_argument("--json", action="store_true")

    p_reset = sub.add_parser("reset", help="Reset daily budget counters")
    p_reset.add_argument("--provider", default="", help="Reset for specific provider")
    p_reset.add_argument("--json", action="store_true")

    p_generate = sub.add_parser("generate", help="Generate via a provider (testing)")
    p_generate.add_argument("--key", required=True, help="Provider key")
    p_generate.add_argument("--prompt", required=True)
    p_generate.add_argument("--system", default="")
    p_generate.add_argument("--temperature", type=float, default=0.7)
    p_generate.add_argument("--max-tokens", type=int, default=512)

    args = parser.parse_args()
    registry = get_provider_registry()

    if args.action == "add":
        cfg = ProviderConfig(
            provider=args.provider,
            model=args.model,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            endpoint=args.endpoint,
            max_requests_day=args.max_requests_day,
            max_tokens_day=args.max_tokens_day,
            max_usd_day=args.max_usd_day,
            max_concurrent=args.max_concurrent,
        )
        result = registry.add_provider(cfg)
        if args.json:
            _print_json(result)
        else:
            print(f"Provider added: {result['key']}")

    elif args.action == "list":
        providers = registry.list_providers()
        if args.json:
            _print_json({"providers": providers})
        else:
            for p in providers:
                status = "enabled" if p["enabled"] else "DISABLED"
                print(f"  {p['key']} ({p['model']}) [{status}] max=${p['max_usd_day']:.2f}/day")

    elif args.action == "disable":
        ok = registry.set_enabled(args.key, False)
        if args.json:
            _print_json({"key": args.key, "disabled": ok})
        else:
            print(f"Provider disabled: {args.key}" if ok else f"Provider not found: {args.key}")

    elif args.action == "budget":
        usage = registry.ledger.daily_usage(args.provider)
        if args.json:
            _print_json(usage)
        else:
            for prov, day in (usage["usage"].items() if isinstance(usage.get("usage"), dict) else []):
                print(f"  {prov}: {day.get('requests', 0)} reqs, {day.get('tokens', 0)} tokens, ${day.get('usd', 0.0):.4f}")

    elif args.action == "reset":
        registry.ledger.reset_daily(args.provider)
        msg = {"status": "reset", "provider": args.provider or "all"}
        if args.json:
            _print_json(msg)
        else:
            print(f"Reset daily budget for: {args.provider or 'all providers'}")

    elif args.action == "generate":
        text, rec = registry.generate(
            args.key, args.prompt,
            system_prompt=args.system,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        print(text)
        print(f"\n--- Usage: {rec.total_tokens} tokens, ${rec.usd_cost:.6f}, {rec.duration_ms:.0f}ms")


if __name__ == "__main__":
    main()
