from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.teacher_base import TeacherBase
from src.trainer_providers import get_provider_registry


class RegistryTeacher(TeacherBase):
    def __init__(self, provider_key: str):
        super().__init__(teacher_name=f"registry:{provider_key}")
        self.provider_key = provider_key
        self.registry = get_provider_registry()

    def check_connection(self) -> dict[str, Any]:
        cfg = self.registry.get_provider(self.provider_key)
        if not cfg:
            return {"ok": False, "error": f"Provider '{self.provider_key}' not registered"}
        if not cfg.enabled:
            return {"ok": False, "error": f"Provider '{self.provider_key}' is disabled"}
        try:
            import os
            key = os.getenv(cfg.api_key_env, "").strip()
            if not key:
                return {"ok": False, "error": f"Missing env var: {cfg.api_key_env}"}
        except Exception as e:
            return {"ok": False, "error": str(e)}
        return {"ok": True, "provider": self.provider_key, "model": cfg.model}

    def generate_text(self, prompt: str, system: str = "", temperature: float = 0.7, max_tokens: int = 512) -> str:
        text, _ = self.registry.generate(
            self.provider_key, prompt,
            system_prompt=system,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return text


def main():
    parser = argparse.ArgumentParser(description="RegistryTeacher — use any registered provider as an Ickle teacher.")
    parser.add_argument("mode", choices=["check", "curriculum", "teach", "batch-sft", "submit"])
    parser.add_argument("--provider", required=True, help="Registered provider key (e.g. 'openai:gpt-4o-mini')")
    parser.add_argument("--topic", default="", help="Topic for curriculum or batch-sft")
    parser.add_argument("--count", type=int, default=5, help="Number of SFT pairs or curriculum prompts")
    parser.add_argument("--prompt", default="", help="User prompt for teach mode")
    parser.add_argument("--response", default="", help="Ickle's answer to evaluate in teach mode")
    parser.add_argument("--api-url", default="", help="Submit via Ickle web API at this URL")
    parser.add_argument("--store", action="store_true", help="Submit turns to local teacher store")

    args = parser.parse_args()
    teacher = RegistryTeacher(args.provider)

    if args.mode == "check":
        result = teacher.check_connection()
        print(json.dumps(result, indent=2))

    elif args.mode == "curriculum":
        if not args.topic:
            raise SystemExit("--topic required for curriculum mode")
        prompts = teacher.generate_curriculum(args.topic, count=args.count)
        for p in prompts:
            print(f"- {p}")

    elif args.mode == "teach":
        if not args.prompt:
            raise SystemExit("--prompt required for teach mode")
        result = teacher.teach_turn(args.prompt, args.response)
        print(json.dumps(result, indent=2))

    elif args.mode == "batch-sft":
        if not args.topic:
            raise SystemExit("--topic required for batch-sft mode")
        pairs = teacher.batch_sft(args.topic, count=args.count)
        for pair in pairs:
            print(f"User: {pair.get('prompt', pair.get('user', ''))}")
            print(f"Ickle: {pair.get('response', pair.get('ickle', pair.get('assistant', '')))}")
            print()

    elif args.mode == "submit":
        if not args.prompt:
            raise SystemExit("--prompt required for submit mode (JSON file path or JSON array of turns)")
        prompt_path = Path(args.prompt)
        if prompt_path.exists():
            with prompt_path.open(encoding="utf-8") as f:
                turns = json.load(f)
            if not isinstance(turns, list):
                raise SystemExit("Error: JSON file must contain a list of turn objects")
        else:
            turns = json.loads(args.prompt)
            if not isinstance(turns, list):
                turns = [turns]

        if args.api_url:
            result = teacher.submit_via_api(turns, topic=args.topic or "registry_teaching", api_url=args.api_url)
        else:
            result = teacher.submit_to_store(turns, topic=args.topic or "registry_teaching")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
