from __future__ import annotations

import argparse
import json
import os
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

from src.teacher_base import TeacherBase


class AnthropicTeacher(TeacherBase):
    def __init__(self, api_key: str = "", model: str = "claude-3-5-sonnet-20241022"):
        super().__init__(teacher_name="anthropic")
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.model = model
        self.base_url = "https://api.anthropic.com/v1"

    def check_connection(self) -> dict[str, Any]:
        if not self.api_key:
            return {"ok": False, "error": "ANTHROPIC_API_KEY not set"}
        return {"ok": True, "model": self.model, "provider": "anthropic"}

    def generate_text(self, prompt: str, system: str = "", temperature: float = 0.7, max_tokens: int = 512) -> str:
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")

        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        url = f"{self.base_url}/messages"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")[:500]
            exc.close()
            raise RuntimeError(f"Anthropic returned {exc.code}: {error_body}")

        content = body.get("content", [])
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                return block.get("text", "")

        return ""


def main():
    parser = argparse.ArgumentParser(description="AnthropicTeacher — generate training data via Claude.")
    parser.add_argument("mode", choices=["check", "curriculum", "teach", "batch-sft", "submit"])
    parser.add_argument("--api-key", default="", help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    parser.add_argument("--model", default="claude-3-5-sonnet-20241022")
    parser.add_argument("--topic", default="", help="Topic for curriculum or batch-sft")
    parser.add_argument("--count", type=int, default=5, help="Number of SFT pairs or curriculum prompts")
    parser.add_argument("--prompt", default="", help="User prompt for teach mode")
    parser.add_argument("--response", default="", help="Ickle's answer to evaluate in teach mode")
    parser.add_argument("--api-url", default="", help="Submit via Ickle web API at this URL")
    parser.add_argument("--store", action="store_true", help="Submit turns to local teacher store")

    args = parser.parse_args()
    teacher = AnthropicTeacher(api_key=args.api_key, model=args.model)

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
            result = teacher.submit_via_api(turns, topic=args.topic or "anthropic_teaching", api_url=args.api_url)
        else:
            result = teacher.submit_to_store(turns, topic=args.topic or "anthropic_teaching")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
