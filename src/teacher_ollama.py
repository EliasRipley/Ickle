"""Ollama Teacher - uses local Ollama models to teach Ickle.

Extends TeacherBase with Ollama's HTTP API (localhost:11434).

Three modes:
  curriculum  - Generate a set of training prompts for a topic
  teach-turn  - Submit a prompt to Ickle, get its answer, have Ollama critique & improve
  batch-sft   - Directly generate User:/Ickle: SFT pairs for a topic
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from src.teacher_base import TeacherBase


DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL = "gemma4:latest"


class OllamaTeacher(TeacherBase):
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_OLLAMA_HOST,
        teacher_name: str = "gemma4",
    ):
        super().__init__(teacher_name=teacher_name)
        self.model = model
        self.host = host.rstrip("/")

    def check_connection(self) -> dict[str, Any]:
        try:
            with urlopen(Request(self.host + "/api/tags"), timeout=10) as resp:
                data = json.loads(resp.read())
            models = [m["name"] for m in data.get("models", [])]
            return {"ok": True, "models": models, "host": self.host}
        except Exception as exc:
            return {"ok": False, "error": str(exc), "host": self.host}

    def generate_text(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system:
            payload["system"] = system

        body = json.dumps(payload).encode("utf-8")
        url = f"{self.host}/api/generate"
        req = Request(url, data=body, headers={"Content-Type": "application/json"})

        for attempt in range(3):
            try:
                with urlopen(req, timeout=180) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                return str(data.get("response", "") or "")
            except Exception as exc:
                if attempt == 2:
                    raise RuntimeError(f"Ollama request failed after 3 attempts: {exc}") from exc
                time.sleep(2.0 * (attempt + 1))

        return ""


def main():
    parser = argparse.ArgumentParser(
        description="Ollama Teacher - use local Ollama models to teach Ickle"
    )
    parser.add_argument("mode", choices=["check", "curriculum", "teach", "batch-sft", "submit"])
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument("--host", default=DEFAULT_OLLAMA_HOST, help="Ollama API host")
    parser.add_argument("--teacher-name", default="gemma4", help="Name recorded as source_model")
    parser.add_argument("--topic", default="", help="Topic for curriculum or batch-sft")
    parser.add_argument("--prompt", default="", help="Prompt for teach-turn mode")
    parser.add_argument("--ickle-answer", default="", help="Ickle's current answer (for teach mode)")
    parser.add_argument("--count", type=int, default=8, help="Number of items to generate")
    parser.add_argument("--tags", default="", help="Comma-separated tags")
    parser.add_argument("--store-dir", default="data/teacher", help="Teacher store directory")
    parser.add_argument("--api-url", default="", help="Submit via HTTP API (e.g. http://localhost:8788)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()
    teacher = OllamaTeacher(model=args.model, host=args.host, teacher_name=args.teacher_name)
    tag_list = [t.strip() for t in (args.tags or "").split(",") if t.strip()]

    if args.mode == "check":
        result = teacher.check_connection()
        if args.json:
            print(json.dumps(result, ensure_ascii=False))
        else:
            if result["ok"]:
                print(f"Connected to Ollama at {result['host']}")
                print(f"Models: {', '.join(result['models'])}")
            else:
                print(f"Failed: {result['error']}")
                sys.exit(1)
        return

    if args.mode == "curriculum":
        if not args.topic:
            print("Error: --topic required for curriculum mode", file=sys.stderr)
            sys.exit(1)
        print(f"Generating {args.count} curriculum prompts for '{args.topic}'...")
        prompts = teacher.generate_curriculum(args.topic, count=args.count, tags=tag_list)
        if args.json:
            print(json.dumps({"topic": args.topic, "prompts": prompts}, ensure_ascii=False))
        else:
            for i, p in enumerate(prompts, 1):
                print(f"  {i}. {p}")
        return

    if args.mode == "teach":
        if not args.prompt:
            print("Error: --prompt required for teach mode", file=sys.stderr)
            sys.exit(1)
        print(f"Teaching turn for: {args.prompt[:80]}...")
        turn = teacher.teach_turn(args.prompt, ickle_answer=args.ickle_answer, tags=tag_list)
        if args.json:
            print(json.dumps(turn, ensure_ascii=False))
        else:
            print(f"Score: {turn['score']}")
            print(f"Feedback: {turn['teacher_feedback']}")
            print(f"Answer: {turn['improved_answer'][:500]}")
        return

    if args.mode == "batch-sft":
        if not args.topic:
            print("Error: --topic required for batch-sft mode", file=sys.stderr)
            sys.exit(1)
        print(f"Generating {args.count} SFT pairs for '{args.topic}'...")
        pairs = teacher.batch_sft(args.topic, count=args.count, tags=tag_list)
        if args.json:
            print(json.dumps({"topic": args.topic, "pairs": pairs}, ensure_ascii=False))
        else:
            for i, pair in enumerate(pairs, 1):
                print(f"\n--- Pair {i} ---")
                print(f"Q: {pair['prompt']}")
                print(f"A: {pair['improved_answer'][:200]}")
        return

    if args.mode == "submit":
        if not args.prompt:
            print("Error: --prompt required for submit mode", file=sys.stderr)
            sys.exit(1)
        prompt_path = Path(args.prompt)
        if prompt_path.exists():
            with prompt_path.open(encoding="utf-8") as f:
                turns = json.load(f)
            if not isinstance(turns, list):
                print("Error: JSON file must contain a list of turn objects", file=sys.stderr)
                sys.exit(1)
        else:
            try:
                turns = json.loads(args.prompt)
                if not isinstance(turns, list):
                    turns = [turns]
            except json.JSONDecodeError:
                print("Error: --prompt must be a JSON file path or valid JSON array", file=sys.stderr)
                sys.exit(1)

        if args.api_url:
            print(f"Submitting {len(turns)} turns via API at {args.api_url}...")
            result = teacher.submit_via_api(turns, topic=args.topic or "ollama_teaching", api_url=args.api_url)
        else:
            print(f"Submitting {len(turns)} turns to local store...")
            result = teacher.submit_to_store(turns, topic=args.topic or "ollama_teaching", store_dir=args.store_dir)

        if args.json:
            print(json.dumps(result, ensure_ascii=False))
        else:
            print(f"Session {result['session_id']}: {result['turns_submitted']} turns submitted")
        return


if __name__ == "__main__":
    import argparse
    main()
