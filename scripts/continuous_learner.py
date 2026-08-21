"""Continuous AI teacher training pipeline for Ickle.
Generates training data from AI teachers (Ollama, Anthropic, OpenAI, etc.),
trains Ickle, and saves checkpoints. Supports resume on restart.

Usage:
    python scripts/continuous_learner.py --topic "general knowledge" --rounds 10
    python scripts/continuous_learner.py --auto --sleep-minutes 30
    python scripts/continuous_learner.py --auto --provider anthropic:claude-3-5-sonnet
    python scripts/continuous_learner.py --auto --provider openai:gpt-4o-mini
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ollama_teacher import OllamaTeacher, DEFAULT_MODEL, DEFAULT_OLLAMA_HOST
from src.teacher_base import TeacherBase
from src.teacher_ingest import TeacherStore


def _create_teacher(
    provider: str = "",
    ollama_model: str = DEFAULT_MODEL,
    ollama_host: str = DEFAULT_OLLAMA_HOST,
    teacher_name: str = "gemma4",
    api_key: str = "",
) -> TeacherBase:
    if provider:
        if provider.startswith("ollama:"):
            model = provider.split(":", 1)[1] if ":" in provider else ollama_model
            return OllamaTeacher(model=model, host=ollama_host, teacher_name=teacher_name)
        elif provider.startswith("anthropic:"):
            from src.teacher_anthropic import AnthropicTeacher
            model = provider.split(":", 1)[1] if ":" in provider else "claude-3-5-sonnet-20241022"
            return AnthropicTeacher(api_key=api_key, model=model)
        else:
            from src.teacher_registry import RegistryTeacher
            return RegistryTeacher(provider)
    return OllamaTeacher(model=ollama_model, host=ollama_host, teacher_name=teacher_name)


AUTONOMOUS_TOPICS = [
    "English grammar and writing",
    "basic mathematics",
    "world geography",
    "cell biology",
    "computer programming fundamentals",
    "world history summary",
    "physics basics",
    "chemistry fundamentals",
    "critical thinking and logic",
    "health and nutrition",
    "environmental science",
    "art and music appreciation",
    "economics basics",
    "civics and government",
    "literature and reading comprehension",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_training_state_path() -> Path:
    return Path("data") / "continuous_learner_state.json"


def _load_state() -> dict[str, Any]:
    p = _get_training_state_path()
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_state(state: dict[str, Any]):
    p = _get_training_state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def _build_model_path(topic: str, round_num: int) -> str:
    slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in topic.lower())[:30]
    return f"models/ickle_{slug}_r{round_num:03d}.pt"


def _build_corpus_path(topic: str, round_num: int) -> str:
    slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in topic.lower())[:30]
    return f"data/corpus_{slug}_r{round_num:03d}.txt"


def generate_training_data(
    teacher: TeacherBase,
    topic: str,
    count: int = 8,
    store_dir: str = "data/teacher",
) -> list[dict[str, Any]]:
    """Generate SFT pairs using the Ollama teacher for a given topic."""
    print(f"  generating {count} SFT pairs for '{topic}'...")
    pairs = teacher.batch_sft(topic, count=count, tags=[topic])
    print(f"  generated {len(pairs)} pairs")
    return pairs


def build_corpus_from_pairs(pairs: list[dict[str, Any]], corpus_path: str) -> str:
    """Build a training corpus file from SFT pairs."""
    lines: list[str] = []
    for pair in pairs:
        prompt = str(pair.get("prompt", "")).strip()
        answer = str(pair.get("improved_answer", "")).strip()
        feedback = str(pair.get("teacher_feedback", "")).strip()
        if prompt and answer:
            lines.append(f"User: {prompt}")
            if feedback:
                lines.append(f"Teacher feedback: {feedback}")
            lines.append(f"Ickle: {answer}")
            lines.append("")
    text = "\n".join(lines)
    p = Path(corpus_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return text


def train_on_corpus(
    corpus_path: str,
    model_out: str,
    *,
    init_model: str = "",
    resume_checkpoint: str = "",
    steps: int = 200,
    cpu_pct: int = 80,
    ram_pct: int = 80,
    gpu_pct: int = 80,
    checkpoint_every: int = 50,
    lr: float = 3e-4,
    batch_size: int = 0,
):
    """Run training on a corpus file with checkpointing."""
    import subprocess

    cmd = [
        sys.executable, "-u", "-m", "src.train",
        "--data", corpus_path,
        "--out", model_out,
        "--steps", str(steps),
        "--cpu-pct", str(cpu_pct),
        "--ram-pct", str(ram_pct),
        "--gpu-pct", str(gpu_pct),
        "--checkpoint-every", str(checkpoint_every),
        "--eval-every", "50",
        "--best-model-path", model_out.replace(".pt", "_best.pt"),
        "--save-best-on-interrupt",
    ]
    if lr != 3e-4:
        cmd.extend(["--lr", str(lr)])
    if batch_size > 0:
        cmd.extend(["--batch-size", str(batch_size)])
    if init_model and not resume_checkpoint:
        cmd.extend(["--init-model", init_model])
    if resume_checkpoint:
        cmd.extend(["--resume-from-checkpoint", resume_checkpoint])
    if os.getenv("ICKLE_TORCH_COMPILE"):
        cmd.append("--compile")

    print(f"  training: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent.parent))
    return proc.returncode == 0


def run_round(
    teacher: TeacherBase,
    topic: str,
    round_num: int,
    *,
    prev_model: str = "",
    resume_checkpoint: str = "",
    pairs_per_round: int = 8,
    train_steps: int = 200,
    cpu_pct: int = 80,
    ram_pct: int = 80,
    gpu_pct: int = 80,
    store_dir: str = "data/teacher",
) -> dict[str, Any]:
    """Run one training round: generate data, train, save checkpoint."""
    print(f"\n=== Round {round_num}: {topic} ===")

    pairs = generate_training_data(teacher, topic, count=pairs_per_round, store_dir=store_dir)

    corpus_path = _build_corpus_path(topic, round_num)
    corpus_text = build_corpus_from_pairs(pairs, corpus_path)
    print(f"  corpus: {corpus_path} ({len(corpus_text)} chars)")

    model_out = _build_model_path(topic, round_num)
    checkpoint_path = model_out.replace(".pt", ".checkpoint.pt")

    if resume_checkpoint and Path(resume_checkpoint).exists():
        print(f"  resuming from checkpoint: {resume_checkpoint}")
        init = ""
        resume = resume_checkpoint
    else:
        init = prev_model
        resume = ""

    success = train_on_corpus(
        corpus_path=corpus_path,
        model_out=model_out,
        init_model=init,
        resume_checkpoint=resume,
        steps=train_steps,
        cpu_pct=cpu_pct, ram_pct=ram_pct, gpu_pct=gpu_pct,
        checkpoint_every=max(20, train_steps // 4),
    )

    checkpoint_for_next = model_out.replace(".pt", ".checkpoint.pt")
    if not Path(checkpoint_for_next).exists():
        checkpoint_for_next = model_out

    return {
        "round": round_num,
        "topic": topic,
        "model": model_out,
        "checkpoint": checkpoint_for_next,
        "corpus": corpus_path,
        "success": success,
        "pairs_generated": len(pairs),
    }


def run_autonomous_loop(
    teacher: TeacherBase,
    *,
    rounds: int = 10,
    sleep_minutes: int = 0,
    pairs_per_round: int = 8,
    train_steps: int = 200,
    cpu_pct: int = 80,
    ram_pct: int = 80,
    gpu_pct: int = 80,
    store_dir: str = "data/teacher",
    topics: list[str] | None = None,
):
    """Run continuous learning: pick topics, train, save state, sleep, repeat."""
    if topics is None:
        topics = AUTONOMOUS_TOPICS

    state = _load_state()
    completed_rounds = int(state.get("completed_rounds", 0))
    current_model = str(state.get("last_model", ""))
    current_checkpoint = str(state.get("last_checkpoint", ""))

    print(f"Continuous learner starting (completed {completed_rounds} rounds)")
    print(f"Current model: {current_model or 'none (fresh)'}")
    print(f"Topics pool: {len(topics)} topics")

    round_num = completed_rounds + 1
    while round_num <= rounds:
        topic_idx = (round_num - 1) % len(topics)
        topic = topics[topic_idx]

        result = run_round(
            teacher=teacher,
            topic=topic,
            round_num=round_num,
            prev_model=current_model,
            resume_checkpoint=current_checkpoint if round_num == completed_rounds + 1 else "",
            pairs_per_round=pairs_per_round,
            train_steps=train_steps,
            cpu_pct=cpu_pct, ram_pct=ram_pct, gpu_pct=gpu_pct,
            store_dir=store_dir,
        )

        if result["success"]:
            current_model = result["model"]
            current_checkpoint = result["checkpoint"]
            completed_rounds = round_num

        state = {
            "completed_rounds": completed_rounds,
            "last_model": current_model,
            "last_checkpoint": current_checkpoint,
            "last_topic": topic,
            "last_round": round_num,
            "updated_utc": _utc_now(),
        }
        _save_state(state)

        if sleep_minutes > 0 and round_num < rounds:
            print(f"\nSleeping {sleep_minutes} minutes before next round...")
            time.sleep(sleep_minutes * 60)

        round_num += 1

    print(f"\n=== Continuous learning complete ({completed_rounds} rounds) ===")
    print(f"Final model: {current_model}")


def run_single_round(args):
    """Run a single training round with explicit parameters."""
    teacher = _create_teacher(
        provider=getattr(args, "provider", ""),
        ollama_model=args.teacher_model,
        ollama_host=args.ollama_host,
        teacher_name=args.teacher_name,
    )

    if args.check:
        result = teacher.check_connection()
        print(json.dumps(result, indent=2))
        if not result.get("ok"):
            sys.exit(1)
        return

    topic = args.topic or "general knowledge"
    result = run_round(
        teacher=teacher,
        topic=topic,
        round_num=args.round,
        prev_model=args.init_model,
        resume_checkpoint=args.resume,
        pairs_per_round=args.pairs,
        train_steps=args.steps,
        cpu_pct=args.cpu_pct,
        ram_pct=args.ram_pct,
        gpu_pct=args.gpu_pct,
        store_dir=args.store_dir,
    )
    print(f"\nResult: {json.dumps(result, indent=2, default=str)}")


def main():
    parser = argparse.ArgumentParser(description="Ickle Continuous AI Teacher Learner")
    parser.add_argument("--mode", default="auto", choices=["auto", "single", "check"])
    parser.add_argument("--topic", default="", help="Topic for single round mode")
    parser.add_argument("--round", type=int, default=1, help="Round number for single mode")
    parser.add_argument("--rounds", type=int, default=10, help="Number of rounds for auto mode")
    parser.add_argument("--pairs", type=int, default=8, help="SFT pairs per round")
    parser.add_argument("--steps", type=int, default=200, help="Training steps per round")
    parser.add_argument("--sleep-minutes", type=int, default=0, help="Sleep between auto rounds (0=no sleep)")
    parser.add_argument("--cpu-pct", type=int, default=80, help="CPU percentage")
    parser.add_argument("--ram-pct", type=int, default=80, help="RAM percentage")
    parser.add_argument("--gpu-pct", type=int, default=80, help="GPU percentage")
    parser.add_argument("--provider", default="", help="Provider key (e.g. 'ollama:llama3', 'anthropic:claude-3-5-sonnet', 'openai:gpt-4o-mini'). Empty = local Ollama default")
    parser.add_argument("--teacher-model", default=DEFAULT_MODEL, help="Ollama teacher model name")
    parser.add_argument("--teacher-name", default="gemma4", help="Teacher name recorded in store")
    parser.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST, help="Ollama API host")
    parser.add_argument("--init-model", default="", help="Initial model for training")
    parser.add_argument("--resume", default="", help="Resume from checkpoint path")
    parser.add_argument("--store-dir", default="data/teacher", help="Teacher store directory")
    parser.add_argument("--check", action="store_true", help="Check teacher connection and exit")
    args = parser.parse_args()

    if args.check or args.mode == "check":
        teacher = _create_teacher(
            provider=args.provider,
            ollama_model=args.teacher_model,
            ollama_host=args.ollama_host,
            teacher_name=args.teacher_name,
        )
        result = teacher.check_connection()
        print(json.dumps(result, indent=2))
        if not result.get("ok"):
            sys.exit(1)
        return

    if args.mode == "single":
        run_single_round(args)
        return

    teacher = _create_teacher(
        provider=args.provider,
        ollama_model=args.teacher_model,
        ollama_host=args.ollama_host,
        teacher_name=args.teacher_name,
    )
    run_autonomous_loop(
        teacher=teacher,
        rounds=args.rounds,
        sleep_minutes=args.sleep_minutes,
        pairs_per_round=args.pairs,
        train_steps=args.steps,
        cpu_pct=args.cpu_pct,
        ram_pct=args.ram_pct,
        gpu_pct=args.gpu_pct,
        store_dir=args.store_dir,
    )


if __name__ == "__main__":
    main()
