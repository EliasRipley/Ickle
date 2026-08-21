"""Initialize a fresh language-first Ickle model.
Trains from scratch on a streamed real-text corpus (smollm-corpus)
to create a model that has basic English language understanding.

Usage:
    python scripts/init_language_model.py --profile laptop --steps 2000 --out models/ickle_language_first.pt
    python scripts/init_language_model.py --profile desktop --steps 5000 --amp bf16 --compile
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Initialize a fresh language-first Ickle model")
    parser.add_argument("--profile", default="laptop", choices=["nano", "laptop", "desktop", "workstation"])
    parser.add_argument("--steps", type=int, default=2000, help="Training steps")
    parser.add_argument("--out", default="models/ickle_language_first.pt", help="Output model path")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=0, help="Batch size (0 = profile default)")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--eval-every", type=int, default=200, help="Evaluate every N steps")
    parser.add_argument("--checkpoint-every", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--warmup-steps", type=int, default=200, help="LR warmup steps")
    parser.add_argument("--amp", default="", choices=["", "bf16", "fp16"], help="Mixed precision")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    parser.add_argument("--spm-vocab", type=int, default=0, help="SentencePiece vocab size (0=auto)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-u", "-m", "src.train",
        "--data", "",
        "--stream-dataset", "HuggingFaceTB/smollm-corpus",
        "--stream-field", "text",
        "--stream-template", "User: Tell me about {text}\n\nIckle: {text}",
        "--stream-max-chars", "500000",
        "--out", str(out_path),
        "--steps", str(args.steps),
        "--profile", args.profile,
        "--lr", str(args.lr),
        "--warmup-steps", str(args.warmup_steps),
        "--eval-every", str(args.eval_every),
        "--checkpoint-every", str(args.checkpoint_every),
        "--grad-accum-steps", str(args.grad_accum),
        "--seed", str(args.seed),
        "--best-model-path", str(out_path).replace(".pt", "_best.pt"),
        "--save-best-on-interrupt",
    ]

    if args.batch_size > 0:
        cmd.extend(["--batch-size", str(args.batch_size)])
    if args.amp:
        cmd.append("--amp")
        cmd.append(args.amp)
    if args.compile:
        cmd.append("--compile")
    if args.spm_vocab > 0:
        cmd.extend(["--spm-vocab-size", str(args.spm_vocab)])

    print(f"Initializing language-first model: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(root))
    if proc.returncode != 0:
        print("Model initialization failed")
        sys.exit(1)

    print(f"\nModel saved: {out_path}")
    print("You can now chat with it:")
    print(f"  python -m src.app chat --model {args.out} --prompt 'Hello, how are you?'")
    print("Or start the launcher menu:")
    print("  python scripts/launch.py")


if __name__ == "__main__":
    main()
