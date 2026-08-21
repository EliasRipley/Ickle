"""Bootstrap a fresh Ickle model using the built-in English seed corpus.
Creates a language-first model from scratch with no external dependencies.

Usage:
    python scripts/bootstrap_fresh.py --steps 3000 --out models/ickle_language_first.pt
    python scripts/bootstrap_fresh.py --profile desktop --steps 5000 --compile
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.english_seed import ENGLISH_BOOTSTRAP_TEXT


def main():
    parser = argparse.ArgumentParser(description="Bootstrap a fresh Ickle model from built-in English seed")
    parser.add_argument("--profile", default="laptop", choices=["nano", "laptop", "desktop", "workstation"])
    parser.add_argument("--steps", type=int, default=3000, help="Training steps")
    parser.add_argument("--out", default="models/ickle_language_first.pt", help="Output model path")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=0, help="Batch size (0 = profile default)")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--warmup-steps", type=int, default=300, help="LR warmup steps")
    parser.add_argument("--eval-every", type=int, default=200, help="Evaluate every N steps")
    parser.add_argument("--checkpoint-every", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--repeat", type=int, default=1000, help="Repeat bootstrap text multiplier")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--spm-vocab", type=int, default=4096, help="SentencePiece vocab size")
    parser.add_argument("--stream-dataset", default="", help="Optional HF dataset to augment with")
    parser.add_argument("--stream-max-chars", type=int, default=500000, help="Max chars to stream")
    parser.add_argument("--layer-drop-rate", type=float, default=0.0, help="Stochastic depth drop rate")
    parser.add_argument("--n-pred-tokens", type=int, default=1, help="Multi-token prediction count")
    parser.add_argument("--ema-decay", type=float, default=0.0, help="EMA decay rate (0 = disable)")
    parser.add_argument("--llrd-decay", type=float, default=1.0, help="Layer-wise LR decay factor")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write the English seed to a text file
    corpus_dir = root / "data"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = corpus_dir / "english_seed_corpus.txt"

    text = "\n".join([ENGLISH_BOOTSTRAP_TEXT] * args.repeat)
    corpus_path.write_text(text, encoding="utf-8")
    print(f"Wrote seed corpus: {corpus_path} ({len(text)} chars)")

    # Build the training command
    import subprocess

    cmd = [
        sys.executable, "-u", "-m", "src.train",
        "--data", str(corpus_path),
        "--out", str(out_path),
        "--steps", str(args.steps),
        "--profile", args.profile,
        "--lr", str(args.lr),
        "--warmup-steps", str(args.warmup_steps),
        "--eval-every", str(args.eval_every),
        "--checkpoint-every", str(args.checkpoint_every),
        "--grad-accum-steps", str(args.grad_accum),
        "--seed", str(args.seed),
        "--spm-vocab-size", str(args.spm_vocab),
        "--best-model-path", str(out_path).replace(".pt", "_best.pt"),
        "--save-best-on-interrupt",
    ]

    if args.batch_size > 0:
        cmd.extend(["--batch-size", str(args.batch_size)])

    if args.stream_dataset:
        cmd.extend([
            "--stream-dataset", args.stream_dataset,
            "--stream-field", "text",
            "--stream-template", "User: Tell me about {text}\n\nIckle: {text}",
            "--stream-max-chars", str(args.stream_max_chars),
        ])

    if args.layer_drop_rate > 0:
        cmd.extend(["--layer-drop-rate", str(args.layer_drop_rate)])
    if args.n_pred_tokens > 1:
        cmd.extend(["--n-pred-tokens", str(args.n_pred_tokens)])
    if args.ema_decay > 0:
        cmd.extend(["--ema-decay", str(args.ema_decay)])
    if args.llrd_decay < 1.0:
        cmd.extend(["--llrd-decay", str(args.llrd_decay)])

    print(f"Training command: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(root))
    if proc.returncode != 0:
        print("Bootstrap training failed")
        sys.exit(1)

    print(f"\nModel saved: {out_path}")
    print(f"Best model saved: {str(out_path).replace('.pt', '_best.pt')}")
    print("\nTry chatting:")
    print(f"  python -m src.app chat --model {args.out} --prompt 'Hello, how are you?'")
    print(f"  python -m src.app serve-web --port 8787")


if __name__ == "__main__":
    main()
