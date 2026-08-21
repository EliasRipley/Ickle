"""Ickle launcher - unified entry point for common tasks.
Provides a simple menu interface for training, chatting, serving, and maintenance.

Usage:
    python scripts/launch.py              # Interactive menu
    python scripts/launch.py --web         # Start web server
    python scripts/launch.py --control     # Start control API
    python scripts/launch.py --train       # Start initial training
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _cmd(*parts: str) -> list[str]:
    return [sys.executable, "-u", *parts]


def _check_ollama() -> bool:
    try:
        import urllib.request
        req = urllib.request.Request("http://127.0.0.1:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def run_web(port: int = 8787):
    print(f"Starting Ickle web UI on http://127.0.0.1:{port}")
    subprocess.run(_cmd("-m", "src.app", "serve-web", "--port", str(port)), cwd=str(ROOT))


def run_control(port: int = 8788):
    print(f"Starting Ickle control API on http://127.0.0.1:{port}")
    subprocess.run(_cmd("-m", "src.app", "serve-control", "--port", str(port)), cwd=str(ROOT))


def run_chat():
    model = input("Model path [models/ickle_best.pt]: ").strip() or "models/ickle_best.pt"
    prompt = input("Prompt: ").strip()
    subprocess.run(_cmd("-m", "src.app", "chat", "--model", model, "--prompt", prompt), cwd=str(ROOT))


def run_preflight():
    subprocess.run(_cmd("-m", "src.app", "preflight"), cwd=str(ROOT))


def run_tidy():
    subprocess.run(_cmd("scripts/tidy_workspace.py"), cwd=str(ROOT))


def run_bootstrap():
    profile = input("Profile [laptop]: ").strip() or "laptop"
    steps = input("Training steps [3000]: ").strip() or "3000"
    subprocess.run(
        _cmd("-m", "scripts.bootstrap_fresh", "--profile", profile, "--steps", steps),
        cwd=str(ROOT),
    )


def run_continuous():
    if not _check_ollama():
        print("\n  WARNING: Ollama does not appear to be running at http://127.0.0.1:11434")
        print("  Start Ollama first, or install from https://ollama.com")
        ok = input("  Continue anyway? [y/N]: ").strip().lower()
        if ok != "y":
            print("  Cancelled.")
            return
    topic = input("Topic [general knowledge]: ").strip() or "general knowledge"
    rounds = input("Number of rounds [5]: ").strip() or "5"
    subprocess.run(
        _cmd("-m", "scripts.continuous_learner", "--mode", "auto", "--rounds", rounds, "--topic", topic),
        cwd=str(ROOT),
    )


def run_reference():
    print("\n  Opening docs/ICKLE_REFERENCE.md ...")
    ref_path = ROOT / "docs" / "ICKLE_REFERENCE.md"
    if ref_path.exists():
        subprocess.run([sys.executable, "-c", f"print(open(r'{ref_path}').read())"], cwd=str(ROOT))
    input("\n  Press Enter to return to menu.")


def show_menu():
    options = {
        "1": ("Start web UI (chat, port 8787)", run_web),
        "2": ("Start control API (tasks, port 8788)", run_control),
        "3": ("Chat (CLI)", run_chat),
        "4": ("Bootstrap fresh model", run_bootstrap),
        "5": ("Continuous AI teacher training", run_continuous),
        "6": ("Run preflight check", run_preflight),
        "7": ("Tidy workspace", run_tidy),
        "8": ("Show full reference docs", run_reference),
        "q": ("Quit", None),
    }

    while True:
        print("\n=== Ickle Launcher ===")
        for key, (label, _) in options.items():
            print(f"  {key}. {label}")
        print("\n  Tip: Use 'python -m src.app help' to see all available commands.")
        choice = input("\nChoice: ").strip().lower()

        if choice == "q":
            print("Goodbye!")
            break

        if choice in options:
            _, action = options[choice]
            if action:
                action()
        else:
            print("Invalid choice")


def main():
    parser = argparse.ArgumentParser(description="Ickle launcher")
    parser.add_argument("--web", action="store_true", help="Start web UI server (port 8787)")
    parser.add_argument("--control", action="store_true", help="Start control API server (port 8788)")
    parser.add_argument("--port", type=int, default=8787, help="Web server port")
    parser.add_argument("--train", action="store_true", help="Start initial training")
    parser.add_argument("--tidy", action="store_true", help="Tidy workspace")
    args = parser.parse_args()

    if args.web:
        run_web(args.port)
    elif args.control:
        run_control()
    elif args.train:
        run_bootstrap()
    elif args.tidy:
        run_tidy()
    else:
        show_menu()


if __name__ == "__main__":
    main()
