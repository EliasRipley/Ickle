from __future__ import annotations

import importlib
import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any


def _check_module(name: str) -> tuple[bool, str]:
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "unknown")
        return True, f"{name} OK (version={version})"
    except Exception as exc:
        return False, f"{name} MISSING ({exc})"


def _check_browser_runtime() -> tuple[bool, str]:
    try:
        from playwright.sync_api import sync_playwright

        from src.browser_runtime import launch_headless_browser

        with sync_playwright() as playwright:
            browser, description = launch_headless_browser(playwright, headless=True)
            browser.close()
        return True, f"Browser automation OK ({description})"
    except Exception as exc:
        return False, f"Browser automation unavailable ({exc})"


def _check_python() -> tuple[bool, str]:
    major, minor = sys.version_info[:2]
    ok = (major, minor) >= (3, 10)
    ver = f"{major}.{minor}"
    return ok, f"Python {ver}" if ok else f"Python {ver} (need >= 3.10)"


def _check_gpu() -> tuple[bool, str]:
    try:
        from src.device_bridge import get_gpu_info
        gpu = get_gpu_info()
        if gpu["available"]:
            names = gpu["names"]
            return True, f"GPU: {', '.join(names)} (backend: {gpu['backend']}, {gpu['count']} device(s))"
        return True, "CPU-only mode available (training will be slower without a GPU accelerator)"
    except Exception as exc:
        return False, f"GPU check failed: {exc}"


def _check_disk_space(paths: list[Path], min_gb: float = 2.0) -> list[tuple[bool, str]]:
    results = []
    for p in paths:
        try:
            p.mkdir(parents=True, exist_ok=True)
            if hasattr(shutil, "disk_usage"):
                usage = shutil.disk_usage(p)
                free_gb = usage.free / (1024 ** 3)
                ok = free_gb >= min_gb
                results.append((ok, f"{p.name}/: {free_gb:.1f} GB free ({'OK' if ok else f'need >= {min_gb} GB'})"))
            else:
                results.append((True, f"{p.name}/: exists (disk check unavailable)"))
        except Exception as exc:
            results.append((False, f"{p.name}/: error ({exc})"))
    return results


def _check_tokenizer() -> tuple[bool, str]:
    try:
        from src.tokenizer import SentencePieceTokenizer, CharTokenizer
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", encoding="utf-8", delete=False) as f:
            f.write("Hello world. Testing tokenizer creation for ILM.")
            tmp = f.name
        try:
            sp = SentencePieceTokenizer.train_from_corpus(corpus_path=tmp, vocab_size=128, model_type="bpe")
            encoded = sp.encode("Hello world")
            decoded = sp.decode(encoded)
            return True, f"SentencePiece BPE OK (vocab={sp.vocab_size}, roundtrip={encoded[0]}...)"
        except Exception:
            try:
                ct = CharTokenizer.from_text("Hello world. Testing.")
                return True, f"CharTokenizer OK (vocab={ct.vocab_size})"
            except Exception as exc:
                return False, f"All tokenizers failed: {exc}"
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    except Exception as exc:
        return False, f"Tokenizer import failed: {exc}"


def _check_model_forward() -> tuple[bool, str]:
    try:
        import torch
        from src.model import ILM, TinyConfig
        cfg = TinyConfig(vocab_size=128, block_size=64, n_embd=32, n_head=2, n_layer=1)
        model = ILM(cfg)
        x = torch.randint(0, 128, (1, 16))
        logits, loss = model(x, x)
        params = sum(p.numel() for p in model.parameters())
        return True, f"Model forward pass OK ({params:,} params, logits shape={list(logits.shape)})"
    except Exception as exc:
        return False, f"Model forward pass FAILED: {exc}"


def _check_training_root() -> tuple[bool, str]:
    try:
        from src.workspace_paths import get_training_root
        root = get_training_root()
        if root.is_dir():
            text_files = [
                path for path in root.rglob("*.txt")
                if path.is_file() and "raw" not in {part.lower() for part in path.relative_to(root).parts}
            ]
            total_bytes = sum(path.stat().st_size for path in text_files)
            total_mb = total_bytes / (1024 * 1024)
            enough_data = total_bytes >= 64 * 1024
            detail = f"Training root: {root} ({len(text_files)} usable text files, {total_mb:.2f} MB)"
            if not enough_data:
                detail += "; need at least 0.06 MB for a bounded experiment"
            return enough_data, detail
        return False, f"Training root missing: {root}"
    except Exception as exc:
        return False, f"Training root check failed: {exc}"


def _check_promotion_benchmark() -> tuple[bool, str]:
    path = Path("data/maintenance/user_chat_benchmark.json")
    try:
        import json

        rows: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return False, f"Promotion benchmark unavailable: {exc}"
    valid = [row for row in rows if isinstance(row, dict) and str(row.get("prompt", "")).strip()]
    if len(valid) < 5:
        return False, f"Promotion benchmark needs at least 5 valid cases (found {len(valid)})"
    return True, f"Promotion benchmark OK ({len(valid)} cases)"


def main() -> int:
    print("=" * 56)
    print("  ILM Setup Wizard")
    print("=" * 56)
    print()

    checks: list[tuple[str, list[tuple[bool, str]]]] = []

    # System
    sys_checks = [_check_python()]
    sys_checks.append(_check_module("torch"))
    sys_checks.append(_check_module("sentencepiece"))
    sys_checks.append(_check_browser_runtime())
    sys_checks.append(_check_module("bs4"))
    sys_checks.append(_check_gpu())
    checks.append(("System", sys_checks))

    # Disk
    dirs = [Path("models"), Path("data"), Path("IckleTraining")]
    disk_checks = _check_disk_space(dirs)
    checks.append(("Directories & Disk", disk_checks))

    # Training data
    checks.append(("Training Data", [_check_training_root()]))

    # Pipeline validation
    pipe_checks = [_check_tokenizer(), _check_model_forward(), _check_promotion_benchmark()]
    checks.append(("Pipeline", pipe_checks))

    # Results
    all_ok = True
    for section, items in checks:
        print(f"  [{section}]")
        for ok, msg in items:
            all_ok = all_ok and ok
            print(f"    {'[OK]' if ok else '[FAIL]'} {msg}")
        print()

    print("-" * 56)
    if all_ok:
        print("  READY — all checks passed.")
        print()
        print("  Next steps:")
        print("    python -m src.app train --data IckleTraining/my_corpus.txt --out models/il.pt")
        print("    python -m src.app chat --model models/il.pt")
        print("    python -m src.app serve-web")
        return 0
    else:
        print("  NOT READY — some checks failed. See above.")
        print()
        print("  Quick fix:")
        print("    pip install torch sentencepiece playwright beautifulsoup4")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
