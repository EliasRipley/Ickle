import argparse
import importlib
import os
import sys

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))


CATEGORIES = {
    "Core": {
        "chat": "CLI chat with a trained model",
        "hub": "Interactive REPL hub interface",
        "train": "Main training loop (pretrain + instruction fine-tuning)",
        "lora-train": "LoRA adapter fine-tuning",
        "dpo-train": "DPO preference alignment training",
        "preflight": "System readiness check",
        "show-profile": "Display hardware profile info",
    },
    "Serving & UI": {
        "app": "Desktop app: chat UI in a native window (recommended for end users)",
        "serve-web": "Start chat web UI server (port 8787), for opening in a browser instead",
        "serve-control": "Start control API server (port 8788) for task queue & training automation",
        "mcp-server": "Run Ickle as an MCP server (stdio) for IDE/agent integration -- requires `pip install -r requirements-mcp.txt`",
    },
    "Federated": {
        "federated-server": "Federated training coordinator",
        "federated-client": "Federated training participant",
        "swarm": "P2P swarm node for delta exchange",
        "torickle": "Delta bundle pack/verify/reassemble",
        "infer": "P2P inference sharing: serve/find/ask peers, view seed:peer ratio",
        "codistill": "Cross-architecture peer teaching: probe/round/trust (no coordinator, no matching model shapes required)",
    },
    "Corpus Building": {
        "build-clean-corpus": "Build clean training corpus from sources",
        "build-smart-corpus": "Build smart filtered corpus",
        "build-base-lm-corpus": "Build base language model corpus",
        "build-feedback-corpus": "Convert hub feedback into training corpus",
        "build-preference-pairs": "Build DPO preference pairs from rated feedback",
        "build-honest-context-package": "Build behavior-focused SFT package",
        "open-dataset-ingest": "Bounded streaming ingest from HuggingFace datasets",
        "sanitize-training-data": "Deduplicate and sanitize training data",
    },
    "Training Automation": {
        "train-cycle": "Repeated build/train/smoke-test cycle",
        "train-autopilot": "Quality-gated growth loop with eval gates",
        "train-intelligence-stack": "Two-stage pretraining pipeline",
        "continual-learn": "(deprecated, use continual-guard) Threshold-triggered continual learning",
        "continual-guard": "Guarded training step with anti-forgetting",
        "ollama-teach": "Ollama teacher (uses local Ollama models)",
        "opencode-teach": "opencode teacher (uses this AI interactively)",
        "anthropic-teach": "Anthropic teacher (uses Claude API)",
        "registry-teach": "Multi-provider teacher (uses any registered provider)",
    },
    "Evaluation": {
        "chat-benchmark": "Run chat benchmark evaluations",
        "honesty-context-eval": "Run honesty context evaluation",
        "reality-check": "Full capability audit report",
    },
    "Model Management": {
        "model-library": "Local-first model package sharing",
        "model-maintain": "Clean up old model checkpoints",
        "quantize": "INT8 post-training quantization",
        "export-onnx": "Export model to ONNX format",
        "knowledge-modules": "Train/compose additive LoRA knowledge modules",
    },
    "Code Agent": {
        "code-index": "Build code repository index",
        "code-agent": "Code-aware agent with repo context",
        "code-corpus": "Build code-focused training corpus",
        "code-repair": "Automated test repair loop",
        "code-eval": "Run code evaluations",
        "code-memory": "Code memory (persistent learnings)",
    },
    "Agent & Skills": {
        "self-improve": "Self-improvement loop",
        "autodidact": "Objective-driven self-generated coding corpus",
        "partner-loop": "Human-first clarify-before-act loop",
        "assist": "Cloud-assist bridge",
        "skill-manager": "Skill lifecycle management",
    },
    "Memory & Data": {
        "research-memory": "Query or list persistent research notes",
        "workspace-check": "Runtime vs training data separation check",
        "training-maintain": "Clean up old training data",
        "supercharge": "Performance tuning diagnostics",
    },
    "Trainer Platform": {
        "trainer-provider": "Register/list trainer providers with budget",
        "trainer-budget": "Manage trainer day-budgets",
        "trainer-operator": "Run task-graph training programs",
    },
}

DESCRIPTIONS = {cmd: desc for cat in CATEGORIES.values() for cmd, desc in cat.items()}


def _print_help():
    print("Ickle — Ickle Language Model")
    print("Unified entry point for all workflows\n")
    print("Usage: python -m src.app <command> [args ...]\n")
    print("Commands:")
    for category, commands in CATEGORIES.items():
        print(f"\n  {category}:")
        for cmd, desc in commands.items():
            print(f"    {cmd:<28s} {desc}")
    print()
    print("Examples:")
    print("  python -m src.app chat --model models/tiny.pt --prompt \"Hello\"")
    print("  python -m src.app train --data data/corpus.txt --out models/tiny.pt --steps 2000")
    print("  python -m src.app serve-web --port 8787")
    print("  python -m src.app help")
    print()
    print("See docs/ICKLE_REFERENCE.md for detailed docs.")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        _print_help()
        return

    command = sys.argv[1]
    if command not in DESCRIPTIONS:
        print(f"Unknown command: {command}\n")
        _print_help()
        sys.exit(1)

    module_imports = {
        "train": "src.train",
        "chat": "src.chat",
        "hub": "src.hub",
        "preflight": "src.preflight_win11",
        "self-improve": "src.self_improve",
        "build-feedback-corpus": "src.build_feedback_corpus",
        "show-profile": "src.show_profile",
        "quantize": "src.quantize_model",
        "autodidact": "src.autodidact",
        "continual-learn": "src.continual_learn",
        "partner-loop": "src.partner_loop",
        "assist": "src.assist",
        "skill-manager": "src.skill_manager",
        "reality-check": "src.reality_check",
        "serve-web": "src.serve_web",
        "serve-control": "src.serve_control",
        "mcp-server": "src.mcp_server",
        "chat-benchmark": "src.chat_benchmark",
        "export-onnx": "src.export_onnx",
        "build-clean-corpus": "src.build_clean_corpus",
        "build-smart-corpus": "src.build_smart_corpus",
        "build-honest-context-package": "src.build_honest_context_package",
        "build-base-lm-corpus": "src.build_base_lm_corpus",
        "train-cycle": "src.train_cycle",
        "train-autopilot": "src.train_autopilot",
        "train-intelligence-stack": "src.train_intelligence_stack",
        "federated-server": "src.federated.server",
        "federated-client": "src.federated.client",
        "workspace-check": "src.workspace_check",
        "model-library": "src.model_library",
        "model-maintain": "src.model_maintain",
        "training-maintain": "src.training_maintain",
        "build-preference-pairs": "src.build_preference_pairs",
        "honesty-context-eval": "src.honesty_context_eval",
        "dpo-train": "src.dpo_train",
        "research-memory": "src.research_memory",
        "open-dataset-ingest": "src.open_dataset_ingest",
        "continual-guard": "src.continual_guard",
        "sanitize-training-data": "src.sanitize_training_data",
        "supercharge": "src.supercharge_ickle",
        "torickle": "src.torickle",
        "swarm": "src.federated.swarm",
        "app": "src.desktop_app",
        "infer": "src.federated.inference_swarm",
        "codistill": "src.federated.codistill",
        "lora-train": "src.lora_train",
        "trainer-provider": "src.trainer_providers_cli",
        "trainer-budget": "src.trainer_providers_cli",
        "trainer-operator": "src.trainer_orchestrator_cli",
        "code-index": "src.repo_index",
        "code-agent": "src.code_agent",
        "code-corpus": "src.code_corpus",
        "code-repair": "src.test_repair_loop",
        "code-eval": "src.code_evals",
        "code-memory": "src.code_memory",
        "ollama-teach": "src.teacher_ollama",
        "opencode-teach": "src.teacher_opencode",
        "anthropic-teach": "src.teacher_anthropic",
        "registry-teach": "src.teacher_registry",
        "knowledge-modules": "src.knowledge_modules",
    }

    module_name = module_imports[command]
    module = importlib.import_module(module_name)
    sys.argv = [module_name.rsplit(".", 1)[-1] + ".py"] + sys.argv[2:]
    if hasattr(module, "main"):
        module.main()
    elif hasattr(module, "run"):
        module.run()


if __name__ == "__main__":
    main()
