# Ickle — Local-First Language Model (ILM)

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)

Ickle is a small GPT-style language model designed to run entirely on personal hardware — a local-first alternative to sending every prompt to a corporate data center. It trains on your own data, runs offline, and can optionally grow through a peer-to-peer federated training network shared with other Ickle users. No datacenter required, and nothing about your device leaves it unless you explicitly turn on network sharing.

## Vision

The long-term goal is an ILM (Ickle Language Model) that individuals — not a company — own and improve, by pooling spare compute from desktops, laptops, and eventually mobile devices (Android first, with iOS and other platforms planned as the network matures) into training rounds that benefit everyone who contributes. Once a model reaches a meaningful improvement, it's shared back to the network for all participants to use.

The project is licensed AGPL v3 specifically because of this goal: the network-copyleft terms mean anyone who builds a service on top of Ickle, including over a network, must release their modifications back to the community. That's meant to prevent the same closed-source corporate capture the project exists to be an alternative to.

## Current status (honest, as of this checkout)

This is an early-stage research/engineering project, not a finished product. Concretely, right now:

- **No trained model checkpoints ship in this checkout.** `models/` is empty by design — train your own from scratch with the commands below, or connect to the federated network once it has active participants.
- **Small local checkpoints, not frontier capability.** Even a fully trained run here produces a small model sized to run on personal hardware, not something comparable to large hosted models. Closing that gap is a long-term, multi-stage goal — see [docs/HEAVY_TRAINING_BLUEPRINT_2026.md](docs/HEAVY_TRAINING_BLUEPRINT_2026.md) for the credible next milestones (reproducible training manifests, contamination-resistant evaluation, promotion gating, secure aggregation) rather than a promise of immediate parity with big-tech models.
- **The P2P federated network and inference-sharing network are both functional but need active participants to be useful.** A single-user install works standalone; the network features (`federated-server`/`federated-client`, `infer serve`/`find`/`ask`) are there for when you want to pool compute with others.
- **The web app is the primary, complete interface.** Chat plus every management feature (training, tasks, models, memory, network status, sharing) is reachable from a single browser tab or the desktop app — see below.

If you're evaluating this project, treat it as infrastructure for an ambitious goal that's still being built, not as a drop-in chatbot replacement today.

## Quick start

```bash
# Setup
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
pip install -r requirements-dev.txt
python -m playwright install firefox

# Preflight check
python -m src.preflight_win11

# Train a tiny model from your own data
python -m src.train --data data/my_corpus.txt --out models/tiny.pt --steps 2000

# Chat locally
python -m src.app chat --model models/tiny.pt --prompt "Hello" --temperature 0.8 --top-k 40

# Interactive hub (REPL chat)
python -m src.app hub

# Desktop app: chat UI in a native window (recommended for most users)
python -m src.app app
```

### Environment variables (all optional)

Ickle runs fully offline with none of these set. If you want them, copy `.env.example` to `.env` and fill in only what you use — `.env` is gitignored and never leaves your machine:

| Variable | Used for |
|---|---|
| `HF_TOKEN` | Higher Hugging Face Hub rate limits when downloading datasets or the vision captioning model. |
| `ANTHROPIC_API_KEY` | Optional: use Claude as a distillation "teacher" model for training-data generation (`src/teacher_anthropic.py`). |
| `ILM_CLOUD_API_KEY` (+`ILM_CLOUD_BASE_URL`/`ILM_CLOUD_ENDPOINT`/`ILM_CLOUD_MODEL`) | Optional cloud-assist bridge to an OpenAI-compatible API (`src/cloud_assist.py`), off unless explicitly configured. |

Other distillation teacher providers each name their own env var per provider config rather than a fixed variable — see `python -m src.app trainer-provider --help`.

## Two-stage pretraining (recommended)

```bash
# Phase 1: raw-text LM pretraining
python -m src.train --data data/corpus.txt --out models/pretrain.pt --pretrain-data data/pretrain.txt --pretrain-steps 3000

# Phase 2: instruction fine-tuning (auto-resumes from pretrain)
python -m src.train --data data/corpus.txt --out models/final.pt --steps 2000
```

## Web UI

The full app — chat plus the Manage panel (training, background tasks, models, memory, hardware dashboard, federated network status, add-ons, and P2P sharing) — is available two ways: as a native desktop window (`app`, no browser needed) or as a regular web server you open in a browser tab (`serve-web`). Both give you the same features; `serve-web` starts its own management API automatically, so a plain browser tab is not a reduced experience.

```bash
# Native window (recommended)
python -m src.app app

# Or: plain web server, open the URL yourself
python -m src.app serve-web --port 8787
# open http://127.0.0.1:8787

# Chat-only, no management panel (smaller footprint)
python -m src.app serve-web --port 8787 --no-control
```

### Chat capabilities

Beyond plain chat, the web UI has three opt-in toggles next to the prompt box:

- **Agent mode** — instead of a single forward pass, Ickle runs an iterative tool-calling loop (`src/agent_loop.py`) that can read the web, search news, search saved memory, and reason step by step before answering. The "Show Ickle's thinking" panel then shows the real step-by-step trace (each tool called, its parameters, and its result or error), not just a single reasoning pass.
- **Image attach** — attach an image (screenshot, photo, document) and Ickle extracts any visible text (OCR) and generates a short description of its contents (`src/tools/image_reader.py`), feeding both into the model as context. Ickle's own model stays text-only; this wraps `easyocr` + a BLIP captioning model the same way `web_read`/`news_search` wrap external capabilities. Needs `pip install -r requirements-vision.txt`.
- **Allow code execution** (only meaningful together with Agent mode) — lets Ickle read, write, and edit files and run shell commands in the local project workspace (`src/code_agent.py`, gated through a workspace-containment check so it can't read/write outside the project root). Off by default; read-only file access (`read_file`/`search_repo`) is available in Agent mode regardless of this toggle, matching how other coding assistants separate "can look" from "can change things."

Responses containing fenced code blocks render as proper monospace blocks with a Copy button, not raw backticks.

## DPO preference alignment

```bash
python -m src.app build-preference-pairs --feedback data/hub_feedback.jsonl --out data/pairs.jsonl
python -m src.app dpo-train --model models/tiny.pt --prefs data/pairs.jsonl --out models/tiny_dpo.pt --steps 300
```

## Federated training

```bash
# Local coordinator (safe development default)
python -m src.app federated-server --host 127.0.0.1 --port 8788 --base-model models/tiny.pt --min-clients 3

# Desktop client
python -m src.app federated-client --server http://127.0.0.1:8788 --local-data data/my_corpus.txt --base-model models/tiny.pt
```

For internet-facing contributors, use TLS (or a TLS reverse proxy) and a registration secret. Ickle refuses a public plaintext listener unless `--allow-insecure-public` is explicitly supplied for isolated development. See [the federated and mobile guide](docs/FEDERATED_MOBILE.md).

## Inference sharing (P2P answer serving)

Contribute isn't only about training ("seed") — peers can also donate spare compute to answer other users' prompts directly ("peer" usage, torrent-ratio style), with no company or coordinator in the loop:

```bash
# Host inference for a model and announce it to the swarm
python -m src.app infer serve --model models/tiny.pt --capacity 2

# Find peers offering inference, then ask one a question -- --bootstrap
# points at a peer you already know about (e.g. the one above, or anyone
# who gave you their host:port); without it there's no one to ask.
python -m src.app infer find --bootstrap peer-host:8791 --model-hash <hash>
python -m src.app infer ask "What is RoPE?" --bootstrap peer-host:8791 --model-hash <hash>

# Check your local seed:peer contribution ratio
python -m src.app infer report
```

The same ratio is also visible in the web app's Manage → Sharing tab. See [docs/INFERENCE_SHARING.md](docs/INFERENCE_SHARING.md) for the protocol and trust model.

## IDE / agent integration (MCP)

Ickle can be summoned directly from an MCP-compatible IDE or agent tool (Claude Code, Cursor, etc.) instead of needing a browser tab — this is the standard way local tools integrate today; Ickle has no npm ecosystem and doesn't need one for this.

```bash
pip install -r requirements-mcp.txt
python -m src.app mcp-server
```

Point your MCP client at it as a stdio server, e.g. in a `.mcp.json`:
```json
{"mcpServers": {"ickle": {"command": "python", "args": ["-m", "src.app", "mcp-server"]}}}
```

Exposes nine tools:

- `chat` — talk to your locally trained model.
- `image_understanding` — OCR + a short description of an image (needs `requirements-vision.txt`).
- `check_capability` — honestly reports whether Ickle supports a given task, reusing the same capability-honesty system the chat path uses.
- `training_status` — read-only progress check on an active training run.
- `read_file`, `search_repo` — read-only access to the local workspace.
- `write_file`, `edit_file`, `run_command` — **write files and run arbitrary shell commands in the local workspace, with no additional prompt or gate from Ickle itself.** Unlike the chat/web-UI agent path (where code execution is off by default and requires an explicit opt-in toggle per request), these MCP tools are deliberately ungated: the calling IDE or agent client's own tool-permission system is the trust boundary here, the same way it already is for other IDE-integrated coding agents (e.g. Claude Code's own file/bash tools). Only point an MCP client at `ickle mcp-server` if you trust it the way you'd trust any other coding agent with shell access on this machine.

## Documentation

| Doc | Contents |
|-----|----------|
| [docs/INDEX.md](docs/INDEX.md) | Full docs index |
| [docs/ICKLE_REFERENCE.md](docs/ICKLE_REFERENCE.md) | Commands, training, federated, model library |
| [docs/HEAVY_TRAINING_BLUEPRINT_2026.md](docs/HEAVY_TRAINING_BLUEPRINT_2026.md) | Honest next milestones toward stronger capability |
| [docs/FEDERATED_MOBILE.md](docs/FEDERATED_MOBILE.md) | Mobile contributor API contract |
| [docs/INFERENCE_SHARING.md](docs/INFERENCE_SHARING.md) | P2P inference sharing: protocol, trust model, seed:peer ratio |
| [docs/CONTINUAL_LEARNING_GUARD.md](docs/CONTINUAL_LEARNING_GUARD.md) | Catastrophic forgetting prevention |
| [docs/ADDITIVE_KNOWLEDGE_MODULES.md](docs/ADDITIVE_KNOWLEDGE_MODULES.md) | Additive modular learning without overwriting the core model |
| [docs/HONEST_CONTEXT_TRAINING_PACKAGE.md](docs/HONEST_CONTEXT_TRAINING_PACKAGE.md) | Behavior-focused SFT + DPO package |
| [docs/WORKSPACE_SEPARATION.md](docs/WORKSPACE_SEPARATION.md) | Runtime vs training data separation |

## Hardware sizing

Ickle derives context size, embedding width, layers, batch size, and thread count from available CPU, RAM, GPU support, and the requested resource percentages. Inspect the result before training:

```bash
python -m src.app show-profile --cpu-pct 70 --ram-pct 70 --gpu-pct 0
```

CPU-only inference and training are supported; training is simply slower without an accelerator. Android uses its own `nano`, `laptop`, and `desktop` mobile profiles.

## Repository structure

```
├── src/               # Core Python modules
│   ├── model.py       # GPT-style ILM architecture
│   ├── train.py       # Training loop
│   ├── tokenizer.py   # SentencePiece + Char tokenizers
│   ├── app.py         # Unified CLI entry (58 subcommands)
│   ├── hub.py         # REPL interface
│   ├── serve_web.py   # Chat + auto-started control API (port 8787)
│   ├── desktop_app.py # Native-window wrapper around serve_web.py (the `app` command)
│   ├── serve_control.py # Control API server (port 8788)
│   ├── federated/     # P2P training protocol
│   ├── tools/         # Desktop automation, timers, Firefox reader, Notepad, etc.
│   └── ilm_chat.py    # Core chat engine with memory, web, & tool routing
├── docs/              # Documentation
├── web/               # Web UI (HTML/CSS/JS) — the primary interface
├── tests/             # Unit tests
├── config/            # TOML policies
├── schemas/           # JSON schemas
├── sql/               # SQLite schemas
├── IckleTraining/     # Training corpus workspace (empty scaffold in this checkout — see below)
└── models/            # Trained checkpoints (gitignored, empty in this checkout)
```

## Single-binary client (PyInstaller)

Build a portable Windows `.exe` with all dependencies bundled (requires PyInstaller):

```bash
pip install -r requirements-dev.txt
scripts\build_client.bat
```

Output: `dist\ickle_client\ickle_client.exe` (~720 MB, includes Python + PyTorch).

Run from anywhere:
```bash
dist\ickle_client\ickle_client.exe --version
dist\ickle_client\ickle_client.exe preflight
dist\ickle_client\ickle_client.exe train --data data/corpus.txt --out models/model.pt --steps 500
dist\ickle_client\ickle_client.exe chat --model models/model.pt --prompt "Hello"
```

## Tests

```bash
python -m unittest discover -s tests -v
```

## License

AGPL v3 — see [LICENSE](LICENSE). This strong network-copyleft license ensures that any entity using Ickle (including over a network) must release their modifications under the same terms, preventing closed-source corporate capture while keeping the project open for all contributors.

## Training data

Canonical training corpora belong in the separate [IckleTraining](https://github.com/EliasRipley/IckleTraining) workspace. This checkout's `IckleTraining/` is an empty scaffold (folder structure only, ready for a fresh corpus) — clone or populate the full corpus workspace before a substantial training run. `python -m src.app workspace-check` reports the active location.

## Technical details

The rest of this README is about what Ickle does and how to run it. This section is for readers who want to know how it's built — skip it if you don't.

- **Subword tokenizer**: SentencePiece/BPE (vocab up to 16K), with char fallback
- **Modern architecture**: RoPE, RMSNorm, SwiGLU MLP, GQA, KV-cache, EMA, LLRD, gradient checkpointing, z-loss, multi-token prediction (Meta 2024)
- **Loss masking**: Only response tokens (after `Ickle:`) contribute to training loss
- **Gradient accumulation**: Simulates larger batches on limited hardware
- **Advanced LR schedule**: Warmup + stable hold + cosine decay to min_lr_ratio
- **INT8 quantization**: Post-training for faster CPU inference
- **ONNX export**: `python -m src.app export-onnx --model models/tiny.pt --out models/tiny.onnx`
- **REPL hub**: `python -m src.app hub` for interactive chat
- **Manage panel**: training, tasks, models, memory, hardware dashboard, network status, and P2P sharing, all from the web UI (`app` or `serve-web`)
- **Skill system**: Learn, store, and activate domain-specific skills
- **Model library**: Local-first package sharing with export/install/validate
- **Autodidact loop**: Self-improvement via objective pass/fail feedback
- **Continual learning guard**: Compartmented mixing + promotion gates against forgetting
- **Additive knowledge modules**: Train LoRA topic modules and compose them at inference time
- **Research memory**: Persistent note-taking across training sessions
- **Ollama teacher**: Generate SFT pairs from any Ollama model
- **Federated training**: P2P swarm-based delta sharing with mobile clients
- **Inference sharing**: peers donate spare compute to answer other users' prompts P2P, tracked via a local seed:peer contribution ratio (`infer serve`/`ask`/`report`)
