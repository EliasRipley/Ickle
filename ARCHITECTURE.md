# Ickle Architecture (clean modular map)

This project is intentionally split into small modules, but organized by responsibility so you can treat it as one system.

The target here is an **ILM (Ickle Language Model)**: user-owned, local-first, and constrained to personal hardware instead of cloud-scale assumptions.

## Single entry point

Use this for everyday usage:

```bash
python -m src.app <command> [...args]
```

Commands:
- `chat`, `hub`, `serve-web`, `serve-control`
- `train`, `train-cycle`, `train-autopilot`, `train-intelligence-stack`, `lora-train`
- `federated-server`, `federated-client`, `swarm`, `torickle`
- plus maintenance/data/eval workflows exposed through `src/app.py`

Current command count is `59` (see `src/app.py` `CATEGORIES`; check `sum(len(v) for v in CATEGORIES.values())` rather than trusting this number if it's been a while — it drifts easily).

## Module map

- `src/model.py` — transformer architecture.
- `src/tokenizer.py` -- tokenizer abstraction (legacy char + sentencepiece scaffold).
- `src/train.py` — training loop and checkpointing.
- `src/chat.py` — inference CLI (supports quantized checkpoints).
- `src/app.py` — unified command router.
- `src/quantize_model.py` — optional dynamic INT8 quantization for CPU-friendly inference.
- `src/hub.py` — interactive user-facing hub and feedback capture.
- `src/build_feedback_corpus.py` — converts rated feedback into training corpus.
- `src/autodidact.py` — builds training corpus from ILM's own objectively successful coding attempts.
- `src/continual_learn.py` -- **deprecated**, superseded by `continual_guard.py`. Simple threshold-triggered retraining loop with no replay buffer, compartment mixing, or promotion gating; kept only for backward compatibility with existing automation.
- `src/partner_loop.py` — human-first clarify→capability→decision→journal control loop.
- `src/agent.py` + `src/clarify.py` + `src/autonomy.py` + `src/capabilities.py` — orchestration, clarification-first behavior, autonomy modes, and honest capability reporting.
- `src/state_store.py` — persistent local single-user memory for preferences and improvement notes.
- `src/user_tooling.py` — user-owned plugin tool loader (`user_tools/*.py`).
- `src/cloud_assist.py` + `src/assist.py` — optional cloud-assist bridge (user-configured, local-first by default).
- `src/skill_system.py` + `src/skill_manager.py` — generic skill lifecycle (learn/store/use) for any domain.
- `src/skill_repair.py` — incident logging + repair playbooks + tool scaffolding when skills fail in production.
- `src/reality_check.py` — explicit claims-vs-reality audit report for implemented/blocked/prototype areas.
- `src/runtime_flags.py` -- persisted runtime feature toggles used by web UI and task worker.
- `src/task_queue.py` + `src/task_actions.py` -- persistent background task queue and autonomous task runners.
- `src/eval_harness.py` -- repeatable evaluation suites used by model gating.
- `src/workspace_paths.py` + `src/workspace_check.py` -- workspace path policy and separation diagnostics for Ickle/IckleTraining.
- `src/serve_web.py` + `src/serve_control.py` + `web/*` -- local servers: `serve_web.py` provides chat UI (chat, streaming, models, sessions, flags, answer maps and local claim review) and `serve_control.py` provides the full control API (tasks, training, memory, maintenance, torickle, swarm, research, commons sync/adoption, and human peer feedback).
- `src/epistemics.py` + `src/federated/knowledge_commons.py` -- the Epistemic Commons: deterministic candidate-claim decomposition and related-source linking around each answer, plus a signed grow-only event set for human support/dispute/correction/adoption. Conflicts are preserved under peer merge, reviews are local unless individually marked shared, and imported peer reviews stay out of generation until the owner explicitly adopts one. See `docs/EPISTEMIC_COMMONS.md`.
- `src/verified_corrections.py` -- turns the owner's own active Epistemic Commons `correct`/`adopt` events into an oversampled `User:`/`Ickle:` corpus that `task_actions.run_continual_guard_task` mixes into every guarded training step by default (`auto_include_verified_corrections`), so a correction can become part of the model itself instead of staying a prompt-time-only patch via `context_for_prompt()`. Still gated by the same anti-forgetting promotion checks as any other training data; still restricted to the owner's own local-authority boundary, never un-adopted peer reviews. `GET /api/consolidation/status`; Control room -> Network -> Epistemic Commons -> "Consolidate corrections now". See `docs/EPISTEMIC_COMMONS.md#from-correction-to-model`.
- `src/friendly_errors.py` -- maps raw exceptions to plain-language messages for surfaces a non-technical user sees (web chat, control API, desktop dashboard). Server-side code still logs the real exception via `traceback.print_exc()`; this only controls what the user is shown. Deliberately not applied to hand-authored `PermissionError` messages (e.g. "Chat is disabled by runtime flags."), which are already safe and specific.
- `src/knowledge_modules.py` vs `src/scoped_knowledge.py` (+`delta_registry.py`/`delta_router.py`) -- both run on every `generate_response()` call and both pick "relevant LoRA knowledge for this prompt," but they are deliberately two different mechanisms, not duplicates: `knowledge_modules.py` selects adapters by keyword/topic overlap against a JSON registry and **permanently merges** their weights into the loaded model for that request; `scoped_knowledge.py` selects deltas via its router's scoring and applies them as **reversible gates** (`federated/lora.py:set_lora_gates`/`reset_lora_gates`), plus surfaces a `memory_context` block of curated facts that gets appended into the prompt (`ilm_chat.py` around the `scoped_knowledge_result` variable) -- a capability `knowledge_modules.py` has no equivalent for. Don't "clean up" one into the other without porting that fact-context feature first.
- `src/ilm_profile.py:apply_cpu_thread_budget()` -- sets OMP/MKL/OpenBLAS thread-count env vars alongside `torch.set_num_threads()` so the `--cpu-pct` hardware-sizing promise holds for NumPy/BLAS-backed code too, not just torch's own intra-op pool. Called at every training/inference entry point instead of a bare `torch.set_num_threads()`.
- `src/user_tool_runner.py` -- subprocess boundary for user-owned plugin tools.
- `src/research_memory.py` -- query/list interface for iterative research memory.
- `src/export_onnx.py` — ONNX export for non-Python inference runtimes.
- `src/build_preference_pairs.py` -- converts rated feedback into prompt/chosen/rejected preference pairs.
- `src/dpo_train.py` -- Direct Preference Optimization (DPO) alignment loop for local models.
- `api/ilm_openapi.yaml` — API contract for local endpoints.
- `src/tools/*` — external capabilities (web read, news scan, minecraft guide, notepad, image OCR/captioning).
- `src/agent_loop.py` -- iterative tool-calling agent loop (D-CoT/Agent-as-Tool style): the model plans and dispatches `<call>` XML tool calls, the loop executes them and feeds results back, up to `MAX_STEPS`. Wired into `ilm_chat.py`'s `generate_response()` behind the web UI's "Agent mode" toggle; its tool set is `_default_tools()` (web_read/news_search/memory_search/think) plus `_code_tools()` (see next entry).
- `src/code_agent.py` -- read/write/edit/run_command/search_repo plus lint/typecheck/test-loop helpers for deterministic code-editing workflows; also a standalone CLI (`code-agent`). Reachable from the live chat/agent path via `agent_loop.py`'s `_code_tools()` (write/edit/run_command gated behind the "Allow code execution" toggle; `agent_loop._resolve_within_workspace()` is the containment check that keeps those wrappers from escaping the project root, since `code_agent.py` itself does none -- its CLI is deliberately general-purpose) and from `src/mcp_server.py` (ungated -- the IDE's own tool-permission system is that surface's trust boundary instead).
- `src/mcp_server.py` -- exposes Ickle as an MCP stdio server (`python -m src.app mcp-server`) so IDEs/agent tools can summon it directly: `chat`, `image_understanding`, `check_capability`, `training_status`, plus the ungated code tools above.
- `src/federated/*` -- federated coordinator, clients, LoRA adapter logic, aggregation, and protocol helpers.
- `src/federated/keys.py` -- Ed25519 identity for messages any peer must be able to verify (offers/inference responses), distinct from the HMAC `SwarmIdentity` used by the torickle protocol.
- `src/federated/inference_swarm.py` -- P2P inference sharing: peers announce signed offers to serve a model and answer other peers' prompts directly (`python -m src.app infer serve/find/ask/report`). See `docs/INFERENCE_SHARING.md`.
- `src/federated/public_dht.py` -- opt-in trackerless public discovery for the model/training swarm. It uses a stable versioned key on the BitTorrent Mainline DHT (BEP 5), BEP 42-compatible node IDs, bounded bencode/KRPC parsing, periodic announcement, and public-endpoint filtering. DHT results remain untrusted until `SwarmNode` probes the Ickle protocol; the DHT never carries model or review content. See `docs/PUBLIC_SWARM.md`.
- `src/federated/nat_traversal.py` -- NAT reachability for `SwarmNode`/`InferenceNode`: STUN (RFC 5389) discovers the machine's actual public IP instead of a local-interface guess, and UPnP IGD asks the router to auto-forward the listening port. Opt-in (`start(attempt_nat_traversal=True)` / CLI `--nat-traversal` / the Control room's "Join swarm" switch) since it makes outbound network calls and a router change; best-effort and never fatal. Public discovery and NAT reachability are reported independently so an outbound-only node is not mislabelled reachable.
- `src/federated/contribution_ledger.py` -- local seed:peer contribution ledger (torrent-ratio-style) backing `infer report`; fed by torickle piece-serving (`src/federated/swarm.py`) and inference serving/consuming.
- `src/federated/codistill.py` -- cross-architecture co-distillation: peers with *different* model sizes/shapes (nano/laptop/desktop) teach each other by exchanging text answers to a shared probe set over the existing `infer serve` transport, instead of weights -- the coordinator-based FedAvg/Multi-Krum/DiLoCo path in `coordinator.py` requires matching LoRA shape against the same base model and can't include heterogeneous peers at all. Text overlap remains an outlier filter for candidate training data, which must still pass `continual_guard.py`; it no longer creates durable reputation merely from conformity. `PeerTrustStore` stays local and per-domain, but moves through explicit human review or future objective outcomes. The live `ask_swarm()` path exposes claim-level common ground and distinct contributions without voting away minority answers. See `docs/CODISTILLATION.md` and `docs/EPISTEMIC_COMMONS.md`.

## Research-backed choices already implemented

- RoPE positional encoding
- RMSNorm
- SwiGLU-style feed-forward
- AdamW + warmup + cosine LR + gradient clipping
- Clarification-first guardrails before tool execution

## Current high-impact priorities

1. Strengthen decentralized networking beyond the new public-DHT/signed-announcement path (Sybil resistance, relay donation for CGNAT/symmetric-NAT users, multi-source scheduling, and abuse controls). Trackerless fresh-node discovery and NAT status are implemented; private/direct bootstraps remain available for restricted networks.
2. Expand benchmark-driven autonomous learning loops and promotion gating.
3. Deepen mobile contribution path (Android hardening and iOS contributor path), including the new inference-serving role.
4. Continue training/inference efficiency work after speculative decoding and LoRA pipeline integration.
5. ~~Reputation-weighted peer selection for plain inference sharing~~ -- done: `infer ask` (`src/federated/inference_swarm.py`) now ranks discovered offers by `PeerTrustStore` domain-specific trust (via `classify_domain()`/`_rank_offers_by_trust()` in `codistill.py`) when a local trust store exists, instead of always picking the first offer found. Still open: wire federated training-round completion into the contribution ledger.
6. ~~Extend co-distillation with per-domain trust and probe-set rotation~~ -- done: `PeerTrustStore` (`src/federated/codistill.py`) tracks an EMA per `(peer_id, domain)` pair instead of one scalar per peer, with transparent migration from old flat-scalar files. Since the Epistemic Commons change, agreement alone no longer moves that EMA; explicit owner feedback does. `select_probes_for_round()` picks a deterministic, date-seeded random subset of the shared probe set each round (`--sample-size`, wired to 14/24 by default in the web UI's queued round) so the full probe list is never all exposed every time.

Detailed phased plan: `docs/HEAVY_TRAINING_BLUEPRINT_2026.md`.


## ILM hardware sizing

- `src/ilm_profile.py` detects CPU, RAM, and accelerator availability and derives a hardware-appropriate configuration.
- Training accepts resource percentages plus explicit `--block-size`, `--n-embd`, `--n-head`, `--n-layer`, and `--batch-size` overrides.
- Android separately defines named `nano`, `laptop`, and `desktop` mobile profiles.


## Autonomy mode design

- Objective: avoid patronizing UX; the user owns this instance and its behavior.
- Modes: `balanced`, `direct`, `power-user` (`src/autonomy.py`) -- currently differ only in tone/description, no mode gates behavior. There used to be a keyword-based "high-risk" refusal (`requires_high_risk_confirmation`) that intercepted prompts before generation; removed, since a fixed refusal string standing in for the model's real output is the same "hardcoded response" problem as any other canned answer, and this is user-owned local software, not a hosted service with third-party users to protect from each other.
- Hub commands: `/mode ...` and `/policy` to make behavior explicit and user-controlled.


## Honesty + persistence design

- Capability honesty: `/can <task>` consults `src/capabilities.py` and reports support status instead of pretending.
- Persistent instance behavior: `src/state_store.py` keeps preferences and improvement notes in local `data/ilm_state.json`.
- This supports a single-user ILM model where improvements persist across sessions.


## Correctness assurance

- `tests/test_core.py` validates capability honesty, autonomy, profile selection, and state persistence.
- `tests/test_user_tooling.py` validates loading/running local user-owned plugin tools.
- `tests/test_autodidact.py` validates self-generated coding corpus filtering rules.
- `tests/test_continual.py` validates retrain-trigger threshold logic.
- `tests/test_partner_loop.py` validates anti-gaslight decision flow and journaling.
- `tests/test_cloud_assist.py` validates optional cloud-assist configuration checks.
- `tests/test_skill_system.py` validates persistent skill registration and retrieval.
- `tests/test_skill_repair.py` validates incident logging, repair planning, and tool scaffolding.
- `tests/test_reality_check.py` validates audit report core coverage.
- `tests/test_sqlite_store.py` validates SQLite event persistence.
- `tests/test_model_maintain.py` validates model/checkpoint archival, `models/candidates/` sweep coverage (fixed: it used to be invisible to cleanup — `Path.glob("*.pt")` only looks at the given directory, never subdirectories, and training's actual output lands in `models/candidates/`), and that `run_data_maintenance()` doesn't crash (it called `asdict()` without importing it).
- `tests/test_federated_keys.py` validates Ed25519 identity sign/verify (including third-party verification and tamper rejection).
- `tests/test_contribution_ledger.py` validates seed:peer ratio math and persistence.
- `tests/test_inference_swarm.py` validates signed offer announce/discover, capacity limiting, and end-to-end signed request/response over HTTP.
- `tests/test_public_dht.py` validates bounded bencode, compact endpoint parsing, the published BEP 42 prefix vector, iterative discovery/announcement, and background status transitions without using the live internet.
- `tests/test_assets.py` validates non-Python web/API assets are present.
- Keep these tests green before adding new tools/policies.


## Multi-file-type architecture

- `config/ilm_policy.toml` stores policy defaults outside Python code.
- `schemas/skill_card.schema.json` defines skill-card contract.
- `sql/skill_events.sql` defines event schema consumed by `src/sqlite_store.py`.
