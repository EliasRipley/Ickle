# Co-distillation (cross-architecture peer teaching)

Ickle's federated path (`src/federated/coordinator.py`) already runs real
FedAvg / trimmed-mean / Multi-Krum aggregation with a DiLoCo-style outer
optimizer. It works well — but only between peers whose LoRA adapter has
**identical shape against the same base model**
(`src/knowledge_modules.py`'s `_base_model_matches` literally checks the
base-model hash). Ickle's own hardware sizing
(`src/ilm_profile.py`, `nano`/`laptop`/`desktop` profiles) deliberately
produces peers with *different* `n_embd`/`n_layer` — a phone-tier nano peer
and a desktop peer can never join the same weight-level federation round,
even though the project exists specifically so people on different hardware
can build a shared intelligence together.

Co-distillation (`src/federated/codistill.py`) closes that gap by exchanging
**text, not weights.** Any two peers that can hold a conversation — regardless
of architecture, size, or tokenizer — can teach each other.

## Protocol

1. Peers already running `infer serve`
   (see `docs/INFERENCE_SHARING.md`) are the teacher pool — no new server, no
   new listener, no new signature scheme. Co-distillation reuses the signed
   Ed25519 `InferenceOffer`/`request_inference` transport verbatim.
2. A learner asks several peers (ranked by locally-observed teaching trust
   **in that probe's domain**, most-trusted first) the same **shared probe
   prompt** (`data/codistill/probe_prompts.json` — 24 prompts spanning
   reasoning, factual knowledge, code, instructions, honesty, math, and
   language, id-prefixed by domain e.g. `code-01`, `math-02`). Each round
   uses a deterministic, date-seeded random subset of the full probe list
   (`select_probes_for_round()`) rather than always exposing every probe, so
   there's nothing static to memorize/game over time.
3. Because multiple peers answer the *same* prompt, their answers can be
   checked against each other: each response's **consensus score** is its
   average text-overlap with the other responses to that probe
   (`score_consensus()`). This is the text-domain analogue of Multi-Krum
   (`src/federated/aggregation.py`'s Byzantine-robust weight aggregator) — a
   lazy, broken, or hostile peer's outlier answer scores low and is dropped,
   instead of being blindly trusted because it answered first.
4. Consensus scores update a local `PeerTrustStore`
   (`data/codistill/peer_trust.json`, an EMA **per `(peer_id, domain)` pair**,
   not one scalar per peer) so future rounds ask proven teachers first —
   this is the "reputation-weighted peer selection" that
   `docs/INFERENCE_SHARING.md` lists as future work, applied here to
   teaching quality specifically. Tracking trust per domain means a peer
   that teaches code well but creative writing poorly is preferred for code
   questions and passed over for creative ones, instead of one flat number
   hiding the difference. `classify_domain()` buckets a live question into
   the same domains a probe's id already encodes, so `infer ask`
   (`src/federated/inference_swarm.py`) and the live `ask_swarm()` path both
   draw on this same per-domain reputation, not just training rounds.
5. Surviving (prompt, response) pairs become an ordinary distillation corpus
   in the exact `DialogPair` format `src/continual_guard.py` already
   consumes. **Nothing is trained on directly from the network** — the
   existing replay-buffer anti-forgetting mixing and regression/promotion
   gates still stand between a peer's answers and anything actually being
   promoted into a live model. A malicious peer whose answers happen to win
   consensus can still only nudge training data; it cannot bypass the gate.

No new trust authority, no new signature scheme, no new corpus format, no
central aggregator anywhere in this path.

## Web UI

The Network tab (`web/index.html`, "Peer teaching") exposes the training-time
path without a terminal: a "Run a teaching round now" button queues a
`codistill_round` background task (`src/task_actions.py:run_codistill_round_task`)
using the same bootstrap peer list the tab's "Known peers" list manages, and
shows the locally-observed trust ranking plus the last round's results
(`GET /api/codistill/status`, backed by `ControlRuntime.get_codistill_status`
in `src/serve_control.py`). The distilled corpus still has to be fed through
`continual-guard run-step` (below) explicitly — the UI stops at "write the
corpus," matching the CLI's boundary between distillation and training.

There is also a live path: `ask_swarm()` in `src/federated/codistill.py`
reuses the exact same signed transport, consensus scoring, and trust store,
but for the user's actual question right now instead of a shared probe set,
querying trust-ranked peers concurrently since a person is waiting on the
answer. Every assistant chat message gets an "Ask the swarm too" button that
calls `POST /api/swarm/ask` (`ControlRuntime.ask_swarm`) and shows each
peer's answer with its consensus/trust score, highlighting the pick. A live
ask still updates the same `PeerTrustStore` a training round reads from, so
the two paths keep improving the same peer-selection signal instead of being
two disconnected features.

## CLI reference

```bash
# See the shared probe set
python -m src.app codistill probes

# Run one teach-and-learn round against the swarm
python -m src.app codistill round --bootstrap 203.0.113.5:8791 \
  --max-teachers-per-probe 5 --min-teachers-per-probe 2 --min-consensus 0.15 \
  --sample-size 14  # 0 (default) uses every probe; N rotates a date-seeded subset

# Inspect locally-observed peer teaching trust, broken down per domain
python -m src.app codistill trust

# Feed the resulting corpus through the existing anti-forgetting pipeline
python -m src.app continual-guard run-step \
  --new-corpus data/codistill/distilled_corpus.txt \
  --core-corpus data/ickle_curated_only.txt \
  --baseline-model models/ickle_clean.pt --out-model models/ickle_candidate.pt \
  --promote-to models/ickle_clean.pt --promotion-gate
```

## Trust model and limits

- Ledger crediting is free: because this reuses `request_inference`, teachers
  accrue `peer_requests_served` and learners accrue `peer_requests_consumed`
  in the existing `ContributionLedger` automatically, same as any other
  `infer ask`.
- `PeerTrustStore` is local and unshared, the same trust model as the
  contribution ledger's seed:peer ratio — each peer forms its own opinion
  from its own observations, nothing is broadcast or globally agreed.
- Consensus requires at least `--min-teachers-per-probe` (default 2)
  responding peers; a probe with too few responders is skipped for that
  round rather than accepting an uncorroborated single answer.
- This is not Byzantine-fault-tolerant in the cryptographic sense — a large
  enough coordinated group of colluding peers could still manufacture
  "consensus" around a bad answer. The regression/promotion gate in
  `continual_guard.py` is the actual backstop: it evaluates the *trained
  result* against held-out core prompts, not the distillation source, so a
  model that got worse from bad peer data is refused promotion regardless of
  how confidently the swarm agreed.
- The probe *file* is still a static, shared file — rotation only changes
  which subset of it is live on a given day, not the underlying content.
  Peers running a round on the same UTC day converge on the same subset
  without any coordinator, since the sampling seed is just the date; a
  peer running an out-of-band CLI round can override it with
  `--sample-size`/deterministic seed for reproducibility.

## Future work

- Weight probe selection so infrequently-covered domains get sampled more,
  instead of a uniform random subset each day.
- Let learners contribute their own probe prompts (with provenance) instead
  of only the shipped set.
