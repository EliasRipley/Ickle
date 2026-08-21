# Inference sharing (P2P answer serving)

Ickle's federated path already lets peers donate compute to *training*. This
adds the other half: peers can also donate spare compute to *answer prompts*
for other users, directly, peer to peer — no coordinator, no company sitting
in the middle of a chat request.

The mental model is the one used by seedbox/Tor-style networks: **seed** is
what you give the network (serving training pieces, completing training
rounds, answering someone else's prompt), **peer** is what you take from it
(asking the network to answer your prompt). `infer report` shows your local
seed:peer ratio so contribution is visible, the same way a torrent client
shows upload/download ratio.

This is a prototype: no payment, no SLA, no cross-network reputation
broadcast, and peer selection is currently "first verified offer that
responds." Treat it as the wire protocol and local trust primitives that a
smarter scheduler can be built on top of later.

## Why a second identity type

`src/federated/identity.py` (`SwarmIdentity`) signs with HMAC using a secret
only the signer holds. That's fine for the existing torickle bundle protocol,
where announcements are looked up through peers who already trust the
DHT-stored blob. Inference responses need a stronger property: **any peer,
including one that has never seen this signer before, must be able to verify
the message came from the claimed peer_id.** HMAC can't do that without
sharing the secret. `src/federated/keys.py` adds a small Ed25519 identity
(`EdIdentity`) for exactly this — `peer_id` is derived from the public key
(`sha256(pubkey)[:40 hex]`), so anyone can recompute and check it.

This is additive: the existing torickle/training-swarm protocol and its
`SwarmIdentity` are untouched.

## Protocol

1. A peer with a loaded model and spare capacity runs:
   ```bash
   python -m src.app infer serve --model models/tiny.pt --capacity 2
   ```
   This starts an HTTP listener (`/infer/v1/`) and periodically announces a
   signed `InferenceOffer` (peer_id, pubkey, model_hash, host:port, capacity,
   context window) into the same Kademlia-style DHT used for torickle bundle
   discovery (`src/federated/peer_discovery.py`), under a separate key
   namespace (`ickle:infer:offer:*`).

2. A peer wanting an answer runs:
   ```bash
   python -m src.app infer find --bootstrap peer-host:8791 --model-hash <hash>     # see who's offering
   python -m src.app infer ask "What is a RoPE embedding?" --bootstrap peer-host:8791
   ```
   `--bootstrap` names a peer you already know about (the DHT has no global
   rendezvous point, so `find`/`ask` have nothing to query without one --
   `serve` announces itself, but something has to tell you where to look).
   `ask` looks up offers (verifying each signature before trusting it),
   picks one, and POSTs the prompt directly to that peer's `/infer/v1/generate`.

3. The response is signed by the serving peer's Ed25519 key. The requester
   verifies it and reports `"verified": false` if the signature doesn't check
   out — that means the message didn't come from who it claims to, not that
   the answer is factually correct. There is no correctness guarantee on
   answers from an untrusted peer; this only proves provenance.

4. Both sides update their local `ContributionLedger`
   (`src/federated/contribution_ledger.py`, stored at
   `data/torickle/contribution_ledger.json`): the server records
   `peer_requests_served`, the client records `peer_requests_consumed`.
   Serving a torickle piece (`src/federated/swarm.py`) already feeds the same
   ledger's `seed_pieces_served`. Wiring federated training-round completion
   into `seed_training_rounds` is a natural follow-up (the ledger API already
   has `record_training_round()`; `src/federated/client.py` doesn't call it
   yet).

## Trust model and limits

- Offers and responses are signed, so you know which peer_id you're talking
  to and that the bytes weren't altered in transit — the same trust level as
  the existing torickle path, just checkable by strangers instead of only by
  the original signer.
- There is currently **no content moderation and no rate-limit beyond
  `--capacity`.** Any peer offering inference will answer any prompt sent to
  it, subject only to concurrency capacity. Treat this like any other P2P
  swarm: don't send secrets in a prompt to an untrusted peer, and don't
  assume an untrusted peer's answer is correct.
- Peer selection: when `data/codistill/peer_trust.json` exists (built up by
  running co-distillation rounds, see `docs/CODISTILLATION.md`), `infer ask`
  ranks discovered offers by locally-observed per-domain teaching trust and
  picks the top-ranked one instead of the first offer found. Without that
  file (a fresh install, or one that has never run a co-distillation round)
  it still falls back to "first offer found."
- Sybil resistance is the same as the rest of the DHT: none yet. A hostile
  actor can flood offers. This is an accepted prototype limitation, matching
  the existing note in `docs/FEDERATED_MOBILE.md`.
- `infer serve` binds a plaintext HTTP listener. For an internet-facing peer,
  put it behind TLS the same way `federated-server` requires for public
  deployments; `--host 127.0.0.1` (LAN/localhost only) is the safe default for
  local experimentation.

## CLI reference

```bash
# Host inference for a model, capacity = max concurrent requests accepted
python -m src.app infer serve --model models/tiny.pt --capacity 2 \
  --bootstrap 203.0.113.5:8790

# List currently-announced offers (optionally filtered by model hash).
# --bootstrap is required to learn about any peer you haven't already seen.
python -m src.app infer find --bootstrap 203.0.113.5:8791 --model-hash <hash>

# Ask the network (or one explicit peer) a single prompt
python -m src.app infer ask "Explain gradient checkpointing" --bootstrap 203.0.113.5:8791 --model-hash <hash>
python -m src.app infer ask "Hello" --peer 203.0.113.5:8791

# Local seed:peer contribution ratio
python -m src.app infer report
```

## Future work

- Feed `federated-client` round completion into `record_training_round()` so
  training contribution and inference contribution show up in one ratio.
- Android inference-serving path (mirrors the existing Android training
  client — see `docs/FEDERATED_MOBILE.md`).
- Optional admission control (registration secret, allow-list) for `infer
  serve`, mirroring `federated-server --registration-secret`.
