# Epistemic Commons: intelligence beyond one generated string

Ickle's first P2P mechanisms share compute, weights, adapters, and complete
answers. The Epistemic Commons adds a different primitive: **inspectable
claims and signed human review**. It treats a generated answer as a proposal
that people and peers can examine, correct, and carry forward without forcing
everyone into one global version of truth.

This is an experimental capability, not a fact checker or truth oracle.

## Why this is a fundamental change

A conventional language model compresses several unlike things into one fluent
string:

- claims recalled from model weights;
- statements related to retrieved sources;
- advice and judgement;
- uncertainty;
- and, in a multi-agent system, agreement with other models.

Fluency hides those boundaries from the person reading the answer. Asking a
model to append a confidence number does not solve the problem: research finds
that verbalized model confidence can be systematically overconfident and is an
unreliable proxy for uncertainty ([Zhou et al., ACL
2024](https://aclanthology.org/2024.acl-long.198/), [Han et al., Findings of ACL
2024](https://aclanthology.org/2024.findings-acl.398/)). Semantic uncertainty
estimated across multiple generated meanings is more informative, but costs
multiple generations and still only detects some confabulations ([Farquhar et
al., Nature 2024](https://www.nature.com/articles/s41586-024-07421-0)).

Ickle therefore does not label model claims with a self-reported probability.
It exposes a separate answer map with observable grounds:

- a retrieved passage is textually related;
- a local human supported, disputed, or corrected the claim;
- signed peers contributed compatible or conflicting reviews;
- or no linked grounds exist yet.

Long-form verification research supports decomposing responses into smaller
claims, while also warning that decomposition itself introduces errors and is
sensitive to atomicity ([Wanner et al., *SEM
2024](https://aclanthology.org/2024.starsem-1.13/), [Lu et al., ACL
2025](https://aclanthology.org/2025.acl-long.254/)). The UI consequently calls
Ickle's deterministic output **candidate claims**, never verified facts.

## Human flow

Every non-empty web-chat answer can show **Inspect candidate claims**. Each
candidate carries one of these labels:

- **Related sources**: retrieved text overlaps the claim. This is not an
  entailment check and does not prove it.
- **Human reviewed**: the owner marked the claim useful/accurate.
- **Human correction**: the owner supplied replacement knowledge.
- **Contested**: the owner disputes it.
- **Peer perspective**: signed peer review exists, but the owner has not
  adopted it and it remains inert.
- **Advice / judgement**: the statement is normative or recommendatory, where
  factual source coverage is not the only relevant question.
- **Open claim**: no source or review is linked.

A correction is stored locally and becomes a future prompt context when the
topic is relevant. This gives the person an immediate way to improve behavior
without waiting for model training. Chat-session persistence includes the
answer map, so closing and reopening a conversation does not erase its
epistemic state.

## P2P protocol

Reviews are events in `data/commons/epistemic.sqlite3`. Every event contains:

- a stable claim id and original text;
- a relation (`support`, `dispute`, `correct`, `adopt`, or `retract`);
- optional correction text and source URL;
- an Ed25519-derived peer id, public key, timestamp, and signature;
- and an explicit `shared` bit.

The event id is the SHA-256 digest of the canonical unsigned body. The
signature covers that id and body. A peer rejects altered, forged, unsigned,
or non-shared remote events.

Replicas merge events by set union on event id. This is deliberately a
grow-only, operation-based CRDT:

- duplicate delivery is harmless;
- sync order does not select a winner;
- a supporting and disputing event both survive;
- a retraction is another signed event and only hides a target written by the
  same author.

The existing Torickle peer listener exposes:

```text
GET  /torickle/v1/commons/events
POST /torickle/v1/commons/events
```

The web app's **Control room -> Network -> Epistemic Commons** panel performs an
explicit bidirectional sync with configured peers. No central registry or
Ickle-operated service is involved.

## Privacy and authority boundaries

The following constraints are structural, not just UI promises:

1. A review is local unless the person ticks **Share this signed review** for
   that individual event.
2. The peer HTTP endpoint exports only events whose signed body says
   `shared=true`.
3. Imported peer events remain inert. They are visible in the Commons panel
   but cannot steer generation until the owner clicks **Use locally**, which
   creates a new locally signed adoption event.
4. Shared data is not remotely deletable. The UI warns that another peer may
   retain a received event even if its author later broadcasts a retraction.
5. Reviews contain claim/correction text and may contain a URL. People should
   not share a review containing private information.
6. Peer batches and response bytes are bounded, cryptographic field lengths
   are validated before verification, and a replica stores at most 10,000
   remote events. Local reviews remain available after that cap.
7. The swarm listener grants no browser CORS origin. Peer sync uses native
   HTTP clients, while arbitrary web pages cannot read or JSON-POST commons
   data through a person's browser.

This is plural ownership: peers may disagree, each person controls what their
own Ickle uses, and no network majority silently overwrites local judgement.

## Swarm deliberation and trust

The live **Ask the swarm too** path now decomposes peer responses into a
descriptive collective view:

- common ground repeated by two or more peers;
- distinct contributions, including minority views;
- and simple possible polarity conflicts.

It still exposes answer-overlap scores, but does not call the highest-overlap
answer true. Work on homogeneous multi-agent debate reports conformity,
contextual destabilization, and consensus collapse, supporting this separation
between agreement and correctness ([Bertalanič and Fortuna,
2026](https://arxiv.org/abs/2605.00914)).

Most importantly, agreement no longer raises durable `PeerTrustStore` scores.
The owner can mark each peer answer **Helpful** or **Not helpful**, and that
human signal updates local per-domain trust. Co-distillation may still use
agreement to reject an obvious textual outlier before placing a candidate in
the existing continual-learning promotion pipeline, but conformity does not
bootstrap reputation.

## From correction to model

Everything above changes what one future answer says, through
`context_for_prompt()` — a correction lives as long as the topic keeps
coming up, but never actually changes the model, so it carries less durable
weight than an ordinary line of scraped conversation log fed into training.
That is backwards for the highest-trust signal Ickle has: an explicit,
signed, local human judgement.

`src/verified_corrections.py` closes that gap. It reads the owner's own
active `correct`/`adopt` events (never an un-adopted peer review — the same
`local_only` boundary `context_for_prompt()` already enforces) and turns each
into an oversampled `User:`/`Ickle:` training pair. Every
`continual_guard_step` — whether triggered from the Training tab, the
Control room's **Consolidate corrections now** button under Epistemic
Commons, or the existing auto-pipeline — folds this corpus in alongside the
usual conversation/smart/focus corpora (`auto_include_verified_corrections`,
on by default), so a correction can become part of the model itself, not
just a patch layered on top of it.

This still goes through the same anti-forgetting promotion gate as any other
training step (`src/continual_guard.py`): a bad correction can fail to
promote just like a bad conversation pair can, and the owner's adoption
decision remains the only thing that puts a claim in scope for this at all —
peer conformity still never does. `GET /api/consolidation/status` reports how
many corrections are currently eligible.

## Code map

- `src/epistemics.py`: deterministic candidate-claim extraction, evidence
  linking, answer maps, and swarm collective views.
- `src/federated/knowledge_commons.py`: signed event schema, SQLite event set,
  convergence, local prompt context, adoption, and peer sync.
- `src/verified_corrections.py`: turns adopted corrections into an
  oversampled training corpus for `continual_guard_step` (see "From
  correction to model" above).
- `src/federated/swarm.py`: peer HTTP transport.
- `src/serve_web.py`: answer-map generation and local-review endpoint.
- `src/serve_control.py`: commons status/sync/adoption, consolidation status,
  and human-governed peer trust endpoints.
- `web/app.js`, `web/index.html`, `web/styles.css`: complete human interface.

## Known limits and honest next experiments

- Candidate-claim extraction is deterministic and cheap, but not a learned
  atomic decomposition policy. Complex sentences may remain too broad or be
  skipped.
- Source linking is lexical/semantic-token overlap, not natural-language
  inference. “Related sources” must never be renamed “verified”.
- Ed25519 proves event integrity and key continuity, not that one human owns
  only one key. The commons deliberately avoids one-peer-one-vote claims.
- Human review can itself be wrong. Conflicts are preserved for this reason,
  and consolidation relies on the same anti-forgetting promotion gate as any
  other training step rather than trusting corrections unconditionally.
- Shared-event sync is explicit and bounded to configured peers; background
  gossip and pagination beyond the newest bounded batch remain future work.
- Consolidation oversamples by a fixed multiplier rather than weighting by
  how many independent peers converged on a correction, or by recency; a
  smarter weighting scheme is future work.
- A stronger optional uncertainty mode could sample multiple independent
  generations and cluster meanings, but must expose its compute cost and avoid
  delaying every answer by default.
