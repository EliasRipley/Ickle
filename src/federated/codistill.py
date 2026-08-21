"""Cross-architecture co-distillation: peers with *different* model sizes and
architectures teach each other without a central coordinator and without
requiring matching weight shapes.

Why this exists: `src/federated/coordinator.py` already runs real FedAvg /
Multi-Krum / DiLoCo aggregation, but it only works when every participant's
LoRA adapter has identical shape against the same base model
(`knowledge_modules.py`'s `_base_model_matches` literally requires the same
base-model hash). Ickle's own hardware sizing (`ilm_profile.py`) deliberately
produces nano/laptop/desktop peers with *different* `n_embd`/`n_layer` --
weight-level federation can never include them together. This module makes
that heterogeneity a non-issue by exchanging text, not weights:

  1. Peers already running `infer serve` (src/federated/inference_swarm.py)
     are asked to answer a shared, versioned set of probe prompts
     (`data/codistill/probe_prompts.json`) -- reusing the existing signed
     Ed25519 offer/request transport verbatim, so ledger crediting
     (`peer_requests_served` / `peer_requests_consumed`) happens for free on
     both sides with no new plumbing.
  2. Because several peers answer the same probe, responses can be checked
     against each other: a response's "consensus score" is its average text
     overlap with the other responses to the same probe. This is a
     text-domain analogue of Multi-Krum (aggregation.py's Byzantine-robust
     weight aggregator) -- an outlier/corrupted/lazy peer's answer scores low
     and gets dropped instead of silently averaged in.
  3. Agreement is recorded as descriptive evidence and an outlier filter,
     but no longer bootstraps durable peer trust. Homogeneous models can
     agree through correlated error or conformity. `PeerTrustStore` is
     updated by explicit owner review (or future objective outcomes), while
     the answer map preserves minority claims for inspection.
  4. Surviving (prompt, response) pairs become an ordinary distillation
     corpus in the same `DialogPair`/`write_pairs_as_corpus` format
     `continual_guard.py` already consumes -- so a bad or malicious peer's
     answers still cannot silently degrade the local model: they must pass
     through the existing replay-buffer anti-forgetting mixing and
     regression/promotion gates before anything is promoted.

No new trust authority, no new signature scheme, no new corpus format. The
human owner remains the authority for their local reputation and knowledge
adoption policy.
"""

from __future__ import annotations

import argparse
import datetime
import random
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.federated.contribution_ledger import LedgerStore
from src.federated.inference_swarm import (
    DEFAULT_DATA_DIR,
    DEFAULT_IDENTITY_PATH,
    InferenceOffer,
    _join_via_bootstrap,
    _parse_addr,
    find_offers,
    request_inference,
)
from src.epistemics import build_collective_view
from src.federated.keys import EdIdentity, ensure_ed_identity
from src.federated.peer_discovery import PeerDiscovery
from src.promotion_gate import has_repeated_word_run, token_overlap_score

DEFAULT_PROBE_SET_PATH = "data/codistill/probe_prompts.json"
DEFAULT_TRUST_STORE_PATH = "data/codistill/peer_trust.json"
DEFAULT_OUT_CORPUS_PATH = "data/codistill/distilled_corpus.txt"
GENERAL_DOMAIN = "general"

# Keyword buckets mirroring the probe-set domain prefixes (data/codistill/probe_prompts.json's
# "code-01"/"math-01"/etc ids) so a live question and a training probe land in the same trust
# bucket. Order matters: first match wins, most-specific keywords first.
_DOMAIN_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("code", ("code", "function", "python", "javascript", "bug", "class ", "variable", "compile", "programming", "script", "algorithm", "regex")),
    ("math", ("calculate", "equation", "percent", "fraction", "solve for", "sum of", "multiply", "divide", "arithmetic", " math ")),
    ("creative", ("story", "poem", "write a", "creative", "imagine", "invent a name", "brainstorm names")),
    ("reason", ("if all", "logically", "puzzle", "riddle", "explain the reasoning", "deduce", "paradox")),
    ("instr", ("step-by-step", "how do i", "how to", "instructions for", "guide me", "walk me through")),
    ("honest", ("are you able", "can you actually", "do you know for certain", "what can't you", "your limitations")),
    ("lang", ("grammar", "rewrite this sentence", "synonym", "translate", "concise", "spelling")),
    ("practical", ("budget", "savings", "remember someone", "rule of thumb", "everyday")),
    ("know", ("what is", "what causes", "difference between", "explain why", "summarize what")),
]


def classify_domain(text: str) -> str:
    """Best-effort keyword classification of a prompt into the same domain
    buckets the probe set uses (see `_DOMAIN_KEYWORDS`), so a live question
    asked via `ask_swarm()` and a training-time probe both draw on the same
    per-domain trust bucket instead of two disconnected reputations. Falls
    back to `GENERAL_DOMAIN` when nothing matches -- deliberately coarse,
    since a wrong bucket only costs some ranking precision, not correctness
    (an unranked peer still gets asked, just without a domain-specific edge)."""
    lowered = f" {re.sub(r'\s+', ' ', str(text or '')).strip().lower()} "
    for domain, keywords in _DOMAIN_KEYWORDS:
        if any(kw in lowered for kw in keywords):
            return domain
    return GENERAL_DOMAIN


def domain_of_probe(probe_id: str) -> str:
    """Probe ids are `<domain>-NN` (e.g. `code-01`, `know-03`) by convention
    in `data/codistill/probe_prompts.json` -- reuse that prefix as the trust
    domain instead of inventing a second taxonomy."""
    prefix = str(probe_id or "").split("-", 1)[0].strip().lower()
    return prefix or GENERAL_DOMAIN

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "how", "in", "is", "it", "of", "on", "or", "that", "the", "this",
    "to", "was", "what", "when", "where", "which", "who", "why", "with",
    "you", "your",
}

# EMA smoothing for peer trust: how much one round's consensus score moves a
# peer's long-run trust. Low so a single bad or lucky round can't swing
# things -- trust should reflect a track record, not one probe set.
TRUST_EMA_ALPHA = 0.25
DEFAULT_TRUST = 0.5


@dataclass
class Probe:
    probe_id: str
    prompt: str


def default_probes() -> list[Probe]:
    """Built-in probe set, used if no probe file is found -- keeps the module
    usable standalone (e.g. in tests) without requiring the data file."""
    return [
        Probe("reason-01", "Explain the difference between correlation and causation with a short example."),
        Probe("know-01", "What is the difference between weather and climate?"),
        Probe("code-01", "Write a Python function that returns the second-largest number in a list."),
        Probe("instr-01", "Give me step-by-step instructions for changing a flat bicycle tire."),
    ]


def load_probes(path: str | Path = DEFAULT_PROBE_SET_PATH) -> list[Probe]:
    p = Path(path)
    if not p.exists():
        return default_probes()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default_probes()
    if not isinstance(raw, list):
        return default_probes()
    probes: list[Probe] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        prompt = str(row.get("prompt", "")).strip()
        if not prompt:
            continue
        probe_id = str(row.get("id", "")).strip() or f"probe-{len(probes)}"
        probes.append(Probe(probe_id=probe_id, prompt=prompt))
    return probes or default_probes()


@dataclass
class TeachingResponse:
    probe_id: str
    prompt: str
    teacher_peer_id: str
    response: str
    consensus_score: float = 0.0
    trust_score: float = DEFAULT_TRUST
    accepted: bool = False
    domain: str = GENERAL_DOMAIN


def _looks_low_quality(text: str) -> bool:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(value) < 8:
        return True
    words = re.findall(r"[a-zA-Z']+", value.lower())
    if len(words) < 4:
        return True
    if has_repeated_word_run(value.lower()):
        return True
    return False


def score_consensus(responses: list[TeachingResponse]) -> None:
    """Mutates `responses` in place, setting `consensus_score` on each to its
    average text-overlap against every *other* response to the same probe.
    A single-response probe gets a neutral 1.0 -- there's nothing to compare
    against, so it shouldn't be penalized as an "outlier of one"."""
    if len(responses) <= 1:
        for r in responses:
            r.consensus_score = 1.0
        return
    texts = [r.response for r in responses]
    for i, r in enumerate(responses):
        others = [texts[j] for j in range(len(texts)) if j != i]
        pair_scores = [
            token_overlap_score(r.response, other, stopwords=_STOPWORDS, empty_a_fallback=0.0)
            for other in others
        ]
        r.consensus_score = round(sum(pair_scores) / max(1, len(pair_scores)), 4)


class PeerTrustStore:
    """Local, per-instance record of how much a peer's teaching has agreed
    with the group historically -- broken down **per domain**, not one
    scalar per peer. A peer might teach code well and creative writing
    poorly; a flat trust score can't express that, and would keep steering
    every kind of question toward whoever happened to win consensus on
    reasoning puzzles. Not shared or broadcast -- each peer forms its own
    opinion from its own observations, the same trust model as the
    ContributionLedger's seed:peer ratio.

    On-disk shape: `{peer_id: {domain: score, ...}, ...}`. A store written by
    the previous flat-scalar version (`{peer_id: score}`) is transparently
    migrated on load into `{domain: "general"}` so existing trust history
    isn't discarded by this change."""

    def __init__(self, path: str | Path = DEFAULT_TRUST_STORE_PATH):
        self.path = Path(path)
        self.trust: dict[str, dict[str, float]] = self._load()

    def _load(self) -> dict[str, dict[str, float]]:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError, ValueError):
                return {}
            if isinstance(data, dict):
                out: dict[str, dict[str, float]] = {}
                for peer_id, value in data.items():
                    if isinstance(value, dict):
                        out[str(peer_id)] = {str(d): float(v) for d, v in value.items()}
                    else:
                        out[str(peer_id)] = {GENERAL_DOMAIN: float(value)}
                return out
        return {}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.trust, indent=2, ensure_ascii=False), encoding="utf-8")

    def get(self, peer_id: str, domain: str = GENERAL_DOMAIN) -> float:
        """Score for `peer_id` in `domain`. Falls back to that peer's overall
        (cross-domain average) score when the domain has no observations yet
        -- a peer who has only ever answered code probes shouldn't look like
        a totally unknown quantity the first time someone asks it math."""
        domains = self.trust.get(peer_id)
        if not domains:
            return DEFAULT_TRUST
        if domain in domains:
            return domains[domain]
        return self.overall(peer_id)

    def overall(self, peer_id: str) -> float:
        domains = self.trust.get(peer_id)
        if not domains:
            return DEFAULT_TRUST
        return round(sum(domains.values()) / len(domains), 4)

    def domains_for(self, peer_id: str) -> dict[str, float]:
        return dict(self.trust.get(peer_id, {}))

    def update(self, peer_id: str, observed_score: float, domain: str = GENERAL_DOMAIN, *, alpha: float = TRUST_EMA_ALPHA):
        domains = self.trust.setdefault(peer_id, {})
        prev = domains.get(domain, DEFAULT_TRUST)
        domains[domain] = round((alpha * observed_score) + ((1 - alpha) * prev), 4)

    def ranked_peer_ids(self, domain: str | None = None) -> list[str]:
        if domain is None:
            return sorted(self.trust, key=lambda pid: self.overall(pid), reverse=True)
        return sorted(self.trust, key=lambda pid: self.get(pid, domain), reverse=True)


def _rank_offers_by_trust(offers: list[InferenceOffer], trust_store: PeerTrustStore, domain: str = GENERAL_DOMAIN) -> list[InferenceOffer]:
    return sorted(offers, key=lambda o: trust_store.get(o.peer_id, domain), reverse=True)


def collect_probe_responses(
    probe: Probe,
    offers: list[InferenceOffer],
    *,
    self_peer_id: str,
    ledger: LedgerStore,
    max_teachers: int,
    timeout: float,
) -> list[TeachingResponse]:
    responses: list[TeachingResponse] = []
    candidates = [o for o in offers if o.peer_id != self_peer_id][:max_teachers]
    domain = domain_of_probe(probe.probe_id)
    for offer in candidates:
        result = request_inference(
            offer,
            probe.prompt,
            max_new_tokens=200,
            temperature=0.3,  # low: we want the peer's considered answer, not a creative sample
            requester_peer_id=self_peer_id,
            ledger=ledger,
            timeout=timeout,
        )
        if not result or not result.get("verified"):
            continue
        text = str(result.get("response", "")).strip()
        if not text or _looks_low_quality(text):
            continue
        responses.append(TeachingResponse(
            probe_id=probe.probe_id,
            prompt=probe.prompt,
            teacher_peer_id=offer.peer_id,
            response=text,
            domain=domain,
        ))
    return responses


def ask_swarm(
    prompt: str,
    *,
    self_peer_id: str,
    peer_discovery: PeerDiscovery,
    ledger: LedgerStore,
    trust_store: PeerTrustStore,
    max_teachers: int = 3,
    timeout: float = 20.0,
) -> dict[str, Any]:
    """Live counterpart to `run_codistillation_round()`: instead of a shared
    probe set feeding a future training corpus, this asks trust-ranked peers
    the *user's actual question right now* and returns their answers plus a
    pick, for a "what does the rest of the swarm think?" action in chat.
    Peers are queried concurrently (unlike the training round, which is fine
    running serially in a background task) because a person is waiting on
    this. Still routes through the same signed-offer transport and updates
    the same trust store, so a live ask and a training round make each other
    smarter over time instead of being two disconnected features."""
    import concurrent.futures

    domain = classify_domain(prompt)
    all_offers = find_offers(peer_discovery)
    candidates = [o for o in _rank_offers_by_trust(all_offers, trust_store, domain) if o.peer_id != self_peer_id][:max_teachers]

    responses: list[TeachingResponse] = []
    if candidates:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(candidates)) as pool:
            futures = {
                pool.submit(
                    request_inference,
                    offer,
                    prompt,
                    max_new_tokens=300,
                    temperature=0.7,
                    requester_peer_id=self_peer_id,
                    ledger=ledger,
                    timeout=timeout,
                ): offer
                for offer in candidates
            }
            for future in concurrent.futures.as_completed(futures):
                offer = futures[future]
                try:
                    result = future.result()
                except Exception:
                    continue
                if not result or not result.get("verified"):
                    continue
                text = str(result.get("response", "")).strip()
                if not text or _looks_low_quality(text):
                    continue
                responses.append(TeachingResponse(
                    probe_id="live",
                    prompt=prompt,
                    teacher_peer_id=offer.peer_id,
                    response=text,
                    domain=domain,
                ))

    score_consensus(responses)
    for r in responses:
        r.trust_score = trust_store.get(r.teacher_peer_id, domain)

    # Agreement is useful descriptive information, but it is not a durable
    # reputation signal: homogeneous models can copy the same error or
    # converge through conformity.  Trust now changes only from an explicit
    # owner review (/api/swarm/feedback) or a future objective evaluation.

    best = None
    if responses:
        best = max(responses, key=lambda r: (r.consensus_score * 0.7) + (r.trust_score * 0.3))

    response_rows = [
        {
            "peer_id": r.teacher_peer_id,
            "response": r.response,
            "consensus_score": r.consensus_score,
            "trust_score": r.trust_score,
        }
        for r in responses
    ]
    return {
        "domain": domain,
        "peers_asked": len(candidates),
        "peers_answered": len(responses),
        "responses": response_rows,
        "best": ({"peer_id": best.teacher_peer_id, "response": best.response} if best else None),
        "representative": ({"peer_id": best.teacher_peer_id, "response": best.response} if best else None),
        "deliberation": build_collective_view(response_rows),
        "trust_policy": "human-or-objective-review; agreement alone does not change trust",
    }


def select_probes_for_round(probes: list[Probe], sample_size: int = 0, seed: int | None = None) -> list[Probe]:
    """Pick the probes a round will actually use. With `sample_size <= 0` (the
    default) every probe runs, matching the old behavior. Otherwise picks a
    deterministic random subset -- deterministic because independent peers
    running a round on the same day, with no coordinator to agree a subset
    with, still need to land on the *same* probes for consensus scoring to
    mean anything; `seed` defaults to today's UTC date so that happens for
    free. This is the probe-set-rotation gap flagged in
    `docs/CODISTILLATION.md`'s future-work section: a fixed, always-fully-used
    probe set is something a peer could eventually learn to game by
    memorizing/gaming the 24 known prompts instead of actually teaching well;
    rotating which subset is live each day removes that incentive."""
    if sample_size <= 0 or sample_size >= len(probes):
        return list(probes)
    if seed is None:
        seed = datetime.date.today().toordinal()
    rng = random.Random(seed)
    ordered = sorted(probes, key=lambda p: p.probe_id)  # stable input order before sampling
    return rng.sample(ordered, sample_size)


def run_codistillation_round(
    *,
    identity: EdIdentity,
    peer_discovery: PeerDiscovery,
    ledger: LedgerStore,
    trust_store: PeerTrustStore,
    probes: list[Probe],
    max_teachers_per_probe: int = 5,
    min_teachers_per_probe: int = 2,
    min_consensus: float = 0.15,
    request_timeout: float = 60.0,
) -> dict[str, Any]:
    """Runs one full teach-and-learn round: for each probe, ask several
    peers ranked by trust *in that probe's domain*, score their agreement,
    describe agreement, and keep candidate responses that clear the outlier bar.
    Returns a report plus the accepted DialogPair-shaped rows; writing them
    to a corpus file is the caller's job (see `main()`/`round` command) so
    this stays testable without touching the filesystem."""
    all_offers = find_offers(peer_discovery)

    accepted_rows: list[dict[str, str]] = []
    probe_reports: list[dict[str, Any]] = []

    for probe in probes:
        domain = domain_of_probe(probe.probe_id)
        ranked_offers = _rank_offers_by_trust(all_offers, trust_store, domain)
        responses = collect_probe_responses(
            probe,
            ranked_offers,
            self_peer_id=identity.peer_id,
            ledger=ledger,
            max_teachers=max_teachers_per_probe,
            timeout=request_timeout,
        )
        if len(responses) < min_teachers_per_probe:
            probe_reports.append({
                "probe_id": probe.probe_id,
                "domain": domain,
                "teachers_responded": len(responses),
                "accepted": False,
                "reason": "not enough teachers responded",
            })
            continue

        score_consensus(responses)
        for r in responses:
            r.trust_score = trust_store.get(r.teacher_peer_id, domain)
            r.accepted = r.consensus_score >= min_consensus

        surviving = [r for r in responses if r.accepted]
        if surviving:
            best = max(surviving, key=lambda r: (r.consensus_score * 0.7) + (r.trust_score * 0.3))
            accepted_rows.append({"user": best.prompt, "assistant": best.response, "source": f"codistill:{best.teacher_peer_id}"})

        probe_reports.append({
            "probe_id": probe.probe_id,
            "domain": domain,
            "teachers_responded": len(responses),
            "teachers_accepted": len(surviving),
            "accepted": bool(surviving),
            "scores": [{"peer_id": r.teacher_peer_id, "consensus": r.consensus_score, "trust": r.trust_score} for r in responses],
            "deliberation": build_collective_view([
                {"peer_id": r.teacher_peer_id, "response": r.response} for r in responses
            ]),
        })

    # Deliberation agreement gates candidate data for the existing promotion
    # pipeline, but never bootstraps long-term reputation from conformity.
    return {
        "peers_discovered": len(all_offers),
        "probes_total": len(probes),
        "probes_taught": len(accepted_rows),
        "rows": accepted_rows,
        "probe_reports": probe_reports,
    }


def write_distillation_corpus(path: str | Path, rows: list[dict[str, str]]):
    from src.continual_guard import DialogPair, write_pairs_as_corpus

    pairs = [DialogPair(user=r["user"], assistant=r["assistant"], source=r.get("source", "")) for r in rows]
    write_pairs_as_corpus(str(path), pairs)


def main():
    parser = argparse.ArgumentParser(description="Ickle co-distillation — cross-architecture peer teaching")
    sub = parser.add_subparsers(dest="command", required=True)

    probes_p = sub.add_parser("probes", help="List the current probe set")
    probes_p.add_argument("--probes-file", default=DEFAULT_PROBE_SET_PATH)

    round_p = sub.add_parser("round", help="Run one co-distillation round against the swarm")
    round_p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    round_p.add_argument("--identity", default=DEFAULT_IDENTITY_PATH)
    round_p.add_argument("--probes-file", default=DEFAULT_PROBE_SET_PATH)
    round_p.add_argument("--out-corpus", default=DEFAULT_OUT_CORPUS_PATH)
    round_p.add_argument("--trust-store", default=DEFAULT_TRUST_STORE_PATH)
    round_p.add_argument("--bootstrap", action="append", default=[], help="Bootstrap peer host:port (repeatable)")
    round_p.add_argument("--max-teachers-per-probe", type=int, default=5)
    round_p.add_argument("--min-teachers-per-probe", type=int, default=2)
    round_p.add_argument("--min-consensus", type=float, default=0.15)
    round_p.add_argument("--timeout", type=float, default=60.0)
    round_p.add_argument(
        "--sample-size", type=int, default=0,
        help="Use only a deterministic (date-seeded) random subset of this many probes "
             "instead of the full set, so a fixed probe list can't be gamed over time. "
             "0 (default) uses every probe, matching the old behavior.",
    )
    round_p.add_argument("--json", action="store_true")

    trust_p = sub.add_parser("trust", help="Show locally-observed peer teaching trust")
    trust_p.add_argument("--trust-store", default=DEFAULT_TRUST_STORE_PATH)

    args = parser.parse_args()

    if args.command == "probes":
        probes = load_probes(args.probes_file)
        for p in probes:
            print(f"{p.probe_id:<16s} {p.prompt}")
        return

    if args.command == "trust":
        store = PeerTrustStore(args.trust_store)
        for peer_id in store.ranked_peer_ids():
            breakdown = ", ".join(f"{d}={v:.2f}" for d, v in sorted(store.domains_for(peer_id).items()))
            print(f"{peer_id[:16]}...  overall={store.overall(peer_id):.3f}  [{breakdown}]")
        if not store.trust:
            print("No peer trust observations yet.")
        return

    identity = ensure_ed_identity(Path(args.identity))
    ledger = LedgerStore(Path(args.data_dir) / "contribution_ledger.json")
    trust_store = PeerTrustStore(args.trust_store)
    probes = select_probes_for_round(load_probes(args.probes_file), args.sample_size)

    peer_discovery = PeerDiscovery()
    for bootstrap_addr in args.bootstrap:
        host, port = _parse_addr(bootstrap_addr)
        peer_discovery.add_bootstrap(host, port)
    _join_via_bootstrap(peer_discovery, args.bootstrap)

    report = run_codistillation_round(
        identity=identity,
        peer_discovery=peer_discovery,
        ledger=ledger,
        trust_store=trust_store,
        probes=probes,
        max_teachers_per_probe=args.max_teachers_per_probe,
        min_teachers_per_probe=args.min_teachers_per_probe,
        min_consensus=args.min_consensus,
        request_timeout=args.timeout,
    )
    write_distillation_corpus(args.out_corpus, report["rows"])
    report["out_corpus"] = args.out_corpus

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"peers_discovered={report['peers_discovered']} probes_taught={report['probes_taught']}/{report['probes_total']}")
        print(f"corpus written: {args.out_corpus}")
        print("Feed it into the anti-forgetting pipeline with:")
        print(f"  python -m src.app continual-guard run-step --new-corpus {args.out_corpus} ...")


if __name__ == "__main__":
    main()
