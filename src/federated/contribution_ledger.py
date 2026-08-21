"""Local seed:peer contribution ledger.

Mirrors the seed/peer idea from swarm file-sharing (Tor's seed = how much a peer
gives to the network, peer = how much they take). Here:

- "seed" credit comes from giving compute/data to the network: serving torickle
  training-delta pieces, completing federated training rounds, and answering
  other peers' inference requests.
- "peer" debt comes from consuming the network: asking other peers to run
  inference for you.

The ratio is informational only (nothing is throttled on it yet) but it is the
number a future admission-control or priority-queue policy would read, and it
gives contributors visibility into whether they're pulling their weight —
exactly the incentive Tor/BitTorrent seed ratios provide.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_LEDGER_PATH = "data/torickle/contribution_ledger.json"

# Relative weights so a small number of "expensive" contributions (a full
# training round) aren't dwarfed by many small ones (serving one inference
# reply). These are deliberately simple and tunable, not physically derived.
WEIGHT_PIECE_SERVED = 1
WEIGHT_TRAINING_ROUND = 200
WEIGHT_INFERENCE_SERVED = 5
WEIGHT_INFERENCE_CONSUMED = 5


@dataclass
class ContributionLedger:
    seed_pieces_served: int = 0
    seed_bytes_served: int = 0
    seed_training_rounds: int = 0
    peer_requests_served: int = 0
    peer_tokens_served: int = 0
    peer_requests_consumed: int = 0
    peer_tokens_consumed: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def record_piece_served(self, nbytes: int = 0):
        self.seed_pieces_served += 1
        self.seed_bytes_served += max(0, int(nbytes))
        self._touch()

    def record_training_round(self):
        self.seed_training_rounds += 1
        self._touch()

    def record_inference_served(self, tokens: int = 0):
        self.peer_requests_served += 1
        self.peer_tokens_served += max(0, int(tokens))
        self._touch()

    def record_inference_consumed(self, tokens: int = 0):
        self.peer_requests_consumed += 1
        self.peer_tokens_consumed += max(0, int(tokens))
        self._touch()

    def _touch(self):
        self.updated_at = time.time()

    @property
    def contributed_score(self) -> int:
        return (
            self.seed_pieces_served * WEIGHT_PIECE_SERVED
            + self.seed_training_rounds * WEIGHT_TRAINING_ROUND
            + self.peer_requests_served * WEIGHT_INFERENCE_SERVED
        )

    @property
    def consumed_score(self) -> int:
        return self.peer_requests_consumed * WEIGHT_INFERENCE_CONSUMED

    def ratio(self) -> float:
        """Seed:peer ratio, torrent-style. >=1.0 means the peer gives back at
        least as much as it takes. A peer that has never consumed anything gets
        an infinite ratio, reported as -1 (unbounded) rather than raising."""
        consumed = self.consumed_score
        if consumed <= 0:
            return -1.0
        return round(self.contributed_score / consumed, 3)

    def summary(self) -> dict[str, Any]:
        ratio = self.ratio()
        return {
            "seed_pieces_served": self.seed_pieces_served,
            "seed_bytes_served": self.seed_bytes_served,
            "seed_training_rounds": self.seed_training_rounds,
            "peer_requests_served": self.peer_requests_served,
            "peer_tokens_served": self.peer_tokens_served,
            "peer_requests_consumed": self.peer_requests_consumed,
            "peer_tokens_consumed": self.peer_tokens_consumed,
            "contributed_score": self.contributed_score,
            "consumed_score": self.consumed_score,
            "ratio": ratio,
            "ratio_display": "unbounded" if ratio < 0 else f"{ratio:.2f}",
            "updated_at": self.updated_at,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed_pieces_served": self.seed_pieces_served,
            "seed_bytes_served": self.seed_bytes_served,
            "seed_training_rounds": self.seed_training_rounds,
            "peer_requests_served": self.peer_requests_served,
            "peer_tokens_served": self.peer_tokens_served,
            "peer_requests_consumed": self.peer_requests_consumed,
            "peer_tokens_consumed": self.peer_tokens_consumed,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "ContributionLedger":
        return ContributionLedger(
            seed_pieces_served=int(d.get("seed_pieces_served", 0)),
            seed_bytes_served=int(d.get("seed_bytes_served", 0)),
            seed_training_rounds=int(d.get("seed_training_rounds", 0)),
            peer_requests_served=int(d.get("peer_requests_served", 0)),
            peer_tokens_served=int(d.get("peer_tokens_served", 0)),
            peer_requests_consumed=int(d.get("peer_requests_consumed", 0)),
            peer_tokens_consumed=int(d.get("peer_tokens_consumed", 0)),
            created_at=float(d.get("created_at", time.time())),
            updated_at=float(d.get("updated_at", time.time())),
        )


class LedgerStore:
    """Loads/saves a ContributionLedger to a JSON file, writing after every
    mutation so a crashed process doesn't lose contribution history."""

    def __init__(self, path: str | Path = DEFAULT_LEDGER_PATH):
        self.path = Path(path)
        self.ledger = self._load()

    def _load(self) -> ContributionLedger:
        if self.path.exists():
            try:
                return ContributionLedger.from_dict(json.loads(self.path.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, KeyError, ValueError):
                pass
        return ContributionLedger()

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.ledger.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    def record_piece_served(self, nbytes: int = 0):
        self.ledger.record_piece_served(nbytes)
        self.save()

    def record_training_round(self):
        self.ledger.record_training_round()
        self.save()

    def record_inference_served(self, tokens: int = 0):
        self.ledger.record_inference_served(tokens)
        self.save()

    def record_inference_consumed(self, tokens: int = 0):
        self.ledger.record_inference_consumed(tokens)
        self.save()

    def summary(self) -> dict[str, Any]:
        return self.ledger.summary()
