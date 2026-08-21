import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Any


@dataclass
class KnowledgeDelta:
    delta_id: str
    version: str = "1.0.0"
    domain_description: str = ""
    description: str = ""
    adapter_path: str = ""
    memory_entries: list[dict] = field(default_factory=list)
    rank: int = 8
    alpha: int = 16
    activation_threshold: float = 0.6
    confidence: float = 1.0
    provenance: str = ""
    evaluation_results: dict[str, Any] = field(default_factory=dict)
    rollback_pointer: str = ""
    enabled: bool = True

    _domain_embedding: Any = field(default=None, repr=False, init=False)

    def to_dict(self) -> dict:
        return {
            "delta_id": self.delta_id, "version": self.version,
            "domain_description": self.domain_description,
            "description": self.description,
            "adapter_path": self.adapter_path,
            "memory_entries": self.memory_entries,
            "rank": self.rank, "alpha": self.alpha,
            "activation_threshold": self.activation_threshold,
            "confidence": self.confidence,
            "provenance": self.provenance,
            "evaluation_results": self.evaluation_results,
            "rollback_pointer": self.rollback_pointer,
            "enabled": self.enabled,
        }

    @staticmethod
    def from_dict(d: dict) -> "KnowledgeDelta":
        return KnowledgeDelta(
            delta_id=d.get("delta_id", ""),
            version=d.get("version", "1.0.0"),
            domain_description=d.get("domain_description", d.get("description", "")),
            description=d.get("description", ""),
            adapter_path=d.get("adapter_path", ""),
            memory_entries=d.get("memory_entries", []),
            rank=d.get("rank", 8), alpha=d.get("alpha", 16),
            activation_threshold=d.get("activation_threshold", 0.6),
            confidence=d.get("confidence", 1.0),
            provenance=d.get("provenance", ""),
            evaluation_results=d.get("evaluation_results", {}),
            rollback_pointer=d.get("rollback_pointer", ""),
            enabled=d.get("enabled", True),
        )


class DeltaRouter:
    def __init__(self, base_model: torch.nn.Module | None = None, tokenizer: Any = None):
        self._model = base_model
        self._tokenizer = tokenizer
        self._deltas: dict[str, KnowledgeDelta] = {}

    @property
    def deltas(self) -> dict[str, KnowledgeDelta]:
        return dict(self._deltas)

    def register(self, delta: KnowledgeDelta):
        self._deltas[delta.delta_id] = delta
        self._precompute_embedding(delta)

    def remove(self, delta_id: str):
        self._deltas.pop(delta_id, None)

    def set_model(self, model: torch.nn.Module, tokenizer: Any = None):
        self._model = model
        if tokenizer is not None:
            self._tokenizer = tokenizer
        for delta in self._deltas.values():
            self._precompute_embedding(delta)

    def _encode_text(self, text: str) -> torch.Tensor:
        if self._model is None or self._tokenizer is None or not text:
            return torch.zeros(1)
        # Domain-routing quality depends on these being the model's real
        # token embeddings, not a stand-in. A prior version derived "tokens"
        # from ord(c) % 256 -- meaningless for any real tokenizer's vocab
        # (SentencePiece IDs don't correlate with character codepoints at
        # all), and could even emit an id >= vocab_size for small models,
        # silently making every delta's cosine similarity noise.
        tokens = self._tokenizer.encode(text.lower())
        if not tokens:
            return torch.zeros(1)
        with torch.no_grad():
            idx = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
            device = next(self._model.parameters()).device
            idx = idx.to(device)
            emb = self._model.token_embedding_table(idx)
            return emb.mean(dim=1).squeeze(0)

    def _precompute_embedding(self, delta: KnowledgeDelta):
        if not delta.enabled or self._model is None:
            return
        emb = self._encode_text(delta.domain_description)
        if emb.norm() > 1e-8:
            delta._domain_embedding = emb

    def compute_activation(self, query: str, delta_id: str) -> float:
        delta = self._deltas.get(delta_id)
        if delta is None or not delta.enabled or self._model is None:
            return 0.0

        q_emb = self._encode_text(query)
        dom_emb = delta._domain_embedding
        if q_emb.norm() < 1e-8 or dom_emb is None or dom_emb.norm() < 1e-8:
            return 0.0

        sim = float(F.cosine_similarity(q_emb.unsqueeze(0), dom_emb.unsqueeze(0), dim=-1).item())
        return max(0.0, min(1.0, sim))

    def route(self, query: str) -> dict[str, dict[str, Any]]:
        results = {}
        for delta_id, delta in self._deltas.items():
            if not delta.enabled:
                continue
            score = self.compute_activation(query, delta_id)
            active = score >= delta.activation_threshold
            results[delta_id] = {
                "score": round(score, 4),
                "active": active,
                "threshold": delta.activation_threshold,
                "delta": delta,
            }
        return results

    def active_deltas(self, query: str, memory_only: bool = False) -> list[tuple[KnowledgeDelta, float]]:
        routing = self.route(query)
        active = [(info["delta"], info["score"]) for info in routing.values() if info["active"]]
        active.sort(key=lambda x: x[1], reverse=True)
        return active
