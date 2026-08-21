import json
from pathlib import Path
from typing import Any

import torch

from src.delta_router import DeltaRouter, KnowledgeDelta
from src.delta_registry import DeltaRegistry
from src.federated.lora import set_lora_gates, reset_lora_gates


class ScopedKnowledgeManager:
    def __init__(self, data_dir: str = "data/deltas"):
        self.registry = DeltaRegistry(data_dir)
        self.router = DeltaRouter()
        self._model: torch.nn.Module | None = None

    def set_model(self, model: torch.nn.Module, tokenizer: Any = None):
        self._model = model
        self.router.set_model(model, tokenizer)
        self._load_all_deltas()

    def _load_all_deltas(self):
        self.router._deltas.clear()
        for entry in self.registry.list_enabled():
            delta = KnowledgeDelta.from_dict(entry)
            self.router.register(delta)

    def register_delta(self, delta_id: str, version: str = "1.0.0",
                       domain_description: str = "", description: str = "",
                       adapter_path: str = "",
                       memory_entries: list[dict] | None = None,
                       rank: int = 8, alpha: int = 16,
                       activation_threshold: float = 0.6,
                       confidence: float = 1.0, provenance: str = "") -> KnowledgeDelta:
        entry = self.registry.register(
            delta_id=delta_id, version=version,
            domain_description=domain_description,
            description=description,
            adapter_path=adapter_path,
            memory_entries=memory_entries,
            rank=rank, alpha=alpha,
            activation_threshold=activation_threshold,
            confidence=confidence, provenance=provenance,
        )
        delta = KnowledgeDelta.from_dict(entry)
        self.router.register(delta)
        return delta

    def activate_for_query(self, query: str) -> dict[str, Any]:
        if self._model is None:
            return {"active_deltas": [], "memory_context": ""}

        routing = self.router.route(query)
        active = self.router.active_deltas(query)

        gates = {}
        for delta, score in active:
            gates[delta.delta_id] = score
            set_lora_gates(self._model, score)

        memory_lines = []
        for delta, score in active:
            for entry in delta.memory_entries[:5]:
                if isinstance(entry, dict):
                    text = entry.get("claim", entry.get("name", entry.get("fact", str(entry))))
                else:
                    text = str(entry)
                if text and text not in memory_lines:
                    memory_lines.append(f"[{delta.description or delta.delta_id}] {text}")

        memory_context = ""
        if memory_lines:
            memory_context = "Relevant knowledge:\n" + "\n".join(memory_lines[:10])

        return {
            "active_deltas": [(delta.delta_id, round(score, 4)) for delta, score in active],
            "gates": {k: round(v, 4) for k, v in gates.items()},
            "memory_context": memory_context,
            "routing": {k: v for k, v in routing.items() if v["active"]},
        }

    def deactivate_all(self):
        if self._model is not None:
            reset_lora_gates(self._model)

    def list_deltas(self) -> list[dict[str, Any]]:
        return self.registry.list_all()

    def inspect_delta(self, delta_id: str) -> str:
        return self.registry.inspect(delta_id)

    def disable_delta(self, delta_id: str) -> bool:
        result = self.registry.disable(delta_id)
        if result:
            self.router.remove(delta_id)
        return result

    def enable_delta(self, delta_id: str) -> bool:
        result = self.registry.enable(delta_id)
        if result:
            entry = self.registry.get(delta_id)
            if entry:
                delta = KnowledgeDelta.from_dict(entry)
                self.router.register(delta)
        return result


_scoped_manager: ScopedKnowledgeManager | None = None


def get_scoped_manager(data_dir: str = "data/deltas") -> ScopedKnowledgeManager:
    global _scoped_manager
    if _scoped_manager is None:
        _scoped_manager = ScopedKnowledgeManager(data_dir)
    return _scoped_manager
