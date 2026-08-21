from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class ClientUpdate:
    client_id: str
    num_examples: int
    delta: dict[str, torch.Tensor]
    metrics: dict[str, Any]


def _stack_for_key(updates: list[ClientUpdate], key: str) -> torch.Tensor:
    return torch.stack([u.delta[key].float() for u in updates], dim=0)


def _flatten_delta(delta: dict[str, torch.Tensor]) -> torch.Tensor:
    flat = [tensor.reshape(-1).float() for _, tensor in sorted(delta.items())]
    if not flat:
        return torch.tensor([], dtype=torch.float32)
    return torch.cat(flat)


def clip_delta_l2(delta: dict[str, torch.Tensor], max_norm: float) -> tuple[dict[str, torch.Tensor], float]:
    if max_norm <= 0:
        return {k: v.clone() for k, v in delta.items()}, 0.0
    vec = _flatten_delta(delta)
    norm = float(torch.linalg.vector_norm(vec).item()) if vec.numel() else 0.0
    if norm == 0.0 or norm <= max_norm:
        return {k: v.clone() for k, v in delta.items()}, norm
    scale = max_norm / norm
    return {k: v * scale for k, v in delta.items()}, norm


def _weighted_average(updates: list[ClientUpdate]) -> dict[str, torch.Tensor]:
    total_weight = max(1.0, float(sum(max(1, u.num_examples) for u in updates)))
    keys = updates[0].delta.keys()
    out: dict[str, torch.Tensor] = {}
    for key in keys:
        acc = torch.zeros_like(updates[0].delta[key], dtype=torch.float32)
        for update in updates:
            w = float(max(1, update.num_examples)) / total_weight
            acc = acc + update.delta[key].float() * w
        out[key] = acc
    return out


def _trim_count(n: int, trim_ratio: float) -> int:
    trim = int(n * max(0.0, min(0.49, trim_ratio)))
    # The coordinator defaults to three clients and a 10% trim ratio.  Plain
    # floor rounding made that configuration trim zero updates, silently
    # reducing the advertised robust aggregation to an ordinary mean exactly
    # when a small swarm is easiest to manipulate.  If trimming is enabled,
    # conservatively remove at least one value per tail whenever three or more
    # clients leave a median/mean to compute.
    if trim == 0 and trim_ratio > 0.0 and n >= 3:
        trim = 1
    return min(trim, (n - 1) // 2)


def _trimmed_mean(updates: list[ClientUpdate], trim_ratio: float) -> dict[str, torch.Tensor]:
    n = len(updates)
    trim = _trim_count(n, trim_ratio)
    keys = updates[0].delta.keys()
    out: dict[str, torch.Tensor] = {}
    for key in keys:
        stack = _stack_for_key(updates, key)
        if trim > 0 and n - (2 * trim) >= 1:
            sorted_vals, _ = torch.sort(stack, dim=0)
            trimmed = sorted_vals[trim : n - trim]
            out[key] = trimmed.mean(dim=0)
        else:
            out[key] = stack.mean(dim=0)
    return out


def _multi_krum_indices(flat_vectors: torch.Tensor, f: int, m: int) -> list[int]:
    n = flat_vectors.size(0)
    if n <= 2:
        return list(range(n))
    f = max(0, min(f, (n - 2) // 2))
    m = max(1, min(m, n - f))

    distances = torch.cdist(flat_vectors, flat_vectors, p=2).pow(2)
    scores: list[tuple[float, int]] = []
    for i in range(n):
        d = torch.cat((distances[i, :i], distances[i, i + 1 :]))
        closest = torch.topk(d, k=max(1, n - f - 2), largest=False).values
        scores.append((float(closest.sum().item()), i))
    scores.sort(key=lambda item: item[0])
    return [idx for _, idx in scores[:m]]


def aggregate_deltas(
    updates: list[ClientUpdate],
    *,
    method: str = "trimmed_mean",
    trim_ratio: float = 0.1,
    max_update_norm: float = 5.0,
    byzantine_f: int = 1,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if not updates:
        raise ValueError("No client updates were provided.")

    client_ids = [str(update.client_id) for update in updates]
    if len(set(client_ids)) != len(client_ids):
        raise ValueError("Only one update per client may be aggregated in a round.")

    expected_keys = set(updates[0].delta)
    expected_shapes = {key: tuple(value.shape) for key, value in updates[0].delta.items()}
    for update in updates:
        if set(update.delta) != expected_keys:
            raise ValueError("All client updates must contain the same tensor keys.")
        for key, tensor in update.delta.items():
            if tuple(tensor.shape) != expected_shapes[key]:
                raise ValueError(f"Tensor shape mismatch for '{key}'.")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Non-finite update values for '{key}'.")

    clipped_updates: list[ClientUpdate] = []
    norms: dict[str, float] = {}
    for update in updates:
        clipped, norm = clip_delta_l2(update.delta, max_update_norm)
        norms[update.client_id] = norm
        clipped_updates.append(
            ClientUpdate(
                client_id=update.client_id,
                num_examples=update.num_examples,
                delta=clipped,
                metrics=update.metrics,
            )
        )

    selected = clipped_updates
    selected_ids = [u.client_id for u in selected]
    method_used = method

    if method.lower() in {"krum", "multi_krum", "multi-krum"} and len(clipped_updates) >= 3:
        flat = torch.stack([_flatten_delta(u.delta) for u in clipped_updates], dim=0)
        choose = max(1, len(clipped_updates) - max(1, byzantine_f))
        indices = _multi_krum_indices(flat, f=byzantine_f, m=choose)
        selected = [clipped_updates[i] for i in indices]
        selected_ids = [u.client_id for u in selected]
        agg = _weighted_average(selected)
        method_used = "multi_krum_weighted"
    elif method.lower() == "weighted_avg":
        agg = _weighted_average(selected)
        method_used = "weighted_avg"
    else:
        agg = _trimmed_mean(selected, trim_ratio=trim_ratio)
        method_used = "trimmed_mean"

    meta = {
        "method": method_used,
        "client_count": len(clipped_updates),
        "selected_count": len(selected),
        "selected_client_ids": selected_ids,
        "update_norms": norms,
    }
    if method_used == "trimmed_mean":
        meta["trimmed_per_tail"] = _trim_count(len(selected), trim_ratio)
    return agg, meta
