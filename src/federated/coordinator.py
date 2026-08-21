from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from src.federated.aggregation import ClientUpdate, aggregate_deltas
from src.federated.diloco import (
    GlobalOptimizer,
    GlobalOptimizerConfig,
    load_global_optimizer,
    save_global_optimizer,
)
from src.federated.local_train import evaluate_adapter_loss, load_base_with_lora
from src.federated.lora import LoRAConfig, get_lora_state_dict
from src.federated.protocol import (
    MAX_CLOCK_SKEW_SECONDS,
    decode_tensor_dict,
    encode_tensor_dict,
    file_sha256,
    generate_client_secret,
    generate_nonce,
    verify_request_signature,
    verify_signature,
    verify_timestamp,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _deep_default(base: dict, default: dict) -> dict:
    out = dict(default)
    for key, value in base.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_default(value, out[key])
        else:
            out[key] = value
    return out


@dataclass
class AggregationOptions:
    method: str = "trimmed_mean"
    trim_ratio: float = 0.1
    max_update_norm: float = 5.0
    byzantine_f: int = 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "trim_ratio": self.trim_ratio,
            "max_update_norm": self.max_update_norm,
            "byzantine_f": self.byzantine_f,
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "AggregationOptions":
        return AggregationOptions(
            method=str(payload.get("method", "trimmed_mean")),
            trim_ratio=float(payload.get("trim_ratio", 0.1)),
            max_update_norm=float(payload.get("max_update_norm", 5.0)),
            byzantine_f=int(payload.get("byzantine_f", 1)),
        )


class FederatedCoordinator:
    def __init__(
        self,
        *,
        base_model_path: str,
        state_dir: str = "data/federated",
        lora_cfg: LoRAConfig | None = None,
        min_clients: int = 3,
        aggregation: AggregationOptions | None = None,
        eval_data_path: str = "",
        eval_iters: int = 8,
        max_regression: float = 0.02,
        diloco_local_steps: int = 100,
        diloco_lr: float = 1.0,
        diloco_beta: float = 0.9,
        diloco_nesterov: bool = True,
        max_examples_per_update: int = 100_000,
    ):
        self.base_model_path = str(Path(base_model_path))
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.state_dir / "coordinator_state.json"
        self.global_adapter_path = self.state_dir / "global_adapter.pt"
        self.global_optimizer_path = self.state_dir / "global_optimizer.pt"
        self.lora_cfg = lora_cfg or LoRAConfig()
        self.min_clients = max(1, int(min_clients))
        self.aggregation = aggregation or AggregationOptions()
        self.eval_data_path = str(eval_data_path).strip()
        self.eval_iters = int(eval_iters)
        self.max_regression = float(max_regression)
        self.diloco_local_steps = int(diloco_local_steps)
        self.max_examples_per_update = max(1, int(max_examples_per_update))
        self.diloco_cfg = GlobalOptimizerConfig(lr=diloco_lr, beta=diloco_beta, nesterov=diloco_nesterov)
        self._lock = threading.RLock()
        self._state = self._load_or_init_state()
        self.global_optimizer = load_global_optimizer(
            str(self.global_optimizer_path), self.diloco_cfg,
        )

    def _default_state(self) -> dict[str, Any]:
        base_hash = file_sha256(self.base_model_path) if Path(self.base_model_path).exists() else ""
        return {
            "base_model_path": self.base_model_path,
            "base_model_sha256": base_hash,
            "lora_config": self.lora_cfg.as_dict(),
            "aggregation": self.aggregation.as_dict(),
            "min_clients": self.min_clients,
            "max_examples_per_update": self.max_examples_per_update,
            "diloco_local_steps": self.diloco_local_steps,
            "diloco_lr": self.diloco_cfg.lr,
            "diloco_beta": self.diloco_cfg.beta,
            "diloco_nesterov": self.diloco_cfg.nesterov,
            "round_id": 1,
            "active_round": {
                "round_id": 1,
                "opened_at_utc": _utc_now(),
                "update_files": [],
                "submitted_client_ids": [],
                "status": "open",
            },
            "completed_rounds": [],
            "clients": {},
            "used_nonces": {},
        }

    def _load_or_init_state(self) -> dict[str, Any]:
        if self.state_path.exists():
            loaded = json.loads(self.state_path.read_text(encoding="utf-8"))
            state = _deep_default(loaded, self._default_state())
        else:
            state = self._default_state()
        self._save_state(state)
        if not self.global_adapter_path.exists():
            self._save_global_adapter(self._fresh_global_adapter())
        else:
            # Older coordinators zeroed both LoRA factors.  With A=0 and B=0,
            # d(B@A)/dA and d(B@A)/dB are both zero, so clients could perform
            # arbitrarily many nominal training steps without changing a
            # single adapter parameter.  Repair that provably inert legacy
            # state by restoring only LoRA-A's normal random initialization;
            # LoRA-B stays zero, so the adapter still starts as an exact no-op.
            existing = self._load_global_adapter()
            if self._is_inert_adapter(existing):
                seeded = self._fresh_global_adapter()
                repaired = {key: value.clone() for key, value in existing.items()}
                for key in repaired:
                    if key.endswith(".lora_a"):
                        repaired[key] = seeded[key].clone()
                self._save_global_adapter(repaired)
        return state

    def _fresh_global_adapter(self) -> dict[str, torch.Tensor]:
        model, _ = load_base_with_lora(self.base_model_path, self.lora_cfg)
        return get_lora_state_dict(model)

    @staticmethod
    def _is_inert_adapter(state: dict[str, torch.Tensor]) -> bool:
        a_tensors = [value for key, value in state.items() if key.endswith(".lora_a")]
        b_tensors = [value for key, value in state.items() if key.endswith(".lora_b")]
        if not a_tensors or not b_tensors:
            return False
        return all(torch.count_nonzero(value).item() == 0 for value in a_tensors + b_tensors)

    def _save_state(self, payload: dict[str, Any]):
        tmp_path = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp_path, self.state_path)

    def _save(self):
        self._save_state(self._state)

    def _load_global_adapter(self) -> dict[str, torch.Tensor]:
        return torch.load(self.global_adapter_path, map_location="cpu")

    def _save_global_adapter(self, state: dict[str, torch.Tensor]):
        tmp_path = self.global_adapter_path.with_suffix(self.global_adapter_path.suffix + ".tmp")
        torch.save(state, tmp_path)
        os.replace(tmp_path, self.global_adapter_path)

    def _round_dir(self, round_id: int) -> Path:
        path = self.state_dir / f"round_{round_id:04d}"
        path.mkdir(parents=True, exist_ok=True)
        (path / "updates").mkdir(parents=True, exist_ok=True)
        return path

    def register_client(
        self,
        *,
        platform: str,
        device_name: str = "",
        capabilities: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            platform = str(platform or "unknown").strip()[:32] or "unknown"
            device_name = str(device_name or "").strip()[:128]
            capabilities = capabilities if isinstance(capabilities, dict) else {}
            if len(json.dumps(capabilities, ensure_ascii=False)) > 16_384:
                raise ValueError("Capabilities metadata is too large (maximum 16 KiB)")
            client_id = f"c_{generate_nonce()}"
            secret = generate_client_secret()
            while client_id in self._state["clients"]:
                client_id = f"c_{generate_nonce()}"
            self._state["clients"][client_id] = {
                "secret": secret,
                "platform": platform,
                "device_name": device_name,
                "capabilities": capabilities,
                "created_at_utc": _utc_now(),
                "last_seen_utc": _utc_now(),
                "revoked": False,
            }
            self._state["used_nonces"].setdefault(client_id, [])
            self._save()
            return {
                "client_id": client_id,
                "client_secret": secret,
                "round": self.get_round_payload(client_id),
            }

    def _validate_client(self, client_id: str) -> dict[str, Any]:
        client = self._state["clients"].get(client_id)
        if not client:
            raise KeyError("Unknown client_id")
        if client.get("revoked"):
            raise PermissionError("Client is revoked")
        return client

    def status(self) -> dict[str, Any]:
        with self._lock:
            active = self._state["active_round"]
            return {
                "round_id": active["round_id"],
                "updates_received": len(active["update_files"]),
                "min_clients": self._state["min_clients"],
                "max_examples_per_update": self._state.get(
                    "max_examples_per_update", self.max_examples_per_update
                ),
                "registered_clients": len(self._state["clients"]),
                "completed_rounds": len(self._state["completed_rounds"]),
                "base_model_path": self._state["base_model_path"],
                "aggregation": self._state["aggregation"],
            }

    def verify_round_request(self, *, client_id: str, nonce: str, timestamp: int, signature: str) -> None:
        client = self._validate_client(client_id)
        if not verify_timestamp(timestamp, max_clock_skew=MAX_CLOCK_SKEW_SECONDS):
            raise PermissionError("Round request timestamp outside allowed clock skew")
        nonces = self._state["used_nonces"].setdefault(client_id, [])
        if nonce in nonces:
            raise PermissionError("Round request replay detected: nonce already used")
        if not verify_request_signature(client["secret"], "GET", "/v1/round", client_id, nonce, timestamp, signature):
            raise PermissionError("Invalid round request signature")
        nonces.append(nonce)
        self._state["used_nonces"][client_id] = nonces[-200:]
        self._save()

    def get_round_payload(self, client_id: str) -> dict[str, Any]:
        with self._lock:
            self._validate_client(client_id)
            active = self._state["active_round"]
            global_adapter = self._load_global_adapter()
            return {
                "round_id": active["round_id"],
                "opened_at_utc": active["opened_at_utc"],
                "base_model_path": self._state["base_model_path"],
                "base_model_sha256": self._state["base_model_sha256"],
                "lora_config": self._state["lora_config"],
                "global_adapter": encode_tensor_dict(global_adapter),
                "diloco_local_steps": self._state.get("diloco_local_steps", 100),
                "submission": {
                    "max_clock_skew_seconds": MAX_CLOCK_SKEW_SECONDS,
                    "signature": "hmac_sha256(canonical_json(payload_without_signature), client_secret)",
                },
            }

    def _use_nonce(self, client_id: str, nonce: str):
        nonces = self._state["used_nonces"].setdefault(client_id, [])
        if nonce in nonces:
            raise ValueError("Replay detected: nonce already used")
        nonces.append(nonce)
        self._state["used_nonces"][client_id] = nonces[-200:]

    def _submitted_client_ids(self, active: dict[str, Any]) -> set[str]:
        """Return unique contributors for a round, including pre-upgrade state files."""
        submitted = {
            str(client_id)
            for client_id in active.get("submitted_client_ids", [])
            if str(client_id).strip()
        }
        if len(submitted) >= len(active.get("update_files", [])):
            return submitted
        for file_path in active.get("update_files", []):
            try:
                payload = torch.load(file_path, map_location="cpu", weights_only=False)
                client_id = str(payload.get("client_id", "")).strip()
                if client_id:
                    submitted.add(client_id)
            except Exception:  # noqa: BLE001
                continue
        active["submitted_client_ids"] = sorted(submitted)
        return submitted

    def submit_update(self, envelope: dict[str, Any]) -> dict[str, Any]:
        required = {"client_id", "round_id", "num_examples", "metrics", "delta", "timestamp", "nonce", "signature"}
        missing = required - set(envelope.keys())
        if missing:
            raise ValueError(f"Missing fields: {', '.join(sorted(missing))}")

        with self._lock:
            client_id = str(envelope["client_id"])
            client = self._validate_client(client_id)

            signed_payload = {k: envelope[k] for k in envelope if k != "signature"}
            if not verify_signature(client["secret"], signed_payload, str(envelope["signature"])):
                raise PermissionError("Invalid signature")

            if not verify_timestamp(int(envelope["timestamp"]), max_clock_skew=MAX_CLOCK_SKEW_SECONDS):
                raise PermissionError("Timestamp outside allowed clock skew")

            active = self._state["active_round"]
            round_id = int(envelope["round_id"])
            if round_id != int(active["round_id"]):
                raise ValueError(f"Round mismatch: expected {active['round_id']}, got {round_id}")

            nonce = str(envelope["nonce"])
            if not nonce or len(nonce) > 128:
                raise ValueError("Invalid nonce")

            submitted_client_ids = self._submitted_client_ids(active)
            if client_id in submitted_client_ids:
                raise ValueError("This client has already submitted an update for the active round")

            try:
                num_examples = int(envelope["num_examples"])
            except (TypeError, ValueError) as exc:
                raise ValueError("num_examples must be an integer") from exc
            max_examples = max(
                1,
                int(self._state.get("max_examples_per_update", self.max_examples_per_update)),
            )
            if not 1 <= num_examples <= max_examples:
                raise ValueError(f"num_examples must be between 1 and {max_examples}")

            if not isinstance(envelope.get("metrics"), dict):
                raise ValueError("metrics must be an object")
            if not isinstance(envelope.get("delta"), dict):
                raise ValueError("delta must be an object")

            global_adapter = self._load_global_adapter()
            delta = decode_tensor_dict(envelope["delta"])
            unexpected_keys = sorted(set(delta) - set(global_adapter))
            if unexpected_keys:
                preview = ", ".join(unexpected_keys[:5])
                raise ValueError(f"Update contains unexpected tensor keys: {preview}")
            normalized_delta: dict[str, torch.Tensor] = {}
            for key, base_tensor in global_adapter.items():
                if key not in delta:
                    normalized_delta[key] = torch.zeros_like(base_tensor)
                    continue
                candidate = delta[key].float()
                if tuple(candidate.shape) != tuple(base_tensor.shape):
                    raise ValueError(f"Shape mismatch for '{key}': {candidate.shape} != {base_tensor.shape}")
                if not torch.isfinite(candidate).all():
                    raise ValueError(f"Non-finite update values for '{key}'")
                normalized_delta[key] = candidate

            # Consume the nonce only once the complete update has passed validation.
            self._use_nonce(client_id, nonce)

            round_dir = self._round_dir(round_id)
            filename = f"{client_id}_{nonce}.pt"
            update_path = round_dir / "updates" / filename
            torch.save(
                {
                    "client_id": client_id,
                    "round_id": round_id,
                    "num_examples": num_examples,
                    "metrics": envelope.get("metrics", {}),
                    "delta": normalized_delta,
                    "submitted_at_utc": _utc_now(),
                },
                update_path,
            )

            active["update_files"].append(str(update_path))
            active["submitted_client_ids"] = sorted(submitted_client_ids | {client_id})
            client["last_seen_utc"] = _utc_now()
            self._save()

            result = {
                "accepted": True,
                "round_id": round_id,
                "update_count": len(active["update_files"]),
                "min_clients": self._state["min_clients"],
            }
            if len(active["update_files"]) >= int(self._state["min_clients"]):
                result["ready_for_aggregation"] = True
            return result

    def _load_round_updates(self, round_id: int) -> list[ClientUpdate]:
        active = self._state["active_round"]
        if int(active["round_id"]) != int(round_id):
            raise ValueError("Can only aggregate active round")
        updates: list[ClientUpdate] = []
        for file_path in active["update_files"]:
            payload = torch.load(file_path, map_location="cpu")
            updates.append(
                ClientUpdate(
                    client_id=str(payload["client_id"]),
                    num_examples=max(1, int(payload.get("num_examples", 1))),
                    delta=payload["delta"],
                    metrics=payload.get("metrics", {}),
                )
            )
        return updates

    def aggregate_active_round(self, force: bool = False) -> dict[str, Any]:
        with self._lock:
            active = self._state["active_round"]
            round_id = int(active["round_id"])
            count = len(active["update_files"])
            if count < int(self._state["min_clients"]) and not force:
                return {
                    "status": "waiting",
                    "round_id": round_id,
                    "updates_received": count,
                    "min_clients": self._state["min_clients"],
                }

            updates = self._load_round_updates(round_id)
            if not updates:
                return {"status": "no_updates", "round_id": round_id}

            agg_delta, agg_meta = aggregate_deltas(
                updates,
                method=self._state["aggregation"]["method"],
                trim_ratio=float(self._state["aggregation"]["trim_ratio"]),
                max_update_norm=float(self._state["aggregation"]["max_update_norm"]),
                byzantine_f=int(self._state["aggregation"]["byzantine_f"]),
            )

            current_adapter = self._load_global_adapter()
            optimizer_state_before = self.global_optimizer.state_dict()
            try:
                candidate_adapter = self.global_optimizer.step(current_adapter, agg_delta)

                eval_before = None
                eval_after = None
                accepted = True
                eval_data_exists = bool(self.eval_data_path and Path(self.eval_data_path).exists())
                if eval_data_exists:
                    eval_before = evaluate_adapter_loss(
                        base_model_path=self.base_model_path,
                        lora_cfg=self.lora_cfg,
                        adapter_state=current_adapter,
                        eval_data_path=self.eval_data_path,
                        eval_iters=self.eval_iters,
                    )
                    eval_after = evaluate_adapter_loss(
                        base_model_path=self.base_model_path,
                        lora_cfg=self.lora_cfg,
                        adapter_state=candidate_adapter,
                        eval_data_path=self.eval_data_path,
                        eval_iters=self.eval_iters,
                    )
                    accepted = bool(eval_after <= (eval_before + self.max_regression))
            except Exception:
                # A failed evaluation leaves the round open for retry.  Its
                # unpromoted update must not remain hidden in memory and gain
                # extra momentum when the retry calls step() again.
                self.global_optimizer.load_state_dict(optimizer_state_before)
                raise

            summary = {
                "round_id": round_id,
                "status": "accepted" if accepted else "rejected",
                "update_count": len(updates),
                "aggregation": agg_meta,
                "eval_before": eval_before,
                "eval_after": eval_after,
                "evaluated": eval_data_exists,
                "completed_at_utc": _utc_now(),
            }

            if accepted:
                self._save_global_adapter(candidate_adapter)
                save_global_optimizer(self.global_optimizer, str(self.global_optimizer_path))
                summary["adapter_path"] = str(self.global_adapter_path)
            else:
                # Promotion gates apply to optimizer state as well as model
                # weights; otherwise a rejected (possibly poisoned) direction
                # influences every later round through momentum.
                self.global_optimizer.load_state_dict(optimizer_state_before)

            active["status"] = "closed"
            active["closed_at_utc"] = _utc_now()
            active["summary"] = summary
            self._state["completed_rounds"].append(active)
            self._state["completed_rounds"] = self._state["completed_rounds"][-200:]

            next_round = round_id + 1
            self._state["round_id"] = next_round
            self._state["active_round"] = {
                "round_id": next_round,
                "opened_at_utc": _utc_now(),
                "update_files": [],
                "submitted_client_ids": [],
                "status": "open",
            }
            self._save()
            return summary
