from __future__ import annotations

import base64
import hashlib
import hmac
import io
import json
import secrets
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


MAX_CLOCK_SKEW_SECONDS = 300


@dataclass
class SubmissionEnvelope:
    client_id: str
    round_id: int
    num_examples: int
    metrics: dict[str, Any]
    delta: dict[str, Any]
    timestamp: int
    nonce: str
    signature: str


def now_epoch_seconds() -> int:
    return int(time.time())


def canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def generate_client_secret() -> str:
    return secrets.token_urlsafe(32)


def generate_nonce() -> str:
    return secrets.token_urlsafe(12)


def sign_payload(secret: str, payload: dict[str, Any]) -> str:
    msg = canonical_json(payload).encode("utf-8")
    key = secret.encode("utf-8")
    return hmac.new(key=key, msg=msg, digestmod=hashlib.sha256).hexdigest()


def verify_signature(secret: str, payload: dict[str, Any], signature: str) -> bool:
    expected = sign_payload(secret, payload)
    return hmac.compare_digest(expected, signature)


def verify_timestamp(epoch_seconds: int, max_clock_skew: int = MAX_CLOCK_SKEW_SECONDS) -> bool:
    return abs(now_epoch_seconds() - int(epoch_seconds)) <= max_clock_skew


def sign_request(secret: str, method: str, path: str, client_id: str, nonce: str, timestamp: int) -> str:
    payload = canonical_json({
        "method": method,
        "path": path,
        "client_id": client_id,
        "nonce": nonce,
        "timestamp": timestamp,
    })
    return sign_payload(secret, payload)


def verify_request_signature(secret: str, method: str, path: str, client_id: str, nonce: str, timestamp: int, signature: str) -> bool:
    expected = sign_request(secret, method, path, client_id, nonce, timestamp)
    return hmac.compare_digest(expected, signature)


def tensor_to_wire(tensor: torch.Tensor) -> dict[str, Any]:
    arr = tensor.detach().cpu().float().numpy().astype(np.float32, copy=False)
    data = base64.b64encode(arr.tobytes(order="C")).decode("ascii")
    return {
        "dtype": "float32",
        "shape": list(arr.shape),
        "data_b64": data,
    }


def tensor_from_wire(payload: dict[str, Any]) -> torch.Tensor:
    dtype = payload.get("dtype", "float32")
    if dtype != "float32":
        raise ValueError(f"Unsupported tensor dtype '{dtype}'")
    shape = payload.get("shape", [])
    raw = base64.b64decode(payload["data_b64"].encode("ascii"))
    arr = np.frombuffer(raw, dtype=np.float32).reshape(shape)
    return torch.from_numpy(arr.copy())


def encode_tensor_dict(tensors: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {name: tensor_to_wire(tensor) for name, tensor in tensors.items()}


def decode_tensor_dict(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, wire in payload.items():
        out[name] = tensor_from_wire(wire)
    return out


def file_sha256(path: str) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def torch_save_to_base64(payload: Any) -> str:
    buf = io.BytesIO()
    torch.save(payload, buf)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def torch_load_from_base64(payload_b64: str) -> Any:
    raw = base64.b64decode(payload_b64.encode("ascii"))
    buf = io.BytesIO(raw)
    return torch.load(buf, map_location="cpu")

