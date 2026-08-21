from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PEER_ID_BYTES = 20
SECRET_BYTES = 32


@dataclass
class SwarmIdentity:
    peer_id: str
    peer_secret: str
    created_at: float
    label: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "peer_id": self.peer_id,
            "peer_secret": self.peer_secret,
            "created_at": self.created_at,
            "label": self.label,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> SwarmIdentity:
        return SwarmIdentity(
            peer_id=str(d["peer_id"]),
            peer_secret=str(d["peer_secret"]),
            created_at=float(d.get("created_at", 0)),
            label=str(d.get("label", "")),
        )

    @property
    def peer_id_bytes(self) -> bytes:
        return bytes.fromhex(self.peer_id)

    def sign(self, payload: dict[str, Any]) -> str:
        msg = _canonical_json(payload).encode("utf-8")
        key = self.peer_secret.encode("utf-8")
        return hmac.new(key=key, msg=msg, digestmod=hashlib.sha256).hexdigest()

    def verify(self, payload: dict[str, Any], signature: str) -> bool:
        return hmac.compare_digest(self.sign(payload), signature)

    def sign_bytes(self, data: bytes) -> str:
        key = self.peer_secret.encode("utf-8")
        return hmac.new(key=key, msg=data, digestmod=hashlib.sha256).hexdigest()

    def verify_bytes(self, data: bytes, signature: str) -> bool:
        return hmac.compare_digest(self.sign_bytes(data), signature)


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def create_identity(label: str = "") -> SwarmIdentity:
    raw_id = secrets.token_bytes(PEER_ID_BYTES)
    peer_id = hashlib.sha256(raw_id).hexdigest()[: PEER_ID_BYTES * 2]
    peer_secret = secrets.token_urlsafe(SECRET_BYTES)
    return SwarmIdentity(peer_id=peer_id, peer_secret=peer_secret, created_at=time.time(), label=label)


def load_identity(path: str | Path) -> SwarmIdentity | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return SwarmIdentity.from_dict(data)
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def save_identity(path: str | Path, identity: SwarmIdentity):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(identity.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def ensure_identity(path: str | Path, label: str = "") -> SwarmIdentity:
    existing = load_identity(path)
    if existing is not None:
        return existing
    identity = create_identity(label=label)
    save_identity(path, identity)
    return identity
