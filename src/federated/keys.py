"""Ed25519 keypair identity for messages that must be verifiable by any peer.

`src.federated.identity.SwarmIdentity` uses a shared HMAC secret, which only the
signer itself can verify with. Inference offers and responses need to be checked
by peers who have never talked to the signer before, so they need real public-key
signatures instead. This module is intentionally standalone (no coupling to
SwarmIdentity) so the existing torickle/training-swarm protocol is untouched.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives import serialization

import hashlib


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass
class EdIdentity:
    """A local Ed25519 keypair. `peer_id` is derived from the public key so any
    peer can recompute it from `pubkey_hex` alone and confirm they match."""

    peer_id: str
    pubkey_hex: str
    privkey_hex: str
    created_at: float
    label: str = ""

    @property
    def _private_key(self) -> Ed25519PrivateKey:
        return Ed25519PrivateKey.from_private_bytes(bytes.fromhex(self.privkey_hex))

    def sign(self, payload: dict[str, Any]) -> str:
        msg = _canonical_json(payload).encode("utf-8")
        return self._private_key.sign(msg).hex()

    def sign_bytes(self, data: bytes) -> str:
        return self._private_key.sign(data).hex()

    def to_dict(self) -> dict[str, Any]:
        return {
            "peer_id": self.peer_id,
            "pubkey_hex": self.pubkey_hex,
            "privkey_hex": self.privkey_hex,
            "created_at": self.created_at,
            "label": self.label,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "EdIdentity":
        return EdIdentity(
            peer_id=str(d["peer_id"]),
            pubkey_hex=str(d["pubkey_hex"]),
            privkey_hex=str(d["privkey_hex"]),
            created_at=float(d.get("created_at", 0)),
            label=str(d.get("label", "")),
        )


def peer_id_from_pubkey(pubkey_hex: str) -> str:
    return hashlib.sha256(bytes.fromhex(pubkey_hex)).hexdigest()[:40]


def verify_payload(pubkey_hex: str, payload: dict[str, Any], signature_hex: str) -> bool:
    """Verify a signature against a claimed public key. Any peer can call this —
    no shared secret required, unlike SwarmIdentity.verify()."""
    try:
        pubkey = Ed25519PublicKey.from_public_bytes(bytes.fromhex(pubkey_hex))
        msg = _canonical_json(payload).encode("utf-8")
        pubkey.verify(bytes.fromhex(signature_hex), msg)
        return True
    except (InvalidSignature, ValueError, TypeError):
        return False


def verify_bytes(pubkey_hex: str, data: bytes, signature_hex: str) -> bool:
    try:
        pubkey = Ed25519PublicKey.from_public_bytes(bytes.fromhex(pubkey_hex))
        pubkey.verify(bytes.fromhex(signature_hex), data)
        return True
    except (InvalidSignature, ValueError, TypeError):
        return False


def create_ed_identity(label: str = "") -> EdIdentity:
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    privkey_hex = private_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    ).hex()
    pubkey_hex = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    ).hex()
    peer_id = peer_id_from_pubkey(pubkey_hex)
    return EdIdentity(
        peer_id=peer_id,
        pubkey_hex=pubkey_hex,
        privkey_hex=privkey_hex,
        created_at=time.time(),
        label=label,
    )


def load_ed_identity(path: str | Path) -> EdIdentity | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return EdIdentity.from_dict(data)
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def save_ed_identity(path: str | Path, identity: EdIdentity):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(identity.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def ensure_ed_identity(path: str | Path, label: str = "") -> EdIdentity:
    existing = load_ed_identity(path)
    if existing is not None:
        return existing
    identity = create_ed_identity(label=label)
    save_ed_identity(path, identity)
    return identity
