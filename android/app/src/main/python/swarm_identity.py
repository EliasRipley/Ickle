import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass


PEER_ID_BYTES = 20
SECRET_BYTES = 32


@dataclass
class SwarmIdentity:
    peer_id: str
    peer_secret: str
    created_at: float
    label: str

    def to_dict(self):
        return {
            "peer_id": self.peer_id,
            "peer_secret": self.peer_secret,
            "created_at": self.created_at,
            "label": self.label,
        }

    @staticmethod
    def from_dict(d):
        return SwarmIdentity(
            peer_id=str(d["peer_id"]),
            peer_secret=str(d["peer_secret"]),
            created_at=float(d.get("created_at", 0)),
            label=str(d.get("label", "")),
        )

    @property
    def peer_id_bytes(self):
        return bytes.fromhex(self.peer_id)

    def sign(self, payload):
        msg = _canonical_json(payload).encode("utf-8")
        return hmac.new(key=self.peer_secret.encode("utf-8"), msg=msg, digestmod=hashlib.sha256).hexdigest()

    def verify(self, payload, signature):
        return hmac.compare_digest(self.sign(payload), signature)


def _canonical_json(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def create_identity(label=""):
    raw_id = secrets.token_bytes(PEER_ID_BYTES)
    peer_id = hashlib.sha256(raw_id).hexdigest()[:PEER_ID_BYTES * 2]
    peer_secret = secrets.token_urlsafe(SECRET_BYTES)
    return SwarmIdentity(peer_id=peer_id, peer_secret=peer_secret, created_at=time.time(), label=label)


def load_identity(path):
    import os
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return SwarmIdentity.from_dict(data)
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def save_identity(path, identity):
    import os
    p = path
    if hasattr(p, 'parent'):
        os.makedirs(str(p.parent) if hasattr(p, 'parent') else os.path.dirname(str(p)), exist_ok=True)
    else:
        os.makedirs(os.path.dirname(str(p)), exist_ok=True)
    with open(str(p), "w", encoding="utf-8") as f:
        json.dump(identity.to_dict(), f, indent=2)


def ensure_identity(path, label=""):
    existing = load_identity(path)
    if existing is not None:
        return existing
    identity = create_identity(label=label)
    save_identity(path, identity)
    return identity
