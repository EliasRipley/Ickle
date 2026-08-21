from __future__ import annotations

import hashlib
import json
import secrets
import socket
import struct
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


K = 20
PING_INTERVAL = 300
STORE_TTL = 3600


def _xor_distance(a: bytes, b: bytes) -> int:
    return int.from_bytes(a, "big") ^ int.from_bytes(b, "big")


def _generate_peer_id() -> bytes:
    return secrets.token_bytes(20)


def _pack_addr(addr: tuple[str, int]) -> bytes:
    host, port = addr
    try:
        ip = socket.inet_pton(socket.AF_INET6, host)
        return b"\x06" + ip + struct.pack(">H", port)
    except OSError:
        ip = socket.inet_pton(socket.AF_INET, host)
        return b"\x04" + ip + struct.pack(">H", port)


def _unpack_addr(data: bytes) -> tuple[str, int]:
    family = data[0]
    if family == 0x06:
        ip = socket.inet_ntop(socket.AF_INET6, data[1:17])
        port = struct.unpack(">H", data[17:19])[0]
    else:
        ip = socket.inet_ntop(socket.AF_INET, data[1:5])
        port = struct.unpack(">H", data[5:7])[0]
    return ip, port


@dataclass
class PeerInfo:
    peer_id: bytes
    address: tuple[str, int]
    last_seen: float = field(default_factory=time.time)
    models: list[str] = field(default_factory=list)
    tags: dict[str, str] = field(default_factory=dict)
    ssl_fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "peer_id": self.peer_id.hex(),
            "host": self.address[0],
            "port": self.address[1],
            "last_seen": self.last_seen,
            "models": self.models,
            "tags": self.tags,
            "ssl_fingerprint": self.ssl_fingerprint,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "PeerInfo":
        return PeerInfo(
            peer_id=bytes.fromhex(d["peer_id"]),
            address=(d["host"], d["port"]),
            last_seen=d.get("last_seen", time.time()),
            models=d.get("models", []),
            tags=d.get("tags", {}),
            ssl_fingerprint=d.get("ssl_fingerprint", ""),
        )

    def encode(self) -> bytes:
        return json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")

    @staticmethod
    def decode(data: bytes) -> "PeerInfo":
        return PeerInfo.from_dict(json.loads(data.decode("utf-8")))


@dataclass
class StoredValue:
    key: bytes
    value: bytes
    publisher: bytes
    expires_at: float


class PeerStore:
    def __init__(self, path: str | None = None):
        self._peers: dict[bytes, PeerInfo] = {}
        self._values: dict[bytes, list[StoredValue]] = defaultdict(list)
        self._path = Path(path) if path else None
        if self._path:
            self._load()

    def _load(self):
        if not self._path or not self._path.exists():
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for pd in data.get("peers", []):
                peer = PeerInfo.from_dict(pd)
                self._peers[peer.peer_id] = peer
        except (json.JSONDecodeError, KeyError):
            pass

    def _save(self):
        if not self._path:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump({"peers": [p.to_dict() for p in self._peers.values()]}, f, indent=2)

    def add_peer(self, peer: PeerInfo):
        self._peers[peer.peer_id] = peer
        self._save()

    def remove_peer(self, peer_id: bytes):
        self._peers.pop(peer_id, None)
        self._save()

    def get_peer(self, peer_id: bytes) -> PeerInfo | None:
        return self._peers.get(peer_id)

    def find_nearest(self, target: bytes, count: int = K) -> list[PeerInfo]:
        peers = sorted(self._peers.values(), key=lambda p: _xor_distance(p.peer_id, target))
        return peers[:count]

    def all_peers(self) -> list[PeerInfo]:
        return list(self._peers.values())

    def store_value(self, key: bytes, value: bytes, publisher: bytes, ttl: int = STORE_TTL):
        stored = StoredValue(key=key, value=value, publisher=publisher, expires_at=time.time() + ttl)
        self._values[key].append(stored)
        self._values[key] = [v for v in self._values[key] if v.expires_at > time.time()]
        self._values[key] = self._values[key][-10:]

    def find_value(self, key: bytes) -> list[bytes]:
        now = time.time()
        self._values[key] = [v for v in self._values.get(key, []) if v.expires_at > now]
        return [v.value for v in self._values.get(key, [])]

    def cleanup(self):
        now = time.time()
        for peer_id, peer in list(self._peers.items()):
            if now - peer.last_seen > STORE_TTL * 2:
                del self._peers[peer_id]
        for key in list(self._values.keys()):
            self._values[key] = [v for v in self._values[key] if v.expires_at > now]
            if not self._values[key]:
                del self._values[key]


class PeerDiscovery:
    def __init__(self, node_id: bytes | None = None, store: PeerStore | None = None):
        self.node_id = node_id or _generate_peer_id()
        self.store = store or PeerStore()
        self._bootstrap_peers: list[tuple[str, int]] = []

    def add_bootstrap(self, host: str, port: int):
        self._bootstrap_peers.append((host, port))

    def announce(self, port: int, models: list[str] | None = None, tags: dict[str, str] | None = None):
        local_ip = self._local_ip()
        local_peer = PeerInfo(
            peer_id=self.node_id,
            address=(local_ip, port),
            last_seen=time.time(),
            models=models or [],
            tags=tags or {},
        )
        self.store.add_peer(local_peer)

    @staticmethod
    def _local_ip() -> str:
        for af, probe in [(socket.AF_INET, "8.8.8.8"), (socket.AF_INET6, "2001:4860:4860::8888")]:
            try:
                s = socket.socket(af, socket.SOCK_DGRAM)
                s.connect((probe, 80))
                ip = s.getsockname()[0]
                s.close()
                return ip
            except OSError:
                continue
        return "127.0.0.1"

def bootstrap_fetch(bootstrap_peers: list[tuple[str, int]], path: str, timeout: float = 5.0) -> list[dict]:
    """GET `path` from each bootstrap peer's HTTP server and return the decoded
    JSON body from every peer that answered.

    `PeerDiscovery.add_bootstrap()` only records a peer's address -- it never
    actually contacts it, so a fresh node (or a one-shot `find`/`ask` CLI
    invocation) never learns anything beyond its own local store no matter how
    many bootstrap peers are configured. This is the missing "ask a known peer
    what it knows" step, over the HTTP server every node type already runs
    (`swarm.py`'s SwarmNode, `inference_swarm.py`'s InferenceNode) -- not a new
    transport, just the client-side call that was never made.
    """
    results: list[dict] = []
    for host, port in bootstrap_peers:
        url = f"http://{host}:{port}{path}"
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            if isinstance(body, dict):
                results.append(body)
        except (urllib.error.URLError, OSError, ValueError, TimeoutError):
            continue
    return results


