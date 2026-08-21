from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from src.federated.contribution_ledger import LedgerStore
from src.federated.identity import SwarmIdentity, ensure_identity, load_identity, save_identity
from src.federated.keys import EdIdentity, ensure_ed_identity, peer_id_from_pubkey, verify_payload
from src.federated.knowledge_commons import EpistemicLedger, MAX_EVENT_BATCH, MAX_SYNC_BYTES
from src.federated.nat_traversal import stun_get_external_address, upnp_add_port_mapping, upnp_remove_port_mapping
from src.federated.node_base import detect_local_ip
from src.federated.peer_discovery import PeerDiscovery, PeerInfo, bootstrap_fetch


DEFAULT_SWARM_PORT = 8790
DEFAULT_DATA_DIR = "data/torickle"
DEFAULT_IDENTITY_PATH = "data/torickle/swarm_identity.json"
# Separate from DEFAULT_IDENTITY_PATH's HMAC SwarmIdentity: bundle
# announcements need to be verifiable by peers who've never talked to the
# signer before, which HMAC (shared-secret) signatures structurally can't
# do -- see BundleAnnouncement.verify() and keys.py's module docstring.
DEFAULT_ED_IDENTITY_PATH = "data/torickle/swarm_ed_identity.json"
ANNOUNCE_TTL = 3600
REQUEST_TIMEOUT = 30
PEER_CLEANUP_INTERVAL_SECONDS = 300


def _bundle_dht_key(bundle_id: str) -> bytes:
    return hashlib.sha256(f"ickle:torickle:bundle:{bundle_id}".encode()).digest()[:20]


def _model_dht_key(model_hash: str) -> bytes:
    return hashlib.sha256(f"ickle:torickle:model:{model_hash}".encode()).digest()[:20]


def _all_bundles_dht_key() -> bytes:
    return hashlib.sha256(b"ickle:torickle:bundle:all").digest()[:20]


def _safe_join(base_dir: Path, relative_name: str) -> Path | None:
    """Resolve a manifest-provided piece path and keep it inside base_dir."""
    rel = Path(str(relative_name or ""))
    if rel.is_absolute():
        return None
    base = base_dir.resolve()
    candidate = (base / rel).resolve()
    try:
        candidate.relative_to(base)
    except ValueError:
        return None
    return candidate


@dataclass
class BundleAnnouncement:
    bundle_id: str
    model_hash: str
    piece_count: int
    total_bytes: int
    payload_sha256: str
    merkle_root: str
    peer_id: str
    host: str
    port: int
    timestamp: float
    signature: str
    use_tls: bool = False
    # Ed25519 public key of the signer, so any peer -- including one that's
    # never seen this signer before -- can verify the signature themselves.
    # HMAC (the old scheme) can't do that: verification needs the signer's
    # shared secret, which a stranger never has. See keys.py's docstring.
    pubkey_hex: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "model_hash": self.model_hash,
            "piece_count": self.piece_count,
            "total_bytes": self.total_bytes,
            "payload_sha256": self.payload_sha256,
            "merkle_root": self.merkle_root,
            "peer_id": self.peer_id,
            "host": self.host,
            "port": self.port,
            "use_tls": self.use_tls,
            "timestamp": self.timestamp,
            "signature": self.signature,
            "pubkey_hex": self.pubkey_hex,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> BundleAnnouncement:
        return BundleAnnouncement(
            bundle_id=str(d["bundle_id"]),
            model_hash=str(d.get("model_hash", "")),
            piece_count=int(d.get("piece_count", 0)),
            total_bytes=int(d.get("total_bytes", 0)),
            payload_sha256=str(d.get("payload_sha256", "")),
            merkle_root=str(d.get("merkle_root", "")),
            peer_id=str(d["peer_id"]),
            host=str(d["host"]),
            port=int(d["port"]),
            use_tls=bool(d.get("use_tls", False)),
            timestamp=float(d.get("timestamp", 0)),
            signature=str(d.get("signature", "")),
            pubkey_hex=str(d.get("pubkey_hex", "")),
        )

    def dht_key(self) -> bytes:
        return _bundle_dht_key(self.bundle_id)

    def verify(self) -> bool:
        """Self-contained: any peer can call this with no prior relationship
        to the signer, since it only needs the pubkey_hex carried in the
        announcement itself -- that's the point of Ed25519 over HMAC here."""
        if not self.pubkey_hex or not self.signature:
            return False
        if peer_id_from_pubkey(self.pubkey_hex) != self.peer_id:
            return False
        payload = self.to_dict()
        payload.pop("signature", None)
        return verify_payload(self.pubkey_hex, payload, self.signature)


class SwarmRequestHandler(BaseHTTPRequestHandler):
    server_version = "IckleSwarm/0.1"

    @property
    def swarm_node(self) -> SwarmNode:
        return self.server.swarm_node  # type: ignore[attr-defined]

    def _json_response(self, status: int, payload: Any):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _binary_response(self, status: int, data: bytes, content_type: str = "application/octet-stream"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _not_found(self, msg: str = "Not found"):
        self._json_response(404, {"error": msg})

    def do_OPTIONS(self):  # noqa: N802
        # Peer-to-peer clients are native HTTP clients and do not need CORS.
        # Deliberately omit Access-Control-Allow-Origin so an arbitrary web
        # page cannot read a person's loopback swarm/commons data or POST to
        # it through their browser. The local Ickle UI talks to serve-control,
        # whose CORS policy is restricted to localhost origins.
        self.send_response(204)
        self.end_headers()

    def _parse_path(self) -> list[str]:
        parsed = urlparse(self.path)
        return [seg for seg in parsed.path.strip("/").split("/") if seg]

    def do_GET(self):  # noqa: N802
        segments = self._parse_path()
        try:
            if len(segments) >= 1 and segments[0] != "torickle":
                return self._not_found("Expected /torickle/v1/...")
            version = segments[1] if len(segments) > 1 else "v1"
            if segments[2:] == ["announce"]:
                return self._handle_list_bundles()
            if len(segments) >= 4 and segments[2] == "manifest":
                return self._handle_get_manifest(segments[3])
            if len(segments) >= 5 and segments[2] == "pieces":
                return self._handle_get_piece(segments[3], segments[4])
            if segments[2:] == ["peers"]:
                return self._handle_list_peers()
            if segments[2:] == ["dht", "bundles"]:
                return self._handle_list_dht_announcements()
            if segments[2:] == ["commons", "events"]:
                return self._handle_list_commons_events()
            return self._json_response(200, {
                "service": "ickle-swarm",
                "version": version,
                "peer_id": self.swarm_node.identity.peer_id,
                "bundles_served": len(self.swarm_node.bundles),
                "peers_known": len(self.swarm_node.peer_discovery.store.all_peers()),
                "commons": self.swarm_node.commons.summary(),
            })
        except IndexError:
            return self._not_found("Invalid path")

    def do_HEAD(self):  # noqa: N802
        segments = self._parse_path()
        try:
            if segments[:3] != ["torickle", "v1", "manifest"]:
                return self._not_found()
            bundle_id = segments[3]
            bundle_dir = self.swarm_node._bundle_dir(bundle_id)
            if not bundle_dir or not (bundle_dir / "manifest.json").exists():
                return self._not_found(f"Bundle {bundle_id} not found")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str((bundle_dir / "manifest.json").stat().st_size))
            self.end_headers()
        except IndexError:
            return self._not_found()

    def _handle_list_bundles(self):
        entries = []
        for bundle_id, info in self.swarm_node.bundles.items():
            bundle_dir = info.bundle_dir
            manifest_path = bundle_dir / "manifest.json"
            if manifest_path.exists():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    entries.append({
                        "bundle_id": bundle_id,
                        "piece_count": manifest.get("piece_count", 0),
                        "total_bytes": manifest.get("total_bytes", 0),
                        "payload_sha256": manifest.get("payload_sha256", ""),
                        "merkle_root": manifest.get("merkle_root", ""),
                        "created_at": manifest.get("created_at_utc", ""),
                    })
                except (json.JSONDecodeError, KeyError):
                    pass
        self._json_response(200, {
            "peer_id": self.swarm_node.identity.peer_id,
            "bundles": entries,
        })

    def _handle_get_manifest(self, bundle_id: str):
        bundle_dir = self.swarm_node._bundle_dir(bundle_id)
        if not bundle_dir:
            return self._not_found(f"Bundle {bundle_id} not found on this peer")
        manifest_path = bundle_dir / "manifest.json"
        if not manifest_path.exists():
            return self._not_found(f"Manifest for {bundle_id} not found")
        data = manifest_path.read_bytes()
        self._binary_response(200, data, "application/json")

    def _handle_get_piece(self, bundle_id: str, piece_index: str):
        bundle_dir = self.swarm_node._bundle_dir(bundle_id)
        if not bundle_dir:
            return self._not_found(f"Bundle {bundle_id} not found on this peer")
        pieces_dir = bundle_dir / "pieces"
        if not pieces_dir.is_dir():
            return self._not_found(f"Pieces directory for {bundle_id} not found")
        try:
            manifest_path = bundle_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            pieces = manifest.get("pieces", [])
            idx = int(piece_index)
            if idx < 0 or idx >= len(pieces):
                return self._not_found(f"Piece index {idx} out of range (0-{len(pieces)-1})")
            file_name = pieces[idx].get("file_name", "")
            if not file_name:
                return self._not_found(f"Piece {idx} has no file_name")
            piece_path = _safe_join(pieces_dir, str(file_name))
            if piece_path is None:
                return self._not_found(f"Invalid piece path for piece {idx}")
            if not piece_path.exists():
                return self._not_found(f"Piece file {file_name} not found")
            data = piece_path.read_bytes()
            self._binary_response(200, data)
            self.swarm_node.ledger.record_piece_served(len(data))
        except (json.JSONDecodeError, KeyError, ValueError, OSError) as exc:
            self._json_response(500, {"error": f"Failed to read piece: {exc}"})

    def _handle_list_dht_announcements(self):
        """All bundle announcements this node currently knows about via its
        local DHT store -- not just bundles it hosts itself (that's
        _handle_list_bundles/GET /torickle/v1/announce). This is what a node
        joining via --bootstrap needs to pull to actually learn about bundles
        it hasn't seen before."""
        raw_values = self.swarm_node.peer_discovery.store.find_value(_all_bundles_dht_key())
        announcements = []
        for raw in raw_values:
            try:
                announcements.append(BundleAnnouncement.from_dict(json.loads(raw.decode("utf-8"))).to_dict())
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
        self._json_response(200, {
            "peer_id": self.swarm_node.identity.peer_id,
            "announcements": announcements,
        })

    def _handle_list_peers(self):
        peers = self.swarm_node.peer_discovery.store.all_peers()
        self._json_response(200, {
            "peer_id": self.swarm_node.identity.peer_id,
            "peers": [p.to_dict() for p in peers],
        })

    def _handle_list_commons_events(self):
        """Expose only reviews their authors explicitly marked as shared."""
        parsed = urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        try:
            after = max(0.0, float((query.get("after", ["0"])[0] or "0")))
            limit = max(1, min(MAX_EVENT_BATCH, int((query.get("limit", [str(MAX_EVENT_BATCH)])[0] or MAX_EVENT_BATCH))))
        except (TypeError, ValueError):
            return self._json_response(400, {"error": "after and limit must be numeric"})
        events = self.swarm_node.commons.public_events(shared_only=True, after=after, limit=limit)
        self._json_response(200, {
            "peer_id": self.swarm_node.ed_identity.peer_id,
            "events": events,
            "conflict_policy": "preserve",
        })

    def do_POST(self):  # noqa: N802
        segments = self._parse_path()
        if len(segments) >= 3 and segments[0] == "torickle" and segments[2] == "announce":
            return self._handle_post_announce()
        if segments == ["torickle", "v1", "commons", "events"]:
            return self._handle_post_commons_events()
        return self._not_found("POST not supported for this path")

    def _handle_post_commons_events(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length <= 0 or content_length > MAX_SYNC_BYTES:
                return self._json_response(400, {"error": "Invalid Content-Length"})
            raw = self.rfile.read(content_length)
            data = json.loads(raw.decode("utf-8"))
            events = data.get("events", []) if isinstance(data, dict) else []
            if not isinstance(events, list):
                return self._json_response(400, {"error": "events must be a list"})
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            return self._json_response(400, {"error": f"Invalid JSON: {exc}"})
        result = self.swarm_node.commons.merge_events(events)
        return self._json_response(200, result)

    def _handle_post_announce(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length <= 0 or content_length > 1_048_576:
                return self._json_response(400, {"error": "Invalid Content-Length"})
            raw = self.rfile.read(content_length)
            data = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, ValueError) as exc:
            return self._json_response(400, {"error": f"Invalid JSON: {exc}"})

        required = {"bundle_id", "peer_id", "signature", "pubkey_hex"}
        missing = required - set(data.keys())
        if missing:
            return self._json_response(400, {"error": f"Missing fields: {', '.join(sorted(missing))}"})

        try:
            ann = BundleAnnouncement.from_dict(data)
        except (KeyError, ValueError) as exc:
            return self._json_response(400, {"error": f"Invalid announcement: {exc}"})

        # Reject unsigned/forged announcements instead of trusting anything
        # POSTed to this endpoint -- verify() is self-contained (checks the
        # embedded pubkey_hex against peer_id and the signature against the
        # payload), so this works even for a peer we've never talked to before.
        if not ann.verify():
            return self._json_response(403, {"error": "Invalid or missing announcement signature"})

        dht_value = json.dumps(ann.to_dict(), sort_keys=True).encode("utf-8")
        dht_keys = {ann.dht_key(), _all_bundles_dht_key()}
        if ann.model_hash:
            dht_keys.add(_model_dht_key(ann.model_hash))
        peer_id_bytes = bytes.fromhex(ann.peer_id) if len(ann.peer_id) == 40 else ann.peer_id.encode("utf-8")[:20]
        for key in dht_keys:
            self.swarm_node.peer_discovery.store.store_value(
                key, dht_value, peer_id_bytes, ANNOUNCE_TTL
            )

        return self._json_response(200, {
            "accepted": True,
            "bundle_id": ann.bundle_id,
            "peer_id": ann.peer_id,
            "stored_in_keys": len(dht_keys),
        })

    def log_message(self, fmt: str, *args):
        pass


@dataclass
class BundleInfo:
    bundle_id: str
    bundle_dir: Path
    manifest: dict[str, Any] = field(default_factory=dict)
    announcement: BundleAnnouncement | None = None


class SwarmNode:
    def __init__(
        self,
        identity: SwarmIdentity,
        data_dir: str | Path = DEFAULT_DATA_DIR,
        peer_discovery: PeerDiscovery | None = None,
        host: str = "0.0.0.0",
        port: int = DEFAULT_SWARM_PORT,
        external_host: str | None = None,
        ledger: LedgerStore | None = None,
        ed_identity: EdIdentity | None = None,
    ):
        self.identity = identity
        self.data_dir = Path(data_dir)
        self.bundles_dir = self.data_dir / "bundles"
        self.bundles_dir.mkdir(parents=True, exist_ok=True)
        # Used to sign/verify BundleAnnouncements (see BundleAnnouncement.verify) --
        # separate from `identity` above, which stays HMAC-based for the DHT
        # peer_id/store namespacing it already handles.
        self.ed_identity = ed_identity or ensure_ed_identity(
            self.data_dir / "swarm_ed_identity.json", label=identity.label
        )
        # Human reviews use the same public peer identity as signed bundle
        # announcements, but live in their own conflict-preserving event set.
        # data_dir is normally data/torickle, so the database sits beside it
        # at data/commons and is shared with the local chat server.
        self.commons = EpistemicLedger(
            self.data_dir.parent / "commons" / "epistemic.sqlite3",
            identity=self.ed_identity,
        )
        self.peer_discovery = peer_discovery or PeerDiscovery()
        self.ledger = ledger or LedgerStore(self.data_dir / "contribution_ledger.json")
        self.host = host
        self.port = port
        self._external_host_explicit = bool(external_host)
        self.external_host = external_host or detect_local_ip()
        self._port_mapped = False
        self._bundles: dict[str, BundleInfo] = {}
        self._server: ThreadingHTTPServer | None = None
        self._server_thread: threading.Thread | None = None
        self._running = False
        self._cleanup_thread: threading.Thread | None = None
        self._cleanup_stop = threading.Event()

    @property
    def bundles(self) -> dict[str, BundleInfo]:
        return dict(self._bundles)

    def _cleanup_loop(self):
        """Periodically drop stale peers and expired DHT values --
        PeerStore.cleanup() previously had no caller anywhere in the
        codebase (only the now-removed, itself-uncalled PeerDiscovery.refresh()
        called it), so peers that went offline were never actually pruned;
        the store just grew forever."""
        while not self._cleanup_stop.wait(PEER_CLEANUP_INTERVAL_SECONDS):
            try:
                self.peer_discovery.store.cleanup()
            except Exception as exc:  # noqa: BLE001 -- a cleanup hiccup must never take the node down
                print(f"  Peer store cleanup failed (non-fatal): {exc}")


    def start(self, *, attempt_nat_traversal: bool = False):
        if self._running:
            return
        self._server = ThreadingHTTPServer((self.host, self.port), SwarmRequestHandler)
        self._server.swarm_node = self  # type: ignore[attr-defined]
        self._server_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._server_thread.start()
        self._running = True
        self.peer_discovery.node_id = self.identity.peer_id_bytes

        if attempt_nat_traversal:
            self._attempt_nat_traversal()

        self.peer_discovery.announce(port=self.port, tags={"peer_id": self.identity.peer_id})
        self._scan_local_bundles()
        self._cleanup_stop.clear()
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
        print(f"Swarm node started on {self.host}:{self.port} (external: {self.external_host}:{self.port})")
        print(f"  Peer ID: {self.identity.peer_id}")
        print(f"  Bundles found: {len(self._bundles)}")

    def _attempt_nat_traversal(self):
        """Best-effort, never fatal: learn our real public address via STUN
        (RFC 5389) and ask the local router to forward our port via UPnP
        IGD. Either step failing (offline, UPnP disabled, symmetric NAT,
        etc.) just leaves the node as reachable as it already was -- it
        never blocks startup or raises."""
        if not self._external_host_explicit:
            stun_result = stun_get_external_address()
            if stun_result:
                public_ip, _public_port = stun_result
                self.external_host = public_ip
                print(f"  STUN: public address is {public_ip} (used as external_host)")
            else:
                print("  STUN: could not determine public address (offline, or UDP blocked) -- "
                      f"keeping local-interface guess {self.external_host}")

        try:
            mapped = upnp_add_port_mapping(self.port, description="Ickle Swarm")
        except Exception as exc:  # noqa: BLE001 -- best-effort, never take down the node
            mapped = False
            print(f"  UPnP port forwarding raised an unexpected error (not just 'unsupported'): {exc}")
        self._port_mapped = mapped
        print(f"  UPnP port forwarding: {'succeeded' if mapped else 'unavailable (router UPnP off, unsupported, or blocked)'}")

    def stop(self):
        self._running = False
        self._cleanup_stop.set()
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)
            self._cleanup_thread = None
        if self._port_mapped:
            try:
                upnp_remove_port_mapping(self.port)
            except Exception as exc:  # noqa: BLE001
                print(f"  UPnP port-mapping cleanup failed (non-fatal): {exc}")
            self._port_mapped = False
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._server_thread:
            self._server_thread.join(timeout=5)
            self._server_thread = None

    def _scan_local_bundles(self):
        if not self.bundles_dir.is_dir():
            return
        for entry in sorted(self.bundles_dir.iterdir()):
            if entry.is_dir():
                manifest_path = entry / "manifest.json"
                if manifest_path.exists():
                    self._register_bundle_dir(entry)

    def _register_bundle_dir(self, bundle_dir: Path) -> str | None:
        manifest_path = bundle_dir / "manifest.json"
        if not manifest_path.exists():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            bundle_id = hashlib.sha256(
                json.dumps(manifest.get("payload_sha256", ""), sort_keys=True).encode()
            ).hexdigest()[:16]
            info = BundleInfo(bundle_id=bundle_id, bundle_dir=bundle_dir, manifest=manifest)
            self._bundles[bundle_id] = info
            return bundle_id
        except (json.JSONDecodeError, KeyError):
            return None

    def _bundle_dir(self, bundle_id: str) -> Path | None:
        info = self._bundles.get(bundle_id)
        if info is not None:
            return info.bundle_dir
        candidate = self.bundles_dir / bundle_id
        if (candidate / "manifest.json").exists():
            self._register_bundle_dir(candidate)
            return candidate
        return None

    def import_bundle(self, bundle_path: str | Path) -> str | None:
        src = Path(bundle_path).resolve()
        manifest_path = src / "manifest.json" if src.is_dir() else Path(bundle_path)
        if manifest_path.suffix == ".json":
            src = manifest_path.parent
        if not (src / "manifest.json").exists():
            print(f"No manifest.json found in {src}")
            return None
        try:
            manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"Could not read manifest.json in {src}: {exc}")
            return None
        bundle_id = hashlib.sha256(
            json.dumps(manifest.get("payload_sha256", ""), sort_keys=True).encode()
        ).hexdigest()[:16]
        dest = self.bundles_dir / bundle_id
        if dest.exists():
            print(f"Bundle {bundle_id} already imported")
            return bundle_id
        import shutil
        shutil.copytree(str(src), str(dest))
        self._register_bundle_dir(dest)
        print(f"Imported bundle {bundle_id} from {src}")
        return bundle_id

    def announce_bundle(self, bundle_id: str, model_hash: str = "") -> BundleAnnouncement | None:
        info = self._bundles.get(bundle_id)
        if info is None:
            info_dir = self._bundle_dir(bundle_id)
            if info_dir is None:
                print(f"Bundle {bundle_id} not found locally")
                return None
        if info is None:
            return None
        manifest = info.manifest or json.loads((info.bundle_dir / "manifest.json").read_text(encoding="utf-8"))
        announcement = BundleAnnouncement(
            bundle_id=bundle_id,
            model_hash=model_hash or manifest.get("metadata", {}).get("model_hash", ""),
            piece_count=int(manifest.get("piece_count", 0)),
            total_bytes=int(manifest.get("total_bytes", 0)),
            payload_sha256=str(manifest.get("payload_sha256", "")),
            merkle_root=str(manifest.get("merkle_root", "")),
            peer_id=self.ed_identity.peer_id,
            host=self.external_host,
            port=self.port,
            timestamp=time.time(),
            signature="",
            pubkey_hex=self.ed_identity.pubkey_hex,
        )
        sig_payload = announcement.to_dict()
        sig_payload.pop("signature", None)
        announcement.signature = self.ed_identity.sign(sig_payload)
        info.announcement = announcement
        dht_value = json.dumps(announcement.to_dict(), sort_keys=True).encode("utf-8")
        dht_keys = {announcement.dht_key(), _all_bundles_dht_key()}
        if announcement.model_hash:
            dht_keys.add(_model_dht_key(announcement.model_hash))
        for key in dht_keys:
            self.peer_discovery.store.store_value(key, dht_value, self.identity.peer_id_bytes, ANNOUNCE_TTL)
        print(f"Announced bundle {bundle_id} (model={announcement.model_hash}, {announcement.piece_count} pieces, {announcement.total_bytes} bytes)")
        return announcement

    def find_bundles(self, model_hash: str = "", max_age: float = ANNOUNCE_TTL) -> list[BundleAnnouncement]:
        keys = [_all_bundles_dht_key()]
        if model_hash:
            keys.insert(0, _model_dht_key(model_hash))
        raw_values: list[bytes] = []
        for key in keys:
            raw_values.extend(self.peer_discovery.store.find_value(key))
        now = time.time()
        dedup: dict[tuple[str, str], BundleAnnouncement] = {}
        for raw in raw_values:
            try:
                d = json.loads(raw.decode("utf-8"))
                ann = BundleAnnouncement.from_dict(d)
                if now - ann.timestamp > max_age:
                    continue
                if model_hash and ann.model_hash != model_hash:
                    continue
                peer = self.peer_discovery.store.get_peer(bytes.fromhex(ann.peer_id))
                if peer is not None:
                    ann.host = peer.address[0]
                    ann.port = peer.address[1]
                    ann.use_tls = bool(peer.ssl_fingerprint)
                dedup_key = (ann.bundle_id, ann.peer_id)
                prev = dedup.get(dedup_key)
                if prev is None or ann.timestamp > prev.timestamp:
                    dedup[dedup_key] = ann
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
        results = list(dedup.values())
        results.sort(key=lambda a: a.timestamp, reverse=True)
        return results

    def query_peer_bundles(self, host: str, port: int, *, use_tls: bool = False) -> list[dict[str, Any]]:
        scheme = "https" if use_tls else "http"
        url = f"{scheme}://{host}:{port}/torickle/v1/announce"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data.get("bundles", [])
        except urllib.error.HTTPError as exc:
            exc.close()
            print(f"Failed to query peer {host}:{port}: HTTP {exc.code}")
            return []
        except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
            print(f"Failed to query peer {host}:{port}: {exc}")
            return []

    def download_manifest(self, host: str, port: int, bundle_id: str, *, use_tls: bool = False) -> dict[str, Any] | None:
        scheme = "https" if use_tls else "http"
        url = f"{scheme}://{host}:{port}/torickle/v1/manifest/{bundle_id}"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            exc.close()
            print(f"Failed to download manifest for {bundle_id} from {host}:{port}: HTTP {exc.code}")
            return None
        except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
            print(f"Failed to download manifest for {bundle_id} from {host}:{port}: {exc}")
            return None

    def download_piece(self, host: str, port: int, bundle_id: str, piece_index: int, *, use_tls: bool = False) -> bytes | None:
        scheme = "https" if use_tls else "http"
        url = f"{scheme}://{host}:{port}/torickle/v1/pieces/{bundle_id}/{piece_index}"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            exc.close()
            print(f"Failed to download piece {piece_index} for {bundle_id} from {host}:{port}: HTTP {exc.code}")
            return None
        except (urllib.error.URLError, OSError) as exc:
            print(f"Failed to download piece {piece_index} for {bundle_id} from {host}:{port}: {exc}")
            return None

    def download_bundle(
        self,
        host: str,
        port: int,
        bundle_id: str,
        out_dir: str | Path | None = None,
        verify: bool = True,
        *,
        use_tls: bool = False,
    ) -> Path | None:
        manifest = self.download_manifest(host, port, bundle_id, use_tls=use_tls)
        if manifest is None:
            return None
        bundle_out = Path(out_dir or self.bundles_dir) / bundle_id
        pieces_out = bundle_out / "pieces"
        pieces_out.mkdir(parents=True, exist_ok=True)
        pieces = manifest.get("pieces", [])
        total = len(pieces)
        for idx, piece_info in enumerate(pieces):
            if not isinstance(piece_info, dict):
                print(f"Invalid piece metadata at index {idx}: expected object")
                import shutil
                shutil.rmtree(bundle_out, ignore_errors=True)
                return None
            file_name = str(piece_info.get("file_name", f"piece_{idx:06d}.bin")).strip()
            data = self.download_piece(host, port, bundle_id, idx, use_tls=use_tls)
            if data is None:
                print(f"Aborting download at piece {idx}/{total}")
                import shutil
                shutil.rmtree(bundle_out, ignore_errors=True)
                return None
            piece_out = _safe_join(pieces_out, file_name)
            if piece_out is None:
                print(f"Invalid piece path at index {idx}: {file_name}")
                import shutil
                shutil.rmtree(bundle_out, ignore_errors=True)
                return None
            piece_out.parent.mkdir(parents=True, exist_ok=True)
            piece_out.write_bytes(data)
            if (idx + 1) % 10 == 0 or idx == total - 1:
                print(f"  Downloaded piece {idx + 1}/{total}")

        manifest_path = bundle_out / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Downloaded bundle {bundle_id} ({total} pieces, {manifest.get('total_bytes', 0)} bytes) to {bundle_out}")

        if verify:
            from src.torickle import verify_manifest
            report = verify_manifest(manifest_path=str(manifest_path))
            if not report.get("valid", False):
                print(f"Verification failed for downloaded bundle: {'; '.join(report.get('errors', []))}")
                import shutil
                shutil.rmtree(bundle_out, ignore_errors=True)
                return None
            print(f"Verification passed (merkle_root={report.get('merkle_root', '')[:16]}...)")

        self._register_bundle_dir(bundle_out)
        return bundle_out

    def reassemble_delta(self, bundle_id: str, out_path: str | Path, strict_verify: bool = True) -> dict[str, Any] | None:
        info = self._bundles.get(bundle_id)
        if info is None:
            info_dir = self._bundle_dir(bundle_id)
            if info_dir is None:
                print(f"Bundle {bundle_id} not found locally")
                return None
        from src.torickle import reassemble_manifest
        manifest_path = (info.bundle_dir if info else info_dir) / "manifest.json"
        try:
            result = reassemble_manifest(
                manifest_path=str(manifest_path),
                out_path=str(out_path),
                strict_verify=strict_verify,
            )
            print(f"Reassembled delta to {out_path}: {result.get('tensor_count', 0)} tensors, verified={result.get('verified', False)}")
            return result
        except Exception as exc:
            print(f"Failed to reassemble bundle {bundle_id}: {exc}")
            return None


def _parse_addr(addr: str) -> tuple[str, int]:
    if ":" in addr:
        host, port_str = addr.rsplit(":", 1)
        return host.strip(), int(port_str)
    return addr.strip(), DEFAULT_SWARM_PORT


def _join_via_bootstrap(peer_discovery: PeerDiscovery, bootstrap_addrs: list[str]):
    """Actually contact each configured bootstrap peer's /torickle/v1/peers
    endpoint and merge what it reports into the local peer store.
    add_bootstrap() alone only records the address; this is the network call
    that makes it mean something."""
    for addr in bootstrap_addrs:
        host, port = _parse_addr(addr)
        peer_discovery.add_bootstrap(host, port)
    if not bootstrap_addrs:
        return
    bootstrap_pairs = [_parse_addr(a) for a in bootstrap_addrs]
    for body in bootstrap_fetch(bootstrap_pairs, "/torickle/v1/peers"):
        for raw_peer in body.get("peers", []):
            try:
                peer_discovery.store.add_peer(PeerInfo.from_dict(raw_peer))
            except (KeyError, ValueError, TypeError):
                continue
    for body in bootstrap_fetch(bootstrap_pairs, "/torickle/v1/dht/bundles"):
        for raw_ann in body.get("announcements", []):
            try:
                ann = BundleAnnouncement.from_dict(raw_ann)
            except (KeyError, ValueError, TypeError):
                continue
            dht_value = json.dumps(ann.to_dict(), sort_keys=True).encode("utf-8")
            dht_keys = {ann.dht_key(), _all_bundles_dht_key()}
            if ann.model_hash:
                dht_keys.add(_model_dht_key(ann.model_hash))
            peer_id_bytes = bytes.fromhex(ann.peer_id) if len(ann.peer_id) == 40 else ann.peer_id.encode("utf-8")[:20]
            for key in dht_keys:
                peer_discovery.store.store_value(key, dht_value, peer_id_bytes, ANNOUNCE_TTL)


def main():
    parser = argparse.ArgumentParser(description="Ickle swarm node — P2P torickle bundle exchange")
    sub = parser.add_subparsers(dest="command", required=True)

    start_parser = sub.add_parser("start", help="Start a swarm node (serve bundles + participate in DHT)")
    start_parser.add_argument("--port", type=int, default=DEFAULT_SWARM_PORT, help="Swarm port")
    start_parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    start_parser.add_argument("--external-host", default="", help="Externally reachable host/IP")
    start_parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Torickle data directory")
    start_parser.add_argument("--identity", default=DEFAULT_IDENTITY_PATH, help="Swarm identity file")
    start_parser.add_argument("--label", default="", help="Label for this peer")
    start_parser.add_argument("--bootstrap", action="append", default=[], help="Bootstrap peer host:port (repeatable)")
    start_parser.add_argument("--model-hash", default="", help="Model hash to announce after import")
    start_parser.add_argument("--daemon", action="store_true", help="Run in background (no stdin loop)")
    start_parser.add_argument(
        "--nat-traversal", action="store_true",
        help="Use STUN to discover this machine's real public IP (instead of "
             "guessing from a local interface) and ask the router via UPnP to "
             "forward --port automatically. Off by default since it makes "
             "outbound network calls and a router change; safe to enable for "
             "any node meant to be reachable from outside your LAN.",
    )

    import_parser = sub.add_parser("import", help="Import a torickle bundle directory")
    import_parser.add_argument("bundle_path", help="Path to torickle bundle directory or manifest.json")
    import_parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    import_parser.add_argument("--identity", default=DEFAULT_IDENTITY_PATH)

    announce_parser = sub.add_parser("announce", help="Announce a bundle to the swarm")
    announce_parser.add_argument("bundle_id", help="Bundle ID to announce")
    announce_parser.add_argument("--model-hash", default="", help="Model hash for discovery")
    announce_parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    announce_parser.add_argument("--identity", default=DEFAULT_IDENTITY_PATH)

    find_parser = sub.add_parser("find", help="Find bundles on the swarm")
    find_parser.add_argument("--model-hash", default="", help="Filter by model hash")
    find_parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    find_parser.add_argument("--identity", default=DEFAULT_IDENTITY_PATH)
    find_parser.add_argument(
        "--bootstrap", action="append", default=[],
        help="Bootstrap peer host:port (repeatable) -- without this, a one-shot `find` has no way to "
             "learn about peers it hasn't seen before, since it never actually joins the swarm.",
    )

    query_parser = sub.add_parser("query-peer", help="Query a peer for its bundle list")
    query_parser.add_argument("peer_addr", help="Peer address host:port")
    query_parser.add_argument("--tls", action="store_true", help="Use HTTPS when connecting to peer")

    pull_parser = sub.add_parser("pull", help="Download a bundle from a peer")
    pull_parser.add_argument("peer_addr", help="Peer address host:port")
    pull_parser.add_argument("bundle_id", help="Bundle ID to download")
    pull_parser.add_argument("--out-dir", default="", help="Output directory (defaults to data-dir/bundles)")
    pull_parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    pull_parser.add_argument("--identity", default=DEFAULT_IDENTITY_PATH)
    pull_parser.add_argument("--no-verify", action="store_true", help="Skip verification after download")
    pull_parser.add_argument("--tls", action="store_true", help="Use HTTPS when connecting to peer")

    list_parser = sub.add_parser("list", help="List locally stored bundles")
    list_parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    list_parser.add_argument("--identity", default=DEFAULT_IDENTITY_PATH)

    reassemble_parser = sub.add_parser("reassemble", help="Reassemble a bundle into a delta .pt file")
    reassemble_parser.add_argument("bundle_id", help="Bundle ID to reassemble")
    reassemble_parser.add_argument("--out", default="", help="Output .pt path")
    reassemble_parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    reassemble_parser.add_argument("--identity", default=DEFAULT_IDENTITY_PATH)
    reassemble_parser.add_argument("--no-strict", action="store_true", help="Allow reassembly even if verification fails")

    args = parser.parse_args()

    identity_path = Path(getattr(args, "identity", DEFAULT_IDENTITY_PATH))
    identity = ensure_identity(identity_path, label=getattr(args, "label", ""))
    data_dir = getattr(args, "data_dir", DEFAULT_DATA_DIR)

    if args.command == "start":
        external = args.external_host or ""
        node = SwarmNode(
            identity=identity,
            data_dir=data_dir,
            host=args.host,
            port=args.port,
            external_host=external or None,
        )
        _join_via_bootstrap(node.peer_discovery, args.bootstrap)
        node.start(attempt_nat_traversal=bool(getattr(args, "nat_traversal", False)))
        if args.model_hash:
            for bundle_id in list(node.bundles.keys()):
                node.announce_bundle(bundle_id, model_hash=args.model_hash)
        print("Swarm node running. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
            node.stop()

    elif args.command == "import":
        node = SwarmNode(identity=identity, data_dir=data_dir)
        bundle_id = node.import_bundle(args.bundle_path)
        if bundle_id:
            print(json.dumps({"bundle_id": bundle_id, "status": "imported"}, indent=2))

    elif args.command == "announce":
        node = SwarmNode(identity=identity, data_dir=data_dir)
        node._scan_local_bundles()
        ann = node.announce_bundle(args.bundle_id, model_hash=args.model_hash)
        if ann:
            print(json.dumps(ann.to_dict(), indent=2, ensure_ascii=False))

    elif args.command == "find":
        node = SwarmNode(identity=identity, data_dir=data_dir)
        _join_via_bootstrap(node.peer_discovery, getattr(args, "bootstrap", []))
        results = node.find_bundles(model_hash=args.model_hash)
        if results:
            print(f"Found {len(results)} bundle(s):")
            for ann in results:
                print(f"  {ann.bundle_id}  model={ann.model_hash}  pieces={ann.piece_count}  "
                      f"peer={ann.host}:{ann.port}  age={int(time.time()-ann.timestamp)}s")
        else:
            print("No bundles found.")

    elif args.command == "query-peer":
        host, port = _parse_addr(args.peer_addr)
        node = SwarmNode(identity=identity, data_dir=data_dir)
        use_tls = getattr(args, "tls", False)
        bundles = node.query_peer_bundles(host, port, use_tls=use_tls)
        print(json.dumps(bundles, indent=2, ensure_ascii=False))

    elif args.command == "pull":
        host, port = _parse_addr(args.peer_addr)
        node = SwarmNode(identity=identity, data_dir=data_dir)
        use_tls = getattr(args, "tls", False)
        out_dir = Path(args.out_dir) if args.out_dir else None
        result = node.download_bundle(
            host, port, args.bundle_id,
            out_dir=out_dir,
            verify=not args.no_verify,
            use_tls=use_tls,
        )
        if result:
            print(f"Bundle saved to {result}")

    elif args.command == "list":
        node = SwarmNode(identity=identity, data_dir=data_dir)
        node._scan_local_bundles()
        if node.bundles:
            print(f"Local bundles ({len(node.bundles)}):")
            for bundle_id, info in sorted(node.bundles.items()):
                m = info.manifest
                pieces = m.get("piece_count", 0)
                total = m.get("total_bytes", 0)
                created = m.get("created_at_utc", "unknown")
                print(f"  {bundle_id}  pieces={pieces}  bytes={total}  created={created}")
        else:
            print("No local bundles found.")

    elif args.command == "reassemble":
        node = SwarmNode(identity=identity, data_dir=data_dir)
        node._scan_local_bundles()
        out = args.out or f"data/torickle/{args.bundle_id}_delta.pt"
        result = node.reassemble_delta(args.bundle_id, out_path=out, strict_verify=not args.no_strict)
        if result:
            print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
