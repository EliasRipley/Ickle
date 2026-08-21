"""P2P inference sharing: peers volunteer spare compute to answer other users'
prompts, the same way they already volunteer compute to train.

This reuses the existing Kademlia-style `PeerDiscovery` DHT (see
`src/federated/peer_discovery.py` and `src/federated/swarm.py`) as the
announcement/rendezvous layer, but defines its own signed message types because
inference offers and responses must be verifiable by peers that have never
talked to the signer before. `SwarmIdentity` (src/federated/identity.py) only
supports HMAC signatures that the signer's own secret can verify, which does not
work for "any peer can check this claim" — hence the Ed25519 identity in
`src/federated/keys.py`.

Protocol, informally:
  1. A peer with spare capacity and a loaded model starts an InferenceNode
     (`infer serve`), which serves HTTP on /infer/v1/* and periodically
     announces a signed InferenceOffer into the DHT.
  2. A peer wanting an answer (`infer ask`) looks up offers for a model hash,
     picks one, and POSTs a prompt to it directly (no coordinator in the loop).
  3. The response is signed by the serving peer's Ed25519 key so the requester
     can at least confirm it came from the peer it thinks it talked to.
  4. Both sides record the exchange in their local ContributionLedger, which is
     the seed:peer ratio surfaced by `infer report`.

This is intentionally a thin, local-first layer: no payment, no SLA, no global
reputation broadcast yet. Peer selection today is "first offer that responds";
ledger-informed peer ranking is future work (see docs/INFERENCE_SHARING.md).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from src.federated.contribution_ledger import LedgerStore
from src.federated.keys import EdIdentity, ensure_ed_identity, verify_payload
from src.federated.node_base import detect_local_ip
from src.federated.swarm import PEER_CLEANUP_INTERVAL_SECONDS
from src.federated.nat_traversal import stun_get_external_address, upnp_add_port_mapping, upnp_remove_port_mapping
from src.federated.peer_discovery import PeerDiscovery, bootstrap_fetch

DEFAULT_INFER_PORT = 8791
DEFAULT_DATA_DIR = "data/torickle"
DEFAULT_IDENTITY_PATH = "data/torickle/ed25519_identity.json"
OFFER_TTL = 900  # offers expire faster than torickle bundle announcements: capacity changes minute to minute
REQUEST_TIMEOUT = 120
MAX_REQUEST_BYTES = 262_144
MAX_PROMPT_CHARS = 8_000


def _offer_dht_key(model_hash: str) -> bytes:
    return hashlib.sha256(f"ickle:infer:offer:{model_hash}".encode()).digest()[:20]


def _all_offers_dht_key() -> bytes:
    return hashlib.sha256(b"ickle:infer:offer:all").digest()[:20]


@dataclass
class InferenceOffer:
    peer_id: str
    pubkey_hex: str
    model_hash: str
    host: str
    port: int
    capacity: int
    context_window: int
    label: str
    timestamp: float
    signature: str = ""
    use_tls: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "peer_id": self.peer_id,
            "pubkey_hex": self.pubkey_hex,
            "model_hash": self.model_hash,
            "host": self.host,
            "port": self.port,
            "capacity": self.capacity,
            "context_window": self.context_window,
            "label": self.label,
            "use_tls": self.use_tls,
            "timestamp": self.timestamp,
            "signature": self.signature,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "InferenceOffer":
        return InferenceOffer(
            peer_id=str(d["peer_id"]),
            pubkey_hex=str(d["pubkey_hex"]),
            model_hash=str(d.get("model_hash", "")),
            host=str(d["host"]),
            port=int(d["port"]),
            capacity=int(d.get("capacity", 1)),
            context_window=int(d.get("context_window", 0)),
            label=str(d.get("label", "")),
            timestamp=float(d.get("timestamp", 0)),
            signature=str(d.get("signature", "")),
            use_tls=bool(d.get("use_tls", False)),
        )

    def dht_key(self) -> bytes:
        return _offer_dht_key(self.model_hash)

    def sign(self, identity: EdIdentity):
        payload = self.to_dict()
        payload.pop("signature", None)
        self.signature = identity.sign(payload)

    def verify(self) -> bool:
        if peer_id_mismatch(self.peer_id, self.pubkey_hex):
            return False
        payload = self.to_dict()
        payload.pop("signature", None)
        return verify_payload(self.pubkey_hex, payload, self.signature)


def peer_id_mismatch(peer_id: str, pubkey_hex: str) -> bool:
    from src.federated.keys import peer_id_from_pubkey

    try:
        return peer_id_from_pubkey(pubkey_hex) != peer_id
    except ValueError:
        return True


GenerateFn = Callable[[str, int, float, int], str]
"""(prompt, max_new_tokens, temperature, top_k) -> response text"""


class InferenceRequestHandler(BaseHTTPRequestHandler):
    server_version = "IckleInfer/0.1"

    @property
    def node(self) -> "InferenceNode":
        return self.server.inference_node  # type: ignore[attr-defined]

    def _json(self, status: int, payload: Any):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _segments(self) -> list[str]:
        return [s for s in urlparse(self.path).path.strip("/").split("/") if s]

    def do_GET(self):  # noqa: N802
        segments = self._segments()
        if segments[:2] != ["infer", "v1"]:
            return self._json(404, {"error": "Expected /infer/v1/..."})
        if segments[2:] == ["offers"]:
            return self._handle_list_offers()
        return self._json(200, {
            "service": "ickle-inference",
            "peer_id": self.node.identity.peer_id,
            "model_hash": self.node.model_hash,
            "capacity": self.node.capacity,
            "in_flight": self.node.in_flight,
            "accepting": self.node.in_flight < self.node.capacity,
        })

    def _handle_list_offers(self):
        """All inference offers this node currently knows about via its local
        DHT store -- including offers announced by other peers, not just its
        own. This is what a `find`/`ask` invocation needs to pull via
        --bootstrap to learn about peers it hasn't seen before, since a
        one-shot CLI process otherwise never joins the swarm at all."""
        offers = find_offers(self.node.peer_discovery)
        self._json(200, {
            "peer_id": self.node.identity.peer_id,
            "offers": [o.to_dict() for o in offers],
        })

    def do_POST(self):  # noqa: N802
        segments = self._segments()
        if segments[:3] != ["infer", "v1", "generate"]:
            return self._json(404, {"error": "Not found"})
        return self._handle_generate()

    def _handle_generate(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length <= 0 or content_length > MAX_REQUEST_BYTES:
                return self._json(400, {"error": "Invalid Content-Length"})
            raw = self.rfile.read(content_length)
            data = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, ValueError) as exc:
            return self._json(400, {"error": f"Invalid JSON: {exc}"})

        prompt = str(data.get("prompt", ""))
        if not prompt.strip():
            return self._json(400, {"error": "Empty prompt"})
        if len(prompt) > MAX_PROMPT_CHARS:
            return self._json(400, {"error": f"Prompt exceeds {MAX_PROMPT_CHARS} chars"})

        request_id = str(data.get("request_id", ""))[:128]
        max_new_tokens = max(1, min(int(data.get("max_new_tokens", 200) or 200), 1024))
        temperature = float(data.get("temperature", 0.8) or 0.8)
        top_k = max(1, int(data.get("top_k", 40) or 40))

        if not self.node.try_acquire_slot():
            return self._json(503, {"error": "At capacity", "capacity": self.node.capacity})
        try:
            started = time.time()
            try:
                response_text = self.node.generate_fn(prompt, max_new_tokens, temperature, top_k)
            except Exception as exc:  # noqa: BLE001
                return self._json(500, {"error": f"Generation failed: {exc}"})
            latency_ms = int((time.time() - started) * 1000)
        finally:
            self.node.release_slot()

        approx_tokens = max(1, len(response_text) // 4)
        self.node.ledger.record_inference_served(approx_tokens)

        payload = {
            "request_id": request_id,
            "peer_id": self.node.identity.peer_id,
            "response": response_text,
            "latency_ms": latency_ms,
        }
        payload["signature"] = self.node.identity.sign(payload)
        return self._json(200, payload)

    def log_message(self, fmt: str, *args):
        pass


class InferenceNode:
    def __init__(
        self,
        identity: EdIdentity,
        generate_fn: GenerateFn,
        model_hash: str = "",
        data_dir: str | Path = DEFAULT_DATA_DIR,
        peer_discovery: PeerDiscovery | None = None,
        ledger: LedgerStore | None = None,
        host: str = "0.0.0.0",
        port: int = DEFAULT_INFER_PORT,
        external_host: str | None = None,
        capacity: int = 2,
        context_window: int = 0,
        label: str = "",
    ):
        self.identity = identity
        self.generate_fn = generate_fn
        self.model_hash = model_hash
        self.data_dir = Path(data_dir)
        self.peer_discovery = peer_discovery or PeerDiscovery()
        self.ledger = ledger or LedgerStore(self.data_dir / "contribution_ledger.json")
        self.host = host
        self.port = port
        self._external_host_explicit = bool(external_host)
        self.external_host = external_host or detect_local_ip()
        self._port_mapped = False
        self.capacity = max(1, capacity)
        self.context_window = context_window
        self.label = label
        self._server: ThreadingHTTPServer | None = None
        self._server_thread: threading.Thread | None = None
        self._running = False
        self._slot_lock = threading.Lock()
        self._in_flight = 0
        self._cleanup_thread: threading.Thread | None = None
        self._cleanup_stop = threading.Event()

    @property
    def in_flight(self) -> int:
        with self._slot_lock:
            return self._in_flight

    def try_acquire_slot(self) -> bool:
        with self._slot_lock:
            if self._in_flight >= self.capacity:
                return False
            self._in_flight += 1
            return True

    def release_slot(self):
        with self._slot_lock:
            self._in_flight = max(0, self._in_flight - 1)

    def _cleanup_loop(self):
        """Same rationale as SwarmNode._cleanup_loop (swarm.py): this node
        has its own independent PeerDiscovery/PeerStore, so it needs its own
        periodic prune too."""
        while not self._cleanup_stop.wait(PEER_CLEANUP_INTERVAL_SECONDS):
            try:
                self.peer_discovery.store.cleanup()
            except Exception as exc:  # noqa: BLE001
                print(f"  Peer store cleanup failed (non-fatal): {exc}")

    def start(self, *, attempt_nat_traversal: bool = False):
        if self._running:
            return
        self._server = ThreadingHTTPServer((self.host, self.port), InferenceRequestHandler)
        self._server.inference_node = self  # type: ignore[attr-defined]
        self._server_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._server_thread.start()
        self._running = True
        self.peer_discovery.node_id = bytes.fromhex(self.identity.peer_id)

        if attempt_nat_traversal:
            self._attempt_nat_traversal()

        self._cleanup_stop.clear()
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()

        print(f"Inference node started on {self.host}:{self.port} (external: {self.external_host}:{self.port})")
        print(f"  Peer ID: {self.identity.peer_id}")
        print(f"  Model hash: {self.model_hash or '(unset)'}  capacity: {self.capacity}")

    def _attempt_nat_traversal(self):
        """Best-effort, never fatal -- see SwarmNode._attempt_nat_traversal
        (swarm.py) for the rationale; this is the same mechanism applied to
        the inference-serving node."""
        if not self._external_host_explicit:
            stun_result = stun_get_external_address()
            if stun_result:
                self.external_host = stun_result[0]
                print(f"  STUN: public address is {self.external_host} (used as external_host)")
        try:
            self._port_mapped = upnp_add_port_mapping(self.port, description="Ickle Inference Node")
        except Exception as exc:  # noqa: BLE001
            self._port_mapped = False
            print(f"  UPnP port forwarding raised an unexpected error (not just 'unsupported'): {exc}")
        print(f"  UPnP port forwarding: {'succeeded' if self._port_mapped else 'unavailable'}")

    def stop(self):
        self._running = False
        self._cleanup_stop.set()
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)
            self._cleanup_thread = None
        if getattr(self, "_port_mapped", False):
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

    @property
    def bound_port(self) -> int:
        """Actual listening port (resolves `port=0` to the OS-assigned port)."""
        if self._server is not None:
            return self._server.server_port
        return self.port

    def build_offer(self) -> InferenceOffer:
        offer = InferenceOffer(
            peer_id=self.identity.peer_id,
            pubkey_hex=self.identity.pubkey_hex,
            model_hash=self.model_hash,
            host=self.external_host,
            port=self.bound_port,
            capacity=self.capacity,
            context_window=self.context_window,
            label=self.label,
            timestamp=time.time(),
        )
        offer.sign(self.identity)
        return offer

    def announce(self) -> InferenceOffer:
        offer = self.build_offer()
        dht_value = json.dumps(offer.to_dict(), sort_keys=True).encode("utf-8")
        keys = {_all_offers_dht_key()}
        if offer.model_hash:
            keys.add(_offer_dht_key(offer.model_hash))
        publisher = bytes.fromhex(offer.peer_id) if len(offer.peer_id) == 40 else offer.peer_id.encode()[:20]
        for key in keys:
            self.peer_discovery.store.store_value(key, dht_value, publisher, OFFER_TTL)
        return offer

    def announce_loop(self, interval: float = 300.0, stop_event: threading.Event | None = None):
        """Re-announce periodically so the offer doesn't expire out of the DHT
        while this node is still willing to serve."""
        stop_event = stop_event or threading.Event()
        while not stop_event.is_set():
            self.announce()
            stop_event.wait(interval)


def find_offers(peer_discovery: PeerDiscovery, model_hash: str = "", max_age: float = OFFER_TTL) -> list[InferenceOffer]:
    keys = [_all_offers_dht_key()]
    if model_hash:
        keys.insert(0, _offer_dht_key(model_hash))
    now = time.time()
    dedup: dict[str, InferenceOffer] = {}
    for key in keys:
        for raw in peer_discovery.store.find_value(key):
            try:
                offer = InferenceOffer.from_dict(json.loads(raw.decode("utf-8")))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
            if now - offer.timestamp > max_age:
                continue
            if model_hash and offer.model_hash != model_hash:
                continue
            if not offer.verify():
                continue
            prev = dedup.get(offer.peer_id)
            if prev is None or offer.timestamp > prev.timestamp:
                dedup[offer.peer_id] = offer
    results = list(dedup.values())
    results.sort(key=lambda o: o.timestamp, reverse=True)
    return results


def request_inference(
    offer: InferenceOffer,
    prompt: str,
    *,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 40,
    requester_peer_id: str = "",
    ledger: LedgerStore | None = None,
    timeout: float = REQUEST_TIMEOUT,
) -> dict[str, Any] | None:
    """Ask a peer's InferenceNode to run a prompt. Returns the response payload
    (with a verified=bool key) or None on a transport-level failure."""
    scheme = "https" if offer.use_tls else "http"
    url = f"{scheme}://{offer.host}:{offer.port}/infer/v1/generate"
    body = json.dumps({
        "request_id": hashlib.sha256(f"{requester_peer_id}:{time.time()}:{prompt[:64]}".encode()).hexdigest()[:24],
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "requester_peer_id": requester_peer_id,
    }).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode("utf-8"))
        except Exception:  # noqa: BLE001
            detail = {"error": str(exc)}
        exc.close()
        detail["http_status"] = exc.code
        return detail
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        print(f"Inference request to {offer.host}:{offer.port} failed: {exc}")
        return None

    sig_payload = {k: v for k, v in payload.items() if k != "signature"}
    payload["verified"] = verify_payload(offer.pubkey_hex, sig_payload, str(payload.get("signature", "")))
    if payload["verified"] and ledger is not None:
        approx_tokens = max(1, len(str(payload.get("response", ""))) // 4)
        ledger.record_inference_consumed(approx_tokens)
    return payload


def make_local_generate_fn(model_path: str) -> GenerateFn:
    """Build a GenerateFn backed by a real local Ickle checkpoint. Imports torch
    and the model bundle lazily so tests can inject a stub generate_fn instead
    of paying the import/load cost."""
    import torch

    from src.ilm_chat import SystemLimits, _generate_model_response, _load_model_bundle
    from src.ilm_profile import apply_cpu_thread_budget

    threads = max(1, (torch.get_num_threads() or 4))
    apply_cpu_thread_budget(threads)
    model, tokenizer = _load_model_bundle(model_path)
    limits = SystemLimits(max_new_tokens=1024, torch_threads=threads)

    def _generate(prompt: str, max_new_tokens: int, temperature: float, top_k: int) -> str:
        args = argparse.Namespace(
            model=model_path,
            max_new=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            speculative=False,
        )
        return _generate_model_response(model, tokenizer, prompt, args, limits)

    return _generate


def _parse_addr(addr: str) -> tuple[str, int]:
    if ":" in addr:
        host, port_str = addr.rsplit(":", 1)
        return host.strip(), int(port_str)
    return addr.strip(), DEFAULT_INFER_PORT


def _join_via_bootstrap(peer_discovery: PeerDiscovery, bootstrap_addrs: list[str]):
    """Actually contact each configured bootstrap peer's /infer/v1/offers
    endpoint and store what it reports into the local DHT store, so a
    subsequent find_offers() call (which only ever reads the local store)
    picks them up. add_bootstrap() alone only records the address."""
    if not bootstrap_addrs:
        return
    bootstrap_pairs = [_parse_addr(a) for a in bootstrap_addrs]
    for body in bootstrap_fetch(bootstrap_pairs, "/infer/v1/offers"):
        for raw_offer in body.get("offers", []):
            try:
                offer = InferenceOffer.from_dict(raw_offer)
            except (KeyError, ValueError, TypeError):
                continue
            dht_value = json.dumps(offer.to_dict(), sort_keys=True).encode("utf-8")
            keys = {_all_offers_dht_key()}
            if offer.model_hash:
                keys.add(_offer_dht_key(offer.model_hash))
            publisher = bytes.fromhex(offer.peer_id) if len(offer.peer_id) == 40 else offer.peer_id.encode()[:20]
            for key in keys:
                peer_discovery.store.store_value(key, dht_value, publisher, OFFER_TTL)


def main():
    parser = argparse.ArgumentParser(description="Ickle inference swarm — P2P prompt answering")
    sub = parser.add_subparsers(dest="command", required=True)

    serve_p = sub.add_parser("serve", help="Host inference for a local model and announce it to the swarm")
    serve_p.add_argument("--model", required=True, help="Path to a trained Ickle checkpoint")
    serve_p.add_argument("--model-hash", default="", help="Model hash for discovery (defaults to sha256 of --model path contents prefix)")
    serve_p.add_argument("--host", default="0.0.0.0")
    serve_p.add_argument("--port", type=int, default=DEFAULT_INFER_PORT)
    serve_p.add_argument("--external-host", default="")
    serve_p.add_argument("--capacity", type=int, default=2, help="Max concurrent inference requests to accept")
    serve_p.add_argument("--label", default="")
    serve_p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    serve_p.add_argument("--identity", default=DEFAULT_IDENTITY_PATH)
    serve_p.add_argument("--bootstrap", action="append", default=[], help="Bootstrap peer host:port (repeatable)")
    serve_p.add_argument("--announce-interval", type=float, default=300.0)

    find_p = sub.add_parser("find", help="Find inference offers on the swarm")
    find_p.add_argument("--model-hash", default="")
    find_p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    find_p.add_argument("--identity", default=DEFAULT_IDENTITY_PATH)
    find_p.add_argument(
        "--bootstrap", action="append", default=[],
        help="Bootstrap peer host:port (repeatable) -- without this, `find` has no way to learn "
             "about offers it hasn't seen before, since a one-shot process never joins the swarm.",
    )

    ask_p = sub.add_parser("ask", help="Send a one-off prompt to a peer offering inference")
    ask_p.add_argument("prompt", help="Prompt text")
    ask_p.add_argument("--model-hash", default="", help="Restrict to peers serving this model")
    ask_p.add_argument("--peer", default="", help="Explicit peer host:port instead of discovery")
    ask_p.add_argument("--max-new-tokens", type=int, default=200)
    ask_p.add_argument("--temperature", type=float, default=0.8)
    ask_p.add_argument("--top-k", type=int, default=40)
    ask_p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ask_p.add_argument(
        "--bootstrap", action="append", default=[],
        help="Bootstrap peer host:port (repeatable), e.g. the peer you're asking. Same reason as `find`.",
    )
    ask_p.add_argument("--identity", default=DEFAULT_IDENTITY_PATH)
    ask_p.add_argument(
        "--trust-store", default="",
        help="Path to a codistill PeerTrustStore (default data/codistill/peer_trust.json). When set "
             "(or left at the default and the file exists), offers are ranked by locally-observed "
             "per-domain teaching trust instead of picking the first offer found -- addresses the "
             "documented 'peer selection is first offer that responds' gap for plain inference asks.",
    )

    report_p = sub.add_parser("report", help="Show local seed:peer contribution ratio")
    report_p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)

    args = parser.parse_args()

    if args.command == "report":
        ledger = LedgerStore(Path(args.data_dir) / "contribution_ledger.json")
        print(json.dumps(ledger.summary(), indent=2, ensure_ascii=False))
        return

    identity = ensure_ed_identity(Path(args.identity), label=getattr(args, "label", ""))
    ledger = LedgerStore(Path(args.data_dir) / "contribution_ledger.json")

    if args.command == "serve":
        generate_fn = make_local_generate_fn(args.model)
        model_hash = args.model_hash or hashlib.sha256(Path(args.model).read_bytes()[:1_000_000]).hexdigest()[:16]
        peer_discovery = PeerDiscovery()
        for bootstrap_addr in args.bootstrap:
            host, port = _parse_addr(bootstrap_addr)
            peer_discovery.add_bootstrap(host, port)
        _join_via_bootstrap(peer_discovery, args.bootstrap)
        node = InferenceNode(
            identity=identity,
            generate_fn=generate_fn,
            model_hash=model_hash,
            data_dir=args.data_dir,
            peer_discovery=peer_discovery,
            ledger=ledger,
            host=args.host,
            port=args.port,
            external_host=args.external_host or None,
            capacity=args.capacity,
            label=args.label,
        )
        node.start()
        node.announce()
        stop_event = threading.Event()
        announce_thread = threading.Thread(
            target=node.announce_loop, args=(args.announce_interval, stop_event), daemon=True
        )
        announce_thread.start()
        print("Inference node running. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
            stop_event.set()
            node.stop()

    elif args.command == "find":
        peer_discovery = PeerDiscovery()
        _join_via_bootstrap(peer_discovery, args.bootstrap)
        offers = find_offers(peer_discovery, model_hash=args.model_hash)
        if offers:
            print(f"Found {len(offers)} offer(s):")
            for o in offers:
                print(f"  {o.peer_id[:16]}...  model={o.model_hash}  capacity={o.capacity}  "
                      f"peer={o.host}:{o.port}  age={int(time.time()-o.timestamp)}s  label={o.label}")
        else:
            print("No inference offers found.")

    elif args.command == "ask":
        peer_discovery = PeerDiscovery()
        if args.peer:
            host, port = _parse_addr(args.peer)
            # An explicit --peer is an implicit bootstrap contact: we still only
            # trust its own *signed* offer (fetched below), never the bare
            # address alone, but the user shouldn't have to separately run
            # `find --bootstrap` first just to reach a peer they already named.
            _join_via_bootstrap(peer_discovery, list(args.bootstrap) + [args.peer])
            offers = find_offers(peer_discovery, model_hash=args.model_hash)
            offer = next((o for o in offers if o.host == host and o.port == port), None)
            if offer is None:
                print(f"No verified offer known for {host}:{port}. It may not be running `infer serve`, "
                      f"or hasn't announced yet.")
                return
        else:
            _join_via_bootstrap(peer_discovery, args.bootstrap)
            offers = find_offers(peer_discovery, model_hash=args.model_hash)
            if not offers:
                print("No inference offers found on the swarm.")
                return
            trust_store_path = Path(args.trust_store) if args.trust_store else Path("data/codistill/peer_trust.json")
            if trust_store_path.exists():
                from src.federated.codistill import PeerTrustStore, _rank_offers_by_trust, classify_domain

                trust_store = PeerTrustStore(trust_store_path)
                domain = classify_domain(args.prompt)
                offers = _rank_offers_by_trust(offers, trust_store, domain)
            offer = offers[0]

        result = request_inference(
            offer,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            requester_peer_id=identity.peer_id,
            ledger=ledger,
        )
        if result is None:
            print("Request failed (transport error).")
            return
        if not result.get("verified", False):
            print("WARNING: response signature could not be verified.")
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
