"""Public, trackerless discovery for the Ickle swarm.

Ickle's application protocol is not BitTorrent.  It does, however, need the
same answer to the zero-contact problem: how does a fresh installation find
another installation without an Ickle-operated directory?  This module uses
the BitTorrent Mainline DHT (BEP 5) as a narrow rendezvous layer:

* every public Ickle node looks up one stable, versioned 20-byte network key;
* reachable nodes announce their Ickle HTTP port under that key;
* returned endpoints are only candidates and must answer Ickle's own probe;
* model bundles and Commons events continue to use Ickle signatures.

The DHT never receives prompts, model data, identities, or review events.  It
only sees the same public IP and listening port that any trackerless swarm
needs to publish.  Nothing in this module runs until the user joins the public
swarm.

Protocol reference: https://www.bittorrent.org/beps/bep_0005.html
Node-ID hardening: https://www.bittorrent.org/beps/bep_0042.html
"""

from __future__ import annotations

import hashlib
import ipaddress
import secrets
import socket
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Callable


PUBLIC_SWARM_LABEL = b"ickle:public-swarm:v1"
PUBLIC_SWARM_INFOHASH = hashlib.sha1(PUBLIC_SWARM_LABEL).digest()
DEFAULT_DHT_ROUTERS = (
    ("dht.libtorrent.org", 25401),
    ("router.bittorrent.com", 6881),
    ("dht.transmissionbt.com", 6881),
    ("router.utorrent.com", 6881),
)
DEFAULT_REFRESH_SECONDS = 15 * 60
MAX_PACKET_BYTES = 65_535
MAX_BENCODE_DEPTH = 12
MAX_BENCODE_ITEMS = 2_048


class BencodeError(ValueError):
    """Raised for malformed or deliberately oversized bencoded packets."""


def bencode(value: Any) -> bytes:
    """Encode the small BEP 5 value subset without adding a dependency."""
    if isinstance(value, bool):
        value = int(value)
    if isinstance(value, int):
        return b"i" + str(value).encode("ascii") + b"e"
    if isinstance(value, str):
        value = value.encode("utf-8")
    if isinstance(value, bytes):
        return str(len(value)).encode("ascii") + b":" + value
    if isinstance(value, (list, tuple)):
        return b"l" + b"".join(bencode(item) for item in value) + b"e"
    if isinstance(value, dict):
        encoded_items: list[tuple[bytes, Any]] = []
        for key, item in value.items():
            key_bytes = key.encode("utf-8") if isinstance(key, str) else key
            if not isinstance(key_bytes, bytes):
                raise TypeError("bencode dictionary keys must be bytes or strings")
            encoded_items.append((key_bytes, item))
        encoded_items.sort(key=lambda pair: pair[0])
        return b"d" + b"".join(bencode(key) + bencode(item) for key, item in encoded_items) + b"e"
    raise TypeError(f"unsupported bencode type: {type(value).__name__}")


def bdecode(data: bytes) -> Any:
    """Decode one bounded bencoded value and reject trailing bytes."""
    if not isinstance(data, bytes) or len(data) > MAX_PACKET_BYTES:
        raise BencodeError("invalid packet size")
    index = 0
    item_count = 0

    def parse(depth: int = 0) -> Any:
        nonlocal index, item_count
        if depth > MAX_BENCODE_DEPTH:
            raise BencodeError("maximum nesting depth exceeded")
        item_count += 1
        if item_count > MAX_BENCODE_ITEMS or index >= len(data):
            raise BencodeError("truncated or oversized value")
        marker = data[index : index + 1]
        if marker == b"i":
            index += 1
            end = data.find(b"e", index)
            if end < 0:
                raise BencodeError("unterminated integer")
            raw = data[index:end]
            if not raw or (raw.startswith(b"-") and len(raw) == 1):
                raise BencodeError("invalid integer")
            if (raw.startswith(b"0") and len(raw) > 1) or raw.startswith(b"-0"):
                raise BencodeError("non-canonical integer")
            try:
                value = int(raw)
            except ValueError as exc:
                raise BencodeError("invalid integer") from exc
            index = end + 1
            return value
        if marker == b"l":
            index += 1
            items = []
            while index < len(data) and data[index : index + 1] != b"e":
                items.append(parse(depth + 1))
            if index >= len(data):
                raise BencodeError("unterminated list")
            index += 1
            return items
        if marker == b"d":
            index += 1
            result = {}
            previous_key: bytes | None = None
            while index < len(data) and data[index : index + 1] != b"e":
                key = parse(depth + 1)
                if not isinstance(key, bytes):
                    raise BencodeError("dictionary key is not bytes")
                if previous_key is not None and key < previous_key:
                    raise BencodeError("dictionary keys are not sorted")
                previous_key = key
                result[key] = parse(depth + 1)
            if index >= len(data):
                raise BencodeError("unterminated dictionary")
            index += 1
            return result
        if marker.isdigit():
            colon = data.find(b":", index)
            if colon < 0:
                raise BencodeError("missing byte-string separator")
            raw_length = data[index:colon]
            if not raw_length or (raw_length.startswith(b"0") and len(raw_length) > 1):
                raise BencodeError("invalid byte-string length")
            try:
                length = int(raw_length)
            except ValueError as exc:
                raise BencodeError("invalid byte-string length") from exc
            if length < 0 or length > MAX_PACKET_BYTES:
                raise BencodeError("byte string too large")
            start = colon + 1
            end = start + length
            if end > len(data):
                raise BencodeError("truncated byte string")
            index = end
            return data[start:end]
        raise BencodeError("unknown value marker")

    decoded = parse()
    if index != len(data):
        raise BencodeError("trailing packet data")
    return decoded


def _crc32c(data: bytes) -> int:
    """Small Castagnoli CRC implementation used by BEP 42 node IDs."""
    crc = 0xFFFFFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0x82F63B78 if crc & 1 else 0)
    return (~crc) & 0xFFFFFFFF


def bep42_node_id(public_ip: str, seed: bytes) -> bytes:
    """Return a deterministic BEP 42-compatible IPv4 node ID.

    DHT implementations that enforce BEP 42 reject arbitrary node IDs.  The
    stable Ickle peer identity supplies the random portion, while the public
    IP supplies the required prefix.  Non-public/invalid addresses fall back
    to a deterministic SHA-1 ID; callers can still use lenient DHT nodes.
    """
    material = seed if len(seed) >= 20 else hashlib.sha1(seed).digest()
    try:
        address = ipaddress.ip_address(public_ip)
    except ValueError:
        return hashlib.sha1(b"ickle:dht-node:" + material).digest()
    if address.version != 4 or not address.is_global:
        return hashlib.sha1(b"ickle:dht-node:" + material).digest()

    random_byte = material[-1]
    octets = bytearray(address.packed)
    mask = (0x03, 0x0F, 0x3F, 0xFF)
    for index, value in enumerate(mask):
        octets[index] &= value
    octets[0] |= (random_byte & 0x07) << 5
    checksum = _crc32c(bytes(octets))

    node_id = bytearray(material[:20])
    node_id[0] = (checksum >> 24) & 0xFF
    node_id[1] = (checksum >> 16) & 0xFF
    node_id[2] = ((checksum >> 8) & 0xF8) | (material[2] & 0x07)
    node_id[19] = random_byte
    return bytes(node_id)


def _is_public_ipv4(host: str) -> bool:
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.version == 4 and address.is_global


def decode_compact_nodes(raw: bytes) -> list[tuple[bytes, tuple[str, int]]]:
    """Decode BEP 5's 26-byte node records, ignoring malformed entries."""
    rows = []
    if not isinstance(raw, bytes):
        return rows
    for offset in range(0, len(raw) - (len(raw) % 26), 26):
        record = raw[offset : offset + 26]
        node_id = record[:20]
        host = socket.inet_ntoa(record[20:24])
        port = struct.unpack("!H", record[24:26])[0]
        if port and _is_public_ipv4(host):
            rows.append((node_id, (host, port)))
    return rows


def decode_compact_peers(values: Any) -> list[tuple[str, int]]:
    """Decode BEP 5 peer values (one or more 6-byte compact endpoints)."""
    if isinstance(values, bytes):
        values = [values]
    if not isinstance(values, list):
        return []
    rows: list[tuple[str, int]] = []
    for value in values:
        if not isinstance(value, bytes):
            continue
        for offset in range(0, len(value) - (len(value) % 6), 6):
            host = socket.inet_ntoa(value[offset : offset + 4])
            port = struct.unpack("!H", value[offset + 4 : offset + 6])[0]
            if port and _is_public_ipv4(host):
                rows.append((host, port))
    return rows


@dataclass(frozen=True)
class DHTContact:
    node_id: bytes | None
    address: tuple[str, int]


@dataclass
class DHTLookupResult:
    endpoints: list[tuple[str, int]]
    nodes_contacted: int
    nodes_responded: int
    announced_to: int
    duration_seconds: float
    error: str = ""


class MainlineDHTClient:
    """Bounded BEP 5 client used for discovery and periodic announcements."""

    def __init__(
        self,
        identity_seed: bytes,
        *,
        public_ip: str = "",
        routers: tuple[tuple[str, int], ...] = DEFAULT_DHT_ROUTERS,
        timeout: float = 0.8,
        max_nodes: int = 32,
        alpha: int = 4,
    ):
        self.identity_seed = identity_seed
        self.node_id = bep42_node_id(public_ip, identity_seed)
        self.routers = routers
        self.timeout = max(0.1, min(5.0, float(timeout)))
        self.max_nodes = max(4, min(128, int(max_nodes)))
        self.alpha = max(1, min(8, int(alpha)))

    def set_public_ip(self, public_ip: str):
        self.node_id = bep42_node_id(public_ip, self.identity_seed)

    def _resolve_routers(self) -> list[DHTContact]:
        contacts: list[DHTContact] = []
        seen: set[tuple[str, int]] = set()
        for host, port in self.routers:
            try:
                infos = socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_DGRAM)
            except OSError:
                continue
            for info in infos:
                address = (str(info[4][0]), int(info[4][1]))
                if address in seen or not _is_public_ipv4(address[0]):
                    continue
                seen.add(address)
                contacts.append(DHTContact(None, address))
        return contacts

    def _rpc(self, address: tuple[str, int], query: bytes, arguments: dict[bytes, Any]) -> dict[bytes, Any] | None:
        transaction = secrets.token_bytes(2)
        packet = bencode({b"a": arguments, b"q": query, b"t": transaction, b"y": b"q"})
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp:
                udp.settimeout(self.timeout)
                udp.sendto(packet, address)
                response, source = udp.recvfrom(MAX_PACKET_BYTES)
        except (OSError, socket.timeout):
            return None
        # A response with the right two-byte transaction ID but from another
        # endpoint is still spoofed traffic. KRPC is UDP, so enforce the node
        # we actually queried before decoding any returned contacts/tokens.
        if str(source[0]) != str(address[0]) or int(source[1]) != int(address[1]):
            return None
        try:
            decoded = bdecode(response)
        except BencodeError:
            return None
        if not isinstance(decoded, dict) or decoded.get(b"t") != transaction or decoded.get(b"y") != b"r":
            return None
        result = decoded.get(b"r")
        return result if isinstance(result, dict) else None

    def _get_peers(self, address: tuple[str, int]) -> dict[bytes, Any] | None:
        return self._rpc(
            address,
            b"get_peers",
            {b"id": self.node_id, b"info_hash": PUBLIC_SWARM_INFOHASH},
        )

    def _announce(self, address: tuple[str, int], token: bytes, port: int) -> bool:
        if not token or not 1 <= port <= 65_535:
            return False
        response = self._rpc(
            address,
            b"announce_peer",
            {
                b"id": self.node_id,
                b"implied_port": 0,
                b"info_hash": PUBLIC_SWARM_INFOHASH,
                b"port": port,
                b"token": token,
            },
        )
        return response is not None

    @staticmethod
    def _distance(contact: DHTContact) -> int:
        if contact.node_id is None:
            return -1
        return int.from_bytes(contact.node_id, "big") ^ int.from_bytes(PUBLIC_SWARM_INFOHASH, "big")

    def discover_and_announce(
        self,
        peer_port: int,
        *,
        should_stop: Callable[[], bool] | None = None,
    ) -> DHTLookupResult:
        started = time.monotonic()
        frontier = self._resolve_routers()
        if not frontier:
            return DHTLookupResult([], 0, 0, 0, time.monotonic() - started, "No DHT bootstrap router resolved.")

        seen: set[tuple[str, int]] = set()
        queued = {contact.address for contact in frontier}
        endpoints: set[tuple[str, int]] = set()
        tokens: list[tuple[DHTContact, bytes]] = []
        responded = 0

        with ThreadPoolExecutor(max_workers=self.alpha, thread_name_prefix="ickle-dht") as executor:
            while frontier and len(seen) < self.max_nodes:
                if should_stop and should_stop():
                    break
                frontier.sort(key=self._distance)
                batch: list[DHTContact] = []
                while frontier and len(batch) < self.alpha and len(seen) + len(batch) < self.max_nodes:
                    contact = frontier.pop(0)
                    if contact.address not in seen:
                        batch.append(contact)
                if not batch:
                    break
                for contact in batch:
                    seen.add(contact.address)
                futures = {executor.submit(self._get_peers, contact.address): contact for contact in batch}
                for future in as_completed(futures):
                    contact = futures[future]
                    try:
                        response = future.result()
                    except Exception:  # a hostile UDP response must not stop discovery
                        response = None
                    if not response:
                        continue
                    responded += 1
                    token = response.get(b"token")
                    if isinstance(token, bytes) and token:
                        tokens.append((contact, token))
                    endpoints.update(decode_compact_peers(response.get(b"values")))
                    for node_id, address in decode_compact_nodes(response.get(b"nodes", b"")):
                        if address in seen or address in queued:
                            continue
                        queued.add(address)
                        frontier.append(DHTContact(node_id, address))

        # The closest token-issuing nodes are the correct place to announce.
        tokens.sort(key=lambda pair: self._distance(pair[0]))
        announced_to = 0
        if 1 <= peer_port <= 65_535 and not (should_stop and should_stop()):
            with ThreadPoolExecutor(max_workers=self.alpha, thread_name_prefix="ickle-dht-announce") as executor:
                futures = [
                    executor.submit(self._announce, contact.address, token, peer_port)
                    for contact, token in tokens[:8]
                ]
                for future in as_completed(futures):
                    try:
                        announced_to += int(bool(future.result()))
                    except Exception:
                        continue

        error = "" if responded else "The public DHT did not answer. UDP may be blocked or the device may be offline."
        return DHTLookupResult(
            endpoints=sorted(endpoints),
            nodes_contacted=len(seen),
            nodes_responded=responded,
            announced_to=announced_to,
            duration_seconds=round(time.monotonic() - started, 3),
            error=error,
        )


class PublicDHTService:
    """Background refresh loop with a small, UI-safe status snapshot."""

    def __init__(
        self,
        client: MainlineDHTClient,
        *,
        refresh_seconds: int = DEFAULT_REFRESH_SECONDS,
    ):
        self.client = client
        self.refresh_seconds = max(30, int(refresh_seconds))
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._peer_port = 0
        self._on_candidates: Callable[[list[tuple[str, int]]], int] | None = None
        self._status: dict[str, Any] = {
            "enabled": False,
            "phase": "off",
            "network_key": PUBLIC_SWARM_INFOHASH.hex(),
            "nodes_contacted": 0,
            "nodes_responded": 0,
            "candidate_endpoints": 0,
            "verified_peers": 0,
            "announced_to": 0,
            "last_refresh_utc": "",
            "last_error": "",
        }

    def start(self, peer_port: int, on_candidates: Callable[[list[tuple[str, int]]], int]):
        if self._thread and self._thread.is_alive():
            return
        self._peer_port = int(peer_port)
        self._on_candidates = on_candidates
        self._stop.clear()
        self._wake.clear()
        self._update(enabled=True, phase="starting", last_error="")
        self._thread = threading.Thread(target=self._run, daemon=True, name="ickle-public-dht")
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=10)
            self._thread = None
        self._update(enabled=False, phase="off")

    def refresh_now(self):
        if self._thread and self._thread.is_alive():
            self._wake.set()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._status)

    def _update(self, **values: Any):
        with self._lock:
            self._status.update(values)

    def _run(self):
        while not self._stop.is_set():
            self._update(phase="searching")
            result = self.client.discover_and_announce(self._peer_port, should_stop=self._stop.is_set)
            verified = 0
            if not self._stop.is_set() and self._on_candidates:
                try:
                    verified = max(0, int(self._on_candidates(result.endpoints)))
                except Exception as exc:  # discovery is best-effort and isolated from the node
                    result.error = f"Peer verification failed: {exc}"
            if result.error:
                phase = "degraded"
            elif verified:
                phase = "connected"
            else:
                phase = "listening"
            self._update(
                phase=phase,
                nodes_contacted=result.nodes_contacted,
                nodes_responded=result.nodes_responded,
                candidate_endpoints=len(result.endpoints),
                verified_peers=verified,
                announced_to=result.announced_to,
                last_refresh_utc=datetime.now(timezone.utc).isoformat(),
                last_error=result.error,
                lookup_seconds=result.duration_seconds,
            )
            self._wake.wait(self.refresh_seconds)
            self._wake.clear()


def lookup_result_dict(result: DHTLookupResult) -> dict[str, Any]:
    """Stable serialization helper used by diagnostics and unit tests."""
    payload = asdict(result)
    payload["endpoints"] = [f"{host}:{port}" for host, port in result.endpoints]
    return payload
