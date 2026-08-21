import socket
import struct
import threading
import unittest

from src.federated.public_dht import (
    BencodeError,
    DHTContact,
    DHTLookupResult,
    MainlineDHTClient,
    PUBLIC_SWARM_INFOHASH,
    PublicDHTService,
    bep42_node_id,
    bdecode,
    bencode,
    decode_compact_nodes,
    decode_compact_peers,
)


def _compact_peer(host: str, port: int) -> bytes:
    return socket.inet_aton(host) + struct.pack("!H", port)


def _compact_node(node_id: bytes, host: str, port: int) -> bytes:
    return node_id + _compact_peer(host, port)


class BencodeTests(unittest.TestCase):
    def test_binary_protocol_round_trip(self):
        value = {
            b"a": {b"id": bytes(range(20)), b"port": 8790},
            b"q": b"announce_peer",
            b"t": b"aa",
            b"y": b"q",
        }
        self.assertEqual(bdecode(bencode(value)), value)

    def test_decoder_rejects_trailing_and_deep_input(self):
        with self.assertRaises(BencodeError):
            bdecode(b"1:ai1e")
        with self.assertRaises(BencodeError):
            bdecode((b"l" * 14) + (b"e" * 14))

    def test_compact_decoders_ignore_private_addresses(self):
        node_id = bytes(range(20))
        nodes = _compact_node(node_id, "8.8.8.8", 6881) + _compact_node(node_id, "192.168.1.2", 6881)
        peers = [_compact_peer("1.1.1.1", 8790), _compact_peer("10.0.0.5", 8790)]
        self.assertEqual(decode_compact_nodes(nodes), [(node_id, ("8.8.8.8", 6881))])
        self.assertEqual(decode_compact_peers(peers), [("1.1.1.1", 8790)])


class MainlineDHTTests(unittest.TestCase):
    def test_public_swarm_key_is_stable_sha1_width(self):
        self.assertEqual(PUBLIC_SWARM_INFOHASH.hex(), "034e41f8a47f1701e233a56493fe2d409bcef030")
        self.assertEqual(len(PUBLIC_SWARM_INFOHASH), 20)

    def test_bep42_node_id_matches_published_vector_prefix(self):
        seed = bytearray(range(20))
        seed[2] = 7
        seed[-1] = 1
        node_id = bep42_node_id("124.31.75.21", bytes(seed))
        # BEP 42's first IPv4 vector requires the first 21 bits to be
        # 0x5fbfb8; the lower three bits of byte 2 remain random.
        self.assertEqual(node_id[:2], bytes.fromhex("5fbf"))
        self.assertEqual(node_id[2] & 0xF8, 0xB8)
        self.assertEqual(node_id[-1], 1)

    def test_iterative_lookup_discovers_and_announces(self):
        first = ("8.8.8.8", 6881)
        second = ("8.8.4.4", 6881)
        candidate_a = ("1.1.1.1", 8790)
        candidate_b = ("9.9.9.9", 8790)
        client = MainlineDHTClient(bytes(range(20)), routers=(), max_nodes=8, alpha=1)
        client._resolve_routers = lambda: [DHTContact(None, first)]
        calls = []

        def fake_get(address):
            calls.append(("get", address))
            if address == first:
                return {
                    b"token": b"first-token",
                    b"nodes": _compact_node(bytes([4]) * 20, second[0], second[1]),
                    b"values": [_compact_peer(*candidate_a)],
                }
            if address == second:
                return {b"token": b"second-token", b"values": [_compact_peer(*candidate_b)]}
            return None

        def fake_announce(address, token, port):
            calls.append(("announce", address, token, port))
            return True

        client._get_peers = fake_get
        client._announce = fake_announce
        result = client.discover_and_announce(8790)

        self.assertEqual(set(result.endpoints), {candidate_a, candidate_b})
        self.assertEqual(result.nodes_contacted, 2)
        self.assertEqual(result.nodes_responded, 2)
        self.assertEqual(result.announced_to, 2)
        self.assertIn(("announce", first, b"first-token", 8790), calls)
        self.assertIn(("announce", second, b"second-token", 8790), calls)

    def test_empty_bootstrap_is_reported_without_raising(self):
        client = MainlineDHTClient(b"identity", routers=())
        result = client.discover_and_announce(8790)
        self.assertEqual(result.nodes_contacted, 0)
        self.assertIn("bootstrap", result.error.lower())


class PublicDHTServiceTests(unittest.TestCase):
    def test_background_service_exposes_real_state(self):
        completed = threading.Event()

        class FakeClient:
            def discover_and_announce(self, peer_port, *, should_stop=None):
                self.port = peer_port
                return DHTLookupResult(
                    endpoints=[("1.1.1.1", 8790)],
                    nodes_contacted=7,
                    nodes_responded=5,
                    announced_to=3,
                    duration_seconds=0.01,
                )

        client = FakeClient()
        service = PublicDHTService(client, refresh_seconds=30)

        def accept(rows):
            self.assertEqual(rows, [("1.1.1.1", 8790)])
            completed.set()
            return 1

        service.start(8790, accept)
        try:
            self.assertTrue(completed.wait(2))
            status = service.status()
            self.assertEqual(status["phase"], "connected")
            self.assertEqual(status["nodes_responded"], 5)
            self.assertEqual(status["verified_peers"], 1)
            self.assertEqual(client.port, 8790)
        finally:
            service.stop()
        self.assertEqual(service.status()["phase"], "off")


if __name__ == "__main__":
    unittest.main()
