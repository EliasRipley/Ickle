import tempfile
import time
import unittest
from pathlib import Path

from src.federated.contribution_ledger import LedgerStore
from src.federated.inference_swarm import (
    InferenceNode,
    InferenceOffer,
    _join_via_bootstrap,
    find_offers,
    request_inference,
)
from src.federated.keys import create_ed_identity
from src.federated.peer_discovery import PeerDiscovery


def _stub_generate(prompt: str, max_new_tokens: int, temperature: float, top_k: int) -> str:
    return f"echo:{prompt}"


class InferenceOfferTests(unittest.TestCase):
    def test_sign_and_verify(self):
        identity = create_ed_identity(label="offer-signer")
        offer = InferenceOffer(
            peer_id=identity.peer_id,
            pubkey_hex=identity.pubkey_hex,
            model_hash="model-1",
            host="127.0.0.1",
            port=8791,
            capacity=2,
            context_window=512,
            label="test",
            timestamp=time.time(),
        )
        offer.sign(identity)
        self.assertTrue(offer.verify())

    def test_tampered_offer_fails_verify(self):
        identity = create_ed_identity()
        offer = InferenceOffer(
            peer_id=identity.peer_id,
            pubkey_hex=identity.pubkey_hex,
            model_hash="model-1",
            host="127.0.0.1",
            port=8791,
            capacity=2,
            context_window=512,
            label="",
            timestamp=time.time(),
        )
        offer.sign(identity)
        offer.capacity = 999
        self.assertFalse(offer.verify())

    def test_offer_with_mismatched_peer_id_fails_verify(self):
        identity = create_ed_identity()
        impostor = create_ed_identity()
        offer = InferenceOffer(
            peer_id=impostor.peer_id,  # claims to be someone else
            pubkey_hex=identity.pubkey_hex,
            model_hash="",
            host="127.0.0.1",
            port=1,
            capacity=1,
            context_window=0,
            label="",
            timestamp=time.time(),
        )
        offer.sign(identity)
        self.assertFalse(offer.verify())

    def test_to_from_dict_roundtrip(self):
        offer = InferenceOffer(
            peer_id="p" * 40,
            pubkey_hex="ab" * 32,
            model_hash="m1",
            host="1.2.3.4",
            port=9000,
            capacity=3,
            context_window=1024,
            label="lbl",
            timestamp=1234.5,
            signature="sig",
        )
        restored = InferenceOffer.from_dict(offer.to_dict())
        self.assertEqual(restored.peer_id, offer.peer_id)
        self.assertEqual(restored.capacity, 3)
        self.assertEqual(restored.signature, "sig")


class InferenceNodeTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _make_node(self, *, capacity: int = 2, peer_discovery: PeerDiscovery | None = None) -> InferenceNode:
        identity = create_ed_identity(label="server")
        ledger = LedgerStore(self.data_dir / "server_ledger.json")
        return InferenceNode(
            identity=identity,
            generate_fn=_stub_generate,
            model_hash="model-x",
            data_dir=self.data_dir,
            peer_discovery=peer_discovery or PeerDiscovery(),
            ledger=ledger,
            host="127.0.0.1",
            port=0,
            external_host="127.0.0.1",
            capacity=capacity,
            label="test-node",
        )

    def test_start_stop(self):
        node = self._make_node()
        node.start()
        self.assertTrue(node._running)
        node.stop()
        self.assertFalse(node._running)

    def test_announce_then_find(self):
        pd = PeerDiscovery()
        node = self._make_node(peer_discovery=pd)
        node.announce()
        offers = find_offers(pd, model_hash="model-x")
        self.assertEqual(len(offers), 1)
        self.assertEqual(offers[0].peer_id, node.identity.peer_id)

    def test_find_filters_by_model_hash(self):
        pd = PeerDiscovery()
        node = self._make_node(peer_discovery=pd)
        node.announce()
        offers = find_offers(pd, model_hash="some-other-model")
        self.assertEqual(len(offers), 0)

    def test_slot_acquire_release_respects_capacity(self):
        node = self._make_node(capacity=1)
        self.assertTrue(node.try_acquire_slot())
        self.assertFalse(node.try_acquire_slot())
        node.release_slot()
        self.assertTrue(node.try_acquire_slot())

    def test_end_to_end_request_and_ledger_updates(self):
        node = self._make_node(capacity=2)
        node.start()
        try:
            offer = node.build_offer()
            requester_identity = create_ed_identity(label="client")
            requester_ledger = LedgerStore(self.data_dir / "client_ledger.json")

            result = request_inference(
                offer,
                "hello there",
                requester_peer_id=requester_identity.peer_id,
                ledger=requester_ledger,
            )
            self.assertIsNotNone(result)
            self.assertTrue(result["verified"], result)
            self.assertEqual(result["response"], "echo:hello there")
            self.assertEqual(result["peer_id"], node.identity.peer_id)

            self.assertEqual(node.ledger.ledger.peer_requests_served, 1)
            self.assertEqual(requester_ledger.ledger.peer_requests_consumed, 1)
        finally:
            node.stop()

    def test_over_capacity_returns_503_like_failure(self):
        node = self._make_node(capacity=1)
        node.start()
        try:
            offer = node.build_offer()
            node.try_acquire_slot()  # occupy the only slot directly
            result = request_inference(offer, "hi")
            self.assertIsNotNone(result)
            self.assertEqual(result.get("http_status"), 503)
        finally:
            node.release_slot()
            node.stop()

    def test_empty_prompt_rejected(self):
        node = self._make_node()
        node.start()
        try:
            offer = node.build_offer()
            result = request_inference(offer, "   ")
            self.assertEqual(result.get("http_status"), 400)
        finally:
            node.stop()

    def test_bootstrap_join_discovers_offer_from_separate_process_view(self):
        """Regression test: `infer find`/`infer ask` used to construct a fresh
        PeerDiscovery() with no way to ever learn about a running `infer serve`
        node -- add_bootstrap() only recorded an address, nothing ever
        contacted it. _join_via_bootstrap() is the fix: a second, independent
        PeerDiscovery (standing in for a separate one-shot CLI process) must
        discover the offer via the node's real HTTP /infer/v1/offers endpoint."""
        serving_node = self._make_node(capacity=2)
        serving_node.start()
        try:
            serving_node.announce()
            port = serving_node._server.server_port

            asker_discovery = PeerDiscovery()
            self.assertEqual(len(find_offers(asker_discovery)), 0, "should start with no knowledge of the peer")

            _join_via_bootstrap(asker_discovery, [f"127.0.0.1:{port}"])

            offers = find_offers(asker_discovery, model_hash="model-x")
            self.assertEqual(len(offers), 1)
            self.assertEqual(offers[0].peer_id, serving_node.identity.peer_id)
            self.assertEqual(offers[0].port, port)
        finally:
            serving_node.stop()


if __name__ == "__main__":
    unittest.main()
