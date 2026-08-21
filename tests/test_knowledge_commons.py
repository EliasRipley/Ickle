import tempfile
import unittest
import urllib.request
from pathlib import Path

from src.federated.keys import create_ed_identity
from src.federated.identity import create_identity
from src.federated.knowledge_commons import (
    EpistemicLedger,
    KnowledgeEvent,
    create_knowledge_event,
    sync_with_peers,
)
from src.federated.swarm import SwarmNode


class SignedEventTests(unittest.TestCase):
    def test_signature_binds_content_and_author(self):
        identity = create_ed_identity("reviewer")
        event = create_knowledge_event(
            identity,
            claim_text="Paris is the capital city of France.",
            relation="support",
            shared=True,
        )
        self.assertTrue(event.verify(require_shared=True))
        tampered = event.to_dict()
        tampered["claim_text"] = "Lyon is the capital city of France."
        self.assertFalse(KnowledgeEvent.from_dict(tampered).verify(require_shared=True))

    def test_malformed_cryptographic_fields_are_rejected_before_verification(self):
        identity = create_ed_identity("reviewer")
        event = create_knowledge_event(
            identity,
            claim_text="A well formed claim for the validation test.",
            relation="support",
            shared=True,
        ).to_dict()
        event["pubkey_hex"] = "aa" * 100_000
        self.assertFalse(KnowledgeEvent.from_dict(event).verify(require_shared=True))

    def test_unshared_event_is_rejected_by_remote_merge(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            a = EpistemicLedger(root / "a.sqlite3", identity=create_ed_identity("a"))
            b = EpistemicLedger(root / "b.sqlite3", identity=create_ed_identity("b"))
            local = a.add_review(
                claim_text="This review stays on its owner's device.",
                relation="support",
                shared=False,
            )
            result = b.merge_events([local])
            self.assertEqual(result, {"accepted": 0, "duplicate": 0, "rejected": 1})
            self.assertEqual(b.summary()["events"], 0)


class ConflictPreservingLedgerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.a = EpistemicLedger(root / "a.sqlite3", identity=create_ed_identity("a"))
        self.b = EpistemicLedger(root / "b.sqlite3", identity=create_ed_identity("b"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_merge_is_idempotent_and_keeps_conflict(self):
        claim = "The new library release supports Python 3.13."
        support = self.a.add_review(claim_text=claim, relation="support", shared=True)
        dispute = self.b.add_review(claim_text=claim, relation="dispute", shared=True)
        first = self.a.merge_events([dispute])
        second = self.a.merge_events([dispute])
        self.b.merge_events([support])
        self.assertEqual(first["accepted"], 1)
        self.assertEqual(second["duplicate"], 1)
        self.assertEqual({row["relation"] for row in self.a.reviews_for_claim(claim)}, {"support", "dispute"})
        self.assertEqual(self.a.summary()["events"], self.b.summary()["events"])

    def test_peer_correction_is_inert_until_owner_adopts_it(self):
        claim = "The service listens on port 9000 by default."
        remote = self.a.add_review(
            claim_text=claim,
            relation="correct",
            correction_text="The service listens on port 8787 by default.",
            shared=True,
        )
        self.b.merge_events([remote])
        self.assertEqual(self.b.context_for_prompt("Which port does the service use?"), "")
        self.b.adopt_event(remote["event_id"])
        context = self.b.context_for_prompt("Which port does the service use by default?")
        self.assertIn("8787", context)
        self.assertIn("owner", context.lower())

    def test_author_can_retract_without_deleting_history(self):
        event = self.a.add_review(
            claim_text="A temporary claim that is long enough for review.",
            relation="support",
            shared=False,
        )
        self.a.retract(event["event_id"])
        self.assertEqual(self.a.reviews_for_claim(event["claim_text"]), [])
        self.assertIsNotNone(self.a.get_event(event["event_id"]))


class PeerSyncTests(unittest.TestCase):
    def test_real_http_sync_is_bidirectional_and_shared_only(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            node = SwarmNode(
                identity=create_identity("server"),
                ed_identity=create_ed_identity("server-ed"),
                data_dir=root / "server" / "torickle",
                host="127.0.0.1",
                port=0,
                external_host="127.0.0.1",
            )
            node.commons.add_review(
                claim_text="The server has a shared review for the client.",
                relation="support",
                shared=True,
            )
            node.start()
            try:
                port = int(node._server.server_address[1])
                client = EpistemicLedger(root / "client" / "commons.sqlite3", identity=create_ed_identity("client"))
                client.add_review(
                    claim_text="The client has a shared dispute for the server.",
                    relation="dispute",
                    shared=True,
                )
                client.add_review(
                    claim_text="This private review must never cross the network.",
                    relation="support",
                    shared=False,
                )

                report = sync_with_peers(client, [f"127.0.0.1:{port}"])

                self.assertEqual(report["peers_reached"], 1)
                self.assertEqual(client.summary()["peer_events"], 1)
                server_claims = {event.claim_text for event in node.commons.list_events()}
                self.assertIn("The client has a shared dispute for the server.", server_claims)
                self.assertNotIn("This private review must never cross the network.", server_claims)
            finally:
                node.stop()

    def test_peer_endpoint_does_not_grant_arbitrary_browser_cors(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            node = SwarmNode(
                identity=create_identity("cors-server"),
                ed_identity=create_ed_identity("cors-server-ed"),
                data_dir=root / "server" / "torickle",
                host="127.0.0.1",
                port=0,
                external_host="127.0.0.1",
            )
            node.start()
            try:
                port = int(node._server.server_address[1])
                request = urllib.request.Request(
                    f"http://127.0.0.1:{port}/torickle/v1/commons/events",
                    headers={"Origin": "https://attacker.example"},
                )
                with urllib.request.urlopen(request, timeout=3) as response:
                    self.assertIsNone(response.headers.get("Access-Control-Allow-Origin"))
            finally:
                node.stop()


if __name__ == "__main__":
    unittest.main()
