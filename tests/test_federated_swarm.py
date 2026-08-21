import json
import os
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from src.federated.identity import SwarmIdentity, create_identity, ensure_identity, load_identity, save_identity
from src.federated.keys import create_ed_identity
from src.federated.swarm import SwarmNode, BundleAnnouncement, _join_via_bootstrap


class IdentityTests(unittest.TestCase):
    def test_create_identity(self):
        identity = create_identity(label="test-peer")
        self.assertEqual(len(identity.peer_id), 40)
        self.assertEqual(len(identity.peer_id_bytes), 20)
        self.assertTrue(len(identity.peer_secret) > 20)
        self.assertEqual(identity.label, "test-peer")
        self.assertGreater(identity.created_at, 0)

    def test_sign_verify(self):
        identity = create_identity()
        payload = {"message": "hello", "round": 1}
        sig = identity.sign(payload)
        self.assertTrue(identity.verify(payload, sig))
        self.assertFalse(identity.verify({"message": "tampered"}, sig))

        wrong = create_identity()
        self.assertFalse(wrong.verify(payload, sig))

    def test_sign_verify_bytes(self):
        identity = create_identity()
        data = b"some binary content"
        sig = identity.sign_bytes(data)
        self.assertTrue(identity.verify_bytes(data, sig))
        self.assertFalse(identity.verify_bytes(b"tampered", sig))

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "identity.json"
            original = create_identity(label="roundtrip-test")
            save_identity(path, original)
            loaded = load_identity(path)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.peer_id, original.peer_id)
            self.assertEqual(loaded.peer_secret, original.peer_secret)
            self.assertEqual(loaded.label, original.label)

    def test_ensure_identity_creates(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "new_identity.json"
            self.assertFalse(path.exists())
            identity = ensure_identity(path, label="auto-created")
            self.assertTrue(path.exists())
            self.assertEqual(identity.label, "auto-created")

    def test_ensure_identity_loads_existing(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "existing.json"
            original = create_identity(label="original")
            save_identity(path, original)
            loaded = ensure_identity(path, label="ignored")
            self.assertEqual(loaded.peer_id, original.peer_id)
            self.assertNotEqual(loaded.label, "ignored")

    def test_to_from_dict(self):
        identity = create_identity(label="dict-test")
        d = identity.to_dict()
        restored = SwarmIdentity.from_dict(d)
        self.assertEqual(restored.peer_id, identity.peer_id)
        self.assertEqual(restored.peer_secret, identity.peer_secret)


class BundleAnnouncementTests(unittest.TestCase):
    def setUp(self):
        self.identity = create_identity(label="announcer")
        self.ed_identity = create_ed_identity(label="announcer")

    def test_create_and_verify(self):
        ann = BundleAnnouncement(
            bundle_id="bundle123",
            model_hash="model_abc",
            piece_count=10,
            total_bytes=64000,
            payload_sha256="abc123",
            merkle_root="def456",
            peer_id=self.ed_identity.peer_id,
            host="192.168.1.1",
            port=8790,
            timestamp=time.time(),
            signature="",
            pubkey_hex=self.ed_identity.pubkey_hex,
        )
        sig_payload = ann.to_dict()
        sig_payload.pop("signature", None)
        ann.signature = self.ed_identity.sign(sig_payload)
        self.assertTrue(ann.verify())
        ann.host = "tampered"
        self.assertFalse(ann.verify())

    def test_verify_rejects_missing_pubkey(self):
        ann = BundleAnnouncement(
            bundle_id="bundle123", model_hash="", piece_count=0, total_bytes=0,
            payload_sha256="", merkle_root="", peer_id=self.ed_identity.peer_id,
            host="192.168.1.1", port=8790, timestamp=time.time(), signature="forged",
        )
        self.assertFalse(ann.verify())

    def test_verify_rejects_peer_id_pubkey_mismatch(self):
        other = create_ed_identity(label="impersonator")
        ann = BundleAnnouncement(
            bundle_id="bundle123", model_hash="", piece_count=0, total_bytes=0,
            payload_sha256="", merkle_root="", peer_id=self.ed_identity.peer_id,
            host="192.168.1.1", port=8790, timestamp=time.time(), signature="",
            pubkey_hex=other.pubkey_hex,
        )
        sig_payload = ann.to_dict()
        sig_payload.pop("signature", None)
        ann.signature = other.sign(sig_payload)
        self.assertFalse(ann.verify())

    def test_to_from_dict(self):
        ann = BundleAnnouncement(
            bundle_id="b1", model_hash="m1", piece_count=5, total_bytes=32000,
            payload_sha256="s1", merkle_root="r1", peer_id="p1",
            host="10.0.0.1", port=8790, timestamp=1000, signature="sig123",
        )
        d = ann.to_dict()
        restored = BundleAnnouncement.from_dict(d)
        self.assertEqual(restored.bundle_id, "b1")
        self.assertEqual(restored.piece_count, 5)
        self.assertEqual(restored.signature, "sig123")

    def test_dht_key(self):
        ann = BundleAnnouncement(
            bundle_id="test-bundle", model_hash="", piece_count=0, total_bytes=0,
            payload_sha256="", merkle_root="", peer_id="p1",
            host="", port=0, timestamp=0, signature="",
        )
        key = ann.dht_key()
        self.assertEqual(len(key), 20)


class SwarmNodeTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.tmpdir.name) / "torickle"
        self.data_dir.mkdir(parents=True)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_start_stop(self):
        identity = create_identity(label="test-peer")
        node = SwarmNode(
            identity=identity,
            data_dir=str(self.data_dir),
            host="127.0.0.1",
            port=0,
            external_host="127.0.0.1",
        )
        node.start()
        self.assertTrue(node._running)
        node.stop()
        self.assertFalse(node._running)

    def test_import_bundle(self):
        identity = create_identity()
        node = SwarmNode(identity=identity, data_dir=str(self.data_dir), host="127.0.0.1", port=0)

        bundle_id = node.import_bundle(str(self.data_dir))
        self.assertIsNone(bundle_id, "Should fail on empty dir")

        bundle_dir = self.data_dir / "test_bundle"
        bundle_dir.mkdir()
        manifest = {
            "torickle_version": "0.1",
            "payload_sha256": "abc123",
            "piece_count": 0,
            "total_bytes": 0,
            "merkle_root": "root",
            "pieces": [],
        }
        (bundle_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        bundle_id = node.import_bundle(str(bundle_dir))
        self.assertIsNotNone(bundle_id)
        self.assertIn(bundle_id, node.bundles)

    def test_scan_local_bundles(self):
        identity = create_identity()
        node = SwarmNode(identity=identity, data_dir=str(self.data_dir), host="127.0.0.1", port=0)

        sub = self.data_dir / "bundles" / "some_bundle"
        sub.mkdir(parents=True)
        manifest = {
            "torickle_version": "0.1",
            "payload_sha256": "xyz789",
            "piece_count": 0,
            "total_bytes": 0,
            "merkle_root": "root",
            "pieces": [],
        }
        (sub / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        node._scan_local_bundles()
        self.assertGreater(len(node.bundles), 0)

    def test_announce_bundle(self):
        identity = create_identity()
        node = SwarmNode(identity=identity, data_dir=str(self.data_dir), host="127.0.0.1", port=0)

        sub = self.data_dir / "bundles" / "test_announce"
        sub.mkdir(parents=True)
        manifest = {
            "torickle_version": "0.1",
            "payload_sha256": "announce_test_hash",
            "piece_count": 3,
            "total_bytes": 12000,
            "merkle_root": "announce_root",
            "pieces": [],
        }
        (sub / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        node._scan_local_bundles()

        bundle_ids = list(node.bundles.keys())
        self.assertGreater(len(bundle_ids), 0)
        ann = node.announce_bundle(bundle_ids[0], model_hash="test_model")
        self.assertIsNotNone(ann)
        self.assertEqual(ann.peer_id, node.ed_identity.peer_id)
        self.assertEqual(ann.model_hash, "test_model")
        self.assertTrue(ann.verify())

    def test_find_bundles_empty(self):
        identity = create_identity()
        node = SwarmNode(identity=identity, data_dir=str(self.data_dir), host="127.0.0.1", port=0)
        results = node.find_bundles(model_hash="nonexistent")
        self.assertEqual(len(results), 0)

    def test_find_bundles_after_announce(self):
        identity = create_identity()
        node = SwarmNode(identity=identity, data_dir=str(self.data_dir), host="127.0.0.1", port=0)
        sub = self.data_dir / "bundles" / "find_test"
        sub.mkdir(parents=True)
        manifest = {
            "torickle_version": "0.1",
            "payload_sha256": "find_hash",
            "piece_count": 0,
            "total_bytes": 0,
            "merkle_root": "find_root",
            "pieces": [],
        }
        (sub / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        node._scan_local_bundles()
        bundle_ids = list(node.bundles.keys())
        self.assertGreater(len(bundle_ids), 0)
        node.announce_bundle(bundle_ids[0], model_hash="model-find")

        by_model = node.find_bundles(model_hash="model-find")
        self.assertGreaterEqual(len(by_model), 1)
        self.assertEqual(by_model[0].model_hash, "model-find")

        all_results = node.find_bundles(model_hash="")
        self.assertGreaterEqual(len(all_results), 1)

    def test_query_peer_bundles_refused(self):
        identity = create_identity()
        node = SwarmNode(identity=identity, data_dir=str(self.data_dir), host="127.0.0.1", port=0)
        bundles = node.query_peer_bundles("127.0.0.1", 1)
        self.assertEqual(len(bundles), 0)

    def test_download_manifest_refused(self):
        identity = create_identity()
        node = SwarmNode(identity=identity, data_dir=str(self.data_dir), host="127.0.0.1", port=0)
        manifest = node.download_manifest("127.0.0.1", 1, "test_bundle")
        self.assertIsNone(manifest)

    def test_peer_serves_bundles(self):
        identity_a = create_identity(label="peer-a")
        identity_b = create_identity(label="peer-b")

        node_a = SwarmNode(identity=identity_a, data_dir=str(self.data_dir), host="127.0.0.1", port=0, external_host="127.0.0.1")
        node_a.start()
        port_a = node_a._server.server_port

        sub = self.data_dir / "bundles" / "serve_test"
        sub.mkdir(parents=True)
        manifest = {
            "torickle_version": "0.1",
            "payload_sha256": "serve_hash",
            "piece_count": 0,
            "total_bytes": 0,
            "merkle_root": "serve_root",
            "pieces": [],
        }
        (sub / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        node_a._scan_local_bundles()
        bundle_ids = list(node_a.bundles.keys())
        self.assertGreater(len(bundle_ids), 0)
        ann = node_a.announce_bundle(bundle_ids[0])

        bundle_list = node_a.query_peer_bundles("127.0.0.1", port_a)
        self.assertGreater(len(bundle_list), 0)
        self.assertEqual(bundle_list[0]["payload_sha256"], "serve_hash")

        manifest2 = node_a.download_manifest("127.0.0.1", port_a, bundle_ids[0])
        self.assertIsNotNone(manifest2)
        self.assertEqual(manifest2["payload_sha256"], "serve_hash")

        node_a.stop()

    def test_bootstrap_join_discovers_peer_and_bundles(self):
        """Regression test: add_bootstrap() alone only recorded an address and
        never contacted it, so a second node could never actually learn about
        the first. _join_via_bootstrap() is the fix -- verify a fresh node
        that bootstraps against a running one immediately knows about both
        the peer itself and any bundle it has announced, without either node
        needing to already know about the other beforehand."""
        identity_a = create_identity(label="bootstrap-a")
        node_a = SwarmNode(identity=identity_a, data_dir=str(self.data_dir), host="127.0.0.1", port=0, external_host="127.0.0.1")
        node_a.start()
        try:
            port_a = node_a._server.server_port

            sub = self.data_dir / "bundles" / "bootstrap_test"
            sub.mkdir(parents=True)
            manifest = {
                "torickle_version": "0.1",
                "payload_sha256": "bootstrap_hash",
                "piece_count": 0,
                "total_bytes": 0,
                "merkle_root": "bootstrap_root",
                "pieces": [],
            }
            (sub / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            node_a._scan_local_bundles()
            bundle_ids = list(node_a.bundles.keys())
            node_a.announce_bundle(bundle_ids[0])

            identity_b = create_identity(label="bootstrap-b")
            data_dir_b = Path(self.tmpdir.name) / "torickle_b"
            data_dir_b.mkdir(parents=True)
            node_b = SwarmNode(identity=identity_b, data_dir=str(data_dir_b), host="127.0.0.1", port=0)

            # This is the actual behavior under test: a bare add_bootstrap()
            # call would leave node_b's store empty; _join_via_bootstrap()
            # must perform the real HTTP fetch-and-merge.
            _join_via_bootstrap(node_b.peer_discovery, [f"127.0.0.1:{port_a}"])

            peer_ids = {p.peer_id for p in node_b.peer_discovery.store.all_peers()}
            self.assertIn(identity_a.peer_id_bytes, peer_ids)

            results = node_b.find_bundles()
            self.assertTrue(any(r.payload_sha256 == "bootstrap_hash" for r in results))
        finally:
            node_a.stop()

    def test_peer_rejects_piece_path_traversal(self):
        identity = create_identity(label="peer-sec")
        node = SwarmNode(identity=identity, data_dir=str(self.data_dir), host="127.0.0.1", port=0, external_host="127.0.0.1")
        node.start()
        try:
            port = node._server.server_port

            outside = self.data_dir / "outside_secret.txt"
            outside.write_text("do-not-serve", encoding="utf-8")

            sub = self.data_dir / "bundles" / "traversal_test"
            (sub / "pieces").mkdir(parents=True)
            manifest = {
                "torickle_version": "0.1",
                "payload_sha256": "traversal_hash",
                "piece_count": 1,
                "total_bytes": 12,
                "merkle_root": "traversal_root",
                "pieces": [
                    {"index": 0, "file_name": "../../../outside_secret.txt", "size_bytes": 12, "sha256": "", "leaf_hash": ""},
                ],
            }
            (sub / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            node._scan_local_bundles()
            bundle_ids = list(node.bundles.keys())
            self.assertGreater(len(bundle_ids), 0)
            bundle_id = bundle_ids[-1]

            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/torickle/v1/pieces/{bundle_id}/0", timeout=5)
            self.assertEqual(ctx.exception.code, 404)
        finally:
            node.stop()
