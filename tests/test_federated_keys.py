import tempfile
import unittest
from pathlib import Path

from src.federated.keys import (
    create_ed_identity,
    ensure_ed_identity,
    load_ed_identity,
    peer_id_from_pubkey,
    save_ed_identity,
    verify_bytes,
    verify_payload,
)


class EdIdentityTests(unittest.TestCase):
    def test_create_identity_shape(self):
        identity = create_ed_identity(label="node-a")
        self.assertEqual(len(identity.peer_id), 40)
        self.assertEqual(len(bytes.fromhex(identity.pubkey_hex)), 32)
        self.assertEqual(len(bytes.fromhex(identity.privkey_hex)), 32)
        self.assertEqual(identity.label, "node-a")
        self.assertEqual(identity.peer_id, peer_id_from_pubkey(identity.pubkey_hex))

    def test_sign_verify_payload_any_peer_can_check(self):
        identity = create_ed_identity()
        payload = {"model_hash": "abc", "capacity": 2}
        sig = identity.sign(payload)
        # Verification only needs the public key, not the identity object.
        self.assertTrue(verify_payload(identity.pubkey_hex, payload, sig))
        self.assertFalse(verify_payload(identity.pubkey_hex, {"model_hash": "tampered"}, sig))

    def test_verify_rejects_wrong_pubkey(self):
        identity = create_ed_identity()
        other = create_ed_identity()
        payload = {"x": 1}
        sig = identity.sign(payload)
        self.assertFalse(verify_payload(other.pubkey_hex, payload, sig))

    def test_sign_verify_bytes(self):
        identity = create_ed_identity()
        data = b"raw response bytes"
        sig = identity.sign_bytes(data)
        self.assertTrue(verify_bytes(identity.pubkey_hex, data, sig))
        self.assertFalse(verify_bytes(identity.pubkey_hex, b"tampered", sig))

    def test_verify_rejects_malformed_signature(self):
        identity = create_ed_identity()
        self.assertFalse(verify_payload(identity.pubkey_hex, {"x": 1}, "not-hex"))
        self.assertFalse(verify_payload("not-hex", {"x": 1}, "aa"))

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ed25519.json"
            original = create_ed_identity(label="roundtrip")
            save_ed_identity(path, original)
            loaded = load_ed_identity(path)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.peer_id, original.peer_id)
            self.assertEqual(loaded.privkey_hex, original.privkey_hex)

    def test_ensure_ed_identity_creates_then_reuses(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ed25519.json"
            self.assertFalse(path.exists())
            first = ensure_ed_identity(path, label="auto")
            self.assertTrue(path.exists())
            second = ensure_ed_identity(path, label="ignored-on-reuse")
            self.assertEqual(first.peer_id, second.peer_id)
            self.assertEqual(second.label, "auto")


if __name__ == "__main__":
    unittest.main()
