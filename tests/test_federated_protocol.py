import unittest

import torch

from src.federated.protocol import (
    decode_tensor_dict,
    encode_tensor_dict,
    now_epoch_seconds,
    sign_payload,
    verify_signature,
    verify_timestamp,
)


class FederatedProtocolTests(unittest.TestCase):
    def test_signature_round_trip(self):
        secret = "test_secret"
        payload = {
            "client_id": "c1",
            "round_id": 7,
            "num_examples": 120,
            "timestamp": 1234567,
            "nonce": "abc",
        }
        signature = sign_payload(secret, payload)
        self.assertTrue(verify_signature(secret, payload, signature))
        self.assertFalse(verify_signature("wrong", payload, signature))

    def test_tensor_wire_round_trip(self):
        src = {
            "a": torch.randn(4, 3),
            "b": torch.randn(2),
        }
        wire = encode_tensor_dict(src)
        out = decode_tensor_dict(wire)
        self.assertEqual(set(src.keys()), set(out.keys()))
        for key in src:
            self.assertTrue(torch.allclose(src[key].float(), out[key].float()))

    def test_timestamp_validation(self):
        now = now_epoch_seconds()
        self.assertTrue(verify_timestamp(now))
        self.assertFalse(verify_timestamp(now - 10_000))


if __name__ == "__main__":
    unittest.main()
