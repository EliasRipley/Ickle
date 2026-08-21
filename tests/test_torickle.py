import json
import unittest
import uuid
from pathlib import Path

import torch

from src.torickle import pack_delta_file, reassemble_manifest, verify_manifest


class TorickleTests(unittest.TestCase):
    @staticmethod
    def _tmp_dir(name: str) -> Path:
        root = Path("data") / ".tmp_tests" / f"{name}_{uuid.uuid4().hex}"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def test_pack_verify_reassemble_roundtrip(self):
        workspace = self._tmp_dir("torickle_roundtrip")
        delta = {
            "layer_a": torch.randn(3, 4),
            "layer_b": torch.randn(5),
        }
        delta_path = workspace / "delta.pt"
        torch.save(delta, delta_path)

        out_dir = workspace / "bundle"
        packed = pack_delta_file(
            delta_path=str(delta_path),
            out_dir=str(out_dir),
            piece_size_bytes=128,
        )
        manifest_path = packed["manifest_path"]
        self.assertTrue(Path(manifest_path).exists())

        verify = verify_manifest(manifest_path=manifest_path)
        self.assertTrue(verify["valid"])
        self.assertGreaterEqual(int(verify["piece_count"]), 1)

        rebuilt_path = workspace / "rebuilt_delta.pt"
        rebuilt = reassemble_manifest(
            manifest_path=manifest_path,
            out_path=str(rebuilt_path),
            strict_verify=True,
        )
        self.assertTrue(Path(rebuilt["out_path"]).exists())

        roundtrip = torch.load(rebuilt_path, map_location="cpu")
        self.assertEqual(set(delta.keys()), set(roundtrip.keys()))
        for key in delta:
            self.assertTrue(torch.allclose(delta[key].float(), roundtrip[key].float()))

    def test_verify_detects_piece_tampering(self):
        workspace = self._tmp_dir("torickle_tamper")
        delta = {
            "x": torch.randn(8, 8),
        }
        delta_path = workspace / "delta.pt"
        torch.save(delta, delta_path)

        out_dir = workspace / "bundle"
        packed = pack_delta_file(
            delta_path=str(delta_path),
            out_dir=str(out_dir),
            piece_size_bytes=96,
        )
        manifest_path = Path(packed["manifest_path"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        first_piece = manifest["pieces"][0]["file_name"]
        first_piece_path = manifest_path.parent / manifest["pieces_dir"] / first_piece
        original = first_piece_path.read_bytes()
        self.assertGreater(len(original), 0)
        flipped = bytes([original[0] ^ 0x01]) + original[1:]
        first_piece_path.write_bytes(flipped)

        verify = verify_manifest(manifest_path=str(manifest_path))
        self.assertFalse(verify["valid"])
        self.assertTrue(any("mismatch" in msg for msg in verify["errors"]))

    def test_pack_accepts_update_file_with_delta_field(self):
        workspace = self._tmp_dir("torickle_update_file")
        delta = {"k": torch.randn(2, 2)}
        update_payload = {
            "client_id": "c_test",
            "round_id": 7,
            "delta": delta,
        }
        update_path = workspace / "update.pt"
        torch.save(update_payload, update_path)

        out_dir = workspace / "bundle"
        packed = pack_delta_file(
            delta_path=str(update_path),
            out_dir=str(out_dir),
            piece_size_bytes=128,
        )
        verify = verify_manifest(manifest_path=packed["manifest_path"])
        self.assertTrue(verify["valid"])


if __name__ == "__main__":
    unittest.main()

