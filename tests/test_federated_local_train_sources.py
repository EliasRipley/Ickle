import tempfile
import unittest
from pathlib import Path

from src.federated.local_train import load_training_text


class FederatedLocalTrainSourceTests(unittest.TestCase):
    def test_load_training_text_from_local_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "tiny.txt"
            p.write_text("hello world\n", encoding="utf-8")
            text, meta = load_training_text(local_data_path=str(p))
            self.assertEqual(text, "hello world\n")
            self.assertEqual(meta.get("data_source"), "local_file")
            self.assertEqual(meta.get("local_data_path"), str(p.resolve()))

    def test_load_training_text_requires_source(self):
        with self.assertRaises(ValueError):
            load_training_text(local_data_path="")


if __name__ == "__main__":
    unittest.main()
