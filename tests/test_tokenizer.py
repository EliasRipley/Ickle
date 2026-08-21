import unittest

from src.tokenizer import CharTokenizer, tokenizer_from_checkpoint


class TokenizerTests(unittest.TestCase):
    def test_char_tokenizer_roundtrip(self):
        tok = CharTokenizer.from_text("hello world")
        ids = tok.encode("hello")
        txt = tok.decode(ids)
        self.assertEqual(txt, "hello")

    def test_legacy_checkpoint_loading(self):
        ckpt = {
            "stoi": {"h": 0, "i": 1, "?": 2},
            "itos": {"0": "h", "1": "i", "2": "?"},
        }
        tok = tokenizer_from_checkpoint(ckpt)
        ids = tok.encode("hi!")
        self.assertEqual(len(ids), 3)
        self.assertEqual(tok.decode(ids[:2]), "hi")


if __name__ == "__main__":
    unittest.main()

