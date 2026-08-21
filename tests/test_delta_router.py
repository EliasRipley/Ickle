import unittest
from unittest import mock

import torch

from src.delta_router import DeltaRouter, KnowledgeDelta
from src.model import ILM, TinyConfig
from src.tokenizer import CharTokenizer


class DeltaRouterRealEmbeddingTests(unittest.TestCase):
    """Regression coverage for a real bug: DeltaRouter._encode_text derived
    "token ids" from ord(c) % 256, which has no relationship to any real
    tokenizer's vocabulary (SentencePiece ids don't correlate with character
    codepoints), and could even produce an id >= vocab_size for a small
    model, making every delta's cosine-similarity routing score meaningless
    noise. It must use the model's real paired tokenizer instead."""

    def setUp(self):
        self.cfg = TinyConfig(vocab_size=32, block_size=16, n_embd=8, n_head=2, n_layer=1)
        self.model = ILM(self.cfg)
        self.tokenizer = CharTokenizer.from_text("hello world domain knowledge routing")

    def test_encode_text_uses_real_tokenizer_not_ord_mod_256(self):
        router = DeltaRouter()
        router.set_model(self.model, self.tokenizer)
        with mock.patch.object(self.tokenizer, "encode", wraps=self.tokenizer.encode) as spy:
            router._encode_text("hello")
        spy.assert_called_once_with("hello")

    def test_without_tokenizer_returns_zero_instead_of_fake_embedding(self):
        router = DeltaRouter()
        router.set_model(self.model, tokenizer=None)
        emb = router._encode_text("hello")
        self.assertEqual(emb.norm().item(), 0.0)

    def test_activation_scoring_works_end_to_end_with_real_tokenizer(self):
        router = DeltaRouter()
        router.set_model(self.model, self.tokenizer)
        delta = KnowledgeDelta(delta_id="d1", domain_description="hello world")
        router.register(delta)
        score = router.compute_activation("hello", "d1")
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
