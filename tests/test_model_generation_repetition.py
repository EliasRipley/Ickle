import unittest

import torch

from src.model import ILM, TinyConfig, _apply_no_repeat_ngram, _apply_repetition_penalty


class RepetitionPenaltyTests(unittest.TestCase):
    """Regression coverage for a real bug: ILM.generate()/generate_streaming()
    had no repetition mitigation at all -- plain temperature/top_k sampling.
    Confirmed live: the default chat settings (temperature=0.25, top_k=20)
    reliably collapsed a freshly trained model into repeating the exact same
    phrase forever ("Feel free to answer well." dozens of times in a row) on
    ordinary prompts like "What can you help with?"."""

    def test_penalty_of_one_is_a_no_op(self):
        logits = torch.tensor([[1.0, 2.0, -1.0, 3.0]])
        generated = torch.tensor([[0, 1]])
        before = logits.clone()
        _apply_repetition_penalty(logits, generated, 1.0)
        self.assertTrue(torch.equal(logits, before))

    def test_penalty_reduces_score_of_already_generated_positive_token(self):
        logits = torch.tensor([[4.0, 4.0]])
        generated = torch.tensor([[0]])
        _apply_repetition_penalty(logits, generated, 2.0)
        self.assertAlmostEqual(logits[0, 0].item(), 2.0, places=4)
        self.assertAlmostEqual(logits[0, 1].item(), 4.0, places=4)

    def test_penalty_reduces_score_of_already_generated_negative_token(self):
        logits = torch.tensor([[-4.0, -4.0]])
        generated = torch.tensor([[0]])
        _apply_repetition_penalty(logits, generated, 2.0)
        # For a negative score, dividing makes it *less* negative (higher);
        # CTRL-style penalty instead multiplies to push it further down.
        self.assertAlmostEqual(logits[0, 0].item(), -8.0, places=4)
        self.assertAlmostEqual(logits[0, 1].item(), -4.0, places=4)


class NoRepeatNgramTests(unittest.TestCase):
    def test_blocks_token_that_would_complete_a_seen_ngram(self):
        # Sequence "... 5 6 5 6" with ngram_size=2: the model is about to
        # pick a token after the trailing "6"; token 5 already followed 6
        # once before, so a bigram block must forbid choosing 5 again.
        generated = torch.tensor([[5, 6, 5, 6]])
        logits = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 0.0]])
        _apply_no_repeat_ngram(logits, generated, ngram_size=2)
        self.assertEqual(logits[0, 5].item(), float("-inf"))

    def test_does_not_block_unrelated_tokens(self):
        generated = torch.tensor([[5, 6, 5, 6]])
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 9.0, 7.0]])
        _apply_no_repeat_ngram(logits, generated, ngram_size=2)
        self.assertEqual(logits[0, 0].item(), 1.0)
        self.assertEqual(logits[0, 6].item(), 7.0)

    def test_short_sequence_is_a_no_op(self):
        generated = torch.tensor([[5]])
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        before = logits.clone()
        _apply_no_repeat_ngram(logits, generated, ngram_size=3)
        self.assertTrue(torch.equal(logits, before))


class GenerationLoopIntegrationTests(unittest.TestCase):
    """End-to-end: a tiny real model, greedy-ish low-temperature sampling
    (the exact regime that triggered the bug), must not get stuck emitting
    the same token forever once repetition guards are in place."""

    def test_generate_does_not_collapse_into_a_single_repeated_token(self):
        cfg = TinyConfig(vocab_size=32, block_size=16, n_embd=16, n_head=2, n_layer=2)
        model = ILM(cfg)
        model.eval()
        idx = torch.zeros((1, 1), dtype=torch.long)
        out = model.generate(idx, max_new_tokens=40, temperature=0.1, top_k=5)
        generated = out[0, 1:].tolist()
        # An untrained model with no guards at very low temperature typically
        # locks onto one dominant token; with both guards active it must not.
        most_common_count = max(generated.count(t) for t in set(generated))
        self.assertLess(most_common_count, len(generated))


if __name__ == "__main__":
    unittest.main()
