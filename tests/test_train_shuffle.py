import unittest

import torch

from src.train import shuffle_in_chunks


class ShuffleInChunksTests(unittest.TestCase):
    """Regression coverage for a real bug: the train/val split takes the
    positional last 10% of the token stream as val_data, but training text
    is built by concatenating sources in a fixed order (streamed dataset,
    then a second streamed dataset, then --bootstrap-english's fixed seed
    text repeated up to 200x by default). Without shuffling first,
    whatever was concatenated last dominates validation -- confirmed as
    the root cause of a real val_loss (0.024) vs. train_loss (4.19)
    anomaly, since a repeated ~9KB boilerplate block landing entirely in
    val is trivially memorizable."""

    def test_preserves_every_token_just_reorders(self):
        encoded = torch.arange(1000)
        shuffled = shuffle_in_chunks(encoded, chunk_size=100, seed=42)
        self.assertEqual(sorted(shuffled.tolist()), list(range(1000)))

    def test_deterministic_given_same_seed(self):
        encoded = torch.arange(1000)
        a = shuffle_in_chunks(encoded, chunk_size=100, seed=42)
        b = shuffle_in_chunks(encoded, chunk_size=100, seed=42)
        self.assertTrue(torch.equal(a, b))

    def test_different_seeds_produce_different_order(self):
        encoded = torch.arange(2000)
        a = shuffle_in_chunks(encoded, chunk_size=100, seed=1)
        b = shuffle_in_chunks(encoded, chunk_size=100, seed=2)
        self.assertFalse(torch.equal(a, b))

    def test_preserves_local_order_within_a_chunk(self):
        # Each chunk's internal token order must stay intact -- only
        # chunk-level macro-order gets shuffled, not individual tokens
        # (which would destroy the local structure a causal LM learns from).
        encoded = torch.arange(1000)
        shuffled = shuffle_in_chunks(encoded, chunk_size=100, seed=7)
        chunks = [shuffled[i : i + 100] for i in range(0, 1000, 100)]
        for chunk in chunks:
            self.assertTrue(torch.equal(chunk, torch.arange(int(chunk[0].item()), int(chunk[0].item()) + 100)))

    def test_tail_heavy_content_gets_redistributed_out_of_the_last_10_percent(self):
        """Directly reproduces the bug scenario: a large block of one
        repeated value (standing in for --bootstrap-english's repeated
        text) appended at the very end of the corpus must not end up
        entirely within the last 10% after shuffling."""
        diverse = torch.arange(8000)  # stand-in for fineweb + oasst1 content
        repeated_tail = torch.full((2000,), 99999)  # stand-in for bootstrap-english x200
        encoded = torch.cat([diverse, repeated_tail])

        shuffled = shuffle_in_chunks(encoded, chunk_size=200, seed=42)
        n = int(0.9 * len(shuffled))
        val_data = shuffled[n:]

        repeated_fraction_in_val = float((val_data == 99999).sum()) / len(val_data)
        # Before the fix this would be 1.0 (val_data is 1000 tokens, entirely
        # inside the 2000-token repeated tail). After shuffling by chunk, the
        # repeated block's ~20% share of the corpus should land roughly
        # proportionally across train/val, not concentrated entirely in val.
        self.assertLess(repeated_fraction_in_val, 0.6)

    def test_short_input_returned_unchanged(self):
        encoded = torch.arange(50)
        result = shuffle_in_chunks(encoded, chunk_size=100, seed=1)
        self.assertTrue(torch.equal(result, encoded))


if __name__ == "__main__":
    unittest.main()
