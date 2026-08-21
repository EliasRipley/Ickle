import unittest

import torch

from src.dpo_train import _clip_pair, _completion_logprob_sum
from src.model import ILM, TinyConfig
from src.tokenizer import CharTokenizer


class DPOTrainTests(unittest.TestCase):
    def test_clip_pair_preserves_completion(self):
        prompt_ids = list(range(100))
        completion_ids = [100, 101, 102, 103]
        full, prompt_len = _clip_pair(prompt_ids, completion_ids, max_total=16, fallback_prompt_id=0)
        self.assertEqual(full[-4:], completion_ids)
        self.assertGreaterEqual(prompt_len, 1)
        self.assertLessEqual(len(full), 16)

    def test_completion_logprob_is_finite(self):
        cfg = TinyConfig(vocab_size=8, block_size=16, n_embd=16, n_head=4, n_layer=2, dropout=0.0)
        model = ILM(cfg)
        tokenizer = CharTokenizer.from_text("abcd ef?")
        with torch.no_grad():
            out = _completion_logprob_sum(
                model,
                prompt="ab",
                completion="cd",
                tokenizer=tokenizer,
            )
        self.assertTrue(torch.isfinite(out).item())


if __name__ == "__main__":
    unittest.main()
