import argparse
import unittest

from src.ilm_chat_generation import _generate_with_reasoning, _generate_model_response
from src.model import ILM, TinyConfig
from src.system_limits import SystemLimits
from src.tokenizer import CharTokenizer


def _tiny_model_and_tokenizer():
    tokenizer = CharTokenizer.from_text("abcdefghij User: Ickle: reason ?")
    cfg = TinyConfig(vocab_size=tokenizer.vocab_size, block_size=32, n_embd=16, n_head=4, n_layer=2, dropout=0.0)
    model = ILM(cfg)
    model.eval()
    return model, tokenizer


class GenerateWithReasoningTests(unittest.TestCase):
    def test_thinking_mode_does_not_crash(self):
        # Regression test: _generate_with_reasoning used `ick` (from
        # src.icklization) without importing it in this module, so any
        # request with thinking_mode=True (the web UI's default) crashed
        # with NameError: name 'ick' is not defined on every single chat
        # message -- caught only by actually sending a real chat request,
        # since no test exercised this function at all.
        model, tokenizer = _tiny_model_and_tokenizer()
        args = argparse.Namespace(
            thinking_mode=True,
            max_new=20,
            temperature=0.7,
            top_k=10,
            speculative=False,
            draft_model="",
        )
        limits = SystemLimits(max_new_tokens=64, torch_threads=1)
        result = _generate_with_reasoning(model, tokenizer, "hi", args, limits)
        self.assertTrue(hasattr(result, "text"))
        self.assertIsInstance(result.text, str)

    def test_thinking_mode_off_still_works(self):
        model, tokenizer = _tiny_model_and_tokenizer()
        args = argparse.Namespace(
            thinking_mode=False,
            max_new=20,
            temperature=0.7,
            top_k=10,
            speculative=False,
            draft_model="",
        )
        limits = SystemLimits(max_new_tokens=64, torch_threads=1)
        result = _generate_with_reasoning(model, tokenizer, "hi", args, limits)
        self.assertIsInstance(result.text, str)

    def test_generate_model_response_plain(self):
        model, tokenizer = _tiny_model_and_tokenizer()
        args = argparse.Namespace(max_new=15, temperature=0.7, top_k=10, speculative=False)
        limits = SystemLimits(max_new_tokens=64, torch_threads=1)
        text = _generate_model_response(model, tokenizer, "hi", args, limits)
        self.assertIsInstance(text, str)


if __name__ == "__main__":
    unittest.main()
