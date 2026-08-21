from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from src.ilm_chat import _generate_model_response
from src.system_limits import SystemLimits


class _FakeTokenizer:
    def encode(self, _: str) -> list[int]:
        return [10, 11]

    def decode(self, ids: list[int]) -> str:
        if ids == [21, 22]:
            return "Ickle: speculative path"
        return "Ickle: standard path"


class _FakeModel:
    class _Cfg:
        block_size = 64

    cfg = _Cfg()

    def generate(self, *_args, **_kwargs):
        return torch.tensor([[10, 11, 31, 32]])

    def parameters(self):
        yield torch.zeros(1)


class ILMChatSpeculativeTests(unittest.TestCase):
    def test_generate_model_response_uses_speculative_when_enabled(self):
        model = _FakeModel()
        tokenizer = _FakeTokenizer()
        args = SimpleNamespace(max_new=8, temperature=0.6, top_k=10, speculative=True, speculative_gamma=4)
        limits = SystemLimits(max_new_tokens=32)
        with mock.patch(
            "src.ilm_chat.speculative_generate_simple",
            return_value=torch.tensor([[10, 11, 21, 22]]),
        ) as patched:
            out = _generate_model_response(model, tokenizer, "hello", args, limits)
        self.assertEqual(out, "speculative path")
        patched.assert_called_once()

    def test_generate_model_response_uses_standard_path_when_speculative_disabled(self):
        model = _FakeModel()
        tokenizer = _FakeTokenizer()
        args = SimpleNamespace(max_new=8, temperature=0.6, top_k=10, speculative=False, speculative_gamma=3)
        limits = SystemLimits(max_new_tokens=32)
        with mock.patch("src.ilm_chat.speculative_generate_simple") as patched:
            out = _generate_model_response(model, tokenizer, "hello", args, limits)
        self.assertEqual(out, "standard path")
        patched.assert_not_called()


if __name__ == "__main__":
    unittest.main()
