import unittest

from src.tokenizer import CharTokenizer
from src.train import build_loss_mask


class BuildLossMaskTests(unittest.TestCase):
    """Regression coverage for a real bug: build_loss_mask used to index the
    raw `text` string with a variable that actually walks the *tokenized*
    ids (`text[j]` where `j` is a token position) -- correct only by
    coincidence for CharTokenizer (1 char == 1 token); with the default
    SentencePiece tokenizer, token count != character count, so the
    "User:" boundary this exists to find effectively never matched, and the
    default full-training path (train.py, unlike lora_train.py) never
    actually applied the mask at all."""

    def test_masks_only_response_spans(self):
        text = "User: hi\nIckle: hello there\nUser: bye\nIckle: goodbye"
        tok = CharTokenizer.from_text(text)
        tokens = tok.encode(text)
        mask = build_loss_mask(tokens, tok, text)
        chars = [tok.decode([t]) for t in tokens]
        rendered = "".join(c if m else "_" for c, m in zip(chars, mask))
        self.assertIn("Ickle: hello there", rendered)
        self.assertIn("Ickle: goodbye", rendered)
        self.assertNotIn("User: hi", rendered)
        self.assertNotIn("User: bye", rendered)

    def test_falls_back_to_all_true_without_markers(self):
        text = "just some raw text with no turn markers at all"
        tok = CharTokenizer.from_text(text)
        tokens = tok.encode(text)
        mask = build_loss_mask(tokens, tok, text)
        self.assertTrue(all(mask))

    def test_ignored_text_argument_does_not_affect_result(self):
        # `text` is kept only for call-signature compatibility -- passing a
        # totally unrelated (even shorter) string must not change the mask,
        # proving the function no longer cross-indexes token positions into it.
        text = "User: hi\nIckle: hello there\nUser: bye\nIckle: goodbye"
        tok = CharTokenizer.from_text(text)
        tokens = tok.encode(text)
        mask_with_real_text = build_loss_mask(tokens, tok, text)
        mask_with_wrong_text = build_loss_mask(tokens, tok, "x")
        self.assertEqual(mask_with_real_text, mask_with_wrong_text)


if __name__ == "__main__":
    unittest.main()
