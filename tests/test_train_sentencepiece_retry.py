import unittest

from src.train import train_sentencepiece_tokenizer_with_retry


def _repetitive_dialog_text(pairs: int = 40) -> str:
    topics = [
        ("Hello", "Hi there, how can I help you today?"),
        ("What is two plus two?", "Two plus two equals four."),
    ]
    lines = []
    for i in range(pairs):
        u, a = topics[i % 2]
        lines.append(f"User: {u}")
        lines.append(f"Ickle: {a}")
    return "\n".join(lines) + "\n"


class SentencePieceRetryTests(unittest.TestCase):
    """Regression coverage for train.py's SentencePiece retry loop. A prior
    version could halve target_size straight past the 128 floor (e.g.
    150 -> 75) without ever re-checking the raise condition at the new size,
    exiting the loop silently with no tokenizer assigned and no exception --
    an UnboundLocalError at the call site. Reproduced with a small/repetitive
    corpus during manual testing, not a hypothetical edge case."""

    def test_small_repetitive_corpus_succeeds_via_retry_instead_of_crashing(self):
        text = _repetitive_dialog_text()
        tokenizer, actual_vocab_size = train_sentencepiece_tokenizer_with_retry(
            text, spm_vocab_size=9600, model_type="bpe"
        )
        self.assertIsNotNone(tokenizer)
        self.assertGreaterEqual(actual_vocab_size, 128)
        self.assertLess(actual_vocab_size, 9600, "should have actually retried down, not just accepted 9600")
        # The returned tokenizer must be immediately usable, not a stub.
        encoded = tokenizer.encode(text)
        self.assertGreater(len(encoded), 0)

    def test_requested_size_already_below_floor_still_succeeds(self):
        # This is the exact bug: if spm_vocab_size starts below 128, the old
        # `while target_size >= 128` loop's body never ran once.
        text = _repetitive_dialog_text()
        tokenizer, actual_vocab_size = train_sentencepiece_tokenizer_with_retry(
            text, spm_vocab_size=50, model_type="bpe"
        )
        self.assertIsNotNone(tokenizer)
        self.assertEqual(actual_vocab_size, 128, "should be clamped up to the floor, not silently skipped")

    def test_degenerate_corpus_raises_clean_system_exit_not_a_raw_traceback(self):
        with self.assertRaises(SystemExit) as ctx:
            train_sentencepiece_tokenizer_with_retry("a a a a a a a a a a", spm_vocab_size=128, model_type="bpe")
        message = str(ctx.exception)
        self.assertIn("Could not train a SentencePiece tokenizer", message)
        self.assertIn("--tokenizer char", message)


if __name__ == "__main__":
    unittest.main()
