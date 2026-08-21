import tempfile
import unittest
from pathlib import Path

from src.partner_loop import PartnerLoop


class PartnerLoopTests(unittest.TestCase):
    def test_vague_prompt_requests_clarification(self):
        loop = PartnerLoop(journal_path="data/test_partner_journal.jsonl")
        d = loop.decide("help with marketing")
        self.assertEqual(d.decision, "ask_clarification")

    def test_unsupported_declares_limit(self):
        loop = PartnerLoop(journal_path="data/test_partner_journal.jsonl")
        d = loop.decide("generate an image of a cat")
        self.assertEqual(d.decision, "declare_limit")

    def test_supported_can_proceed(self):
        loop = PartnerLoop(journal_path="data/test_partner_journal.jsonl")
        d = loop.decide("read website")
        self.assertEqual(d.decision, "proceed")

    def test_journal_written(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "journal.jsonl"
            loop = PartnerLoop(journal_path=str(path))
            d = loop.decide("read website")
            loop.journal(d)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
