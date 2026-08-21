import tempfile
import unittest
from pathlib import Path

from src.federated.contribution_ledger import ContributionLedger, LedgerStore


class ContributionLedgerTests(unittest.TestCase):
    def test_fresh_ledger_has_unbounded_ratio(self):
        ledger = ContributionLedger()
        self.assertEqual(ledger.ratio(), -1.0)
        self.assertEqual(ledger.summary()["ratio_display"], "unbounded")

    def test_serving_without_consuming_stays_unbounded(self):
        ledger = ContributionLedger()
        ledger.record_piece_served(4096)
        ledger.record_training_round()
        ledger.record_inference_served(120)
        self.assertGreater(ledger.contributed_score, 0)
        self.assertEqual(ledger.consumed_score, 0)
        self.assertEqual(ledger.ratio(), -1.0)

    def test_ratio_reflects_give_vs_take(self):
        ledger = ContributionLedger()
        ledger.record_inference_served(100)  # contributed_score = 5
        ledger.record_inference_consumed(100)  # consumed_score = 5
        self.assertEqual(ledger.ratio(), 1.0)

        ledger.record_inference_consumed(100)  # consumed_score = 10, contributed still 5
        self.assertLess(ledger.ratio(), 1.0)

    def test_training_round_dominates_many_small_requests(self):
        ledger = ContributionLedger()
        ledger.record_training_round()
        heavy_contribution = ledger.contributed_score
        light_ledger = ContributionLedger()
        for _ in range(10):
            light_ledger.record_inference_served(10)
        self.assertGreater(heavy_contribution, light_ledger.contributed_score)

    def test_to_from_dict_roundtrip(self):
        ledger = ContributionLedger()
        ledger.record_piece_served(10)
        ledger.record_inference_consumed(50)
        restored = ContributionLedger.from_dict(ledger.to_dict())
        self.assertEqual(restored.seed_pieces_served, ledger.seed_pieces_served)
        self.assertEqual(restored.peer_tokens_consumed, ledger.peer_tokens_consumed)


class LedgerStoreTests(unittest.TestCase):
    def test_persists_across_instances(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ledger.json"
            store = LedgerStore(path)
            store.record_inference_served(42)
            store.record_piece_served(1000)

            reloaded = LedgerStore(path)
            self.assertEqual(reloaded.ledger.peer_requests_served, 1)
            self.assertEqual(reloaded.ledger.peer_tokens_served, 42)
            self.assertEqual(reloaded.ledger.seed_bytes_served, 1000)

    def test_missing_file_starts_fresh(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "nonexistent" / "ledger.json"
            store = LedgerStore(path)
            self.assertEqual(store.summary()["ratio_display"], "unbounded")

    def test_corrupt_file_starts_fresh(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ledger.json"
            path.write_text("not json", encoding="utf-8")
            store = LedgerStore(path)
            self.assertEqual(store.ledger.seed_pieces_served, 0)


if __name__ == "__main__":
    unittest.main()
