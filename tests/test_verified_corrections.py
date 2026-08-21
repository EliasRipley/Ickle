import tempfile
import unittest
from pathlib import Path

from src.federated.keys import create_ed_identity
from src.federated.knowledge_commons import EpistemicLedger
from src.verified_corrections import (
    build_verified_corrections_corpus_file,
    collect_verified_corrections,
    verified_corrections_status,
)


class VerifiedCorrectionsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.owner_id = self.root / "owner.sqlite3"
        self.owner = EpistemicLedger(self.owner_id, identity=create_ed_identity("owner"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_local_correction_is_collected(self):
        self.owner.add_review(
            claim_text="The tokenizer has a fixed 4K vocabulary.",
            relation="correct",
            correction_text="The tokenizer supports up to a 16K vocabulary.",
            shared=False,
        )
        pairs = collect_verified_corrections(ledger=self.owner)
        self.assertEqual(len(pairs), 1)
        self.assertIn("fixed 4K vocabulary", pairs[0].user)
        self.assertEqual(pairs[0].assistant, "The tokenizer supports up to a 16K vocabulary.")

    def test_support_and_dispute_relations_are_excluded(self):
        self.owner.add_review(claim_text="Some claim.", relation="support", shared=False)
        self.owner.add_review(claim_text="Another claim.", relation="dispute", shared=False)
        pairs = collect_verified_corrections(ledger=self.owner)
        self.assertEqual(pairs, [])

    def test_unadopted_peer_correction_is_not_collected(self):
        peer_id = self.root / "peer.sqlite3"
        peer = EpistemicLedger(peer_id, identity=create_ed_identity("peer"))
        shared_event = peer.add_review(
            claim_text="Peer claim needing correction.",
            relation="correct",
            correction_text="Peer's correction.",
            shared=True,
        )
        self.owner.merge_events([shared_event])
        pairs = collect_verified_corrections(ledger=self.owner)
        self.assertEqual(pairs, [], "peer review must stay inert until the owner adopts it")

    def test_adopting_a_peer_review_makes_it_eligible(self):
        peer_id = self.root / "peer2.sqlite3"
        peer = EpistemicLedger(peer_id, identity=create_ed_identity("peer2"))
        shared_event = peer.add_review(
            claim_text="Another peer claim.",
            relation="correct",
            correction_text="Peer's other correction.",
            shared=True,
        )
        self.owner.merge_events([shared_event])
        self.owner.adopt_event(shared_event["event_id"], shared=False)
        pairs = collect_verified_corrections(ledger=self.owner)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].assistant, "Peer's other correction.")

    def test_retracted_correction_is_excluded(self):
        added = self.owner.add_review(
            claim_text="Retracted claim.",
            relation="correct",
            correction_text="Retracted correction text.",
            shared=False,
        )
        self.owner.add_review(
            claim_text="Retracted claim.",
            relation="retract",
            target_event_id=added["event_id"],
            shared=False,
        )
        pairs = collect_verified_corrections(ledger=self.owner)
        self.assertEqual(pairs, [])

    def test_duplicate_corrections_deduplicate(self):
        for _ in range(2):
            self.owner.add_review(
                claim_text="Same claim text.",
                relation="correct",
                correction_text="Same correction text.",
                shared=False,
            )
        pairs = collect_verified_corrections(ledger=self.owner)
        self.assertEqual(len(pairs), 1)

    def test_build_corpus_file_oversamples_and_writes_dialog_format(self):
        self.owner.add_review(
            claim_text="Claim one.",
            relation="correct",
            correction_text="Correction one.",
            shared=False,
        )
        self.owner.add_review(
            claim_text="Claim two.",
            relation="adopt",
            correction_text="Correction two.",
            shared=False,
        )
        out_path = self.root / "corrections_corpus.txt"
        stats = build_verified_corrections_corpus_file(
            out_path=str(out_path),
            ledger=self.owner,
            oversample=3,
        )
        self.assertEqual(stats["distinct_corrections"], 2)
        self.assertEqual(stats["written_pairs"], 6)
        text = out_path.read_text(encoding="utf-8")
        self.assertEqual(text.count("User:"), 6)
        self.assertEqual(text.count("Correction one."), 3)
        self.assertEqual(text.count("Correction two."), 3)

    def test_build_corpus_file_respects_max_pairs(self):
        self.owner.add_review(
            claim_text="Claim one.",
            relation="correct",
            correction_text="Correction one.",
            shared=False,
        )
        out_path = self.root / "corrections_corpus_capped.txt"
        stats = build_verified_corrections_corpus_file(
            out_path=str(out_path),
            ledger=self.owner,
            oversample=10,
            max_pairs=2,
        )
        self.assertEqual(stats["written_pairs"], 2)

    def test_status_reports_eligible_count_and_sample(self):
        self.owner.add_review(
            claim_text="Status claim.",
            relation="correct",
            correction_text="Status correction.",
            shared=False,
        )
        status = verified_corrections_status(ledger=self.owner)
        self.assertEqual(status["eligible_corrections"], 1)
        self.assertEqual(len(status["sample"]), 1)


if __name__ == "__main__":
    unittest.main()
