import tempfile
import unittest
from pathlib import Path

from src.epistemics import build_collective_view, stable_claim_id
from src.federated.keys import create_ed_identity
from src.federated.knowledge_commons import EpistemicLedger
from src.disagreement_curriculum import (
    build_hedge_corpus_file,
    disagreement_queue,
    disagreement_status,
    record_conflicts,
)


def _conflicting_responses():
    return [
        {"peer_id": "peer-a", "response": "The library does not support async writes."},
        {"peer_id": "peer-b", "response": "The library does support async writes."},
    ]


class DisagreementCurriculumTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.path = str(self.root / "disagreements.json")
        self.owner = EpistemicLedger(self.root / "owner.sqlite3", identity=create_ed_identity("owner"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_record_conflicts_creates_new_entry(self):
        view = build_collective_view(_conflicting_responses())
        result = record_conflicts(view["possible_conflicts"], source="live_ask", path=self.path)
        self.assertEqual(result["new"], 1)
        self.assertEqual(result["updated"], 0)
        queue = disagreement_queue(path=self.path, ledger=self.owner)
        self.assertEqual(len(queue), 1)
        self.assertEqual(queue[0]["peer_count"], 2)
        self.assertIn("live_ask", queue[0]["sources"])

    def test_record_conflicts_merges_repeat_observation(self):
        view = build_collective_view(_conflicting_responses())
        record_conflicts(view["possible_conflicts"], source="live_ask", path=self.path)
        more_responses = _conflicting_responses() + [
            {"peer_id": "peer-c", "response": "The library does not support async writes."},
        ]
        view2 = build_collective_view(more_responses)
        result = record_conflicts(view2["possible_conflicts"], source="codistill_round", path=self.path)
        self.assertEqual(result["new"], 0)
        self.assertEqual(result["updated"], 1)
        queue = disagreement_queue(path=self.path, ledger=self.owner)
        self.assertEqual(len(queue), 1)
        self.assertEqual(queue[0]["peer_count"], 3)
        self.assertEqual(queue[0]["times_observed"], 2)
        self.assertEqual(set(queue[0]["sources"]), {"live_ask", "codistill_round"})

    def test_single_peer_conflict_excluded_by_min_peer_count(self):
        record_conflicts(
            [{"representative": "Solo claim.", "variants": ["Solo claim."], "peer_ids": ["only-peer"]}],
            source="live_ask",
            path=self.path,
        )
        self.assertEqual(disagreement_queue(path=self.path, ledger=self.owner), [])

    def test_resolving_via_commons_removes_it_from_the_open_queue(self):
        view = build_collective_view(_conflicting_responses())
        record_conflicts(view["possible_conflicts"], source="live_ask", path=self.path)
        self.assertEqual(len(disagreement_queue(path=self.path, ledger=self.owner)), 1)

        representative = view["possible_conflicts"][0]["representative"]
        self.owner.add_review(
            claim_text=representative,
            relation="correct",
            correction_text="Resolved: it does support async writes as of v2.",
            shared=False,
        )
        self.assertEqual(disagreement_queue(path=self.path, ledger=self.owner), [])

    def test_status_reports_open_and_resolved_counts(self):
        view = build_collective_view(_conflicting_responses())
        record_conflicts(view["possible_conflicts"], source="live_ask", path=self.path)
        status = disagreement_status(path=self.path, ledger=self.owner)
        self.assertEqual(status["total_tracked"], 1)
        self.assertEqual(status["open_count"], 1)
        self.assertEqual(status["resolved_count"], 0)
        self.assertEqual(len(status["top"]), 1)

    def test_build_hedge_corpus_writes_dialog_pairs_for_open_disagreements(self):
        view = build_collective_view(_conflicting_responses())
        record_conflicts(view["possible_conflicts"], source="live_ask", path=self.path)
        out_path = self.root / "hedges.txt"
        stats = build_hedge_corpus_file(
            out_path=str(out_path),
            disagreements_path=self.path,
            oversample=2,
            ledger=self.owner,
        )
        self.assertEqual(stats["distinct_disagreements"], 1)
        self.assertEqual(stats["written_pairs"], 2)
        text = out_path.read_text(encoding="utf-8")
        self.assertEqual(text.count("User:"), 2)
        self.assertIn("independent peers disagree", text)

    def test_build_hedge_corpus_excludes_resolved_disagreements(self):
        view = build_collective_view(_conflicting_responses())
        record_conflicts(view["possible_conflicts"], source="live_ask", path=self.path)
        representative = view["possible_conflicts"][0]["representative"]
        self.owner.add_review(
            claim_text=representative,
            relation="correct",
            correction_text="Resolved.",
            shared=False,
        )
        out_path = self.root / "hedges_after_resolve.txt"
        stats = build_hedge_corpus_file(
            out_path=str(out_path),
            disagreements_path=self.path,
            ledger=self.owner,
        )
        self.assertEqual(stats["written_pairs"], 0)

    def test_claim_id_alignment_with_commons_uses_same_stable_id_function(self):
        view = build_collective_view(_conflicting_responses())
        conflict = view["possible_conflicts"][0]
        self.assertEqual(conflict["cluster_id"], stable_claim_id(conflict["representative"]))


if __name__ == "__main__":
    unittest.main()
