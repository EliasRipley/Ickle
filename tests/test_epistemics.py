import unittest

from src.epistemics import build_answer_map, build_collective_view, extract_candidate_claims


class _Reviews:
    def reviews_for_claim(self, claim_text, claim_id="", limit=40):
        if "Paris" in claim_text:
            return [
                {
                    "relation": "correct",
                    "correction_text": "Paris is France's capital, but not its largest region.",
                    "source_url": "https://example.test/france",
                    "is_local": True,
                }
            ]
        return []


class _PeerReviews:
    def reviews_for_claim(self, claim_text, claim_id="", limit=40):
        return [
            {
                "relation": "correct",
                "correction_text": "A peer-suggested replacement.",
                "source_url": "",
                "is_local": False,
            }
        ]


class CandidateClaimTests(unittest.TestCase):
    def test_extracts_prose_but_skips_heading_question_and_code(self):
        text = """## Result
Paris is the capital city of France. Would you like more detail?
```python
print('not a prose claim')
```
You should verify travel rules before booking.
"""
        claims = extract_candidate_claims(text)
        self.assertEqual(len(claims), 2)
        self.assertEqual(claims[0]["kind"], "statement")
        self.assertEqual(claims[1]["kind"], "advice")

    def test_answer_map_calls_related_evidence_related_not_verified(self):
        passport = build_answer_map(
            prompt="What is the capital of France?",
            response="Paris is the capital city of France.",
            evidence_items=[
                {
                    "claim": "The capital and most populous city of France is Paris.",
                    "source_url": "https://example.test/paris",
                    "source_title": "Paris reference",
                    "score": 0.9,
                }
            ],
        )
        self.assertEqual(passport["claims"][0]["status"], "source_linked")
        self.assertNotIn("verified", str(passport).lower())

    def test_local_correction_has_priority_over_source_overlap(self):
        passport = build_answer_map(
            prompt="Tell me about Paris",
            response="Paris is the capital of France and its largest region.",
            evidence_items=[{"claim": "Paris is the capital of France", "source_url": "https://example.test"}],
            review_lookup=_Reviews(),
        )
        claim = passport["claims"][0]
        self.assertEqual(claim["status"], "corrected")
        self.assertEqual(claim["reviews"]["corrections"][0]["is_local"], True)

    def test_unadopted_peer_correction_is_labeled_as_perspective(self):
        passport = build_answer_map(
            prompt="Explain a claim",
            response="This is a substantive candidate claim for inspection.",
            review_lookup=_PeerReviews(),
        )
        self.assertEqual(passport["claims"][0]["status"], "peer_perspective")


class CollectiveViewTests(unittest.TestCase):
    def test_preserves_common_and_distinct_claims_without_declaring_truth(self):
        view = build_collective_view(
            [
                {"peer_id": "a", "response": "Paris is the capital city of France."},
                {"peer_id": "b", "response": "The capital city of France is Paris."},
                {"peer_id": "c", "response": "France has several major wine regions."},
            ]
        )
        self.assertGreaterEqual(view["summary"]["common_claims"], 1)
        self.assertGreaterEqual(view["summary"]["distinct_claims"], 1)
        self.assertIn("not correctness", view["caveat"])


if __name__ == "__main__":
    unittest.main()
