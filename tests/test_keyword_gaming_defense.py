import unittest

from src.promotion_gate import (
    _composite_response_score,
    keyword_stuffing_penalty,
)


class KeywordStuffingPenaltyTests(unittest.TestCase):
    """Regression coverage for a real gaming vector: promotion_gate.py,
    continual_guard.py, and eval_harness.py all gate model promotion on a
    keyword-hit ratio. A response that's just the target keywords stitched
    together with no real content ("keyword salad") used to score close to
    1.0 on keyword coverage and, since it has no repeated words, also
    scored well on the lexical-diversity quality metric -- a candidate
    model could pass promotion gates without producing a coherent, on-topic
    answer at all."""

    def test_keyword_salad_is_penalized(self):
        keywords = ["hi", "hello", "hey", "assist", "greeting"]
        salad = "hi hello hey assist greeting"
        penalty = keyword_stuffing_penalty(salad, keywords)
        self.assertGreater(penalty, 0.0)

    def test_genuine_response_using_keywords_in_context_is_not_penalized(self):
        keywords = ["hi", "hello", "assist"]
        genuine = (
            "Hi there! Hello, happy to help -- I can assist you with "
            "setting up your project step by step, just tell me the goal."
        )
        penalty = keyword_stuffing_penalty(genuine, keywords)
        self.assertEqual(penalty, 0.0)

    def test_repeating_one_keyword_is_penalized(self):
        keywords = ["hello", "assist", "greeting"]
        spam = "hello hello hello hello hello hello hello"
        penalty = keyword_stuffing_penalty(spam, keywords)
        self.assertGreater(penalty, 0.0)

    def test_composite_score_of_keyword_salad_loses_to_genuine_response(self):
        prompt = "Hello!"
        keywords = ["hi", "hello", "hey", "assist", "greeting"]
        anti_keywords = ["probability", "event", "sample", "outcome", "subset"]

        salad_result = _composite_response_score(
            "hi hello hey assist greeting", prompt, keywords, anti_keywords
        )
        genuine_result = _composite_response_score(
            "Hey, hello! I'm here to assist -- happy to greet you and help out.",
            prompt, keywords, anti_keywords,
        )
        self.assertGreater(genuine_result["score"], salad_result["score"])
        self.assertGreater(salad_result["stuffing_penalty"], 0.0)


if __name__ == "__main__":
    unittest.main()
