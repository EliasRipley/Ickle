import unittest
from pathlib import Path
from unittest import mock
from uuid import uuid4

from src.chat_benchmark import _load_cases, _score_response, run_benchmark


class ChatBenchmarkTests(unittest.TestCase):
    def test_concise_correct_calculation_is_not_marked_low_quality(self):
        score = _score_response(
            "What is 17 multiplied by 6?",
            "17 × 6 = 102.",
            ["17", "6", "102"],
            required_keywords=["102"],
        )
        self.assertGreaterEqual(score["quality_score"], 0.8)

    def test_load_cases(self):
        root = Path("data/.tmp_tests")
        root.mkdir(parents=True, exist_ok=True)
        p = root / f"bench_{uuid4().hex}.json"
        p.write_text(
            '[{"name":"gorilla","prompt":"Where do gorillas live?","keywords":["gorilla","africa","forest"]}]',
            encoding="utf-8",
        )
        cases = _load_cases(p)
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].name, "gorilla")

    def test_run_benchmark_scores_keyword_hits(self):
        case_json = '[{"name":"gorilla","prompt":"Where do gorillas live?","keywords":["gorilla","africa","forest"]}]'
        root = Path("data/.tmp_tests")
        root.mkdir(parents=True, exist_ok=True)
        p = root / f"bench_{uuid4().hex}.json"
        p.write_text(case_json, encoding="utf-8")
        cases = _load_cases(p)

        def fake_generate_response(args):
            if "gorillas" in args.prompt.lower():
                return "Gorillas live in forests of central Africa."
            return "Unknown."

        with mock.patch("src.chat_benchmark.generate_response", side_effect=fake_generate_response):
            out = run_benchmark(model="models/fake.pt", cases=cases, enable_memory=False, enable_web_tools=False)
        self.assertEqual(out["case_count"], 1)
        self.assertGreater(out["avg_score"], 0.6)

    def test_required_keyword_caps_a_plausible_but_wrong_answer(self):
        score = _score_response(
            "What is 17 multiplied by 6?",
            "The product is 96 after multiplying the two numbers.",
            ["product", "multiply"],
            required_keywords=["102"],
        )
        self.assertEqual(score["required_score"], 0.0)
        self.assertEqual(score["score"], 0.0)

    def test_forbidden_claim_caps_score(self):
        score = _score_response(
            "Does a fair coin guarantee heads next?",
            "A fair coin has equal probability, so it guarantees heads next.",
            ["fair", "coin", "probability"],
            forbidden_keywords=["guarantees heads"],
        )
        self.assertGreater(score["forbidden_hits"], 0)
        self.assertLessEqual(score["score"], 0.05)


if __name__ == "__main__":
    unittest.main()
