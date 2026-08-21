import json
import unittest
from pathlib import Path
from uuid import uuid4

from src.conversation_focus import build_focus_corpus_file, merge_dialog_corpora, parse_dialog_pairs


class ConversationFocusTests(unittest.TestCase):
    @staticmethod
    def _tmpdir() -> Path:
        root = Path("data/.tmp_tests") / f"focus_{uuid4().hex}"
        root.mkdir(parents=True, exist_ok=False)
        return root

    def test_parse_dialog_pairs_filters_low_signal_and_bad_grammar(self):
        td = self._tmpdir()
        src = td / "pairs.txt"
        src.write_text(
            "\n".join(
                [
                    "User: Where do gorillas live?",
                    "Ickle: Gorillas are inhabits the Albertine Rift montane cloud forests.",
                    "",
                    "User: Where do gorillas live?",
                    "Ickle: Gorillas live in tropical forests of equatorial Africa.",
                    "",
                    "User: What is photosynthesis?",
                    "Ickle: I can help with that. Share the exact outcome you want and any constraints.",
                    "",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        pairs = parse_dialog_pairs(src, max_pairs=20)
        self.assertEqual(len(pairs), 1)
        self.assertIn("equatorial africa", pairs[0].assistant.lower())

    def test_build_focus_corpus_prefers_non_evasive_candidates(self):
        td = self._tmpdir()
        candidate = td / "candidate.txt"
        candidate.write_text(
            "\n".join(
                [
                    "User: Where do gorillas live?",
                    "Ickle: Gorillas live in tropical forests of equatorial Africa.",
                    "",
                    "User: Where do gorillas live?",
                    "Ickle: I can help with that. Share the exact outcome you want and any constraints.",
                    "",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        benchmark = td / "bench.json"
        benchmark.write_text(
            json.dumps(
                [
                    {
                        "name": "gorilla_habitat",
                        "prompt": "Where do gorillas live?",
                        "keywords": ["gorilla", "africa", "forest"],
                    }
                ]
            ),
            encoding="utf-8",
        )
        out = td / "focus.txt"
        report = build_focus_corpus_file(
            out_path=str(out),
            benchmark_file=str(benchmark),
            candidate_paths=[str(candidate)],
            max_pairs=20,
            seed=7,
        )
        text = out.read_text(encoding="utf-8").lower()
        self.assertGreaterEqual(int(report["written_pairs"]), 1)
        self.assertIn("where do gorillas live?", text)
        self.assertIn("equatorial africa", text)
        self.assertNotIn("share the exact outcome", text)

    def test_merge_dialog_corpora_deduplicates(self):
        td = self._tmpdir()
        a = td / "a.txt"
        b = td / "b.txt"
        a.write_text(
            "User: What is entropy?\nIckle: Entropy is a measure of disorder.\n\n",
            encoding="utf-8",
        )
        b.write_text(
            "\n".join(
                [
                    "User: What is entropy?",
                    "Ickle: Entropy is a measure of disorder.",
                    "",
                    "User: What causes earthquakes?",
                    "Ickle: Earthquakes happen when stress is released along faults.",
                    "",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        out = td / "merged.txt"
        report = merge_dialog_corpora(
            corpus_paths=[str(a), str(b)],
            out_path=str(out),
            max_pairs_per_source=50,
            max_total_pairs=50,
            seed=7,
        )
        text = out.read_text(encoding="utf-8").lower()
        self.assertEqual(int(report["written_pairs"]), 2)
        self.assertEqual(text.count("user: what is entropy?"), 1)


if __name__ == "__main__":
    unittest.main()
