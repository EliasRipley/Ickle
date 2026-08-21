import unittest
from uuid import uuid4
from pathlib import Path

from src.build_smart_corpus import build_smart_corpus


class BuildSmartCorpusTests(unittest.TestCase):
    def test_build_smart_corpus_filters_code_and_keeps_knowledge(self):
        root = Path("data/.tmp_tests") / f"smart_{uuid4().hex}"
        root.mkdir(parents=True, exist_ok=False)
        (root / "open_oasst1_stream.txt").write_text(
            "\n".join(
                [
                    "User: Where do gorillas live?",
                    "Ickle: Gorillas live in forests in central and eastern Africa.",
                    "",
                    "User: Write a Go websocket server.",
                    "Ickle: package main import ( \"github.com/gorilla/websocket\" )",
                    "",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (root / "open_openhermes_2_5_stream.txt").write_text(
            "\n".join(
                [
                    "User: Explain photosynthesis.",
                    "Ickle: Photosynthesis uses light energy to convert carbon dioxide and water into sugars and oxygen.",
                    "",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        pairs, stats = build_smart_corpus(training_root=root, max_pairs=20, seed=7)
        self.assertGreaterEqual(stats["selected_pairs"], 2)
        corpus_text = "\n".join([f"User: {p.user}\nIckle: {p.assistant}" for p in pairs]).lower()
        self.assertIn("where do gorillas live", corpus_text)
        self.assertNotIn("package main", corpus_text)

    def test_build_smart_corpus_avoids_duplication_and_meta_pairs(self):
        root = Path("data/.tmp_tests") / f"smart_meta_{uuid4().hex}"
        root.mkdir(parents=True, exist_ok=False)
        (root / "open_oasst1_stream.txt").write_text(
            "\n".join(
                [
                    "User: What is photosynthesis?",
                    "Ickle: Photosynthesis uses light to make sugars from water and carbon dioxide.",
                    "",
                    "User: How do earthquakes happen?",
                    "Ickle: Earthquakes happen when stress is released along faults between tectonic plates.",
                    "",
                    "User: Can you help?",
                    "Ickle: Yes. I will list practical options and note the consequences of each.",
                    "",
                    "User: What's your view on this?",
                    "Ickle: As an AI language model, I do not have personal opinions.",
                    "",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (root / "open_openhermes_2_5_stream.txt").write_text("", encoding="utf-8")

        pairs, _stats = build_smart_corpus(training_root=root, max_pairs=50, seed=7)
        pair_set = {(p.user.lower(), p.assistant.lower()) for p in pairs}
        self.assertEqual(len(pair_set), len(pairs))
        corpus_text = "\n".join([f"User: {p.user}\nIckle: {p.assistant}" for p in pairs]).lower()
        self.assertNotIn("as an ai language model", corpus_text)
        self.assertNotIn("yes. i will list practical options", corpus_text)


if __name__ == "__main__":
    unittest.main()
