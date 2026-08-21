import unittest
import json
from pathlib import Path
import tempfile

from src.web_topics import (
    _collect_known_training_topics,
    _dedupe_preserve_order,
    _default_auto_pipeline_model_path,
    _default_topic_queue_path,
    _load_subject_catalog,
    _pick_subject_seed_topics,
    _topic_is_covered,
    _topic_from_benchmark_row,
)


class ServeWebTests(unittest.TestCase):
    def test_default_auto_pipeline_model_path_includes_topic_and_task(self):
        path = _default_auto_pipeline_model_path("t_abc123def456", "Ocean currents and El Nino")
        self.assertTrue(path.startswith("models/ickle_auto_"))
        self.assertTrue(path.endswith(".pt"))
        self.assertIn("ocean_currents_and_el_nino", path)
        self.assertIn("t_abc123def4", path)

    def test_default_auto_pipeline_model_path_is_unique_per_task(self):
        a = _default_auto_pipeline_model_path("t_task_a", "medieval Islamic astronomy")
        b = _default_auto_pipeline_model_path("t_task_b", "medieval Islamic astronomy")
        self.assertNotEqual(a, b)

    def test_default_topic_queue_path_contains_slug(self):
        path = _default_topic_queue_path("data/.tmp_tests/serve_web", "Ocean currents and El Nino")
        p = Path(path)
        self.assertTrue(str(p).endswith("ocean_currents_and_el_nino.txt"))
        self.assertIn("topic_queues", str(p))

    def test_topic_from_benchmark_row_prefers_keywords(self):
        topic = _topic_from_benchmark_row(
            {
                "name": "earthquake_basic",
                "prompt": "What causes earthquakes?",
                "keywords": ["tectonic", "plate", "fault", "stress"],
            }
        )
        self.assertEqual(topic, "tectonic plate fault")

    def test_topic_from_benchmark_row_handles_memory_followup(self):
        topic = _topic_from_benchmark_row(
            {
                "name": "memory_followup_2",
                "prompt": "And what is that relative to UTC?",
                "keywords": ["utc+09:00", "utc", "9"],
            }
        )
        self.assertIn("Japan Standard Time", topic)

    def test_dedupe_preserve_order(self):
        out = _dedupe_preserve_order(["  Alpha  ", "beta", "alpha", "Beta", "", "gamma"])
        self.assertEqual(out, ["Alpha", "beta", "gamma"])

    def test_topic_is_covered_detects_overlap(self):
        known = ["gorilla africa forest and habitat", "plate tectonics and earthquakes"]
        self.assertTrue(_topic_is_covered("gorilla africa forest", known))
        self.assertFalse(_topic_is_covered("quantum field theory", known))

    def test_collect_known_training_topics_reads_topic_queues_and_queue_file(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            topic_dir = root / "topic_queues"
            topic_dir.mkdir(parents=True, exist_ok=True)
            (topic_dir / "colosseum_in_ancient_rome.txt").write_text(
                "User: We are studying Colosseum in ancient Rome. What from Colosseum is most reliable?\n",
                encoding="utf-8",
            )
            (root / "queued_wikipedia_learning.txt").write_text(
                "User: Tell me about gorilla habitats and gorilla social behavior from the internet.\n",
                encoding="utf-8",
            )
            out = _collect_known_training_topics(str(root))
        text = " | ".join(out).lower()
        self.assertIn("colosseum in ancient rome", text)
        self.assertIn("gorilla habitats and gorilla social behavior", text)

    def test_pick_subject_seed_topics_prefers_uncovered(self):
        catalog = {
            "version": "x",
            "subjects": [
                {"name": "life", "topics": ["gorilla habitat", "cell biology"]},
                {"name": "math", "topics": ["probability theory", "linear algebra"]},
            ],
        }
        selected, meta = _pick_subject_seed_topics(
            catalog=catalog,
            known_topics=["gorilla social behavior", "probability theory"],
            max_topics=3,
        )
        joined = " | ".join(selected).lower()
        self.assertIn("cell biology", joined)
        self.assertIn("linear algebra", joined)
        self.assertIn("subjects", meta)

    def test_load_subject_catalog_accepts_domain_schema(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "subject_catalog.json").write_text(
                json.dumps(
                    {
                        "version": "1",
                        "domains": [
                            {
                                "name": "Applied science",
                                "subjects": [
                                    {"name": "Medicine and health", "topics": ["cardiology", "epidemiology"]},
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            out = _load_subject_catalog(str(root))
        flat = list(out.get("subjects") or [])
        self.assertTrue(any(str(x.get("name", "")).startswith("Applied science:") for x in flat))
        self.assertTrue(any("cardiology" in list(x.get("topics") or []) for x in flat))


if __name__ == "__main__":
    unittest.main()
