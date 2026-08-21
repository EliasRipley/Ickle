import unittest
import uuid
from pathlib import Path

from src.ilm_memory import ILMMemory


class ILMMemoryTests(unittest.TestCase):
    @staticmethod
    def _tmp_dir() -> Path:
        root = Path("data") / ".tmp_tests"
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"mem_{uuid.uuid4().hex}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def test_add_fact_deduplicates_same_entry(self):
        td = self._tmp_dir()
        mem = ILMMemory(memory_dir=str(td))
        mem.add_fact("The project codename is Atlas.", category="project", source="user")
        mem.add_fact("The   project codename is Atlas.", category="project", source="user")
        facts = mem.get_facts(category="project", limit=10)
        self.assertEqual(len(facts), 1)

    def test_search_facts_matches_by_token_overlap(self):
        td = self._tmp_dir()
        mem = ILMMemory(memory_dir=str(td))
        mem.add_fact("Atlas milestone is Friday.", category="project")
        mem.add_fact("Use concise bullet points for status updates.", category="preference")
        results = mem.search_facts("When is the atlas milestone?", limit=2)
        self.assertTrue(results)
        self.assertIn("Atlas milestone is Friday.", results[0]["fact"])

    def test_remember_conversation_truncates_long_entries(self):
        td = self._tmp_dir()
        mem = ILMMemory(memory_dir=str(td))
        long_text = "x" * 5000
        mem.remember_conversation(long_text, long_text, {})
        recent = mem.get_recent_context(limit=1)
        self.assertEqual(len(recent), 1)
        self.assertLessEqual(len(recent[0]["user_input"]), 1200)
        self.assertLessEqual(len(recent[0]["ickle_response"]), 1800)

    def test_add_web_learning_avoids_duplicate_topic_facts(self):
        td = self._tmp_dir()
        mem = ILMMemory(memory_dir=str(td))
        mem.add_web_learning(
            url="https://example.com",
            title="Example Domain",
            key_facts=["Fact A", "Fact A", "Fact B"],
            topic="Example",
        )
        topic = mem.get_web_learning("Example")
        self.assertEqual(sorted(topic["facts"]), ["Fact A", "Fact B"])

    def test_search_web_facts_prefers_relevant_topic(self):
        td = self._tmp_dir()
        mem = ILMMemory(memory_dir=str(td))
        mem.add_web_learning(
            url="https://example.com/literature",
            title="Literature",
            key_facts=["Romanticism is a literary movement in the late 18th century."],
            topic="English literature",
        )
        mem.add_web_learning(
            url="https://example.com/space",
            title="Black hole",
            key_facts=["An event horizon is a boundary in spacetime around a black hole."],
            topic="Black holes",
        )
        rows = mem.search_web_facts("What is an event horizon?", limit=3)
        self.assertTrue(rows)
        self.assertIn("event horizon", rows[0]["fact"].lower())

    def test_research_notes_can_be_searched(self):
        td = self._tmp_dir()
        mem = ILMMemory(memory_dir=str(td))
        sid = mem.add_research_note(
            topic="English literature",
            question="What is Romanticism?",
            finding="Romanticism emphasized emotion and individual imagination.",
            source_url="https://en.wikipedia.org/wiki/Romanticism",
            source_title="Romanticism",
            tags=["wikipedia", "literature"],
            confidence=0.8,
        )
        self.assertTrue(sid)
        rows = mem.search_research_notes("romanticism imagination", limit=3)
        self.assertTrue(rows)
        self.assertIn("romanticism", rows[0]["finding"].lower())
        sessions = mem.list_research_sessions(limit=5)
        self.assertTrue(any(s.get("session_id") == sid for s in sessions))

    def test_search_web_facts_matches_accented_terms(self):
        td = self._tmp_dir()
        mem = ILMMemory(memory_dir=str(td))
        mem.add_web_learning(
            url="https://example.com/el-nino",
            title="ENSO",
            key_facts=["El Niño is the warm phase of ENSO and can shift weather patterns."],
            topic="ocean currents and El Nino",
        )
        rows = mem.search_web_facts("what is el nino", limit=3)
        self.assertTrue(rows)
        self.assertIn("enso", rows[0]["fact"].lower())

    def test_search_research_notes_uses_topic_hint_for_sparse_overlap(self):
        td = self._tmp_dir()
        mem = ILMMemory(memory_dir=str(td))
        mem.add_research_note(
            topic="ocean currents and El Nino",
            question="What is ENSO?",
            finding="El Niño is the warm phase of ENSO and can shift global weather patterns.",
            source_url="https://example.com/enso",
            source_title="ENSO",
            tags=["web", "enso"],
            confidence=0.8,
        )
        rows = mem.search_research_notes(
            "What is El Nino and how does it affect weather?",
            limit=3,
            topic_hint="el nino weather",
        )
        self.assertTrue(rows)
        self.assertIn("el niño", rows[0]["finding"].lower())

    def test_search_research_notes_hint_matches_finding_content(self):
        td = self._tmp_dir()
        mem = ILMMemory(memory_dir=str(td))
        mem.add_research_note(
            topic="medieval Islamic astronomy",
            question="What is important about astronomy in the medieval Islamic world?",
            finding="The astrolabe was used to measure celestial positions and assist navigation.",
            source_url="https://example.com/astrolabe",
            source_title="Astrolabe",
            tags=["wikipedia"],
            confidence=0.8,
        )
        rows = mem.search_research_notes(
            "What is an astrolabe used for?",
            limit=3,
            topic_hint="astrolabe used",
        )
        self.assertTrue(rows)
        self.assertIn("astrolabe", rows[0]["finding"].lower())

    def test_prune_nonsense_clears_short_term_and_noisy_entries(self):
        td = self._tmp_dir()
        mem = ILMMemory(memory_dir=str(td))
        mem.remember_conversation("hello", "hi there", {})
        mem.add_fact("Wikipedia does not have an article with this exact name", category="web", confidence=0.9)
        mem.add_fact("Useful project note for launch readiness.", category="project", confidence=0.9)
        mem.add_research_note(
            topic="test",
            question="q",
            finding="From Wikipedia, the free encyclopedia blah",
            source_url="https://example.com",
            source_title="Noise",
            confidence=0.9,
        )
        stats = mem.prune_nonsense(clear_short_term=True, min_fact_confidence=0.5, min_research_confidence=0.5)
        self.assertGreaterEqual(stats["facts_removed"], 1)
        self.assertGreaterEqual(stats["research_notes_removed"], 1)
        self.assertFalse(mem.get_recent_context(limit=5))


if __name__ == "__main__":
    unittest.main()
