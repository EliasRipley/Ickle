import unittest
from types import SimpleNamespace
from unittest import mock

from src.knowledge_modules import ResolvedKnowledgeModule
from src.ilm_chat_utils import _is_contextual_followup
from src.ilm_chat import (
    _extract_response_text,
    _is_noisy_memory_fact,
    _looks_low_quality_response,
    _memory_knowledge_response,
    _prompt_relevance_score,
    _topic_overlap_count,
    detect_web_request,
    generate_response,
)


class ILMChatTests(unittest.TestCase):
    def test_low_quality_response_detects_repeated_fragments(self):
        self.assertTrue(_looks_low_quality_response("lelelelelelelele"))
        self.assertTrue(_looks_low_quality_response("word " + "x" * 40))

    def test_generate_response_uses_local_reasoning_without_loading_model(self):
        args = SimpleNamespace(
            model="models/ickle_v5_bpe.best.pt",
            prompt="What is 17 multiplied by 6?",
            max_new=120,
            max_new_limit=240,
            temperature=0.5,
            top_k=20,
            torch_threads=1,
            skill="",
            enable_memory=False,
            enable_web_tools=False,
        )
        with mock.patch("src.ilm_chat._load_model_bundle") as mocked_loader:
            out = generate_response(args)
        self.assertEqual(out["response"], "17 \u00d7 6 = 102.")
        mocked_loader.assert_not_called()

    def test_detect_web_request_direct_url(self):
        prompt = "Please check https://example.com/docs?page=1."
        self.assertEqual(detect_web_request(prompt), "https://example.com/docs?page=1")

    def test_detect_web_request_domain_phrase(self):
        prompt = "Can you visit example.org/help for me?"
        self.assertEqual(detect_web_request(prompt), "https://example.org/help")

    def test_extract_response_text_strips_speaker_and_turn_markers(self):
        raw = "Ickle: Sure, let's do that.\nUser: next question"
        self.assertEqual(_extract_response_text(raw), "Sure, let's do that.")

    def test_extract_response_text_drops_heading_noise(self):
        raw = "BASIC GREETINGS\nHello there."
        self.assertEqual(_extract_response_text(raw), "Hello there.")

    def test_noise_memory_fact_detection(self):
        self.assertTrue(_is_noisy_memory_fact("Wikipedia does not have an article with this exact name"))
        self.assertFalse(_is_noisy_memory_fact("Probability theory studies uncertainty and random events."))

    def test_prompt_relevance_score_higher_for_related_response(self):
        prompt = "What time is it in Japan right now?"
        related = "Japan uses UTC plus 9 and I can check current time with a tool."
        unrelated = "I can compare complexity, performance, maintainability, and risk."
        self.assertGreater(_prompt_relevance_score(prompt, related), _prompt_relevance_score(prompt, unrelated))

    def test_prompt_relevance_score_handles_accented_tokens(self):
        prompt = "What is El Nino?"
        related = "El Niño is a climate pattern in the Pacific Ocean."
        self.assertGreater(_prompt_relevance_score(prompt, related), 0.3)

    def test_memory_knowledge_response_supports_what_causes_prompt(self):
        fake_memory = mock.Mock()
        fake_memory.search_research_notes.return_value = []
        fake_memory.search_web_facts.return_value = [
            {
                "fact": "Major ocean currents are driven by winds and density differences in seawater.",
                "topic": "ocean currents",
                "source_title": "Ocean current",
            }
        ]
        fake_memory.search_facts.return_value = []
        out = _memory_knowledge_response(fake_memory, "What causes major ocean currents?")
        self.assertIsNotNone(out)
        self.assertIn("ocean currents", out.lower())

    def test_generate_response_uses_model_path_after_memory_evidence_context(self):
        fake_memory = mock.Mock()
        fake_memory.search_research_notes.return_value = [
            {
                "finding": "An astrolabe is used to measure celestial angles for navigation.",
                "source_title": "Astrolabe",
                "topic": "medieval astronomy",
                "question": "What is an astrolabe used for?",
            }
        ]
        fake_memory.search_web_facts.return_value = []
        fake_memory.search_facts.return_value = []
        fake_memory.get_owner_info.return_value = {}
        fake_memory.get_recent_context.return_value = []

        args = SimpleNamespace(
            model="models/ickle_brain_base_v3.pt",
            prompt="What is an astrolabe used for?",
            max_new=120,
            max_new_limit=240,
            temperature=0.5,
            top_k=20,
            torch_threads=1,
            skill="",
            enable_memory=True,
            enable_web_tools=False,
        )

        with (
            mock.patch("src.ilm_chat.get_memory", return_value=fake_memory),
            mock.patch("src.ilm_chat._load_model_bundle", return_value=(object(), object())),
            mock.patch(
                "src.ilm_chat._generate_model_response",
                return_value="An astrolabe measures celestial angles for navigation.",
            ),
        ):
            out = generate_response(args)
        self.assertIn("astrolabe", out["response"].lower())

    def test_topic_overlap_count_ignores_generic_words(self):
        prompt = "Explain mitosis in simple terms."
        off_topic = "Linear algebra studies vectors and matrices in simple terms."
        on_topic = "Mitosis is cell division where one cell splits into two."
        self.assertEqual(_topic_overlap_count(prompt, off_topic), 0)
        self.assertGreater(_topic_overlap_count(prompt, on_topic), 0)

    def test_contextual_followup_detection(self):
        self.assertTrue(_is_contextual_followup("And what is that relative to UTC?"))
        self.assertFalse(_is_contextual_followup("Explain operating systems."))

    def test_generate_response_followup_prefers_model_not_memory_shortcut(self):
        fake_memory = mock.Mock()
        fake_memory.search_research_notes.return_value = [
            {"finding": "Relative plate motion can be lateral.", "topic": "plate tectonics", "question": "q"}
        ]
        fake_memory.search_web_facts.return_value = [
            {"fact": "Relative plate motion can be lateral.", "topic": "plate tectonics", "source_title": "test"}
        ]
        fake_memory.search_facts.return_value = []
        fake_memory.get_owner_info.return_value = {}
        fake_memory.get_recent_context.return_value = [
            {
                "user_input": "Ickle can you please tell me the time in Japan?",
                "ickle_response": "Japan uses UTC plus 9.",
            }
        ]

        args = SimpleNamespace(
            model="models/ickle_brain_base_v3.pt",
            prompt="And what about that in one line?",
            max_new=120,
            max_new_limit=240,
            temperature=0.5,
            top_k=20,
            torch_threads=1,
            skill="",
            enable_memory=True,
            enable_web_tools=False,
        )

        with (
            mock.patch("src.ilm_chat.get_memory", return_value=fake_memory),
            mock.patch("src.ilm_chat._load_model_bundle", return_value=(object(), object())),
            mock.patch(
                "src.ilm_chat._generate_model_response",
                return_value="One line summary: Japan is UTC plus 9 hours.",
            ),
        ):
            out = generate_response(args)
        self.assertEqual(out["response"], "One line summary: Japan is UTC plus 9 hours.")

    def test_generate_response_returns_dict_structure(self):
        fake_memory = mock.Mock()
        fake_memory.get_recent_context.return_value = [
            {
                "user_input": "Ickle can you please tell me the time in Japan?",
                "ickle_response": "Japan uses UTC plus 9, and I can check current time with tools.",
            }
        ]
        fake_memory.search_research_notes.return_value = []
        fake_memory.search_web_facts.return_value = []
        fake_memory.search_facts.return_value = []
        fake_memory.get_owner_info.return_value = {}

        args = SimpleNamespace(
            model="models/ickle_brain_base_v3.pt",
            prompt="And what is that relative to UTC?",
            max_new=120,
            max_new_limit=240,
            temperature=0.5,
            top_k=20,
            torch_threads=1,
            skill="",
            enable_memory=True,
            enable_web_tools=False,
        )

        with (
            mock.patch("src.ilm_chat.get_memory", return_value=fake_memory),
            mock.patch("src.ilm_chat._load_model_bundle", return_value=(object(), object())),
            mock.patch(
                "src.ilm_chat._generate_model_response",
                return_value="Japan is UTC plus 9 hours.",
            ),
        ):
            out = generate_response(args)
        self.assertIsInstance(out, dict)
        self.assertIn("response", out)
        self.assertIn("model", out)
        self.assertIn("reasoning", out)

    def test_low_quality_response_is_flagged_not_replaced(self):
        """Ickle's real generated text is always what's shown -- a weak/
        garbled response is never swapped for a canned template string or
        synthesized web-snippet prose (that used to happen here and made
        gated answers indistinguishable from hardcoded ones). The quality
        gate now only flags the result via low_confidence."""
        garbled = "word word word " * 20  # trips _looks_low_quality_response's repetition check
        args = SimpleNamespace(
            model="models/ickle_brain_base_v3.pt",
            prompt="Tell me about oceans.",
            max_new=120,
            max_new_limit=240,
            temperature=0.5,
            top_k=20,
            torch_threads=1,
            skill="",
            enable_memory=False,
            enable_web_tools=False,
        )
        with (
            mock.patch("src.ilm_chat._load_model_bundle", return_value=(object(), object())),
            mock.patch("src.ilm_chat._generate_model_response", return_value=garbled),
        ):
            out = generate_response(args)
        self.assertEqual(out["response"], garbled)
        self.assertTrue(out["low_confidence"])

    def test_raw_output_bypasses_quality_gate(self):
        """Regression coverage for the "Show raw output" toggle: with
        raw_output=True, even response text that would normally trip the
        quality/relevance gate and get replaced with a canned uncertainty
        message must be returned verbatim -- added specifically so a
        freshly trained model's real (possibly weak) output isn't
        indistinguishable from a hardcoded response."""
        garbled = "word word word " * 20
        args = SimpleNamespace(
            model="models/ickle_brain_base_v3.pt",
            prompt="Tell me about oceans.",
            max_new=120,
            max_new_limit=240,
            temperature=0.5,
            top_k=20,
            torch_threads=1,
            skill="",
            enable_memory=False,
            enable_web_tools=False,
            raw_output=True,
        )
        with (
            mock.patch("src.ilm_chat._load_model_bundle", return_value=(object(), object())),
            mock.patch("src.ilm_chat._generate_model_response", return_value=garbled),
        ):
            out = generate_response(args)
        self.assertEqual(out["response"], garbled)

    def test_generate_response_passes_selected_knowledge_modules_to_loader(self):
        args = SimpleNamespace(
            model="models/ickle_brain_base_v3.pt",
            prompt="Explain mitosis in one sentence.",
            max_new=120,
            max_new_limit=240,
            temperature=0.5,
            top_k=20,
            torch_threads=1,
            skill="",
            enable_memory=False,
            enable_web_tools=False,
            knowledge_registry="data/knowledge/module_registry.json",
            knowledge_modules="",
            knowledge_max_modules=2,
            knowledge_auto=True,
        )

        selected = [ResolvedKnowledgeModule(module_id="bio", path="models/bio_module.pt", weight=1.0)]

        with (
            mock.patch("src.ilm_chat.resolve_runtime_modules", return_value=selected),
            mock.patch("src.ilm_chat._load_model_bundle", return_value=(object(), object())) as mocked_loader,
            mock.patch("src.ilm_chat._generate_model_response", return_value="Mitosis is cell division into two cells."),
        ):
            out = generate_response(args)

        self.assertIn("mitosis", out["response"].lower())
        mocked_loader.assert_called_once()
        self.assertEqual(mocked_loader.call_args.kwargs.get("module_specs"), selected)

    def test_generate_response_uses_auto_web_knowledge_for_topic_prompts(self):
        args = SimpleNamespace(
            model="models/ickle_brain_base_v3.pt",
            prompt="What is an event horizon?",
            max_new=120,
            max_new_limit=240,
            temperature=0.5,
            top_k=20,
            torch_threads=1,
            skill="",
            enable_memory=False,
            enable_web_tools=True,
            knowledge_registry="data/knowledge/module_registry.json",
            knowledge_modules="",
            knowledge_max_modules=2,
            knowledge_auto=True,
            auto_web_knowledge=True,
            web_knowledge_max_sources=2,
            web_knowledge_timeout_ms=5000,
        )

        topic_payload = {
            "success": True,
            "topic": "event horizon",
            "sources": [
                {
                    "url": "https://example.com/space",
                    "title": "Event Horizon Basics",
                    "relevance": 0.88,
                    "facts": ["An event horizon is the boundary around a black hole."],
                    "evidence_items": [],
                    "quality": {"score": 0.7},
                }
            ],
            "facts": ["An event horizon is the boundary around a black hole."],
        }

        with (
            mock.patch("src.ilm_chat.resolve_runtime_modules", return_value=[]),
            mock.patch("src.ilm_chat._load_model_bundle", return_value=(object(), object())),
            mock.patch("src.ilm_chat.collect_topic_web_evidence", return_value=topic_payload) as mocked_collect,
            mock.patch("src.ilm_chat._generate_model_response", return_value="Event horizons are boundaries around black holes.") as mocked_generate,
        ):
            out = generate_response(args)

        self.assertIn("event horizon", out["response"].lower())
        mocked_collect.assert_called_once()
        rendered_prompt = mocked_generate.call_args.args[2]
        self.assertIn("Topic: event horizon", rendered_prompt)


if __name__ == "__main__":
    unittest.main()
