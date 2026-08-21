import unittest

from src.knowledge_extraction import extract_structured_knowledge


class FallbackMethodTaggingTests(unittest.TestCase):
    def test_no_teacher_is_tagged_as_fallback(self):
        result = extract_structured_knowledge("Some arbitrary corpus text with no real topic focus.")
        self.assertEqual(result["method"], "fallback")

    def test_fallback_still_produces_a_domain_description_field(self):
        result = extract_structured_knowledge("How AP reported in all formats from tornado-stricken regions")
        self.assertEqual(result["method"], "fallback")
        self.assertIn("AP reported", result["domain_description"])

    def test_teacher_raising_falls_back_and_is_tagged_accordingly(self):
        class BrokenTeacher:
            def generate_text(self, **kwargs):
                raise RuntimeError("teacher unavailable")

        result = extract_structured_knowledge("Some text.", teacher=BrokenTeacher())
        self.assertEqual(result["method"], "fallback")

    def test_teacher_returning_valid_json_is_tagged_as_teacher(self):
        class WorkingTeacher:
            def generate_text(self, **kwargs):
                return '{"entities": [], "relationships": [], "facts": [], "domain_description": "Acoustics and signal processing", "skills_identified": []}'

        result = extract_structured_knowledge("Some text about acoustics.", teacher=WorkingTeacher())
        self.assertEqual(result["method"], "teacher")
        self.assertEqual(result["domain_description"], "Acoustics and signal processing")


if __name__ == "__main__":
    unittest.main()
