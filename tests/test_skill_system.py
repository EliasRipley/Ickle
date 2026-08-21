import tempfile
import unittest

from src.skill_system import SkillCard, SkillCardValidationError, SkillRegistry, validate_skill_card


class SkillCardValidationTests(unittest.TestCase):
    """Regression coverage for a real gap: reality_check.py's
    "non_python_artifacts" check only tested that schemas/skill_card.schema.json
    existed on disk -- no skill card was ever actually validated against it,
    at registration or anywhere else. An empty name or a card missing a
    required field would silently persist."""

    def test_empty_name_is_rejected(self):
        with self.assertRaises(SkillCardValidationError):
            validate_skill_card({
                "name": "",
                "description": "x",
                "corpus_path": "c",
                "model_path": "m",
                "activation_prompt": "a",
            })

    def test_missing_required_field_is_rejected(self):
        with self.assertRaises(SkillCardValidationError):
            validate_skill_card({
                "name": "french",
                "description": "x",
                "corpus_path": "c",
                # model_path missing
                "activation_prompt": "a",
            })

    def test_extra_field_is_rejected(self):
        with self.assertRaises(SkillCardValidationError):
            validate_skill_card({
                "name": "french",
                "description": "x",
                "corpus_path": "c",
                "model_path": "m",
                "activation_prompt": "a",
                "unexpected_field": "nope",
            })

    def test_register_skill_with_empty_name_raises(self):
        with tempfile.TemporaryDirectory() as td:
            reg = SkillRegistry(root=td)
            card = SkillCard(
                name="",
                description="x",
                corpus_path="c",
                model_path="m",
                activation_prompt="a",
            )
            with self.assertRaises(SkillCardValidationError):
                reg.register_skill(card)
            self.assertEqual(reg.list_skills(), [])


class SkillSystemTests(unittest.TestCase):
    def test_register_and_get_skill(self):
        with tempfile.TemporaryDirectory() as td:
            reg = SkillRegistry(root=td)
            card = SkillCard(
                name="french",
                description="French language skill",
                corpus_path="data/french.txt",
                model_path="models/french.pt",
                activation_prompt="Use French mode",
            )
            reg.register_skill(card)
            self.assertIn("french", reg.list_skills())
            loaded = reg.get_skill("french")
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.model_path, "models/french.pt")
            self.assertEqual(reg.activation_prompt("french"), "Use French mode")


if __name__ == "__main__":
    unittest.main()
