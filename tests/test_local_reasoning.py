import unittest

from src.local_reasoning import local_reasoning_response


class LocalReasoningTests(unittest.TestCase):
    def test_arithmetic(self):
        self.assertEqual(local_reasoning_response("What is 17 multiplied by 6?"), "17 \u00d7 6 = 102.")

    def test_arithmetic_rejects_code(self):
        self.assertIsNone(local_reasoning_response("Calculate __import__('os').system('whoami')"))

    def test_transitive_age_comparison(self):
        response = local_reasoning_response(
            "Alice is older than Bob, and Bob is older than Cara. Who is oldest and who is youngest?"
        )
        self.assertEqual(response, "Alice is the oldest, and Cara is the youngest.")

    def test_greeting_is_not_a_canned_response(self):
        # Regression coverage: this used to short-circuit to a fixed string
        # ("Hi. What would you like to work on?") before the model ever ran --
        # instant, un-generated, and indistinguishable from real output. A
        # greeting isn't a computable/checkable fact, so it must fall through
        # to the model like any other prompt.
        self.assertIsNone(local_reasoning_response("Hello"))

    def test_trivia_questions_fall_through_to_the_model(self):
        # These used to be hardcoded answers too (probability bounds, coin
        # flips, AGPL licensing) -- knowledge questions, not computation.
        self.assertIsNone(local_reasoning_response("What range can the probability of an event take?"))
        self.assertIsNone(local_reasoning_response("A fair coin landed heads five times. Is tails guaranteed next?"))


if __name__ == "__main__":
    unittest.main()
