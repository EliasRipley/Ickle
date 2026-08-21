import unittest

from src.continual_learn import should_trigger_retrain


class ContinualLearnTests(unittest.TestCase):
    def test_trigger_threshold(self):
        self.assertFalse(should_trigger_retrain(100, 110, 20))
        self.assertTrue(should_trigger_retrain(100, 125, 20))


if __name__ == "__main__":
    unittest.main()
