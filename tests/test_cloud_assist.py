import os
import unittest

from src.cloud_assist import cloud_is_configured, cloud_status_text


class CloudAssistTests(unittest.TestCase):
    def test_status_without_key(self):
        old = os.environ.pop("ILM_CLOUD_API_KEY", None)
        try:
            self.assertFalse(cloud_is_configured())
            self.assertIn("NOT configured", cloud_status_text())
        finally:
            if old is not None:
                os.environ["ILM_CLOUD_API_KEY"] = old


if __name__ == "__main__":
    unittest.main()
