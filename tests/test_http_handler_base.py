import unittest

from src.http_handler_base import DEFAULT_MAX_JSON_BYTES, is_local_cors_origin


class IsLocalCorsOriginTests(unittest.TestCase):
    def test_localhost_and_loopback_origins_are_allowed(self):
        for origin in [
            "http://127.0.0.1:8787",
            "https://127.0.0.1:9999",
            "http://localhost:8787",
            "HTTP://127.0.0.1:1",
        ]:
            with self.subTest(origin=origin):
                self.assertTrue(is_local_cors_origin(origin))

    def test_external_and_empty_origins_are_rejected(self):
        for origin in [
            "",
            "https://evil.example.com",
            "http://127.0.0.1.evil.com",
            "http://10.0.0.5:8787",
            "null",
        ]:
            with self.subTest(origin=origin):
                self.assertFalse(is_local_cors_origin(origin))

    def test_default_json_size_cap_is_24_mib(self):
        # Accommodates a base64-encoded chat image attachment; documented
        # here so a future change to this constant is a deliberate edit,
        # not an accidental one.
        self.assertEqual(DEFAULT_MAX_JSON_BYTES, 24 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
