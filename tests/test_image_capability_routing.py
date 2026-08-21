import unittest
from unittest import mock

from src.capabilities import check_capability


class ImageCapabilityRoutingTests(unittest.TestCase):
    """Regression test for a real routing bug: IMAGE_ALIASES originally listed
    the generic "image"/"photo"/"picture" aliases before the more specific
    captioning phrases, so a query like "describe this image" matched the
    bare "image" substring first and got routed to OCR instead of
    captioning. Caught by manual testing, not by the original (missing)
    test coverage for this new code."""

    def test_captioning_phrases_route_to_captioning_not_ocr(self):
        # Mocked so this test's outcome doesn't depend on whether
        # requirements-vision.txt happens to be installed in the environment
        # it runs in -- it's testing alias ROUTING, not dependency detection
        # (that's covered separately below).
        with mock.patch(
            "src.tools.image_reader.image_tools_available",
            return_value={"ocr": False, "captioning": False},
        ):
            for phrase in [
                "describe this image",
                "describe this photo",
                "what is in this photo",
                "what's in this image",
                "image recognition",
                "identify this image",
            ]:
                with self.subTest(phrase=phrase):
                    report = check_capability(phrase)
                    self.assertIn("description", report.summary.lower(), phrase)

    def test_ocr_phrases_route_to_ocr_not_captioning(self):
        with mock.patch(
            "src.tools.image_reader.image_tools_available",
            return_value={"ocr": False, "captioning": False},
        ):
            for phrase in [
                "extract text from image",
                "read image",
                "read this image",
                "ocr",
                "text in this image",
            ]:
                with self.subTest(phrase=phrase):
                    report = check_capability(phrase)
                    self.assertIn("text extraction", report.summary.lower(), phrase)

    def test_image_generation_request_is_not_falsely_claimed_as_supported(self):
        """The bug that mattered most: a bare "image" substring alias
        previously made this claim OCR support for a request Ickle cannot
        fulfill at all (these tools only read existing images, never
        generate one)."""
        for phrase in ["generate an image of a cat", "create a picture of a sunset", "draw me a photo"]:
            with self.subTest(phrase=phrase):
                report = check_capability(phrase)
                self.assertFalse(report.supported, phrase)

    def test_reports_supported_when_dependencies_are_installed(self):
        with mock.patch(
            "src.tools.image_reader.image_tools_available",
            return_value={"ocr": True, "captioning": True},
        ):
            self.assertTrue(check_capability("read image").supported)
            self.assertTrue(check_capability("describe this image").supported)

    def test_reports_unsupported_with_actionable_suggestion_when_missing(self):
        with mock.patch(
            "src.tools.image_reader.image_tools_available",
            return_value={"ocr": False, "captioning": False},
        ):
            report = check_capability("read image")
            self.assertFalse(report.supported)
            self.assertIn("requirements-vision.txt", report.suggestion)


if __name__ == "__main__":
    unittest.main()
