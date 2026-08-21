"""Image understanding as a bolt-on tool, not a core-model capability.

Ickle's own model (src/model.py) is and stays a text-only decoder transformer
-- there is no plan to retrain it as multimodal. Instead, like web_read/
news_research/etc., this wraps external libraries and returns plain text
that gets fed into the text-only model as context. Both functions are
optional: `pip install -r requirements-vision.txt` to enable them. Without
that, callers get a clear ImageToolsUnavailable error instead of a crash or
a silently wrong answer -- consistent with the project's capability-honesty
principle (src/capabilities.py).
"""

from __future__ import annotations

import threading


class ImageToolsUnavailable(RuntimeError):
    def __init__(self, missing: str):
        super().__init__(
            f"Image understanding needs the optional '{missing}' package. "
            f"Install with: pip install -r requirements-vision.txt"
        )


_ocr_reader = None
_ocr_lock = threading.Lock()


def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is not None:
        return _ocr_reader
    with _ocr_lock:
        if _ocr_reader is None:
            try:
                import easyocr
            except ImportError as exc:
                raise ImageToolsUnavailable("easyocr") from exc
            # English only by default; gpu=False keeps this usable on any
            # machine (matches the project's CPU-first, no-accelerator-required
            # stance elsewhere -- e.g. src/device_bridge.py's CPU fallback).
            _ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _ocr_reader


def extract_text_from_image(image_path: str) -> str:
    """Extract any visible text from an image (screenshot, photo of a
    document/whiteboard/sign, etc.) via OCR. Returns an empty string if no
    text is found -- that's a normal result, not an error."""
    reader = _get_ocr_reader()
    results = reader.readtext(image_path, detail=0, paragraph=True)
    return "\n".join(str(line).strip() for line in results if str(line).strip())


_captioner = None
_captioner_lock = threading.Lock()
DEFAULT_CAPTION_MODEL = "Salesforce/blip-image-captioning-base"


def _get_captioner():
    """Returns (processor, model). Uses the BLIP model classes directly
    rather than transformers' pipeline("image-to-text", ...) convenience
    wrapper -- that pipeline task name was silently renamed to
    "image-text-to-text" in a newer transformers release and broke this
    with a raw KeyError, found via live testing, not just unit tests.
    Going directly to the model classes sidesteps that pipeline-task-name
    churn entirely."""
    global _captioner
    if _captioner is not None:
        return _captioner
    with _captioner_lock:
        if _captioner is None:
            try:
                from transformers import BlipForConditionalGeneration, BlipProcessor
            except ImportError as exc:
                raise ImageToolsUnavailable("transformers") from exc
            # Downloaded from Hugging Face on first use and cached locally,
            # the same on-demand-fetch pattern src/open_dataset_ingest.py
            # already uses for training corpora -- not bundled with Ickle.
            processor = BlipProcessor.from_pretrained(DEFAULT_CAPTION_MODEL)
            model = BlipForConditionalGeneration.from_pretrained(DEFAULT_CAPTION_MODEL)
            _captioner = (processor, model)
    return _captioner


def describe_image(image_path: str) -> str:
    """Generate a short natural-language description of what's in an image
    ("a person walking a dog on a beach"), for images where the content
    matters more than any text in it. Complements extract_text_from_image
    rather than replacing it -- callers typically want both."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImageToolsUnavailable("Pillow") from exc

    processor, model = _get_captioner()
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    output_ids = model.generate(**inputs, max_new_tokens=40)
    caption = processor.decode(output_ids[0], skip_special_tokens=True)
    return caption.strip()


def image_tools_available() -> dict[str, bool]:
    """Runtime check (not a static claim) of which optional vision
    dependencies are actually importable right now."""
    availability = {"ocr": False, "captioning": False}
    try:
        import easyocr  # noqa: F401

        availability["ocr"] = True
    except ImportError:
        pass
    try:
        import transformers  # noqa: F401
        from PIL import Image  # noqa: F401

        availability["captioning"] = True
    except ImportError:
        pass
    return availability
