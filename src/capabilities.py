from dataclasses import dataclass

from src.icklization import ick


@dataclass
class CapabilityReport:
    supported: bool
    summary: str
    suggestion: str = ""


def _build_known() -> dict[str, CapabilityReport]:
    descs = ick.capability_descriptions()
    suggestions = ick.capability_suggestions()
    known: dict[str, CapabilityReport] = {}
    for key, desc in descs.items():
        known[key] = CapabilityReport(
            supported=True,
            summary=desc,
            suggestion=suggestions.get(key, ""),
        )
    return known


ALIASES = {
    "timer": "timer",
    "time it": "timer",
    "start timer": "timer",
    "set timer": "timer",
    "stopwatch": "timer",
    "countdown": "timer",
    "notepad": "write_notepad",
    "write note": "write_notepad",
    "web": "web_read",
    "read website": "web_read",
    "minecraft": "minecraft_guide",
    "news": "news_research",
    "desktop": "desktop_control",
    "desktop control": "desktop_control",
    "screenshot": "desktop_control",
    "take screenshot": "desktop_control",
    "move mouse": "desktop_control",
    "click": "desktop_control",
    "type": "desktop_control",
    "press key": "desktop_control",
}

# Image capabilities are checked at runtime, not claimed statically like the
# ALIASES-based ones above -- they depend on the optional
# requirements-vision.txt dependencies actually being installed, and
# capability-honesty means not claiming support that isn't really there.
#
# Deliberately no bare "image"/"photo"/"picture" catch-all: an earlier
# version had one, and it made check_capability("generate an image of a
# cat") falsely claim OCR support for an image-*generation* request Ickle
# cannot do at all (these tools only read existing images, never create
# one) -- a real capability-honesty violation caught by test_partner_loop.py
# unexpectedly failing after this was added. Every alias here must describe
# reading/understanding an existing image, not just mention "image" at all;
# a false "not supported" for a borderline phrase is far less harmful than
# a false "supported" for something Ickle genuinely can't do.
IMAGE_ALIASES = {
    # Order matters: substring matching below checks these in order and
    # takes the first hit, so the more specific "describe/recognize/
    # identify" phrases must come before anything shorter they contain.
    "describe image": "captioning", "describe the image": "captioning", "describe this image": "captioning",
    "describe this photo": "captioning", "describe this picture": "captioning",
    "what's in this image": "captioning", "what is in this image": "captioning",
    "what is in this photo": "captioning", "what's in this photo": "captioning",
    "image recognition": "captioning", "identify image": "captioning", "identify this image": "captioning",
    "read image": "ocr", "read the image": "ocr", "read this image": "ocr",
    "screenshot text": "ocr", "ocr": "ocr", "extract text from image": "ocr",
    "text in the image": "ocr", "text in this image": "ocr",
}


def _check_image_capability(kind: str) -> CapabilityReport:
    from src.tools.image_reader import image_tools_available

    available = image_tools_available()
    if available.get(kind):
        if kind == "ocr":
            return CapabilityReport(True, "Can extract visible text from an image (OCR).")
        return CapabilityReport(True, "Can generate a short description of what's in an image.")
    return CapabilityReport(
        False,
        f"Image {'text extraction' if kind == 'ocr' else 'description'} needs the optional vision "
        f"dependencies, which aren't installed in this environment.",
        "Install with: pip install -r requirements-vision.txt",
    )


def check_capability(task_text: str) -> CapabilityReport:
    lower = task_text.strip().lower()

    image_key = next((mapped for alias, mapped in IMAGE_ALIASES.items() if alias in lower), None)
    if image_key:
        return _check_image_capability(image_key)

    known = _build_known()
    key = next((mapped for alias, mapped in ALIASES.items() if alias in lower), None)
    if not key:
        return CapabilityReport(
            False,
            ick.unknown_capability_summary(),
            ick.unknown_capability_suggestion(),
        )
    return known.get(key, CapabilityReport(False, ick.unknown_capability_summary()))
