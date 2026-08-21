from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_HAVE_AUTOGUI = False
try:
    import pyautogui as _pag
    _HAVE_AUTOGUI = True
except ImportError:
    pass


ALLOWED_ACTIONS = frozenset({
    "screenshot",
    "click",
    "type_text",
    "key_press",
    "move_mouse",
    "get_mouse_position",
    "get_screen_size",
    "desktop_send",
})


class DesktopControlBlocked(Exception):
    pass


@dataclass
class DesktopActionResult:
    success: bool
    action: str
    message: str
    data: Any = None


def _require_autogui():
    if not _HAVE_AUTOGUI:
        raise DesktopControlBlocked(
            "pyautogui is not installed. Run: pip install pyautogui"
        )


def screenshot(region: tuple[int, int, int, int] | None = None, save_path: str | None = None) -> DesktopActionResult:
    _require_autogui()
    try:
        img = _pag.screenshot(region=region)
        if save_path:
            img.save(save_path)
            return DesktopActionResult(True, "screenshot", f"Screenshot saved to {save_path}", {"path": save_path})
        return DesktopActionResult(True, "screenshot", "Screenshot captured", {"size": img.size})
    except Exception as e:
        return DesktopActionResult(False, "screenshot", str(e))


def move_mouse(x: int, y: int, duration: float = 0.5) -> DesktopActionResult:
    _require_autogui()
    try:
        _pag.moveTo(x, y, duration=duration)
        return DesktopActionResult(True, "move_mouse", f"Moved mouse to ({x}, {y})")
    except Exception as e:
        return DesktopActionResult(False, "move_mouse", str(e))


def click(x: int, y: int, button: str = "left", clicks: int = 1) -> DesktopActionResult:
    _require_autogui()
    try:
        _pag.click(x, y, button=button, clicks=clicks)
        return DesktopActionResult(True, "click", f"Clicked ({x}, {y}) with {button} button")
    except Exception as e:
        return DesktopActionResult(False, "click", str(e))


def type_text(text: str, interval: float = 0.05) -> DesktopActionResult:
    _require_autogui()
    try:
        text = str(text)[:5000]
        _pag.typewrite(text, interval=interval)
        return DesktopActionResult(True, "type_text", f"Typed {len(text)} characters")
    except Exception as e:
        return DesktopActionResult(False, "type_text", str(e))


def key_press(keys: str) -> DesktopActionResult:
    _require_autogui()
    try:
        _pag.hotkey(*keys.split("+"))
        return DesktopActionResult(True, "key_press", f"Pressed keys: {keys}")
    except Exception as e:
        return DesktopActionResult(False, "key_press", str(e))


def get_mouse_position() -> DesktopActionResult:
    _require_autogui()
    try:
        x, y = _pag.position()
        return DesktopActionResult(True, "get_mouse_position", f"({x}, {y})", {"x": x, "y": y})
    except Exception as e:
        return DesktopActionResult(False, "get_mouse_position", str(e))


def get_screen_size() -> DesktopActionResult:
    _require_autogui()
    try:
        w, h = _pag.size()
        return DesktopActionResult(True, "get_screen_size", f"{w}x{h}", {"width": w, "height": h})
    except Exception as e:
        return DesktopActionResult(False, "get_screen_size", str(e))


def desktop_send(text: str) -> DesktopActionResult:
    _require_autogui()
    try:
        _pag.write(text, interval=0.02)
        _pag.press("enter")
        return DesktopActionResult(True, "desktop_send", f"Sent: {text[:80]}")
    except Exception as e:
        return DesktopActionResult(False, "desktop_send", str(e))


DISPATCH = {
    "screenshot": screenshot,
    "click": click,
    "type_text": type_text,
    "key_press": key_press,
    "move_mouse": move_mouse,
    "get_mouse_position": get_mouse_position,
    "get_screen_size": get_screen_size,
    "desktop_send": desktop_send,
}
