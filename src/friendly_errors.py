"""Plain-language error messages for surfaces a non-technical user sees.

Ickle's web chat and control API used to forward `str(exc)` straight into
JSON responses, which the chat UI then rendered as if the assistant itself
had said it — a raw `KeyError: 'chat_model'` reads as terrifying nonsense to
someone who isn't a developer. The real exception should still be logged
server-side (callers keep doing `traceback.print_exc()`); this module only
controls what the *user* sees.
"""

from __future__ import annotations

_MESSAGES: tuple[tuple[type[BaseException], str], ...] = (
    (FileNotFoundError, "Ickle couldn't find one of its files. Try picking a different model, or restart Ickle."),
    (PermissionError, "Ickle doesn't have permission to do that on this device."),
    (TimeoutError, "That took too long and timed out. Please try again."),
    (ConnectionError, "Ickle lost its connection. Please try again."),
    (MemoryError, "This device ran out of memory for that request. Try a shorter message or a smaller model."),
    (InterruptedError, "That request was interrupted. Please try again."),
    (ValueError, "Ickle couldn't understand that request. Please try rephrasing it."),
    (KeyError, "Ickle couldn't understand that request. Please try rephrasing it."),
    (TypeError, "Ickle couldn't understand that request. Please try rephrasing it."),
)

_DEFAULT_MESSAGE = "Something went wrong on Ickle's end. The details were saved to the local log for troubleshooting."


def friendly_error_message(exc: BaseException) -> str:
    """Map an exception to a plain-language message safe to show end users.

    Server-side code should still log the real exception (e.g. via
    `traceback.print_exc()`) before calling this — this function is only
    about what gets sent back over the API/rendered in the UI.
    """
    for exc_type, message in _MESSAGES:
        if isinstance(exc, exc_type):
            return message
    return _DEFAULT_MESSAGE
