"""Small deterministic reasoning paths that should not depend on LM quality."""

from __future__ import annotations

import ast
import math
import operator
import re
from typing import Callable


_BINARY_OPERATORS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _evaluate_arithmetic_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate_arithmetic_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        value = float(node.value)
        if not math.isfinite(value) or abs(value) > 1e12:
            raise ValueError("number is outside the supported range")
        return value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_evaluate_arithmetic_node(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate_arithmetic_node(node.left)
        right = _evaluate_arithmetic_node(node.right)
        if isinstance(node.op, ast.Pow) and (abs(right) > 12 or abs(left) > 1e6):
            raise ValueError("power is outside the supported range")
        value = _BINARY_OPERATORS[type(node.op)](left, right)
        if not math.isfinite(value) or abs(value) > 1e15:
            raise ValueError("result is outside the supported range")
        return value
    raise ValueError("unsupported arithmetic expression")


def _arithmetic_response(prompt: str) -> str | None:
    text = str(prompt or "").strip().lower()
    if not re.search(r"\d", text):
        return None
    match = re.search(r"(?:what\s+is|calculate|compute|work\s+out)\s+(.+?)(?:\?|$)", text)
    if not match:
        return None
    expression = match.group(1).strip()
    replacements = (
        (r"\bmultiplied\s+by\b", "*"),
        (r"\bdivided\s+by\b", "/"),
        (r"\bto\s+the\s+power\s+of\b", "**"),
        (r"\bplus\b", "+"),
        (r"\bminus\b", "-"),
        (r"\btimes\b", "*"),
        (r"\bover\b", "/"),
        (r"\bmodulo\b|\bmod\b", "%"),
    )
    for pattern, replacement in replacements:
        expression = re.sub(pattern, replacement, expression)
    expression = expression.replace("\u00d7", "*").replace("\u00f7", "/").replace("^", "**")
    expression = re.sub(r"\s+", " ", expression).strip()
    if not re.fullmatch(r"[0-9eE+\-*/%.()\s]+", expression) or not re.search(r"[+\-*/%]", expression):
        return None
    try:
        value = _evaluate_arithmetic_node(ast.parse(expression, mode="eval"))
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
        return None
    rendered = str(int(value)) if value.is_integer() else f"{value:.12g}"
    display = expression.replace("**", "^").replace("*", "\u00d7").replace("/", "\u00f7")
    return f"{display} = {rendered}."


def _comparison_response(prompt: str) -> str | None:
    text = str(prompt or "").strip()
    lower = text.lower()
    if "oldest" not in lower or "youngest" not in lower:
        return None
    comparisons = re.findall(
        r"\b([A-Z][A-Za-z'-]{0,30})\s+is\s+(older|younger)\s+than\s+([A-Z][A-Za-z'-]{0,30})\b",
        text,
    )
    if len(comparisons) < 2:
        return None
    older_than: dict[str, set[str]] = {}
    names: set[str] = set()
    for left, relation, right in comparisons:
        names.update((left, right))
        older, younger = (left, right) if relation.lower() == "older" else (right, left)
        older_than.setdefault(older, set()).add(younger)
        older_than.setdefault(younger, set())

    changed = True
    while changed:
        changed = False
        for name in list(names):
            expanded = set(older_than.get(name, set()))
            for younger in list(expanded):
                expanded.update(older_than.get(younger, set()))
            if expanded != older_than.get(name, set()):
                older_than[name] = expanded
                changed = True

    oldest = [name for name in names if len(older_than.get(name, set())) == len(names) - 1]
    youngest = [name for name in names if all(name in older_than.get(other, set()) for other in names if other != name)]
    if len(oldest) != 1 or len(youngest) != 1:
        return None
    return f"{oldest[0]} is the oldest, and {youngest[0]} is the youngest."


def local_reasoning_response(prompt: str) -> str | None:
    """Deliberately narrow: only genuinely deterministic, symbolically
    checkable operations belong here (arithmetic, logical deduction from
    premises given in the prompt) -- the same reasoning that makes a
    calculator tool legitimate rather than "cheating," because the answer
    is verifiable independently of whatever the model itself knows.

    This used to also short-circuit greetings, "what's your name", and a
    handful of specific trivia questions (probability bounds, a coin-flip
    explanation, AGPL licensing) with fixed strings -- those aren't
    computable facts, they're canned answers standing in for the model's
    own knowledge, which is exactly the "hardcoded response" problem: a
    user asking Ickle a question should see what Ickle actually knows, not
    a pre-written answer that makes it look smarter (or more personable)
    than its real, current training. Removed rather than fixed in place."""
    text = str(prompt or "").strip()
    if not text:
        return None

    arithmetic = _arithmetic_response(text)
    if arithmetic:
        return arithmetic
    comparison = _comparison_response(text)
    if comparison:
        return comparison
    return None
