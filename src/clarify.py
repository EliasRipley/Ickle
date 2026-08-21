from dataclasses import dataclass

from src.icklization import ick


@dataclass
class ClarificationResult:
    needs_clarification: bool
    question: str = ""


def _social_missing_questions() -> dict[str, str]:
    return {
        "platform": ick.clarification_question("social", "platform"),
        "audience": ick.clarification_question("social", "audience"),
        "goal": ick.clarification_question("social", "goal"),
        "offer": ick.clarification_question("social", "offer"),
    }


def _minecraft_missing_questions() -> dict[str, str]:
    return {
        "edition": ick.clarification_question("minecraft", "edition"),
        "platform": ick.clarification_question("minecraft", "platform"),
        "experience": ick.clarification_question("minecraft", "experience"),
        "goal": ick.clarification_question("minecraft", "goal"),
    }


def detect_vague_prompt(prompt: str) -> bool:
    """Only flags prompts matching a genuinely ambiguous directive (e.g.
    "handle it", "you decide") via vague_patterns. Used to also treat any
    prompt shorter than vague_min_length as vague regardless of content --
    that caught plain short messages ("Hello", "Thanks", "Why?") with no
    actual ambiguity to clarify, forcing a fixed clarification-question
    string in place of a real answer. Length alone isn't ambiguity."""
    p = prompt.strip().lower()
    if not p:
        return False
    patterns = ick.detection_list("vague_patterns")
    return any(pattern in p for pattern in patterns)


def social_brief_clarification(
    platform: str | None,
    audience: str | None,
    goal: str | None,
    offer: str | None,
) -> ClarificationResult:
    questions = _social_missing_questions()
    missing = []
    if not platform:
        missing.append(questions["platform"])
    if not audience:
        missing.append(questions["audience"])
    if not goal:
        missing.append(questions["goal"])
    if not offer:
        missing.append(questions["offer"])

    if missing:
        return ClarificationResult(
            needs_clarification=True,
            question=ick.clarification_prefix("social") + " ".join(missing),
        )

    return ClarificationResult(needs_clarification=False)


def minecraft_brief_clarification(
    edition: str | None,
    platform: str | None,
    experience: str | None,
    goal: str | None,
) -> ClarificationResult:
    questions = _minecraft_missing_questions()
    missing = []
    if not edition:
        missing.append(questions["edition"])
    if not platform:
        missing.append(questions["platform"])
    if not experience:
        missing.append(questions["experience"])
    if not goal:
        missing.append(questions["goal"])

    if missing:
        return ClarificationResult(
            needs_clarification=True,
            question=ick.clarification_prefix("minecraft") + " ".join(missing),
        )
    return ClarificationResult(needs_clarification=False)
