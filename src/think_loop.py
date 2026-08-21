import re
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ThinkConfig:
    think_budget: int = 0
    research_enabled: bool = True
    min_confidence: float = 0.5
    max_iterations: int = 5

    def budget_remaining(self, spent: int) -> int:
        if self.think_budget <= 0:
            return 1_000_000
        return max(0, self.think_budget - spent)


def assess_self_knowledge(response: str, prompt: str) -> dict[str, Any]:
    evasive_patterns = [
        r"i don't know", r"i'm not sure", r"i cannot", r"i can't",
        r"i have no (information|knowledge|idea|data)",
        r"i am unable", r"it is unclear", r"not certain",
        r"i don't have (enough )?(information|context|knowledge|data)",
        r"beyond my (knowledge|training|capability)",
        r"i (wasn't|was not) trained on",
        r"i lack (the )?(specific |necessary )?(information|knowledge|data|context)",
    ]
    lower = response.lower()
    evasion_count = sum(1 for p in evasive_patterns if re.search(p, lower))

    unique_words = len(set(re.findall(r"[a-z]{3,}", lower)))
    word_count = max(1, len(re.findall(r"[a-z]+", lower)))
    uniqueness = unique_words / word_count

    prompt_words = set(re.findall(r"[a-z]{3,}", prompt.lower()))
    response_words = set(re.findall(r"[a-z]{3,}", response.lower()))
    topic_overlap = len(prompt_words & response_words) / max(1, len(prompt_words))

    has_facts = bool(re.search(r"(?:is|are|was|were)\s+(?:a|an|the)\s", lower))
    has_specifics = bool(re.search(r"\b(?:January|February|\d{4}|\d+%|\d+\s*(?:BC|AD|CE|BCE))\b", response))

    knowledge_score = 1.0
    if evasion_count >= 2:
        knowledge_score -= 0.4
    elif evasion_count >= 1:
        knowledge_score -= 0.2

    if uniqueness < 0.3:
        knowledge_score -= 0.2
    if topic_overlap < 0.1:
        knowledge_score -= 0.2
    if not has_facts and not has_specifics:
        knowledge_score -= 0.1

    knowledge_score = max(0.0, knowledge_score)

    return {
        "knowledge_score": round(knowledge_score, 4),
        "evasion_detected": evasion_count > 0,
        "evasion_count": evasion_count,
        "topic_overlap": round(topic_overlap, 4),
        "needs_research": knowledge_score < 0.5 or evasion_count > 0,
    }


def score_confidence(response: str, prompt: str, iterations: int = 1) -> dict[str, Any]:
    lower = response.lower()

    unique_words = len(set(re.findall(r"[a-z]{3,}", lower)))
    word_count = max(1, len(re.findall(r"[a-z]+", lower)))
    uniqueness = min(1.0, unique_words / max(1, word_count * 0.6))

    facts_count = len(re.findall(
        r"(?:is|are|was|were|has|have|had)\s+(?:a|an|the|not)|"
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}|"
        r"\b\d{4}\b|"
        r"\b\d+\s*(?:BC|AD|CE|BCE|percent|%|kg|km|miles|years|people|dollars)",
        response
    ))

    sentences = [s.strip() for s in re.split(r"[.!?]+", response) if len(s.strip()) > 10]
    sentence_count = len(sentences)

    prompt_words = set(re.findall(r"[a-z]{3,}", prompt.lower()))
    response_words = set(re.findall(r"[a-z]{3,}", response.lower()))
    relevance = len(prompt_words & response_words) / max(1, len(prompt_words))

    evasive = sum(1 for p in [
        r"i don't know", r"i'm not sure", r"i cannot", r"i can't",
        r"i have no (information|knowledge|idea)",
    ] if re.search(p, lower))

    confidence = (
        uniqueness * 0.20
        + min(1.0, facts_count / 5.0) * 0.30
        + min(1.0, sentence_count / 4.0) * 0.15
        + relevance * 0.25
        - evasive * 0.15
    )

    confidence = max(0.0, min(1.0, confidence))
    if iterations > 1:
        confidence = min(1.0, confidence + 0.05 * (iterations - 1))

    return {
        "confidence": round(confidence, 4),
        "uniqueness": round(uniqueness, 4),
        "facts_count": facts_count,
        "sentence_count": sentence_count,
        "relevance": round(relevance, 4),
        "iterations": iterations,
    }


class ThinkingLoop:
    def __init__(self, config: ThinkConfig | None = None):
        self.config = config or ThinkConfig()
        self._iteration_history: list[dict[str, Any]] = []
        self._best_response = ""
        self._best_confidence = 0.0
        self._tokens_spent = 0

    def reset(self):
        self._iteration_history = []
        self._best_response = ""
        self._best_confidence = 0.0
        self._tokens_spent = 0

    def should_research(self, prompt: str, response: str) -> bool:
        if not self.config.research_enabled:
            return False
        if not self.config.budget_remaining(self._tokens_spent):
            return False
        assessment = assess_self_knowledge(response, prompt)
        return assessment["needs_research"]

    def should_iterate(self, confidence: dict[str, Any]) -> bool:
        if self._iteration_history and len(self._iteration_history) >= self.config.max_iterations:
            return False
        if not self.config.budget_remaining(self._tokens_spent):
            return False
        return confidence["confidence"] < self.config.min_confidence

    def record_iteration(self, response: str, confidence: dict[str, Any], tokens: int):
        self._tokens_spent += tokens
        self._iteration_history.append({
            "response": response[:500],
            "confidence": confidence,
            "tokens": tokens,
            "total_tokens_spent": self._tokens_spent,
        })
        if confidence["confidence"] > self._best_confidence:
            self._best_response = response
            self._best_confidence = confidence["confidence"]

    def final_result(self, response: str, confidence: dict[str, Any] | None = None) -> dict[str, Any]:
        if confidence is None:
            confidence = score_confidence(response, "")
        if self._best_confidence > confidence["confidence"]:
            response = self._best_response
            confidence = score_confidence(response, "", iterations=len(self._iteration_history))
        return {
            "response": response,
            "confidence": confidence["confidence"],
            "iterations": len(self._iteration_history),
            "tokens_spent": self._tokens_spent,
            "budget_remaining": self.config.budget_remaining(self._tokens_spent),
        }
