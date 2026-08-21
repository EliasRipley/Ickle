"""Abstract Teacher base class - AI-agnostic teaching interface for Ickle.

This is the pipeline that any AI teacher (Ollama, opencode, GPT, Claude, etc.)
can implement. It was always meant to exist - the original ollama_teacher.py
just skipped the abstraction layer."""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from typing import Any


CURRICULUM_PROMPT = """Generate {count} training prompts for Ickle about {topic}.

Make them progressively harder:
- First few: basic definitions and explanations
- Middle: application and comparison
- Last: synthesis, edge cases, or creative application

Output ONLY a JSON array: ["prompt 1", "prompt 2", ...]"""

CURRICULUM_SYSTEM = "You design training curricula for small language models. Generate prompts that test knowledge and reasoning about a topic at progressively harder levels."

BATCH_SFT_SYSTEM = """You are an expert curriculum designer creating supervised fine-tuning data for a small
language model called Ickle. Generate high-quality training examples.

Rules:
- Generate diverse prompts and clear, accurate answers
- Cover different aspects of the topic
- Answers should be concise but thorough
- Output ONLY a JSON array of objects, no other text."""

BATCH_SFT_PROMPT = """Generate {count} User:/Ickle: training pairs about {topic}.

Each pair should have:
- A clear prompt/question
- A high-quality, accurate answer

Output ONLY a JSON array like:
[
  {{"prompt": "What is X?", "answer": "X is...", "tags": ["tag1", "tag2"]}},
  ...
]"""


def extract_json(text: str) -> Any | None:
    text = str(text or "").strip()
    if not text:
        return None
    text = re.sub(r"```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n\s*```", "", text)
    text = text.strip()
    for pattern in (r"\{[\s\S]*\}", r"\[[\s\S]*\]"):
        match = re.search(pattern, text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                continue
    return None


def parse_teaching_json(raw: str) -> dict[str, Any] | None:
    result = extract_json(raw)
    if not result:
        return None
    if "improved_answer" not in result:
        return None
    return {
        "improved_answer": str(result.get("improved_answer", "")).strip(),
        "teacher_feedback": str(result.get("teacher_feedback", "")).strip(),
        "score": max(0.0, min(1.0, float(result.get("score", 0.7)))),
        "raw": raw[:500],
    }


class TeacherBase(ABC):
    """Abstract base for any AI teacher (Ollama, opencode, GPT, Claude, etc.).

    Subclasses must implement generate_text(). The higher-level methods
    (curriculum, teach_turn, batch_sft) have default implementations that
    call generate_text() - override them if the teacher has specific prompting needs.
    """

    def __init__(self, teacher_name: str = "generic"):
        self.teacher_name = teacher_name

    @abstractmethod
    def check_connection(self) -> dict[str, Any]:
        ...

    @abstractmethod
    def generate_text(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> str:
        ...

    def generate_curriculum(
        self,
        topic: str,
        count: int = 8,
        *,
        tags: list[str] | None = None,
    ) -> list[str]:
        raw = self.generate_text(
            CURRICULUM_PROMPT.format(topic=topic, count=count),
            system=CURRICULUM_SYSTEM,
            temperature=0.5,
            max_tokens=800,
        )
        data = extract_json(raw)
        if isinstance(data, list):
            return [str(x) for x in data if str(x).strip()]
        if isinstance(data, dict):
            for key in ("prompts", "questions", "items", "curriculum"):
                if isinstance(data.get(key), list):
                    return [str(x) for x in data[key] if str(x).strip()]
        lines = []
        for match in re.finditer(r'(?:^[-*]\s+|^\d+\.\s+)["\']?(.+?)["\']?$', raw, re.MULTILINE):
            lines.append(match.group(1).strip())
        if lines:
            return lines
        for match in re.finditer(r'"([^"]+)"', raw):
            lines.append(match.group(1).strip())
        return lines[:count]

    def teach_turn(
        self,
        prompt: str,
        ickle_answer: str = "",
        *,
        tags: list[str] | None = None,
        retries: int = 2,
    ) -> dict[str, Any]:
        teach_prompt = (
            "Review this prompt and the answer given by Ickle (the student language model).\n\n"
            f"User prompt: {prompt}\n\n"
            f"Ickle's current answer: {ickle_answer or '(Ickle has not yet answered this prompt)'}\n\n"
            "Provide a better answer and brief feedback. Output ONLY JSON:\n"
            '{{"improved_answer": "<your improved answer>", '
            '"teacher_feedback": "<1-2 sentences of critique>", "score": <0.0-1.0>}}'
        )
        teach_system = (
            "You are an expert teacher AI. Your job is to review answers from a very small language model "
            "called Ickle and provide improved answers that will be used as supervised fine-tuning data.\n\n"
            "Rules:\n"
            "- The improved_answer must be clear, accurate, concise, and well-formatted\n"
            "- The teacher_feedback should be 1-2 sentences explaining what was improved and why\n"
            "- The score should reflect the quality of the improved answer (0.0-1.0)\n"
            "- Output ONLY valid JSON, no other text."
        )
        for attempt in range(retries + 1):
            raw = self.generate_text(
                teach_prompt,
                system=teach_system,
                temperature=0.3,
                max_tokens=2000,
            )
            result = parse_teaching_json(raw)
            if result:
                result["prompt"] = prompt
                result["ickle_answer"] = ickle_answer
                result["tags"] = tags or []
                result["source_model"] = self.teacher_name
                return result
            time.sleep(1.0)
        raise RuntimeError(
            f"Failed to get valid teaching JSON from {self.teacher_name} after {retries + 1} attempts"
        )

    def batch_sft(
        self,
        topic: str,
        count: int = 6,
        *,
        tags: list[str] | None = None,
        retries: int = 2,
    ) -> list[dict[str, Any]]:
        sft_prompt = BATCH_SFT_PROMPT.format(topic=topic, count=count)
        for attempt in range(retries + 1):
            raw = self.generate_text(
                sft_prompt,
                system=BATCH_SFT_SYSTEM,
                temperature=0.5,
                max_tokens=4000,
            )
            data = extract_json(raw)
            pairs: list[dict[str, Any]] = []
            if isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    result = self._parse_sft_item(item, tags, topic)
                    if result:
                        pairs.append(result)
                if pairs:
                    return pairs
            if isinstance(data, dict):
                for key in ("pairs", "examples", "items", "qa_pairs"):
                    items = data.get(key)
                    if isinstance(items, list):
                        for item in items:
                            if not isinstance(item, dict):
                                continue
                            result = self._parse_sft_item(item, tags, topic)
                            if result:
                                pairs.append(result)
                        if pairs:
                            return pairs
            time.sleep(1.0)
        raise RuntimeError(
            f"Failed to get valid batch SFT data from {self.teacher_name} after {retries + 1} attempts"
        )

    def _parse_sft_item(
        self,
        item: dict[str, Any],
        tags: list[str] | None,
        topic: str,
    ) -> dict[str, Any] | None:
        prompt = str(item.get("prompt", item.get("question", ""))).strip()
        answer = str(item.get("answer", item.get("response", item.get("improved_answer", "")))).strip()
        if not prompt or not answer:
            return None
        item_tags = item.get("tags", [])
        if not isinstance(item_tags, list):
            item_tags = list(tags or [])
        return {
            "prompt": prompt,
            "ickle_answer": "",
            "improved_answer": answer,
            "teacher_feedback": f"Generated {topic} training example by {self.teacher_name}",
            "score": 0.8,
            "tags": item_tags or tags or [topic],
            "source_model": self.teacher_name,
        }

    def submit_to_store(
        self,
        turns: list[dict[str, Any]],
        *,
        topic: str = "",
        store_dir: str = "data/teacher",
    ) -> dict[str, Any]:
        from src.teacher_ingest import TeacherStore

        store = TeacherStore(store_dir)
        session = store.start_session(
            topic=topic or f"{self.teacher_name}_teaching",
            source_model=self.teacher_name,
            tags=[topic] if topic else [],
        )
        session_id = session.session_id
        submitted = 0
        for turn in turns:
            store.add_turn(
                session_id=session_id,
                prompt=str(turn.get("prompt", "")),
                ickle_answer=str(turn.get("ickle_answer", "")),
                teacher_feedback=str(turn.get("teacher_feedback", "")),
                improved_answer=str(turn.get("improved_answer", "")),
                score=float(turn.get("score", 0.7)),
                tags=turn.get("tags") if isinstance(turn.get("tags"), list) else [],
                source_model=str(turn.get("source_model", self.teacher_name)),
            )
            submitted += 1
        store.close_session(session_id)
        return {"session_id": session_id, "turns_submitted": submitted, "topic": topic}

    def submit_via_api(
        self,
        turns: list[dict[str, Any]],
        *,
        topic: str = "",
        api_url: str = "http://localhost:8788",
    ) -> dict[str, Any]:
        import json
        from urllib.request import Request, urlopen

        base = api_url.rstrip("/")
        session_req = Request(
            f"{base}/api/teach/sessions",
            data=json.dumps({"topic": topic, "source_model": self.teacher_name}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(session_req, timeout=30) as resp:
            session = json.loads(resp.read())
        session_id = str(session.get("session_id", ""))
        if not session_id:
            raise RuntimeError("Failed to create teaching session via API")

        submitted = 0
        for turn in turns:
            turn_payload = {
                "prompt": str(turn.get("prompt", "")),
                "ickle_answer": str(turn.get("ickle_answer", "")),
                "teacher_feedback": str(turn.get("teacher_feedback", "")),
                "improved_answer": str(turn.get("improved_answer", "")),
                "score": float(turn.get("score", 0.7)),
                "tags": turn.get("tags") if isinstance(turn.get("tags"), list) else [],
                "source_model": str(turn.get("source_model", self.teacher_name)),
            }
            turn_req = Request(
                f"{base}/api/teach/sessions/{session_id}/turn",
                data=json.dumps(turn_payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urlopen(turn_req, timeout=30) as resp:
                resp.read()
            submitted += 1

        close_req = Request(
            f"{base}/api/teach/sessions/{session_id}/close",
            data=b"{}",
            headers={"Content-Type": "application/json"},
        )
        with urlopen(close_req, timeout=30) as resp:
            close_result = json.loads(resp.read())

        return {
            "session_id": session_id,
            "turns_submitted": submitted,
            "topic": topic,
            "closed": close_result,
        }
