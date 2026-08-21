import json
from typing import Any


EXTRACTION_PROMPT = """You are a knowledge extraction system. Given the following text, extract structured information.

Output ONLY a JSON object with these fields:
- "entities": list of key entities (people, places, events, concepts, dates) — each with "name", "type", and brief context
- "relationships": list of relationships between entities — each with "from", "to", "relation", and brief context
- "facts": list of factual claims from the text — each with "claim", "confidence" (0-1), and whether it's "stable" (unlikely to change) or "updateable"
- "domain_description": a concise natural-language description of what domain this knowledge covers (used for semantic routing)
- "skills_identified": list of any reasoning skills this material could teach (e.g., "causal reasoning", "comparative analysis", "timeline construction") — or [] if none

Do not include markdown formatting. Output valid JSON only.

Text to analyze:
---
{text}
---
"""

EXTRACTION_SYSTEM = "You are an expert knowledge extraction system. You identify entities, relationships, facts, and reasoning patterns in text. You output structured JSON. You are precise and avoid hallucination."


def extract_structured_knowledge(text: str, teacher=None) -> dict[str, Any]:
    if teacher is None:
        return _fallback_extraction(text)

    try:
        result = teacher.generate_text(
            prompt=EXTRACTION_PROMPT.format(text=text[:8000]),
            system=EXTRACTION_SYSTEM,
            temperature=0.1,
            max_tokens=1024,
        )
        parsed = _safe_parse_json(result)
        if parsed and isinstance(parsed, dict):
            return _normalize_extraction(parsed)
    except Exception:
        pass

    return _fallback_extraction(text)


def _safe_parse_json(text: str) -> Any:
    text = str(text or "").strip()
    if not text:
        return None
    for attempt in range(3):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            if "```json" in text:
                start = text.find("```json") + 7
                end = text.find("```", start)
                if end > start:
                    text = text[start:end].strip()
                    continue
            if "{" in text and "}" in text:
                start = text.find("{")
                end = text.rfind("}") + 1
                if end > start:
                    text = text[start:end]
                    continue
            break
    return None


def _normalize_extraction(parsed: dict) -> dict[str, Any]:
    entities = []
    for e in parsed.get("entities", []) or []:
        if isinstance(e, dict):
            entities.append({
                "name": str(e.get("name", "")),
                "type": str(e.get("type", "unknown")),
                "context": str(e.get("context", "")),
            })
    relationships = []
    for r in parsed.get("relationships", []) or []:
        if isinstance(r, dict):
            relationships.append({
                "from": str(r.get("from", "")),
                "to": str(r.get("to", "")),
                "relation": str(r.get("relation", "")),
                "context": str(r.get("context", "")),
            })
    facts = []
    for f in parsed.get("facts", []) or []:
        if isinstance(f, dict):
            facts.append({
                "claim": str(f.get("claim", "")),
                "confidence": float(f.get("confidence", 0.7)),
                "stable": bool(f.get("stable", True)),
            })
    return {
        "entities": entities,
        "relationships": relationships,
        "facts": facts,
        "domain_description": str(parsed.get("domain_description", "")),
        "skills_identified": parsed.get("skills_identified", []) or [],
        "summary": {
            "total_entities": len(entities),
            "total_relationships": len(relationships),
            "total_facts": len(facts),
            "skills_count": len(parsed.get("skills_identified", []) or []),
        },
        "method": "teacher",
    }


def _fallback_extraction(text: str) -> dict[str, Any]:
    # No teacher means no real topic classification happened -- the first N
    # words of an arbitrary corpus sample are not a "domain description" of
    # anything, just whatever text happened to be first (a scraped news
    # article, a Wikipedia stub, mid-sentence). Callers that display this to
    # a person (e.g. train.py's --auto-register) must check "method" and use
    # an honest label instead of presenting this echo as a real summary.
    words = text.split()
    return {
        "entities": [],
        "relationships": [],
        "facts": [],
        "domain_description": " ".join(words[:30]) if words else text[:200],
        "skills_identified": [],
        "summary": {"total_entities": 0, "total_relationships": 0, "total_facts": 0, "skills_count": 0},
        "method": "fallback",
    }


def generate_delta_spec(text: str, delta_id: str, teacher=None) -> dict[str, Any]:
    knowledge = extract_structured_knowledge(text, teacher=teacher)

    facts_for_memory = []
    for f in knowledge.get("facts", []):
        facts_for_memory.append({
            "name": f["claim"][:100],
            "type": "fact",
            "confidence": f["confidence"],
            "stable": f["stable"],
        })

    return {
        "delta_id": delta_id,
        "version": "1.0.0",
        "domain_description": knowledge.get("domain_description", text[:200]),
        "description": knowledge.get("domain_description", delta_id),
        "memory_entries": facts_for_memory,
        "skills_identified": knowledge.get("skills_identified", []),
        "knowledge_summary": knowledge.get("summary", {}),
    }
