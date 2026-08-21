import json
from dataclasses import asdict, dataclass
from pathlib import Path

import jsonschema

SKILL_CARD_SCHEMA_PATH = Path("schemas/skill_card.schema.json")


class SkillCardValidationError(ValueError):
    pass


def _load_skill_card_schema() -> dict:
    with SKILL_CARD_SCHEMA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_skill_card(payload: dict) -> None:
    """Validate a skill card payload against schemas/skill_card.schema.json.

    Previously reality_check.py's "non_python_artifacts" check only tested
    that the schema *file* existed on disk -- no skill card, at registration
    or load time, was ever actually checked against it. A card missing a
    required field, or with an empty name (the schema's minLength: 1), would
    silently persist and only surface as a confusing error much later."""
    schema = _load_skill_card_schema()
    try:
        jsonschema.validate(instance=payload, schema=schema)
    except jsonschema.ValidationError as exc:
        raise SkillCardValidationError(str(exc.message)) from exc


@dataclass
class SkillCard:
    name: str
    description: str
    corpus_path: str
    model_path: str
    activation_prompt: str


class SkillRegistry:
    """Persistent registry for ILM-acquired skills."""

    def __init__(self, root: str = "data/skills"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"
        if not self.index_path.exists():
            self._write_index({"skills": {}})

    def _read_index(self) -> dict:
        with self.index_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _write_index(self, payload: dict):
        with self.index_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def register_skill(self, card: SkillCard):
        payload = asdict(card)
        validate_skill_card(payload)
        idx = self._read_index()
        idx.setdefault("skills", {})[card.name] = payload
        self._write_index(idx)

    def list_skills(self) -> list[str]:
        idx = self._read_index()
        return sorted(idx.get("skills", {}).keys())

    def get_skill(self, name: str) -> SkillCard | None:
        idx = self._read_index()
        raw = idx.get("skills", {}).get(name)
        if not raw:
            return None
        return SkillCard(**raw)

    def activation_prompt(self, name: str) -> str:
        card = self.get_skill(name)
        if not card:
            raise KeyError(f"Unknown skill '{name}'")
        return card.activation_prompt

    def model_path(self, name: str) -> str:
        card = self.get_skill(name)
        if not card:
            raise KeyError(f"Unknown skill '{name}'")
        return card.model_path
