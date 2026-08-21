import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from src.knowledge_modules import (
    ResolvedKnowledgeModule,
    apply_lora_modules_to_model,
    resolve_runtime_modules,
)


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 2, bias=False)


class KnowledgeModuleTests(unittest.TestCase):
    def test_apply_lora_modules_adds_weight_delta(self):
        with tempfile.TemporaryDirectory() as tmp:
            module_path = Path(tmp) / "math_module.pt"
            lora_a = torch.tensor([[1.0, 2.0, 3.0], [0.5, 0.0, 1.0]], dtype=torch.float32)
            lora_b = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float32)
            torch.save(
                {
                    "lora_state": {
                        "proj.lora_a": lora_a,
                        "proj.lora_b": lora_b,
                    },
                    "lora": {"rank": 2, "alpha": 2},
                },
                module_path,
            )

            model = _ToyModel()
            with torch.no_grad():
                model.proj.weight.zero_()

            module = ResolvedKnowledgeModule(module_id="math", path=str(module_path), weight=0.5)
            report = apply_lora_modules_to_model(model, [module])

            expected = (lora_b @ lora_a) * 0.5
            self.assertTrue(torch.allclose(model.proj.weight, expected))
            self.assertEqual(report["modules_applied"], 1)
            self.assertEqual(report["applied_layers"], 1)

    def test_resolve_runtime_modules_prefers_relevant_topics(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            mod_a = base / "biology.pt"
            mod_b = base / "finance.pt"
            stub_state = {"proj.lora_a": torch.zeros(1, 1), "proj.lora_b": torch.zeros(1, 1)}
            torch.save({"lora_state": stub_state, "lora": {"rank": 1, "alpha": 1}}, mod_a)
            torch.save({"lora_state": stub_state, "lora": {"rank": 1, "alpha": 1}}, mod_b)

            registry = {
                "version": 1,
                "modules": [
                    {
                        "id": "bio",
                        "path": str(mod_a),
                        "topics": ["photosynthesis", "plants", "biology"],
                        "description": "Plant processes and cell energy transfer",
                        "enabled": True,
                    },
                    {
                        "id": "fin",
                        "path": str(mod_b),
                        "topics": ["stocks", "bonds", "interest rates"],
                        "description": "Financial markets",
                        "enabled": True,
                    },
                ],
            }
            registry_path = base / "registry.json"
            registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")

            resolved = resolve_runtime_modules(
                prompt="Explain photosynthesis in plants in simple terms.",
                base_model="models/base.pt",
                registry_path=str(registry_path),
                max_modules=1,
                auto_select=True,
            )
            self.assertEqual(len(resolved), 1)
            self.assertEqual(resolved[0].module_id, "bio")

            explicit = resolve_runtime_modules(
                prompt="irrelevant prompt",
                base_model="models/base.pt",
                registry_path=str(registry_path),
                explicit_module_ids=["fin"],
                max_modules=1,
                auto_select=False,
            )
            self.assertEqual(len(explicit), 1)
            self.assertEqual(explicit[0].module_id, "fin")


if __name__ == "__main__":
    unittest.main()
