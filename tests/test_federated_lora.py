import unittest

import torch
import torch.nn as nn

from src.federated.lora import LoRAConfig, get_lora_state_dict, inject_lora, load_lora_state_dict


class TinyToy(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 8, bias=False)
        self.other = nn.Linear(8, 4, bias=False)

    def forward(self, x):
        return self.other(self.proj(x))


class FederatedLoRATests(unittest.TestCase):
    def test_inject_and_round_trip_state(self):
        model = TinyToy()
        cfg = LoRAConfig(rank=4, alpha=8, dropout=0.0, target_modules=("proj",))
        replaced = inject_lora(model, cfg)
        self.assertEqual(replaced, ["proj"])

        state = get_lora_state_dict(model)
        self.assertIn("proj.lora_a", state)
        self.assertIn("proj.lora_b", state)

        patched = {k: v + 1.0 for k, v in state.items()}
        load_lora_state_dict(model, patched, strict=True)
        loaded = get_lora_state_dict(model)
        self.assertTrue(torch.allclose(loaded["proj.lora_a"], patched["proj.lora_a"]))
        self.assertTrue(torch.allclose(loaded["proj.lora_b"], patched["proj.lora_b"]))


if __name__ == "__main__":
    unittest.main()

