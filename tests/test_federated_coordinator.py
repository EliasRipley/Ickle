import tempfile
import unittest
from pathlib import Path

import torch

from src.federated.coordinator import FederatedCoordinator
from src.federated.protocol import encode_tensor_dict, generate_nonce, now_epoch_seconds, sign_payload
from src.model import ILM, TinyConfig
from src.tokenizer import CharTokenizer


class FederatedCoordinatorSubmissionTests(unittest.TestCase):
    def _coordinator(self, root: Path) -> FederatedCoordinator:
        tokenizer = CharTokenizer.from_text("User: hello\nIckle: hello there\n")
        cfg = TinyConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=32,
            n_embd=16,
            n_head=2,
            n_layer=1,
        )
        model_path = root / "base.pt"
        payload = {"config": vars(cfg), "model_state": ILM(cfg).state_dict()}
        payload.update(tokenizer.checkpoint_payload())
        torch.save(payload, model_path)
        return FederatedCoordinator(
            base_model_path=str(model_path),
            state_dir=str(root / "state"),
            min_clients=2,
            max_examples_per_update=50,
        )

    @staticmethod
    def _envelope(coordinator: FederatedCoordinator, registration: dict, *, examples: int, nonce: str) -> dict:
        adapter = coordinator._load_global_adapter()
        payload = {
            "client_id": registration["client_id"],
            "round_id": 1,
            "num_examples": examples,
            "metrics": {"loss": 1.0},
            "delta": encode_tensor_dict({key: torch.zeros_like(value) for key, value in adapter.items()}),
            "timestamp": now_epoch_seconds(),
            "nonce": nonce,
        }
        payload["signature"] = sign_payload(registration["client_secret"], payload)
        return payload

    def test_only_one_submission_per_client_counts_in_a_round(self):
        with tempfile.TemporaryDirectory() as tmp:
            coordinator = self._coordinator(Path(tmp))
            client = coordinator.register_client(platform="test")
            first = self._envelope(coordinator, client, examples=10, nonce=generate_nonce())
            accepted = coordinator.submit_update(first)
            second = self._envelope(coordinator, client, examples=10, nonce=generate_nonce())
            with self.assertRaisesRegex(ValueError, "already submitted"):
                coordinator.submit_update(second)
        self.assertEqual(accepted["update_count"], 1)

    def test_rejects_unbounded_reported_example_weight(self):
        with tempfile.TemporaryDirectory() as tmp:
            coordinator = self._coordinator(Path(tmp))
            client = coordinator.register_client(platform="test")
            envelope = self._envelope(coordinator, client, examples=51, nonce=generate_nonce())
            with self.assertRaisesRegex(ValueError, "between 1 and 50"):
                coordinator.submit_update(envelope)


if __name__ == "__main__":
    unittest.main()
