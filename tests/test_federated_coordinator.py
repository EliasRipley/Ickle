import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from src.federated.coordinator import FederatedCoordinator
from src.federated.local_train import LocalTrainConfig, evaluate_adapter_loss, train_local_delta
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

    def test_initial_adapter_is_noop_but_has_a_trainable_lora_factor(self):
        with tempfile.TemporaryDirectory() as tmp:
            coordinator = self._coordinator(Path(tmp))
            adapter = coordinator._load_global_adapter()

        a_tensors = [value for key, value in adapter.items() if key.endswith(".lora_a")]
        b_tensors = [value for key, value in adapter.items() if key.endswith(".lora_b")]
        self.assertTrue(a_tensors)
        self.assertTrue(b_tensors)
        self.assertTrue(any(torch.count_nonzero(value).item() > 0 for value in a_tensors))
        self.assertTrue(all(torch.count_nonzero(value).item() == 0 for value in b_tensors))

    def test_one_local_step_changes_the_fresh_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            coordinator = self._coordinator(Path(tmp))
            initial = coordinator._load_global_adapter()
            delta, metrics = train_local_delta(
                base_model_path=coordinator.base_model_path,
                lora_cfg=coordinator.lora_cfg,
                global_adapter_state=initial,
                train_text="User: hello\nIckle: hello there\n" * 8,
                train_cfg=LocalTrainConfig(
                    steps=1,
                    batch_size=1,
                    lr=1e-2,
                    torch_threads=1,
                ),
            )

        self.assertEqual(metrics["steps_ran"], 1.0)
        self.assertTrue(any(torch.count_nonzero(value).item() > 0 for value in delta.values()))

    def test_restart_repairs_legacy_all_zero_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            coordinator = self._coordinator(root)
            adapter = coordinator._load_global_adapter()
            torch.save(
                {key: torch.zeros_like(value) for key, value in adapter.items()},
                coordinator.global_adapter_path,
            )

            restarted = self._coordinator(root)
            repaired = restarted._load_global_adapter()

        a_tensors = [value for key, value in repaired.items() if key.endswith(".lora_a")]
        b_tensors = [value for key, value in repaired.items() if key.endswith(".lora_b")]
        self.assertTrue(any(torch.count_nonzero(value).item() > 0 for value in a_tensors))
        self.assertTrue(all(torch.count_nonzero(value).item() == 0 for value in b_tensors))

    def test_adapter_evaluation_uses_repeatable_minibatches(self):
        with tempfile.TemporaryDirectory() as tmp:
            coordinator = self._coordinator(Path(tmp))
            adapter = coordinator._load_global_adapter()
            eval_text = "User: hello\nIckle: hello there\n" * 12

            torch.manual_seed(1)
            first = evaluate_adapter_loss(
                base_model_path=coordinator.base_model_path,
                lora_cfg=coordinator.lora_cfg,
                adapter_state=adapter,
                eval_text=eval_text,
                eval_iters=3,
                batch_size=2,
                torch_threads=1,
            )
            torch.manual_seed(999)
            second = evaluate_adapter_loss(
                base_model_path=coordinator.base_model_path,
                lora_cfg=coordinator.lora_cfg,
                adapter_state=adapter,
                eval_text=eval_text,
                eval_iters=3,
                batch_size=2,
                torch_threads=1,
            )

        self.assertEqual(first, second)

    def test_rejected_round_does_not_poison_optimizer_momentum(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            coordinator = self._coordinator(root)
            eval_path = root / "eval.txt"
            eval_path.write_text("evaluation data", encoding="utf-8")
            coordinator.eval_data_path = str(eval_path)

            client = coordinator.register_client(platform="test")
            envelope = self._envelope(
                coordinator, client, examples=10, nonce=generate_nonce()
            )
            adapter = coordinator._load_global_adapter()
            envelope["delta"] = encode_tensor_dict(
                {key: torch.ones_like(value) for key, value in adapter.items()}
            )
            envelope["signature"] = sign_payload(
                client["client_secret"],
                {key: value for key, value in envelope.items() if key != "signature"},
            )
            coordinator.submit_update(envelope)

            with mock.patch(
                "src.federated.coordinator.evaluate_adapter_loss",
                side_effect=[1.0, 2.0],
            ):
                summary = coordinator.aggregate_active_round(force=True)

            self.assertEqual(summary["status"], "rejected")
            self.assertEqual(coordinator.global_optimizer.velocity, {})
            self.assertFalse(coordinator.global_optimizer_path.exists())


if __name__ == "__main__":
    unittest.main()
