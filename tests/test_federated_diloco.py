import unittest

import torch

from src.federated.diloco import GlobalOptimizer, GlobalOptimizerConfig


class GlobalOptimizerTests(unittest.TestCase):
    def test_first_step_moves_toward_client_model_delta(self):
        optimizer = GlobalOptimizer(
            GlobalOptimizerConfig(lr=1.0, beta=0.0, nesterov=False)
        )
        current = {"weight": torch.tensor([1.0, -1.0])}
        client_delta = {"weight": torch.tensor([0.25, -0.5])}

        updated = optimizer.step(current, client_delta)

        self.assertTrue(
            torch.allclose(updated["weight"], current["weight"] + client_delta["weight"])
        )

    def test_momentum_accumulates_in_the_client_update_direction(self):
        optimizer = GlobalOptimizer(
            GlobalOptimizerConfig(lr=1.0, beta=0.5, nesterov=False)
        )
        current = {"weight": torch.tensor([0.0])}
        delta = {"weight": torch.tensor([1.0])}

        first = optimizer.step(current, delta)
        second = optimizer.step(first, delta)

        self.assertTrue(torch.allclose(first["weight"], torch.tensor([1.0])))
        self.assertTrue(torch.allclose(second["weight"], torch.tensor([2.5])))
        self.assertTrue(torch.allclose(optimizer.velocity["weight"], torch.tensor([1.5])))


if __name__ == "__main__":
    unittest.main()
