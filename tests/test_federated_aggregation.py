import unittest

import torch

from src.federated.aggregation import ClientUpdate, aggregate_deltas


class FederatedAggregationTests(unittest.TestCase):
    def test_trimmed_mean_reduces_outlier_effect(self):
        base = torch.tensor([1.0, 1.0, 1.0])
        updates = [
            ClientUpdate(client_id="a", num_examples=10, delta={"x": base.clone()}, metrics={}),
            ClientUpdate(client_id="b", num_examples=10, delta={"x": base.clone()}, metrics={}),
            ClientUpdate(client_id="c", num_examples=10, delta={"x": base.clone()}, metrics={}),
            ClientUpdate(client_id="z", num_examples=10, delta={"x": torch.tensor([100.0, 100.0, 100.0])}, metrics={}),
        ]
        agg, meta = aggregate_deltas(
            updates,
            method="trimmed_mean",
            trim_ratio=0.25,
            max_update_norm=1000.0,
        )
        self.assertEqual(meta["method"], "trimmed_mean")
        self.assertTrue(torch.allclose(agg["x"], torch.tensor([1.0, 1.0, 1.0])))

    def test_weighted_avg_prefers_larger_client(self):
        updates = [
            ClientUpdate(client_id="small", num_examples=1, delta={"x": torch.tensor([0.0])}, metrics={}),
            ClientUpdate(client_id="big", num_examples=9, delta={"x": torch.tensor([1.0])}, metrics={}),
        ]
        agg, meta = aggregate_deltas(updates, method="weighted_avg", max_update_norm=1000.0)
        self.assertEqual(meta["method"], "weighted_avg")
        self.assertTrue(torch.allclose(agg["x"], torch.tensor([0.9])))

    def test_rejects_duplicate_client_updates(self):
        updates = [
            ClientUpdate(client_id="same", num_examples=1, delta={"x": torch.tensor([1.0])}, metrics={}),
            ClientUpdate(client_id="same", num_examples=1, delta={"x": torch.tensor([2.0])}, metrics={}),
        ]
        with self.assertRaisesRegex(ValueError, "one update per client"):
            aggregate_deltas(updates)

    def test_rejects_inconsistent_tensor_keys(self):
        updates = [
            ClientUpdate(client_id="a", num_examples=1, delta={"x": torch.tensor([1.0])}, metrics={}),
            ClientUpdate(client_id="b", num_examples=1, delta={"y": torch.tensor([1.0])}, metrics={}),
        ]
        with self.assertRaisesRegex(ValueError, "same tensor keys"):
            aggregate_deltas(updates)

    def test_rejects_non_finite_values(self):
        updates = [
            ClientUpdate(client_id="a", num_examples=1, delta={"x": torch.tensor([1.0])}, metrics={}),
            ClientUpdate(client_id="b", num_examples=1, delta={"x": torch.tensor([float("nan")])}, metrics={}),
        ]
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            aggregate_deltas(updates)


if __name__ == "__main__":
    unittest.main()
