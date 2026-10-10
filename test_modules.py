import unittest

import torch

from viscon_modules import (
    ReferenceAwareQueryRefinement,
    SimilarityConditionedVisualRegularization,
)


class ReferenceRefinementTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        self.module = ReferenceAwareQueryRefinement(
            8, 12, num_heads=2, ffn_dim=16, dropout=0.0
        )
        self.composed = torch.randn(2, 3, 8, requires_grad=True)
        self.reference = torch.randn(2, 5, 12)

    def test_shape_and_initial_residual(self):
        output = self.module(self.composed, self.reference)
        self.assertEqual(output.shape, self.composed.shape)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.allclose(output, self.module.output_norm(self.composed), atol=1e-6))

    def test_backward(self):
        self.module(self.composed, self.reference).square().sum().backward()
        self.assertIsNotNone(self.composed.grad)
        self.assertTrue(torch.isfinite(self.composed.grad).all())
        self.assertIsNotNone(self.module.reference_ffn[-1].weight.grad)

    def test_reference_dependence_after_residual_initialization(self):
        with torch.no_grad():
            self.module.reference_ffn[-1].weight.normal_(std=0.1)
        self.module.eval()
        first = self.module(self.composed, self.reference)
        second = self.module(self.composed, self.reference.flip(0))
        self.assertFalse(torch.allclose(first, second))

    def test_invalid_heads(self):
        with self.assertRaises(ValueError):
            ReferenceAwareQueryRefinement(7, 12, num_heads=2)


class VisualRegularizationTests(unittest.TestCase):
    def setUp(self):
        self.module = SimilarityConditionedVisualRegularization()
        self.reference = torch.tensor([[1., 0.]] * 3, requires_grad=True)
        self.target = torch.tensor([[1., 0.], [.85, .5267827], [0., 1.]], requires_grad=True)
        self.query = torch.tensor([[.8, .6]] * 3, requires_grad=True)

    def test_weights_and_loss(self):
        loss, stats = self.module(self.query, self.reference, self.target)
        self.assertTrue(torch.allclose(stats["sample_weight"], torch.tensor([2., 1.2, 0.])))
        self.assertTrue(torch.allclose(loss, torch.tensor(.2), atol=1e-6))

    def test_supervision_detached(self):
        loss, stats = self.module(self.query, self.reference, self.target)
        loss.backward()
        self.assertIsNotNone(self.query.grad)
        self.assertTrue(torch.isfinite(self.query.grad).all())
        self.assertIsNone(self.reference.grad)
        self.assertIsNone(self.target.grad)
        self.assertTrue(all(not value.requires_grad for value in stats.values()))

    def test_zero_weight_batch(self):
        loss, _ = self.module(self.query, self.reference, torch.tensor([[0., 1.]] * 3))
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertTrue(torch.isfinite(self.query.grad).all())

    def test_invalid_thresholds_and_weights(self):
        for parameters in (
            {"high_threshold": .8, "medium_threshold": .9},
            {"high_weight": 1.0, "medium_weight": 1.2},
            {"low_weight": -0.1},
        ):
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                SimilarityConditionedVisualRegularization(**parameters)


if __name__ == "__main__":
    unittest.main()
