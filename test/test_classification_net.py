from math import log
from unittest import TestCase

import torch

from craftdl import ClassificationNet


class IdentityNet(ClassificationNet):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ClassificationNetTest(TestCase):
    def setUp(self) -> None:
        self.net = IdentityNet()

        self.binary_inputs = torch.tensor([[0.0], [log(3.0)], [log(3.0)], [-log(3.0)]])
        self.multiclass_inputs = torch.tensor(
            [
                [log(2.0), log(4.0), log(8.0), log(2.0)],
                [log(4.0), log(4.0), log(8.0), log(16.0)],
            ]
        )

    def test_predict_prob_binary(self) -> None:
        probs = self.net.predict_prob(self.binary_inputs)
        expected_probs = torch.tensor([[0.5], [0.75], [0.75], [0.25]])
        self.assertTrue(torch.isclose(probs, expected_probs).all())

    def test_predict_prob_multiclass(self) -> None:
        probs = self.net.predict_prob(self.multiclass_inputs)
        expected_probs = torch.tensor(
            [[0.125, 0.25, 0.5, 0.125], [0.125, 0.125, 0.25, 0.5]]
        )
        self.assertTrue((probs == expected_probs).all())

    def test_predict_binary(self) -> None:
        predictions = self.net.predict(self.binary_inputs)
        print(predictions)
        expected_predictions = torch.tensor([0, 1, 1, 0], dtype=torch.long)
        self.assertTrue(torch.isclose(predictions, expected_predictions).all())

    def test_predict_multiclass(self) -> None:
        predictions = self.net.predict(self.multiclass_inputs)
        self.assertEqual(predictions[0], 2)
        self.assertEqual(predictions[1], 3)

    def test_predict_meshgrid(self) -> None:
        x1_grid, x2_grid, predictions = self.net.predict_grid(0, 1, 0, 4, 2)
        expected_x1_grid = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        expected_x2_grid = torch.tensor([[0.0, 4.0], [0.0, 4.0]])
        expected_predictions = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)

        self.assertTrue((x1_grid == expected_x1_grid).all())
        self.assertTrue((x2_grid == expected_x2_grid).all())
        self.assertTrue((predictions == expected_predictions).all())

    def test_accuracy(self) -> None:
        inputs = torch.tensor(
            [
                [0.5, 0.3, -0.2],  # 0
                [0.3, 0.5, -0.2],  # 1
                [0.3, -0.2, 0.5],  # 2
                [0.3, 0.5, -0.2],  # 1
                [-0.2, 0.3, 0.5],  # 2
                [0.5, -0.2, 0.3],  # 0
            ]
        )
        targets = torch.tensor([0, 1, 1, 2, 2, 1])
        accuracy = self.net.accuracy(inputs=inputs, targets=targets)

        self.assertEqual(accuracy, 0.5)
