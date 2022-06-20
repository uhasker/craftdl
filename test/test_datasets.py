import os
from unittest import TestCase

import torch

from craftdl import circle, circles_dataset, dataset_from_file, train_test_split


class DatasetsTest(TestCase):
    def test_circle_no_std(self) -> None:
        x1s, x2s = circle(r=1.0, n=4, std=0.0)
        x1s_expected = [1.0, 0.0, -1.0, 0.0]
        x2s_expected = [0.0, 1.0, 0.0, -1.0]
        for i in range(4):
            self.assertAlmostEqual(float(x1s[i]), x1s_expected[i], places=4)
            self.assertAlmostEqual(float(x2s[i]), x2s_expected[i], places=4)

    def test_circle(self) -> None:
        x1s, x2s = circle(r=1.0, n=4, std=0.1)
        self.assertEqual(tuple(x1s.size()), (4,))
        self.assertEqual(tuple(x2s.size()), (4,))

    def test_two_circles_dataset_no_std(self) -> None:
        inputs, targets = circles_dataset(rs=[1.0, 2.0], ns=[4, 4], std=0.0)
        x1s_expected = [1.0, 0.0, -1.0, 0.0, 2.0, 0.0, -2.0, 0.0]
        x2s_expected = [0.0, 1.0, 0.0, -1.0, 0.0, 2.0, 0.0, -2.0]
        targets_expected = [0, 0, 0, 0, 1, 1, 1, 1]
        for i in range(8):
            self.assertAlmostEqual(float(inputs[i][0]), x1s_expected[i], places=4)
            self.assertAlmostEqual(float(inputs[i][1]), x2s_expected[i], places=4)
            self.assertEqual(targets[i], targets_expected[i])

    def test_dataset_from_file(self) -> None:
        filename = "example"
        with open(filename, "w", encoding="utf-8") as file:
            file.write("0.5 0.25 1.5 0\n0.75 0.5 0.25 0\n0.125 0.25 0.125 1")
        inputs, targets = dataset_from_file(filename=filename)
        x1s_expected = [0.5, 0.75, 0.125]
        x2s_expected = [0.25, 0.5, 0.25]
        x3s_expected = [1.5, 0.25, 0.125]
        targets_expected = [0, 0, 1]
        for i in range(3):
            self.assertEqual(float(inputs[i][0]), x1s_expected[i])
            self.assertEqual(float(inputs[i][1]), x2s_expected[i])
            self.assertEqual(float(inputs[i][2]), x3s_expected[i])
            self.assertEqual(targets[i], targets_expected[i])
        os.remove(filename)

    def test_split_train_test(self) -> None:
        inputs = torch.rand(size=(10, 6))
        targets = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
        (inputs_train, targets_train), (inputs_test, targets_test) = train_test_split(
            inputs=inputs, targets=targets
        )
        self.assertEqual(tuple(inputs_train.size()), (8, 6))
        self.assertEqual(tuple(targets_train.size()), (8,))
        self.assertEqual(tuple(inputs_test.size()), (2, 6))
        self.assertEqual(tuple(targets_test.size()), (2,))
