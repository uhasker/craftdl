from unittest import TestCase

import torch
from torch.nn import Linear, ReLU

from craftdl import LinearNet, CraftdlException


class LinearNetTest(TestCase):
    def test_linear_net_no_hidden_dims(self) -> None:
        linear_net = LinearNet(input_dim=12, hidden_dims=[], output_dim=4)
        print(linear_net.layers)

        self.assertEqual(len(linear_net.layers), 1)
        self.assertIsInstance(linear_net.layers[0], Linear)
        self.assertEqual(linear_net.layers[0].in_features, 12)
        self.assertEqual(linear_net.layers[0].out_features, 4)

    def test_linear_net_single_hidden_dim(self) -> None:
        linear_net = LinearNet(input_dim=12, hidden_dims=[6], output_dim=4)
        print(linear_net.layers)

        self.assertEqual(len(linear_net.layers), 3)

        # layers[0]
        self.assertIsInstance(linear_net.layers[0], Linear)
        self.assertEqual(linear_net.layers[0].in_features, 12)
        self.assertEqual(linear_net.layers[0].out_features, 6)

        # layers[1]
        self.assertIsInstance(linear_net.layers[1], ReLU)

        # layers[2]
        self.assertIsInstance(linear_net.layers[2], Linear)
        self.assertEqual(linear_net.layers[2].in_features, 6)
        self.assertEqual(linear_net.layers[2].out_features, 4)

    def test_linear_net_single_hidden_dim_int(self) -> None:
        linear_net = LinearNet(input_dim=12, hidden_dims=6, output_dim=4)
        print(linear_net.layers)

        self.assertEqual(len(linear_net.layers), 3)

        # layers[0]
        self.assertIsInstance(linear_net.layers[0], Linear)
        self.assertEqual(linear_net.layers[0].in_features, 12)
        self.assertEqual(linear_net.layers[0].out_features, 6)

        # layers[1]
        self.assertIsInstance(linear_net.layers[1], ReLU)

        # layers[2]
        self.assertIsInstance(linear_net.layers[2], Linear)
        self.assertEqual(linear_net.layers[2].in_features, 6)
        self.assertEqual(linear_net.layers[2].out_features, 4)

    def test_linear_net_multiple_hidden_dims(self) -> None:
        linear_net = LinearNet(input_dim=12, hidden_dims=[6, 8, 16], output_dim=4)
        print(linear_net.layers)

        self.assertEqual(len(linear_net.layers), 7)

        # layers[0]
        self.assertIsInstance(linear_net.layers[0], Linear)
        self.assertEqual(linear_net.layers[0].in_features, 12)
        self.assertEqual(linear_net.layers[0].out_features, 6)

        # layers[1]
        self.assertIsInstance(linear_net.layers[1], ReLU)

        # layers[2]
        self.assertIsInstance(linear_net.layers[2], Linear)
        self.assertEqual(linear_net.layers[2].in_features, 6)
        self.assertEqual(linear_net.layers[2].out_features, 8)

        # layers[3]
        self.assertIsInstance(linear_net.layers[3], ReLU)

        # layers[4]
        self.assertIsInstance(linear_net.layers[4], Linear)
        self.assertEqual(linear_net.layers[4].in_features, 8)
        self.assertEqual(linear_net.layers[4].out_features, 16)

        # layers[5]
        self.assertIsInstance(linear_net.layers[5], ReLU)

        # layers[6]
        self.assertIsInstance(linear_net.layers[6], Linear)
        self.assertEqual(linear_net.layers[6].in_features, 16)
        self.assertEqual(linear_net.layers[6].out_features, 4)

    def test_linear_net_fit_bce(self) -> None:
        linear_net = LinearNet(input_dim=2, hidden_dims=[8], output_dim=1)

        xor_inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float)
        xor_targets = torch.tensor([0, 1, 1, 0], dtype=torch.long)

        epoch_losses = linear_net.fit(
            xor_inputs, xor_targets, n_epochs=1000, loss_name="bce", lr=1.0
        )
        print(epoch_losses[-1])
        self.assertTrue(epoch_losses[-1] < 0.1)

    def test_linear_net_fit_ce(self) -> None:
        linear_net = LinearNet(input_dim=2, hidden_dims=[8], output_dim=4)

        xor_inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float)
        xor_targets_multiclass = torch.tensor([0, 1, 2, 3], dtype=torch.long)

        epoch_losses = linear_net.fit(
            xor_inputs,
            xor_targets_multiclass,
            n_epochs=2000,
            loss_name="crossentropy",
            lr=0.1,
        )
        print(epoch_losses[-1])
        self.assertTrue(epoch_losses[-1] < 0.1)

    def test_raise_input_dim(self) -> None:
        linear_net = LinearNet(input_dim=2, hidden_dims=[8], output_dim=4)
        self.assertRaises(CraftdlException, linear_net, torch.tensor([[0.0, 1.0, 2.0]]))

    def test_raise_no_loss(self) -> None:
        linear_net = LinearNet(input_dim=2, hidden_dims=[8], output_dim=4)
        self.assertRaises(
            CraftdlException,
            linear_net.fit,
            inputs=torch.tensor([[0.0]]),
            targets=torch.tensor([[0]]),
            n_epochs=1,
            loss_name="noloss",
        )
