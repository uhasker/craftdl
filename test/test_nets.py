from unittest import TestCase

import torch

from craftdl import LinearClassificationNet


class LinearClassificationNetTest(TestCase):
    def test_linear_classification_net_fit_bce(self) -> None:
        linear_net = LinearClassificationNet(input_dim=2, hidden_dims=[8], output_dim=1)

        xor_inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float)
        xor_targets = torch.tensor([0, 1, 1, 0], dtype=torch.long)

        epoch_losses = linear_net.fit(xor_inputs, xor_targets, n_epochs=1000, lr=1.0)
        print(epoch_losses[-1])
        self.assertTrue(epoch_losses[-1] < 0.5)

    def test_linear_classification_net_fit_bce_loss_name(self) -> None:
        linear_net = LinearClassificationNet(input_dim=2, hidden_dims=[8], output_dim=1)

        xor_inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float)
        xor_targets = torch.tensor([0, 1, 1, 0], dtype=torch.long)

        epoch_losses = linear_net.fit(
            xor_inputs, xor_targets, n_epochs=1000, loss_name="bce", lr=1.0
        )
        print(epoch_losses[-1])
        self.assertTrue(epoch_losses[-1] < 0.5)

    def test_linear_classification_net_fit_multiclass(self) -> None:
        linear_net = LinearClassificationNet(input_dim=2, hidden_dims=[8], output_dim=4)

        xor_inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float)
        xor_targets_multiclass = torch.tensor([0, 1, 2, 3], dtype=torch.long)

        epoch_losses = linear_net.fit(
            xor_inputs,
            xor_targets_multiclass,
            n_epochs=2000,
            lr=0.1,
        )
        print(epoch_losses[-1])
        self.assertTrue(epoch_losses[-1] < 0.5)
