from typing import Tuple

import torch
import torch.nn as nn


class ClassificationNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def predict_prob(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Obtain the probabilities corresponding to the inputs.

        Each probability corresponds to an input.

        If we are dealing with a binary classification problem the dimension of each
        row is 1 and contains the probability that the corresponding input should have
        class 1.

        If we are dealing with a multiclass classification problem the dimension of
        each row equals the number of classes. Each value in the row contains the
        probability for the respective class.

        :param inputs: A tensor of size (batch_size, input_dim). Each row corresponds
            to a data point.
        :return: The tensor containing the probabilities. This tensor has size
            (batch_size, class_dim) for multiclass classification or (batch_size, 1)
            for binary classification.
        """
        logits = self(inputs)
        if logits.size()[1] == 1:
            return nn.Sigmoid()(logits)
        else:
            return nn.Softmax(dim=1)(logits)

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Obtain the class predictions for some inputs.

        :param inputs: A tensor of size (batch_size, input_dim). Each row corresponds
            to a data point.
        :return: The tensor containing the predicted classes. This tensor has size
            (batch_size,).
        """
        probs = self.predict_prob(inputs)
        if probs.size()[1] == 1:
            probs = self.predict_prob(inputs).flatten()
            return (probs > 0.5).long()
        else:
            return probs.argmax(dim=1)

    def predict_grid(
        self, x1_min: float, x1_max: float, x2_min: float, x2_max: float, n: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Obtain class predictions for a grid. This is useful for plotting decision surfaces.

        :param x1_min: Minimum value for x1.
        :param x1_max: Maximum value for x2.
        :param x2_min: Minimum value for x2.
        :param x2_max: Maximum value for x2.
        :param n: The number of points along each axis.
        :return: Three 2D tensors representing the values for x1, x2 and the predictions.
        """
        x1s = torch.linspace(x1_min, x1_max, n)
        x2s = torch.linspace(x2_min, x2_max, n)

        x1_grid, x2_grid = torch.meshgrid(x1s, x2s, indexing="ij")
        x1_flat = torch.flatten(x1_grid)
        x2_flat = torch.flatten(x2_grid)

        inputs = torch.stack([x1_flat, x2_flat], dim=1)
        predictions = self.predict(inputs)
        predictions = predictions.reshape(tuple(x1_grid.size()))
        return x1_grid, x2_grid, predictions

    def accuracy(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Obtain the accuracy of the model for inputs with given targets.

        :param inputs: The tensor representing the inputs.
        :param targets: The tensor representing the targets.
        :return: The accuracy.
        """
        predicts = self.predict(inputs)
        correct = torch.sum(predicts == targets)
        return float(correct / len(inputs))
