from typing import List, Union, Tuple, Any, Callable, Optional

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm  # type: ignore

from craftdl.exception import CraftdlException
from craftdl.util import _SimpleDataset


def _linear_dim_pairs(
    input_dim: int, hidden_dims: List[int], output_dim: int
) -> List[Tuple[int, int]]:
    dims = [input_dim] + hidden_dims + [output_dim]
    return [(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]


def _insert_relu_between(values: List[Any]) -> List[Any]:
    result = []
    for i in range(len(values) - 1):
        result.append(values[i])
        result.append(nn.ReLU())
    result.append(values[-1])
    return result


class LinearNet(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dims: Union[int, List[int]], output_dim: int
    ) -> None:
        super().__init__()

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        dim_pairs = _linear_dim_pairs(input_dim, hidden_dims, output_dim)
        layers = [nn.Linear(in_dim, out_dim) for in_dim, out_dim in dim_pairs]
        layers = _insert_relu_between(layers)

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2 and x.size()[1] != self.input_dim:
            raise CraftdlException(
                f"The input dimension expected by the network is {self.input_dim}, "
                f"but the input dimension of the provided tensor is {x.size()[1]}. "
                f"Either the model input dimension or the passed tensor are wrong."
            )

        logits = self.layers(x)
        return logits

    def fit(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        n_epochs: int,
        loss_name: str,
        lr: float = 0.01,
        batch_size: int = 64,
        verbose: bool = True,
    ) -> List[float]:
        """
        Fit the model on a dataset.

        :param inputs: The inputs of the dataset.
        :param targets: The target of the dataset.
        :param n_epochs: The number of epochs to fit for.
        :param loss_name: The name of the loss function. This must be either "bce"
            (for binary crossentropy) or "crossentropy" (for regular crossentropy).
        :param lr: The learning rate to use.
        :param batch_size: The batch size to use.
        :param verbose: Indicates whether to show progress lines.
        :return: A list of epoch losses.
        """
        dataset = _SimpleDataset(inputs, targets)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = SGD(self.parameters(), lr=lr)

        loss_fun: Optional[Callable] = None
        if loss_name == "bce":
            loss_fun = nn.BCEWithLogitsLoss()
        elif loss_name == "crossentropy":
            loss_fun = nn.CrossEntropyLoss()
        else:
            raise CraftdlException(f"No loss {loss_name} known")

        epoch_losses = []
        epoch_enumerable = tqdm(range(n_epochs)) if verbose else range(n_epochs)
        for _ in epoch_enumerable:
            epoch_loss = 0.0
            for inputs, targets in dataloader:
                pred = self(inputs)
                if loss_name == "bce":
                    pred = pred.flatten()
                    targets = targets.float()
                loss = loss_fun(pred, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.item())
            epoch_losses.append(epoch_loss)
        return epoch_losses
