from typing import Union, List, Optional

import torch

from craftdl.classification_net import ClassificationNet
from craftdl.linear_net import LinearNet


class LinearClassificationNet(LinearNet, ClassificationNet):
    def __init__(
        self, input_dim: int, hidden_dims: Union[int, List[int]], output_dim: int
    ) -> None:
        super().__init__(input_dim, hidden_dims, output_dim)

    def fit(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        n_epochs: int,
        loss_name: Optional[str] = None,
        lr: float = 0.01,
        batch_size: int = 64,
        verbose: bool = True,
    ) -> List[float]:
        if loss_name is None:
            loss_name = "bce" if self.output_dim == 1 else "crossentropy"
        return super().fit(inputs, targets, n_epochs, loss_name, lr, batch_size)
