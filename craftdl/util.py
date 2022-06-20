from typing import Tuple

import torch
from torch.utils.data import Dataset


class _SimpleDataset(Dataset):
    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        assert inputs.dtype == torch.float
        self.inputs = inputs

        assert targets.dtype == torch.long
        self.targets = targets

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[item], self.targets[item]
