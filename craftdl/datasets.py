import math
from typing import Tuple, List

import torch


def circle(r: float, n: int, std: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a circle with added Gaussian noise.

    :param r: The radius of the circle.
    :param n: The number of points to create.
    :param std: The standard deviation of the Gaussian noise.
    :return: Two tensors representing the values for x1 and x2.
    """
    start_angle, end_angle = 0, 2 * math.pi * (n - 1) / n
    angles = torch.linspace(start_angle, end_angle, n)

    x1s_noise = (std**0.5) * torch.randn(n)
    x1s = r * torch.cos(angles) + x1s_noise

    x2s_noise = (std**0.5) * torch.randn(n)
    x2s = r * torch.sin(angles) + x2s_noise
    return x1s, x2s


def circles_dataset(
    rs: List[float],
    ns: List[int],
    std: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a dataset with n concentric circles.

    :param rs: The radii of the circles.
    :param ns: The number of points to draw for the circles.
    :param std: The standard deviation of the Gaussian noise.
    :return: Two tensors representing the inputs and the targets.
    """
    n_circles = len(ns)
    all_x1s, all_x2s, all_ys = [], [], []
    for i in range(n_circles):
        x1s, x2s = circle(rs[i], ns[i], std)
        ys = torch.full(size=(ns[i],), fill_value=i)
        all_x1s.append(x1s)
        all_x2s.append(x2s)
        all_ys.append(ys)

    x1s = torch.cat(tuple(all_x1s), dim=0)
    x2s = torch.cat(tuple(all_x2s), dim=0)

    inputs = torch.stack([x1s, x2s], dim=1)
    targets = torch.cat(tuple(all_ys), dim=0)
    return inputs, targets


def dataset_from_file(
    filename: str, encoding: str = "utf-8"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load a dataset from a file.

    Each line in the file should contain a list of number separated by whitespace
    representing the respective data point. The numbers before the last number indicate
    the coordinates of the data point. The last number indicates the class of the
    data point.

    Consider the file containing the following content:
    0.1 0.3 0.5 0
    0.4 0.7 0.8 1
    0.6 0.3 0.8 2

    This would be interpreted as the following dataset:
    inputs = [[0.1, 0.3, 0.5], [0.4, 0.7, 0.8], [0.6, 0.3, 0.8]]
    targets = [0, 1, 2]

    :param filename: The name of the file.
    :param encoding: The encoding of the file.
    :return: Two tensors representing the inputs and the targets.
    """
    inputs, targets = [], []
    with open(filename, "r", encoding=encoding) as file:
        for row in file.readlines():
            row_nums = [float(num) for num in row.split()]
            inputs.append(row_nums[:-1])
            targets.append(row_nums[-1])

    inputs_tensor = torch.tensor(inputs, dtype=torch.float)
    targets_tensor = torch.tensor(targets, dtype=torch.long)
    return inputs_tensor, targets_tensor


def train_test_split(
    inputs: torch.Tensor, targets: torch.Tensor, p_train: float = 0.8
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Create a train-test split from a dataset.

    :param inputs: The inputs of the dataset.
    :param targets: The targets of the dataset.
    :param p_train: The percentage of point to put into the training dataset (should be
        a number between 0 and 1).
    :return: Two tuples. The first tuple contains the training dataset inputs and
        targets while the second tuple contains the testing dataset inputs and targets.
    """
    n = len(inputs)

    permutation = torch.randperm(n)
    permuted_inputs = inputs[permutation]
    permuted_targets = targets[permutation]

    split_idx = int(p_train * n)

    inputs_train, targets_train = (
        permuted_inputs[:split_idx],
        permuted_targets[:split_idx],
    )
    inputs_test, targets_test = (
        permuted_inputs[split_idx:],
        permuted_targets[split_idx:],
    )
    return (inputs_train, targets_train), (inputs_test, targets_test)
