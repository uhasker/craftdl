from typing import Tuple, List

import matplotlib  # type: ignore
import torch
from matplotlib.collections import PathCollection  # type: ignore
from matplotlib.contour import QuadContourSet  # type: ignore
from matplotlib.lines import Line2D  # type: ignore

from craftdl.exception import CraftdlException


def _check_data(data: torch.Tensor) -> None:
    if data.dim() != 2:
        raise CraftdlException(
            f"Data tensor must be 2D to plot, but you provided a "
            f"{data.dim()}D tensor with size {data.size()}"
        )


def plot_data(data: torch.Tensor, test_mode: bool = False) -> PathCollection:
    """
    Plot a 2D dataset.

    :param data: The dataset.
    :param test_mode: Whether to use a test backend.
    :return: The underlying matplotlib object.
    """
    _check_data(data)

    if test_mode:  # pragma: no cover
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    result = plt.scatter(data[:, 0], data[:, 1])
    plt.show()
    return result


def plot_labels(
    data: torch.Tensor,
    labels: torch.Tensor,
    cmap: str = "RdYlGn",
    test_mode: bool = False,
) -> PathCollection:
    """
    Plot a 2D dataset together with labels.

    :param data: The dataset.
    :param labels: The labels.
    :param cmap: The colormap.
    :param test_mode: Whether to use a test backend.
    :return: The underlying matplotlib object.
    """
    _check_data(data)

    if test_mode:  # pragma: no cover
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    result = plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=cmap)
    plt.show()
    return result


def plot_losses(losses: List[float], test_mode: bool = False) -> Line2D:
    """
    Plot a list of epoch losses.

    :param losses: The epoch losses.
    :param test_mode: Whether to use a test backend.
    :return: The underlying matplotlib object.
    """
    if test_mode:  # pragma: no cover
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    result = plt.plot(losses)[0]
    plt.show()
    return result


def plot_decision_surface(
    x1_grid: torch.Tensor,
    x2_grid: torch.Tensor,
    predictions: torch.Tensor,
    cmap: str = "RdYlGn",
    test_mode: bool = False,
) -> QuadContourSet:
    """
    Plot a decision surface.

    The arguments for this function are usually obtained by calling predict_grid on
    a model. For example you might do the following:

    x1_grid, x2_grid, predictions = model.predict_grid(...)

    plot_decision_surface(x1_grid, x2_grid, predictions)

    :param x1_grid: The x1 values.
    :param x2_grid: The x2 values.
    :param predictions: The predictions values.
    :param cmap: The colormap.
    :param test_mode: Whether to use a test backend.
    :return: The underlying matplotlib object.
    """
    if test_mode:  # pragma: no cover
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    result = plt.contourf(x1_grid, x2_grid, predictions, cmap=cmap)
    plt.colorbar(result)
    plt.show()
    return result


def plot_labels_with_decision_surface(
    data: torch.Tensor,
    labels: torch.Tensor,
    x1_grid: torch.Tensor,
    x2_grid: torch.Tensor,
    predictions: torch.Tensor,
    surface_alpha: float = 0.25,
    cmap: str = "RdYlGn",
    test_mode: bool = False,
) -> Tuple[QuadContourSet, PathCollection]:
    """
    Plot a dataset together with labels and a decision surface.

    See the plot_data and plot_decision_surface documentations for further information.
    """
    _check_data(data)

    if test_mode:  # pragma: no cover
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    countourf_result = plt.contourf(
        x1_grid, x2_grid, predictions, cmap=cmap, alpha=surface_alpha
    )
    plt.colorbar(countourf_result)
    scatter_result = plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=cmap)
    plt.show()
    return scatter_result, countourf_result
