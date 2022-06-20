from unittest import TestCase

import torch

from craftdl import (
    plot_data,
    plot_labels,
    plot_decision_surface,
    plot_labels_with_decision_surface,
    CraftdlException,
    plot_losses,
)


class PlotTest(TestCase):
    def setUp(self) -> None:
        self.data = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.labels = torch.tensor([0, 0, 0, 1])

        self.x1_grid = torch.tensor([[0, 1], [0, 1]])
        self.x2_grid = torch.tensor([[0, 0], [1, 1]])
        self.predictions = torch.tensor([[0, 1], [2, 3]])

    def test_plot_data(self) -> None:
        result = plot_data(self.data, test_mode=True)
        plotted = torch.tensor(result.get_offsets())
        self.assertTrue((plotted == self.data).all())

    def test_plot_labels(self) -> None:
        result = plot_labels(self.data, self.labels, test_mode=True)
        plotted = torch.tensor(result.get_offsets())
        self.assertTrue((plotted == self.data).all())

    def test_plot_losses(self) -> None:
        result = plot_losses([1.0, 0.5, 0.0], test_mode=True)
        xdata = result.get_xdata()
        ydata = result.get_ydata()
        self.assertTrue(list(xdata) == [0.0, 1.0, 2.0])
        self.assertTrue(list(ydata) == [1.0, 0.5, 0.0])

    def test_plot_decision_surface(self) -> None:
        result = plot_decision_surface(
            self.x1_grid, self.x2_grid, self.predictions, test_mode=True
        )
        self.assertEqual(len(result.allsegs), 8)

    def test_plot_data_with_decision_surface(self) -> None:
        labels_result, surf_result = plot_labels_with_decision_surface(
            data=self.data,
            labels=self.labels,
            x1_grid=self.x1_grid,
            x2_grid=self.x2_grid,
            predictions=self.predictions,
            test_mode=True,
        )
        plotted = torch.tensor(labels_result.get_offsets())
        self.assertTrue((plotted == self.data).all())
        self.assertEqual(len(surf_result.allsegs), 8)

    def test_plot_data_bad_dim(self) -> None:
        self.assertRaises(CraftdlException, plot_data, torch.Tensor([0.0]))
