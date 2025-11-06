"""Plotting functions for bias-related outputs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from dataeval.data._images import Images
from dataeval.outputs._bias import BalanceOutput, CoverageOutput, DiversityOutput
from dataeval.plots import plot
from dataeval.plots.utils import heatmap
from dataeval.protocols import Dataset
from dataeval.utils._array import as_numpy, channels_first_to_last

__all__ = []


@plot.register
def plot_balance_output(
    output: BalanceOutput,
    row_labels: Sequence[Any] | NDArray[Any] | None = None,
    col_labels: Sequence[Any] | NDArray[Any] | None = None,
    plot_classwise: bool = False,
) -> Figure:
    """
    Plot a heatmap of balance information.

    Parameters
    ----------
    output : BalanceOutput
        The balance output object to plot
    row_labels : ArrayLike or None, default None
        List/Array containing the labels for rows in the histogram
    col_labels : ArrayLike or None, default None
        List/Array containing the labels for columns in the histogram
    plot_classwise : bool, default False
        Whether to plot per-class balance instead of global balance

    Returns
    -------
    matplotlib.figure.Figure
    """
    if plot_classwise:
        if row_labels is None:
            row_labels = output.class_names
        if col_labels is None:
            col_labels = output.factor_names

        fig = heatmap(
            output.classwise,
            row_labels,
            col_labels,
            xlabel="Factors",
            ylabel="Class",
            cbarlabel="Normalized Mutual Information",
        )
    else:
        # Combine balance and factors results
        data = np.concatenate(
            [
                output.balance[np.newaxis, 1:],
                output.factors,
            ],
            axis=0,
        )
        # Create a mask for the upper triangle of the symmetrical array, ignoring the diagonal
        mask = np.triu(data + 1, k=0) < 1
        # Finalize the data for the plot, last row is last factor x last factor so it gets dropped
        heat_data = np.where(mask, np.nan, data)[:-1]
        # Creating label array for heat map axes
        heat_labels = output.factor_names

        if row_labels is None:
            row_labels = heat_labels[:-1]
        if col_labels is None:
            col_labels = heat_labels[1:]

        fig = heatmap(heat_data, row_labels, col_labels, cbarlabel="Normalized Mutual Information")

    return fig


@plot.register
def plot_diversity_output(
    output: DiversityOutput,
    row_labels: Sequence[Any] | NDArray[Any] | None = None,
    col_labels: Sequence[Any] | NDArray[Any] | None = None,
    plot_classwise: bool = False,
) -> Figure:
    """
    Plot a heatmap of diversity information.

    Parameters
    ----------
    output : DiversityOutput
        The diversity output object to plot
    row_labels : ArrayLike or None, default None
        List/Array containing the labels for rows in the histogram
    col_labels : ArrayLike or None, default None
        List/Array containing the labels for columns in the histogram
    plot_classwise : bool, default False
        Whether to plot per-class balance instead of global balance

    Returns
    -------
    matplotlib.figure.Figure
    """
    from dataclasses import asdict

    if plot_classwise:
        if row_labels is None:
            row_labels = output.class_names
        if col_labels is None:
            col_labels = output.factor_names

        fig = heatmap(
            output.classwise,
            row_labels,
            col_labels,
            xlabel="Factors",
            ylabel="Class",
            cbarlabel=f"Normalized {asdict(output.meta())['arguments']['method'].title()} Index",
        )

    else:
        # Creating label array for heat map axes
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 8))
        heat_labels = ["class_labels"] + list(output.factor_names)
        ax.bar(heat_labels, output.diversity_index)
        ax.set_xlabel("Factors")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        fig.tight_layout()

    return fig


@plot.register
def plot_coverage_output(
    output: CoverageOutput,
    images: Images[Any] | Dataset[Any],
    top_k: int = 6,
) -> Figure:
    """
    Plot the top k images together for visualization.

    Parameters
    ----------
    output : CoverageOutput
        The coverage output object to plot
    images : Images or Dataset
        Original images (not embeddings) in (N, C, H, W) or (N, H, W) format
    top_k : int, default 6
        Number of images to plot (plotting assumes groups of 3)

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    images_obj = Images(images) if isinstance(images, Dataset) else images
    if np.max(output.uncovered_indices) > len(images_obj):
        raise ValueError(
            f"Uncovered indices {output.uncovered_indices} specify images "
            f"unavailable in the provided number of images {len(images_obj)}."
        )

    # Determine which images to plot
    selected_indices = output.uncovered_indices[:top_k]

    # Plot the images
    num_images = min(top_k, len(selected_indices))

    rows = int(np.ceil(num_images / 3))
    cols = min(3, num_images)
    fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))

    # Flatten axes using numpy array explicitly for compatibility
    axs_flat = np.asarray(axs).flatten()

    for image, ax in zip(images_obj[:num_images], axs_flat):
        image = channels_first_to_last(as_numpy(image))
        ax.imshow(image)
        ax.axis("off")

    for ax in axs_flat[num_images:]:
        ax.axis("off")

    fig.tight_layout()
    return fig
