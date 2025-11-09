"""Shared helper functions for plotting backends."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from dataeval_plots.protocols import (
        Indexable,
        PlottableBalance,
        PlottableDiversity,
        PlottableDriftMVDC,
    )

__all__ = [
    "f_out",
    "project_steps",
    "calculate_projection",
    "normalize_reference_outputs",
    "prepare_balance_data",
    "prepare_diversity_data",
    "prepare_drift_data",
    "prepare_coverage_images",
    "normalize_image_to_uint8",
    "image_to_base64_png",
    "calculate_subplot_grid",
    "validate_class_names",
    "CHANNELWISE_METRICS",
]

# Constants
CHANNELWISE_METRICS = ["mean", "std", "var", "skew", "zeros", "brightness", "contrast", "darkness", "entropy"]


def f_out(n_i: NDArray[Any], x: NDArray[Any]) -> NDArray[Any]:
    """
    Calculates the line of best fit based on its free parameters.

    Parameters
    ----------
    n_i : NDArray
        Array of sample sizes
    x : NDArray
        Array of inverse power curve coefficients

    Returns
    -------
    NDArray
        Data points for the line of best fit
    """
    return x[0] * n_i ** (-x[1]) + x[2]


def project_steps(params: NDArray[Any], projection: NDArray[Any]) -> NDArray[Any]:
    """
    Projects the measures for each value of X.

    Parameters
    ----------
    params : NDArray
        Inverse power curve coefficients used to calculate projection
    projection : NDArray
        Steps to extrapolate

    Returns
    -------
    NDArray
        Extrapolated measure values at each projection step
    """
    return 1 - f_out(projection, params)


def calculate_projection(steps: NDArray[Any]) -> NDArray[Any]:
    """
    Calculate the projection array for extrapolation.

    Parameters
    ----------
    steps : NDArray
        Array of step values from the output

    Returns
    -------
    NDArray
        Projection array for extrapolation
    """
    last_X = steps[-1]
    geomshape = (0.01 * last_X, last_X * 4, len(steps))
    return np.geomspace(*geomshape).astype(np.int64)


def normalize_reference_outputs(
    reference_outputs: Sequence[Any] | Any | None,
) -> list[Any]:
    """
    Normalize reference outputs to a list.

    Parameters
    ----------
    reference_outputs : Sequence, single object, or None
        Reference outputs to normalize

    Returns
    -------
    list
        List of reference outputs (empty if None provided)
    """
    if reference_outputs is None:
        return []
    if not isinstance(reference_outputs, (list, tuple)):
        return [reference_outputs]
    return list(reference_outputs)


def prepare_balance_data(
    output: PlottableBalance,
    row_labels: Sequence[Any] | NDArray[Any] | None = None,
    col_labels: Sequence[Any] | NDArray[Any] | None = None,
    plot_classwise: bool = False,
) -> tuple[NDArray[Any], NDArray[Any] | Sequence[Any], NDArray[Any] | Sequence[Any], str, str, str]:
    """
    Prepare balance data for plotting across all backends.

    Parameters
    ----------
    output : PlottableBalance
        The balance output object to plot
    row_labels : ArrayLike or None, default None
        List/Array containing the labels for rows in the histogram
    col_labels : ArrayLike or None, default None
        List/Array containing the labels for columns in the histogram
    plot_classwise : bool, default False
        Whether to plot per-class balance instead of global balance

    Returns
    -------
    tuple
        (data, row_labels, col_labels, xlabel, ylabel, title)
    """
    if plot_classwise:
        if row_labels is None:
            row_labels = output.class_names
        if col_labels is None:
            col_labels = output.factor_names

        data = output.classwise
        xlabel = "Factors"
        ylabel = "Class"
        title = "Classwise Balance"
    else:
        # Combine balance and factors results
        data = np.concatenate(
            [
                output.balance[np.newaxis, 1:],
                output.factors,
            ],
            axis=0,
        )
        # Create a mask for the upper triangle
        mask = np.triu(data + 1, k=0) < 1
        data = np.where(mask, np.nan, data)[:-1]

        if row_labels is None:
            row_labels = output.factor_names[:-1]
        if col_labels is None:
            col_labels = output.factor_names[1:]

        xlabel = ""
        ylabel = ""
        title = "Balance Heatmap"

    return data, row_labels, col_labels, xlabel, ylabel, title


def prepare_diversity_data(
    output: PlottableDiversity,
    row_labels: Sequence[Any] | NDArray[Any] | None = None,
    col_labels: Sequence[Any] | NDArray[Any] | None = None,
    plot_classwise: bool = False,
) -> tuple[NDArray[Any], NDArray[Any] | Sequence[Any], NDArray[Any] | Sequence[Any], str, str, str, str]:
    """
    Prepare diversity data for plotting across all backends.

    Parameters
    ----------
    output : PlottableDiversity
        The diversity output object to plot
    row_labels : ArrayLike or None, default None
        List/Array containing the labels for rows in the histogram
    col_labels : ArrayLike or None, default None
        List/Array containing the labels for columns in the histogram
    plot_classwise : bool, default False
        Whether to plot per-class diversity instead of global diversity

    Returns
    -------
    tuple
        (data, row_labels, col_labels, xlabel, ylabel, title, method_name)
        data is None for non-classwise bar charts
    """
    method_name = output.meta().arguments["method"].title()

    if plot_classwise:
        if row_labels is None:
            row_labels = output.class_names
        if col_labels is None:
            col_labels = output.factor_names

        data = output.classwise
        xlabel = "Factors"
        ylabel = "Class"
        title = "Classwise Diversity"
    else:
        # Bar chart - no heatmap data
        heat_labels = ["class_labels"] + list(output.factor_names)
        data = np.ndarray(0)  # unused
        row_labels = heat_labels
        col_labels = []  # unused
        xlabel = "Factors"
        ylabel = "Diversity Index"
        title = "Diversity Index by Factor"

    return data, row_labels, col_labels, xlabel, ylabel, title, method_name


def prepare_drift_data(
    output: PlottableDriftMVDC,
) -> tuple[Any, Any, Any, NDArray[Any], bool]:
    """
    Prepare drift detection data for plotting across all backends.

    Parameters
    ----------
    output : PlottableDriftMVDC
        The drift MVDC output object to plot

    Returns
    -------
    tuple
        (resdf, trndf, tstdf, driftx, is_sufficient)
        resdf: Full results dataframe
        trndf: Training/reference data
        tstdf: Test/analysis data
        driftx: Indices where drift was detected
        is_sufficient: Whether there's enough data to plot (>= 3 rows)
    """
    resdf = output.to_dataframe()
    is_sufficient = resdf.shape[0] >= 3

    if not is_sufficient:
        return resdf, None, None, np.array([]), False

    trndf = resdf[resdf["chunk"]["period"] == "reference"]
    tstdf = resdf[resdf["chunk"]["period"] == "analysis"]
    driftx = np.where(resdf["domain_classifier_auroc"]["alert"].values)[0]  # type: ignore

    return resdf, trndf, tstdf, driftx, True


def prepare_coverage_images(
    output: Any,  # PlottableCoverage
    images: Indexable | None,
    top_k: int,
) -> tuple[list[Any], int, int, int]:
    """
    Prepare and validate coverage images for plotting.

    Parameters
    ----------
    output : PlottableCoverage
        The coverage output object to plot
    images : Indexable or None
        Original images (not embeddings)
    top_k : int
        Number of images to plot

    Returns
    -------
    tuple
        (selected_images, num_images, rows, cols)

    Raises
    ------
    ValueError
        If images is None or indices are out of bounds
    """
    if images is None:
        raise ValueError("images parameter is required for coverage plotting")

    if np.max(output.uncovered_indices) > len(images):
        raise ValueError(
            f"Uncovered indices {output.uncovered_indices} specify images "
            f"unavailable in the provided number of images {len(images)}."
        )

    # Determine which images to plot
    selected_indices = output.uncovered_indices[:top_k]
    num_images = min(top_k, len(selected_indices))

    # Calculate grid layout (3 columns)
    rows = int(np.ceil(num_images / 3))
    cols = min(3, num_images)

    # Get selected images
    selected_images = list(images[:num_images])

    return selected_images, num_images, rows, cols


def normalize_image_to_uint8(img_np: NDArray[Any]) -> NDArray[Any]:
    """
    Normalize image array to 0-255 uint8 range.

    Parameters
    ----------
    img_np : NDArray
        Image array in HWC format

    Returns
    -------
    NDArray
        Image array in uint8 format (0-255 range)
    """
    if img_np.max() <= 1.0:
        return (img_np * 255).astype(np.uint8)
    return img_np.astype(np.uint8)


def image_to_base64_png(img_np: NDArray[Any]) -> str:
    """
    Convert numpy image array to base64 encoded PNG string.

    Parameters
    ----------
    img_np : NDArray
        Image array in uint8 format

    Returns
    -------
    str
        Base64 encoded PNG data URL string (data:image/png;base64,...)
    """
    import base64
    from io import BytesIO

    from PIL import Image

    # Convert to PIL Image
    pil_img = Image.fromarray(img_np)

    # Convert to base64
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return f"data:image/png;base64,{img_str}"


def calculate_subplot_grid(num_items: int, cols_per_row: int = 3) -> tuple[int, int]:
    """
    Calculate grid layout for subplots.

    Parameters
    ----------
    num_items : int
        Number of items to plot
    cols_per_row : int, default 3
        Number of columns per row

    Returns
    -------
    tuple
        (rows, cols) for subplot grid
    """
    import math

    rows = math.ceil(num_items / cols_per_row)
    cols = min(num_items, cols_per_row)
    return rows, cols


def validate_class_names(measures: NDArray[Any], class_names: Sequence[str] | None) -> None:
    """
    Validate that class names align with measures.

    Parameters
    ----------
    measures : NDArray
        Measures array (multiclass, first dimension is classes)
    class_names : Sequence[str] or None
        List of class names

    Raises
    ------
    IndexError
        If class name count does not align with measures
    """
    if class_names is not None and len(measures) != len(class_names):
        raise IndexError("Class name count does not align with measures")
