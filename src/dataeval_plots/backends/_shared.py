"""Shared helper functions for plotting backends."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from dataeval_plots.protocols import (
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
    "normalize_image_to_uint8",
    "image_to_base64_png",
    "image_to_hwc",
    "calculate_subplot_grid",
    "validate_class_names",
    "parse_dataset_item",
    "format_label_from_target",
    "merge_metadata",
    "draw_bounding_boxes",
    "extract_boxes_and_labels",
    "CHANNELWISE_METRICS",
]

# Constants
CHANNELWISE_METRICS = [
    "mean",
    "std",
    "var",
    "skew",
    "zeros",
    "brightness",
    "contrast",
    "darkness",
    "entropy",
]


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
) -> tuple[
    NDArray[Any],
    NDArray[Any] | Sequence[Any],
    NDArray[Any] | Sequence[Any],
    str,
    str,
    str,
]:
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
) -> tuple[
    NDArray[Any],
    NDArray[Any] | Sequence[Any],
    NDArray[Any] | Sequence[Any],
    str,
    str,
    str,
    str,
]:
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


def image_to_hwc(image: NDArray[Any]) -> NDArray[Any]:
    """Convert image array to Height-Width-Channel (HWC) format.

    Parameters
    ----------
    image : NDArray[Any]
        Input image array

    Returns
    -------
    NDArray[Any]
        Image in HWC format
    """
    image = np.asarray(image)
    if image.ndim == 2:
        return image[:, :, np.newaxis]  # Grayscale to HWC
    if image.shape[0] in {1, 3, 4}:
        return np.transpose(image, (1, 2, 0))  # Channels-first to HWC
    return image  # Assume already HWC


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


def parse_dataset_item(datum: Any) -> tuple[Any, Any | None, dict[str, Any]]:
    """
    Parse a dataset item into image, target, and metadata components.

    Parameters
    ----------
    datum : Any
        Dataset item that can be:
        - Just an image (array-like)
        - Tuple of (image,)
        - Tuple of (image, target)
        - Tuple of (image, target, metadata)

    Returns
    -------
    tuple
        (image, target, metadata) where:
        - image: The image array
        - target: Target labels/boxes/etc. or None if not present
        - metadata: Dictionary of metadata (empty dict if not present)
    """
    if isinstance(datum, tuple):
        if len(datum) == 1:
            return datum[0], None, {}
        if len(datum) == 2:
            return datum[0], datum[1], {}
        if len(datum) >= 3:
            # Extract metadata - convert to dict if it's not already
            meta = datum[2] if isinstance(datum[2], dict) else {}
            return datum[0], datum[1], meta

    # Single item - just the image
    return datum, None, {}


def format_label_from_target(
    target: Any,
    index2label: dict[int, str] | None = None,
) -> str | None:
    """
    Format a human-readable label string from various target types.

    Parameters
    ----------
    target : Any
        Target can be:
        - Array of pseudo probabilities or one-hot encoded labels
        - Dict with keys 'boxes', 'labels', 'scores' (object detection)
        - Object with attributes 'boxes', 'labels', 'scores'
        - None
    index2label : dict[int, str] or None
        Mapping from class indices to class names

    Returns
    -------
    str or None
        Formatted label string or None if target is None
    """
    if target is None:
        return None

    # Handle dict or object with boxes/labels/scores (object detection format)
    boxes = None
    labels = None

    if isinstance(target, dict):
        boxes = target.get("boxes")
        labels = target.get("labels")
        target.get("scores")
    elif hasattr(target, "boxes") and hasattr(target, "labels"):
        boxes = getattr(target, "boxes", None)
        labels = getattr(target, "labels", None)
        getattr(target, "scores", None)

    # If we found object detection format
    if boxes is not None and labels is not None:
        boxes_arr = np.asarray(boxes)
        labels_arr = np.asarray(labels)

        if len(boxes_arr) == 0:
            return "No objects"

        # Check if labels are one-hot or probabilities (2D array)
        label_indices = np.argmax(labels_arr, axis=1) if labels_arr.ndim == 2 else labels_arr.astype(int)

        # Count objects per class
        unique_labels, counts = np.unique(label_indices, return_counts=True)

        # Format label string
        label_parts = []
        for lbl, cnt in zip(unique_labels, counts):
            if index2label and lbl in index2label:
                label_parts.append(f"{index2label[lbl]}: {cnt}")
            else:
                label_parts.append(f"Class {lbl}: {cnt}")

        return ", ".join(label_parts)

    # Handle array of probabilities or one-hot encoding (classification)
    target_arr = np.asarray(target)

    if target_arr.ndim == 0:
        # Scalar label
        label_idx = int(target_arr)
        if index2label and label_idx in index2label:
            return index2label[label_idx]
        return f"Class {label_idx}"

    if target_arr.ndim == 1 and len(target_arr) > 0:
        # 1D array - could be probabilities or one-hot
        if target_arr.dtype in (np.float32, np.float64):
            # Check if it's one-hot or probabilities
            if np.allclose(target_arr.sum(), 1.0) and np.max(target_arr) <= 1.0:
                # Probabilities or one-hot
                label_idx = int(np.argmax(target_arr))
                confidence = target_arr[label_idx]

                if index2label and label_idx in index2label:
                    return f"{index2label[label_idx]} ({confidence:.2f})"
                return f"Class {label_idx} ({confidence:.2f})"
        else:
            # Integer array - direct label
            if len(target_arr) == 1:
                label_idx = int(target_arr[0])
                if index2label and label_idx in index2label:
                    return index2label[label_idx]
                return f"Class {label_idx}"

    return None


def merge_metadata(
    base_metadata: dict[str, Any],
    additional_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Merge base metadata with additional metadata.

    Parameters
    ----------
    base_metadata : dict[str, Any]
        Base metadata from the dataset item
    additional_metadata : dict[str, Any] or None
        Additional metadata to merge in

    Returns
    -------
    dict[str, Any]
        Merged metadata dictionary
    """
    if additional_metadata is None:
        return base_metadata

    merged = base_metadata.copy()
    merged.update(additional_metadata)
    return merged


def extract_boxes_and_labels(
    target: Any,
) -> tuple[NDArray[Any] | None, NDArray[Any] | None, NDArray[Any] | None]:
    """
    Extract bounding boxes, labels, and scores from object detection targets.

    Parameters
    ----------
    target : Any
        Target that may contain object detection information:
        - Dict with keys 'boxes', 'labels', 'scores'
        - Object with attributes 'boxes', 'labels', 'scores'

    Returns
    -------
    tuple[NDArray | None, NDArray | None, NDArray | None]
        (boxes, labels, scores) where each is None if not found.
        boxes are in XYXY format.
    """
    boxes = None
    labels = None
    scores = None

    if isinstance(target, dict):
        boxes = target.get("boxes")
        labels = target.get("labels")
        scores = target.get("scores")
    elif hasattr(target, "boxes") and hasattr(target, "labels"):
        boxes = getattr(target, "boxes", None)
        labels = getattr(target, "labels", None)
        scores = getattr(target, "scores", None)

    if boxes is not None:
        boxes = np.asarray(boxes)
    if labels is not None:
        labels = np.asarray(labels)
    if scores is not None:
        scores = np.asarray(scores)

    return boxes, labels, scores


def draw_bounding_boxes(
    image: NDArray[Any],
    boxes: NDArray[Any],
    labels: NDArray[Any] | None = None,
    scores: NDArray[Any] | None = None,
    index2label: dict[int, str] | None = None,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> NDArray[Any]:
    """
    Draw bounding boxes on an image.

    Parameters
    ----------
    image : NDArray
        Image array in HWC format (uint8)
    boxes : NDArray
        Bounding boxes in XYXY format, shape (N, 4)
    labels : NDArray or None, default None
        Label indices or probabilities for each box
    scores : NDArray or None, default None
        Confidence scores for each box
    index2label : dict[int, str] or None, default None
        Mapping from class indices to class names
    color : tuple[int, int, int], default (0, 255, 0)
        RGB color for bounding boxes
    thickness : int, default 2
        Line thickness for bounding boxes

    Returns
    -------
    NDArray
        Image with bounding boxes drawn (copy of input)

    Notes
    -----
    This function requires opencv-python (cv2) to be installed.
    If cv2 is not available, it falls back to using PIL which provides
    basic rectangle drawing but with less features.
    """
    try:
        import cv2  # type: ignore
    except ImportError:
        cv2 = None

    # Make a copy to avoid modifying the original
    img_with_boxes = image.copy()

    if len(boxes) == 0:
        return img_with_boxes

    # Process labels to get class indices
    if labels is not None:
        label_indices = np.argmax(labels, axis=1) if labels.ndim == 2 else labels.astype(int)
    else:
        label_indices = None

    if cv2:
        # Use OpenCV for better quality boxes with labels
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.astype(int)

            # Draw rectangle
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, thickness)

            # Build label text
            label_text = None
            if label_indices is not None:
                label_idx = label_indices[i]
                if index2label and label_idx in index2label:
                    label_text = index2label[label_idx]
                else:
                    label_text = f"Class {label_idx}"

                # Add score if available
                if scores is not None and i < len(scores):
                    label_text += f" {scores[i]:.2f}"

            # Draw label background and text
            if label_text:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_thickness = 1
                (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)

                # Draw background rectangle
                cv2.rectangle(
                    img_with_boxes,
                    (x1, y1 - text_height - baseline - 5),
                    (x1 + text_width, y1),
                    color,
                    -1,  # Filled
                )

                # Draw text
                cv2.putText(
                    img_with_boxes,
                    label_text,
                    (x1, y1 - baseline - 2),
                    font,
                    font_scale,
                    (255, 255, 255),  # White text
                    font_thickness,
                )
    else:
        # Fallback to PIL for basic box drawing
        from PIL import Image, ImageDraw, ImageFont

        # Convert to PIL Image
        pil_img = Image.fromarray(img_with_boxes)
        draw = ImageDraw.Draw(pil_img)

        # Try to load a font, fallback to default if not available
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except Exception:
            font = ImageFont.load_default()

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.astype(int)

            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=tuple(color), width=thickness)

            # Build label text
            label_text = None
            if label_indices is not None:
                label_idx = label_indices[i]
                if index2label and label_idx in index2label:
                    label_text = index2label[label_idx]
                else:
                    label_text = f"Class {label_idx}"

                # Add score if available
                if scores is not None and i < len(scores):
                    label_text += f" {scores[i]:.2f}"

            # Draw label text
            if label_text:
                # Draw text background
                try:
                    bbox = draw.textbbox((x1, y1 - 15), label_text, font=font)
                    draw.rectangle(bbox, fill=tuple(color))
                    draw.text((x1, y1 - 15), label_text, fill=(255, 255, 255), font=font)
                except Exception:
                    # Fallback for older PIL versions
                    draw.text((x1, y1 - 15), label_text, fill=tuple(color), font=font)

        # Convert back to numpy
        img_with_boxes = np.array(pil_img)

    return img_with_boxes
