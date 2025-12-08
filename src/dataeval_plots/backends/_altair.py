"""Altair plotting backend."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dataeval_plots.backends._base import BasePlottingBackend
from dataeval_plots.backends._shared import (
    CHANNELWISE_METRICS,
    calculate_projection,
    draw_bounding_boxes,
    extract_boxes_and_labels,
    format_label_from_target,
    image_to_base64_png,
    image_to_hwc,
    merge_metadata,
    normalize_image_to_uint8,
    normalize_reference_outputs,
    parse_dataset_item,
    prepare_balance_data,
    prepare_diversity_data,
    prepare_drift_data,
    project_steps,
    validate_class_names,
)
from dataeval_plots.protocols import (
    Dataset,
    PlottableBalance,
    PlottableDiversity,
    PlottableDriftMVDC,
    PlottableStats,
    PlottableSufficiency,
)


class AltairBackend(BasePlottingBackend):
    """Altair implementation of plotting backend."""

    def _plot_balance(
        self,
        output: PlottableBalance,
        figsize: tuple[int, int] | None = None,
        row_labels: Sequence[Any] | NDArray[Any] | None = None,
        col_labels: Sequence[Any] | NDArray[Any] | None = None,
        plot_classwise: bool = False,
    ) -> Any:  # alt.Chart
        """
        Plot a heatmap of balance information.

        Parameters
        ----------
        output : PlottableBalance
            The balance output object to plot
        figsize : tuple[int, int] | None, default None
            Figure size in inches (width, height)
        row_labels : ArrayLike or None, default None
            List/Array containing the labels for rows in the histogram
        col_labels : ArrayLike or None, default None
            List/Array containing the labels for columns in the histogram
        plot_classwise : bool, default False
            Whether to plot per-class balance instead of global balance

        Returns
        -------
        alt.Chart
            Altair heatmap chart
        """
        import altair as alt
        import pandas as pd

        # Use shared helper to prepare data
        data, row_labels, col_labels, xlabel, ylabel, title = prepare_balance_data(
            output, row_labels, col_labels, plot_classwise
        )

        # Convert to long format for Altair
        rows, cols = data.shape
        heatmap_data = []
        for i in range(rows):
            for j in range(cols):
                # For non-classwise (triangular) plots, only include upper triangle cells
                # For classwise plots, include all non-NaN values
                if not np.isnan(data[i, j]):
                    heatmap_data.append(
                        {
                            "row": str(row_labels[i]),
                            "col": str(col_labels[j]),
                            "value": float(data[i, j]),
                            "row_idx": i,
                            "col_idx": j,
                        }
                    )

        df = pd.DataFrame(heatmap_data)

        # Determine chart dimensions
        if figsize is not None:
            width_inches, height_inches = figsize
            width = int(width_inches * 100)
            height = int(height_inches * 100)
        else:
            width = 400
            height = 400

        # Create heatmap with proper ordering
        # For triangular heatmaps, we need to preserve the diagonal structure
        chart = (
            alt.Chart(df)
            .mark_rect()
            .encode(
                x=alt.X("col:N", title=xlabel, axis=alt.Axis(labelAngle=-45), sort=list(col_labels)),
                y=alt.Y("row:N", title=ylabel, sort=list(row_labels)),
                color=alt.Color(
                    "value:Q",
                    scale=alt.Scale(scheme="viridis", domain=[0, 1]),
                    title=["Normalized", "Mutual", "Information"],  # Multi-line title
                ),
                tooltip=["row:N", "col:N", alt.Tooltip("value:Q", format=".2f")],
            )
            .properties(width=width, height=height, title=title)
        )

        # Add text labels
        text = (
            alt.Chart(df)
            .mark_text(baseline="middle")
            .encode(
                x=alt.X("col:N", sort=list(col_labels)),
                y=alt.Y("row:N", sort=list(row_labels)),
                text=alt.Text("value:Q", format=".2f"),
                color=alt.condition(alt.datum.value > 0.5, alt.value("white"), alt.value("black")),
            )
        )

        return chart + text

    def _plot_diversity(
        self,
        output: PlottableDiversity,
        figsize: tuple[int, int] | None = None,
        row_labels: Sequence[Any] | NDArray[Any] | None = None,
        col_labels: Sequence[Any] | NDArray[Any] | None = None,
        plot_classwise: bool = False,
    ) -> Any:  # alt.Chart
        """
        Plot a heatmap or bar chart of diversity information.

        Parameters
        ----------
        output : PlottableDiversity
            The diversity output object to plot
        figsize : tuple[int, int] | None, default None
            Figure size in inches (width, height)
        row_labels : ArrayLike or None, default None
            List/Array containing the labels for rows in the histogram
        col_labels : ArrayLike or None, default None
            List/Array containing the labels for columns in the histogram
        plot_classwise : bool, default False
            Whether to plot per-class balance instead of global balance

        Returns
        -------
        alt.Chart
            Altair chart (heatmap or bar chart)
        """
        import altair as alt
        import pandas as pd

        # Use shared helper to prepare data
        data, row_labels, col_labels, xlabel, ylabel, title, method_name = prepare_diversity_data(
            output, row_labels, col_labels, plot_classwise
        )

        # Determine chart dimensions
        if figsize is not None:
            width_inches, height_inches = figsize
            width = int(width_inches * 100)
            height = int(height_inches * 100)
        else:
            width = 400 if plot_classwise else 500
            height = 400

        if plot_classwise:
            # Create heatmap similar to balance
            rows, cols = data.shape
            heatmap_data = []
            for i in range(rows):
                for j in range(cols):
                    heatmap_data.append(
                        {"row": str(row_labels[i]), "col": str(col_labels[j]), "value": float(data[i, j])}
                    )

            df = pd.DataFrame(heatmap_data)

            chart = (
                alt.Chart(df)
                .mark_rect()
                .encode(
                    x=alt.X("col:N", title=xlabel, axis=alt.Axis(labelAngle=-45)),
                    y=alt.Y("row:N", title=ylabel),
                    color=alt.Color(
                        "value:Q",
                        scale=alt.Scale(scheme="viridis", domain=[0, 1]),
                        title=["Normalized", method_name, "Index"],
                    ),
                    tooltip=["row:N", "col:N", alt.Tooltip("value:Q", format=".2f")],
                )
                .properties(width=width, height=height, title=title)
            )

            text = (
                alt.Chart(df)
                .mark_text(baseline="middle")
                .encode(
                    x=alt.X("col:N"),
                    y=alt.Y("row:N"),
                    text=alt.Text("value:Q", format=".2f"),
                    color=alt.condition(alt.datum.value > 0.5, alt.value("white"), alt.value("black")),
                )
            )

            return chart + text
        # Bar chart for diversity indices
        # DataFrame-based: get diversity values from factors DataFrame
        diversity_values = output.factors["diversity_value"].to_list()
        df = pd.DataFrame({"factor": row_labels, "diversity": diversity_values})

        return (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("factor:N", title=xlabel, axis=alt.Axis(labelAngle=-45)),
                y=alt.Y("diversity:Q", title=ylabel),
                tooltip=["factor:N", alt.Tooltip("diversity:Q", format=".3f")],
            )
            .properties(width=width, height=height, title=title)
        )

    def _plot_sufficiency(
        self,
        output: PlottableSufficiency,
        figsize: tuple[int, int] | None = None,
        class_names: Sequence[str] | None = None,
        show_error_bars: bool = True,
        show_asymptote: bool = True,
        reference_outputs: Sequence[PlottableSufficiency] | PlottableSufficiency | None = None,
    ) -> list[Any]:  # list[alt.Chart]
        """
        Plotting function for data sufficiency tasks.

        Parameters
        ----------
        output : PlottableSufficiency
            The sufficiency output object to plot
        figsize : tuple[int, int] | None, default None
            Figure size in inches (width, height)
        class_names : Sequence[str] | None, default None
            List of class names
        show_error_bars : bool, default True
            True if error bars should be plotted, False if not
        show_asymptote : bool, default True
            True if asymptote should be plotted, False if not
        reference_outputs : Sequence[PlottableSufficiency] | PlottableSufficiency, default None
            Singular or multiple SufficiencyOutput objects to include in plots

        Returns
        -------
        list[alt.Chart]
            List of Altair charts for each measure
        """
        import altair as alt
        import pandas as pd

        # Determine chart dimensions
        if figsize is not None:
            width_inches, height_inches = figsize
            width = int(width_inches * 100)
            height = int(height_inches * 100)
        else:
            width = 500
            height = 400

        # Extrapolation parameters
        projection = calculate_projection(output.steps)

        # Wrap reference
        reference_outputs = normalize_reference_outputs(reference_outputs)

        charts = []

        for name, measures in output.averaged_measures.items():
            if measures.ndim > 1:
                # Multi-class plotting
                validate_class_names(measures, class_names)

                for i, values in enumerate(measures):
                    class_name = str(i) if class_names is None else class_names[i]

                    # Prepare data
                    plot_data = []

                    # Actual measurements
                    for step, value in zip(output.steps, values):
                        plot_data.append(
                            {
                                "step": int(step),
                                "value": float(value),
                                "type": "Model Results",
                                "series": f"{name}_{class_name}",
                            }
                        )

                    # Projection curve
                    proj_values = project_steps(output.params[name][i], projection)
                    for step, value in zip(projection, proj_values):
                        plot_data.append(
                            {
                                "step": int(step),
                                "value": float(value),
                                "type": "Potential Model Results",
                                "series": f"{name}_{class_name}",
                            }
                        )

                    df = pd.DataFrame(plot_data)

                    # Create chart
                    potential_df = df[df["type"] == "Potential Model Results"]
                    line = (
                        alt.Chart(potential_df)  # type: ignore[arg-type]
                        .mark_line()
                        .encode(
                            x=alt.X("step:Q", scale=alt.Scale(type="log"), title="Steps"),
                            y=alt.Y("value:Q", title=name),
                            color=alt.Color("type:N", legend=alt.Legend(title="Series")),
                            tooltip=["step:Q", alt.Tooltip("value:Q", format=".4f")],
                        )
                    )

                    results_df = df[df["type"] == "Model Results"]
                    points = (
                        alt.Chart(results_df)  # type: ignore[arg-type]
                        .mark_point(size=100)
                        .encode(
                            x=alt.X("step:Q", scale=alt.Scale(type="log")),
                            y=alt.Y("value:Q"),
                            color=alt.Color("type:N"),
                            tooltip=["step:Q", alt.Tooltip("value:Q", format=".4f")],
                        )
                    )

                    chart = (line + points).properties(
                        width=width, height=height, title=f"{name} Sufficiency - Class {class_name}"
                    )

                    # Add asymptote if requested
                    if show_asymptote:
                        bound = 1 - output.params[name][i][2]
                        asymptote_df = pd.DataFrame(
                            {
                                "step": [projection[0], projection[-1]],
                                "value": [bound, bound],
                                "type": [f"Asymptote: {bound:.4g}", f"Asymptote: {bound:.4g}"],
                            }
                        )
                        asymptote = (
                            alt.Chart(asymptote_df)
                            .mark_line(strokeDash=[5, 5])
                            .encode(x="step:Q", y="value:Q", color=alt.Color("type:N"))
                        )
                        chart = chart + asymptote

                    charts.append(chart)
            else:
                # Single-class plotting
                plot_data = []

                # Actual measurements
                for step, value in zip(output.steps, measures):
                    plot_data.append({"step": int(step), "value": float(value), "type": "Model Results"})

                # Projection curve
                proj_values = project_steps(output.params[name], projection)
                for step, value in zip(projection, proj_values):
                    plot_data.append({"step": int(step), "value": float(value), "type": "Potential Model Results"})

                df = pd.DataFrame(plot_data)

                # Create chart
                potential_df = df[df["type"] == "Potential Model Results"]
                line = (
                    alt.Chart(potential_df)  # type: ignore[arg-type]
                    .mark_line()
                    .encode(
                        x=alt.X("step:Q", scale=alt.Scale(type="log"), title="Steps"),
                        y=alt.Y("value:Q", title=name),
                        color=alt.Color("type:N", legend=alt.Legend(title="Series")),
                        tooltip=["step:Q", alt.Tooltip("value:Q", format=".4f")],
                    )
                )

                results_df = df[df["type"] == "Model Results"]
                points = (
                    alt.Chart(results_df)  # type: ignore[arg-type]
                    .mark_point(size=100)
                    .encode(
                        x=alt.X("step:Q", scale=alt.Scale(type="log")),
                        y=alt.Y("value:Q"),
                        color=alt.Color("type:N"),
                        tooltip=["step:Q", alt.Tooltip("value:Q", format=".4f")],
                    )
                )

                chart = (line + points).properties(width=width, height=height, title=f"{name} Sufficiency")

                # Add asymptote if requested
                if show_asymptote:
                    bound = 1 - output.params[name][2]
                    asymptote_df = pd.DataFrame(
                        {
                            "step": [projection[0], projection[-1]],
                            "value": [bound, bound],
                            "type": [f"Asymptote: {bound:.4g}", f"Asymptote: {bound:.4g}"],
                        }
                    )
                    asymptote = (
                        alt.Chart(asymptote_df)
                        .mark_line(strokeDash=[5, 5])
                        .encode(x="step:Q", y="value:Q", color=alt.Color("type:N"))
                    )
                    chart = chart + asymptote

                charts.append(chart)

        return charts

    def _plot_stats(
        self,
        output: PlottableStats,
        figsize: tuple[int, int] | None = None,
        log: bool = True,
        channel_limit: int | None = None,
        channel_index: int | Iterable[int] | None = None,
    ) -> Any:  # alt.VConcatChart | alt.HConcatChart
        """
        Plots the statistics as a set of histograms.

        Parameters
        ----------
        output : PlottableStats
            The stats output object to plot
        figsize : tuple[int, int] | None, default None
            Figure size in inches (width, height) - applied to overall grid size
        log : bool, default True
            If True, plots the histograms on a logarithmic scale.
        channel_limit : int or None, default None
            The maximum number of channels to plot. If None, all channels are plotted.
        channel_index : int, Iterable[int] or None, default None
            The index or indices of the channels to plot. If None, all channels are plotted.

        Returns
        -------
        alt.VConcatChart or alt.HConcatChart
            Altair chart with histogram grid
        """
        import altair as alt
        import pandas as pd

        max_channels, ch_mask = output._get_channels(channel_limit, channel_index)
        factors = output.factors(exclude_constant=True)

        if not factors:
            # Return empty chart
            return alt.Chart(pd.DataFrame()).mark_point()

        # Determine individual histogram dimensions (for 3-column grid)
        if figsize is not None:
            width_inches, height_inches = figsize
            width = int(width_inches * 100)
            height = int(height_inches * 100)
            hist_width = width // 3  # 3 columns
            hist_height = height // ((len(factors) + 2) // 3)  # rows based on number of metrics
        else:
            hist_width = 250
            hist_height = 200

        charts = []

        if max_channels == 1:
            # Single channel histogram
            for metric_name, metric_values in factors.items():
                df = pd.DataFrame({"value": metric_values.flatten(), "metric": metric_name})

                chart = (
                    alt.Chart(df)
                    .mark_bar()
                    .encode(
                        x=alt.X("value:Q", bin=alt.Bin(maxbins=20), title="Values"),
                        y=alt.Y("count()", scale=alt.Scale(type="log" if log else "linear"), title="Counts"),
                        tooltip=["count()"],
                    )
                    .properties(width=hist_width, height=hist_height, title=metric_name)
                )
                charts.append(chart)
        else:
            # Multi-channel histogram - use shared constant
            for metric_name, metric_values in factors.items():
                if metric_name in CHANNELWISE_METRICS:
                    # Reshape for channel-wise data
                    data = metric_values[ch_mask].reshape(-1, max_channels)

                    plot_data = []
                    for ch_idx in range(max_channels):
                        for val in data[:, ch_idx]:
                            plot_data.append(
                                {"value": float(val), "channel": f"Channel {ch_idx}", "metric": metric_name}
                            )

                    df = pd.DataFrame(plot_data)

                    chart = (
                        alt.Chart(df)
                        .mark_bar(opacity=0.7)
                        .encode(
                            x=alt.X("value:Q", bin=alt.Bin(maxbins=20), title="Values"),
                            y=alt.Y("count()", scale=alt.Scale(type="log" if log else "linear"), title="Counts"),
                            color=alt.Color("channel:N", legend=alt.Legend(title="Channel")),
                            tooltip=["channel:N", "count()"],
                        )
                        .properties(width=hist_width, height=hist_height, title=metric_name)
                    )
                    charts.append(chart)
                else:
                    # Non-channelwise metric
                    df = pd.DataFrame({"value": metric_values.flatten(), "metric": metric_name})

                    chart = (
                        alt.Chart(df)
                        .mark_bar()
                        .encode(
                            x=alt.X("value:Q", bin=alt.Bin(maxbins=20), title="Values"),
                            y=alt.Y("count()", scale=alt.Scale(type="log" if log else "linear"), title="Counts"),
                            tooltip=["count()"],
                        )
                        .properties(width=hist_width, height=hist_height, title=metric_name)
                    )
                    charts.append(chart)

        # Arrange in grid (3 columns)
        rows = []
        for i in range(0, len(charts), 3):
            row_charts = charts[i : i + 3]
            if len(row_charts) == 1:
                rows.append(row_charts[0])
            else:
                rows.append(alt.hconcat(*row_charts))

        if len(rows) == 1:
            return rows[0]
        return alt.vconcat(*rows)

    def _plot_drift_mvdc(
        self,
        output: PlottableDriftMVDC,
        figsize: tuple[int, int] | None = None,
    ) -> Any:  # alt.Chart
        """
        Render the roc_auc metric over the train/test data in relation to the threshold.

        Parameters
        ----------
        output : PlottableDriftMVDC
            The drift MVDC output object to plot
        figsize : tuple[int, int] | None, default None
            Figure size in inches (width, height)

        Returns
        -------
        alt.Chart
            Altair line chart with drift detection
        """
        import altair as alt
        import pandas as pd

        # Determine chart dimensions
        if figsize is not None:
            width_inches, height_inches = figsize
            width = int(width_inches * 100)
            height = int(height_inches * 100)
        else:
            width = 600
            height = 400

        # Use shared helper to prepare drift data
        resdf, _, _, _, is_sufficient = prepare_drift_data(output)

        if not is_sufficient:
            # Not enough data to plot
            return alt.Chart(pd.DataFrame()).mark_point().properties(title="Insufficient data for drift detection plot")

        # Prepare data for plotting
        plot_data = []

        for idx, row in resdf.iterrows():
            period = row["chunk"]["period"]
            value = row["domain_classifier_auroc"]["value"]
            upper_thr = row["domain_classifier_auroc"]["upper_threshold"]
            lower_thr = row["domain_classifier_auroc"]["lower_threshold"]
            alert = row["domain_classifier_auroc"]["alert"]

            plot_data.append(
                {
                    "index": idx,
                    "value": float(value),
                    "upper_threshold": float(upper_thr),
                    "lower_threshold": float(lower_thr),
                    "period": "train" if period == "reference" else "test",
                    "alert": bool(alert),
                }
            )

        df = pd.DataFrame(plot_data)

        # Create base chart
        base = alt.Chart(df).encode(x=alt.X("index:Q", title="Chunk Index"))

        # Threshold lines
        upper_line = base.mark_line(strokeDash=[5, 5], color="red").encode(
            y=alt.Y("upper_threshold:Q", title="ROC AUC")
        )

        lower_line = base.mark_line(strokeDash=[5, 5], color="red").encode(y="lower_threshold:Q")

        # Train and test lines
        train_df = df[df["period"] == "train"]
        test_df = df[df["period"] == "test"]

        train_line = (
            alt.Chart(train_df)  # type: ignore[arg-type]
            .mark_line(color="blue")
            .encode(
                x="index:Q",
                y=alt.Y("value:Q", scale=alt.Scale(domain=[0, 1.1])),
                tooltip=["index:Q", alt.Tooltip("value:Q", format=".4f")],
            )
        )

        test_line = (
            alt.Chart(test_df)  # type: ignore[arg-type]
            .mark_line(color="green")
            .encode(x="index:Q", y="value:Q", tooltip=["index:Q", alt.Tooltip("value:Q", format=".4f")])
        )

        # Drift markers
        drift_df = df[df["alert"]]
        drift_points = (
            alt.Chart(drift_df)  # type: ignore[arg-type]
            .mark_point(shape="diamond", size=100, color="magenta", filled=True)
            .encode(x="index:Q", y="value:Q", tooltip=["index:Q", alt.Tooltip("value:Q", format=".4f")])
        )

        # Combine all layers
        return (upper_line + lower_line + train_line + test_line + drift_points).properties(
            width=width, height=height, title="Domain Classifier, Drift Detection"
        )

    def _plot_image_grid(
        self,
        dataset: Dataset,
        indices: Sequence[int],
        images_per_row: int = 3,
        figsize: tuple[int, int] | None = None,
        show_labels: bool = False,
        show_metadata: bool = False,
        additional_metadata: Sequence[dict[str, Any]] | None = None,
    ) -> Any:
        """
        Plot a grid of images from a dataset using Altair.

        Parameters
        ----------
        dataset : Dataset
            MAITE-compatible dataset containing images
        indices : Sequence[int]
            Indices of images to plot from the dataset
        images_per_row : int, default 3
            Number of images to display per row
        figsize : tuple[int, int] or None, default None
            Figure size in inches (width, height)
        show_labels : bool, default False
            Whether to display labels extracted from targets
        show_metadata : bool, default False
            Whether to display metadata from the dataset items
        additional_metadata : Sequence[dict[str, Any]] or None, default None
            Additional metadata to display for each image (must match length of indices)

        Returns
        -------
        altair.Chart
            Altair chart with concatenated image subplots

        Raises
        ------
        ValueError
            If additional_metadata length doesn't match indices length
        """
        import altair as alt
        import pandas as pd

        # Validate additional_metadata length
        if additional_metadata is not None and len(additional_metadata) != len(indices):
            raise ValueError(
                f"additional_metadata length ({len(additional_metadata)}) must match indices length ({len(indices)})"
            )

        num_images = len(indices)
        num_rows = (num_images + images_per_row - 1) // images_per_row

        # Get index2label mapping if available
        index2label = dataset.metadata.get("index2label") if hasattr(dataset, "metadata") else None

        # Determine dimensions
        if figsize is not None:
            # figsize is in inches (width, height), convert to pixels
            width_inches, height_inches = figsize
            width = int(width_inches * 100)
            height = int(height_inches * 100)
            img_width = width // images_per_row
            img_height = height // num_rows if num_rows > 0 else height
        else:
            # Auto-detect size based on first image dimensions
            # Get the first image to determine dimensions
            datum = dataset[indices[0]]
            image, _, _ = parse_dataset_item(datum)
            image_hwc = image_to_hwc(image)
            first_img_height, first_img_width = image_hwc.shape[:2]

            # Use actual image dimensions with slim borders (5% padding on top/bottom)
            padding_factor = 0.05
            img_width = first_img_width
            img_height = int(first_img_height * (1 + 2 * padding_factor))  # Add top and bottom padding

        # Prepare all image data with grid positions
        image_data = []
        for i, idx in enumerate(indices):
            # Get dataset item and parse it
            datum = dataset[idx]
            image, target, metadata = parse_dataset_item(datum)

            # Merge with additional metadata if provided
            if additional_metadata is not None:
                metadata = merge_metadata(metadata, additional_metadata[i])

            # Convert image to base64
            image_hwc = image_to_hwc(image)
            image_uint8 = normalize_image_to_uint8(image_hwc)

            # Check if we have object detection targets and should draw boxes
            boxes, labels, scores = extract_boxes_and_labels(target)
            if boxes is not None and len(boxes) > 0:
                # Draw bounding boxes
                image_uint8 = draw_bounding_boxes(
                    image_uint8,
                    boxes,
                    labels,
                    scores,
                    index2label,
                )

            img_base64 = image_to_base64_png(image_uint8)

            row = i // images_per_row
            col = i % images_per_row

            # Build title from labels and metadata
            title_parts = []

            if show_labels and target is not None:
                label_str = format_label_from_target(target, index2label)
                if label_str:
                    title_parts.append(label_str)

            if show_metadata and metadata:
                # Format metadata as key: value pairs
                metadata_strs = [f"{k}: {v}" for k, v in metadata.items()]
                title_parts.extend(metadata_strs)

            # Create title or use empty string
            title = " | ".join(title_parts) if title_parts else ""

            image_data.append({"image": img_base64, "row": row, "col": col, "idx": idx, "title": title})

        df = pd.DataFrame(image_data)

        # Create image chart
        image_chart = (
            alt.Chart(df)
            .mark_image(width=img_width, height=img_height)
            .encode(
                url="image:N",
                x=alt.X("col:O", axis=None, scale=alt.Scale(padding=0)),
                y=alt.Y("row:O", axis=None, scale=alt.Scale(padding=0)),
            )
        )

        # Add text labels if any titles are present
        if show_labels or show_metadata:
            text_chart = (
                alt.Chart(df)
                .mark_text(align="center", baseline="top", dy=5, fontSize=8)
                .encode(
                    text="title:N",
                    x=alt.X("col:O", axis=None, scale=alt.Scale(padding=0)),
                    y=alt.Y("row:O", axis=None, scale=alt.Scale(padding=0)),
                )
            )
            chart = (image_chart + text_chart).properties(
                width=img_width * images_per_row, height=img_height * num_rows
            )
        else:
            chart = image_chart.properties(width=img_width * images_per_row, height=img_height * num_rows)

        return chart
