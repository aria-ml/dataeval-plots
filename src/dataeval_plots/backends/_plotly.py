"""Plotly plotting backend."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from numpy.typing import NDArray

from dataeval_plots.backends._base import BasePlottingBackend
from dataeval_plots.backends._shared import (
    CHANNELWISE_METRICS,
    calculate_projection,
    calculate_subplot_grid,
    draw_bounding_boxes,
    extract_boxes_and_labels,
    format_label_from_target,
    image_to_hwc,
    merge_metadata,
    normalize_image_to_uint8,
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


class PlotlyBackend(BasePlottingBackend):
    """Plotly implementation of plotting backend with interactive visualizations."""

    def _plot_balance(
        self,
        output: PlottableBalance,
        figsize: tuple[int, int] | None = None,
        row_labels: Sequence[Any] | NDArray[Any] | None = None,
        col_labels: Sequence[Any] | NDArray[Any] | None = None,
        plot_classwise: bool = False,
    ) -> Any:  # go.Figure
        """
        Plot a heatmap of balance information.

        Parameters
        ----------
        output : PlottableBalance
            The balance output object to plot
        figsize : tuple[int, int] or None, default None
            Figure size in pixels (width, height). If None, defaults to 600x600.
        row_labels : ArrayLike or None, default None
            List/Array containing the labels for rows in the histogram
        col_labels : ArrayLike or None, default None
            List/Array containing the labels for columns in the histogram
        plot_classwise : bool, default False
            Whether to plot per-class balance instead of global balance

        Returns
        -------
        plotly.graph_objects.Figure
        """
        import numpy as np
        import plotly.graph_objects as go

        # Use shared helper to prepare data
        data, row_labels, col_labels, xlabel, ylabel, title = prepare_balance_data(
            output, row_labels, col_labels, plot_classwise
        )

        # Create heatmap with annotations
        # For triangular heatmaps (non-classwise), we mask NaN values to show only upper triangle
        text = [[f"{val:.2f}" if not np.isnan(val) else "" for val in row] for row in data]

        # Create custom hover text that handles NaN values properly
        # Ensure we only iterate up to the length of the labels to avoid index errors
        hovertext = []
        customdata = []
        for i, row in enumerate(data):
            if i >= len(row_labels):
                break
            hovertext_row = []
            customdata_row = []
            for j, val in enumerate(row):
                if j >= len(col_labels):
                    break
                if not np.isnan(val):
                    hovertext_row.append(f"Row: {row_labels[i]}<br>Col: {col_labels[j]}<br>Value: {val:.2f}")
                    customdata_row.append(val)
                else:
                    hovertext_row.append("")
                    customdata_row.append(np.nan)
            hovertext.append(hovertext_row)
            customdata.append(customdata_row)

        # Replace NaN with None for better Plotly handling (shows as gaps)
        z_data = [[None if np.isnan(val) else val for val in row] for row in data]

        fig = go.Figure(
            data=go.Heatmap(
                z=z_data,
                x=[str(label) for label in col_labels],
                y=[str(label) for label in row_labels],
                colorscale="Viridis",
                zmin=0,
                zmax=1,
                text=text,
                texttemplate="%{text}",
                textfont={"size": 10},
                colorbar={"title": {"text": "Normalized Mutual Information", "side": "right"}},
                hovertext=hovertext,
                hoverinfo="text",
            )
        )

        # Set figure size
        if figsize is not None:
            width_inches, height_inches = figsize
            width = int(width_inches * 100)
            height = int(height_inches * 100)
        else:
            width = 600
            height = 600

        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            width=width,
            height=height,
            xaxis={"tickangle": -45},
            yaxis={"autorange": "reversed"},  # Reverse y-axis to show first row at top
        )

        return fig

    def _plot_diversity(
        self,
        output: PlottableDiversity,
        figsize: tuple[int, int] | None = None,
        row_labels: Sequence[Any] | NDArray[Any] | None = None,
        col_labels: Sequence[Any] | NDArray[Any] | None = None,
        plot_classwise: bool = False,
    ) -> Any:  # go.Figure
        """
        Plot a heatmap or bar chart of diversity information.

        Parameters
        ----------
        output : PlottableDiversity
            The diversity output object to plot
        figsize : tuple[int, int] or None, default None
            Figure size in pixels (width, height). If None, defaults to 600x600 for heatmap or 700x500 for bar chart.
        row_labels : ArrayLike or None, default None
            List/Array containing the labels for rows in the histogram
        col_labels : ArrayLike or None, default None
            List/Array containing the labels for columns in the histogram
        plot_classwise : bool, default False
            Whether to plot per-class balance instead of global balance

        Returns
        -------
        plotly.graph_objects.Figure
        """
        import plotly.graph_objects as go

        # Use shared helper to prepare data
        data, row_labels, col_labels, xlabel, ylabel, title, method_name = prepare_diversity_data(
            output, row_labels, col_labels, plot_classwise
        )

        if plot_classwise:
            # Create heatmap with annotations
            text = [[f"{val:.2f}" for val in row] for row in data]

            fig = go.Figure(
                data=go.Heatmap(
                    z=data,
                    x=[str(label) for label in col_labels],
                    y=[str(label) for label in row_labels],
                    colorscale="Viridis",
                    zmin=0,
                    zmax=1,
                    text=text,
                    texttemplate="%{text}",
                    textfont={"size": 10},
                    colorbar={
                        "title": {
                            "text": f"Normalized {method_name} Index",
                            "side": "right",
                        }
                    },
                    hovertemplate="Row: %{y}<br>Col: %{x}<br>Value: %{z:.2f}<extra></extra>",
                )
            )

            # Set figure size for heatmap
            if figsize is not None:
                width_inches, height_inches = figsize
                width = int(width_inches * 100)
                height = int(height_inches * 100)
            else:
                width = 600
                height = 600

            fig.update_layout(
                title=title,
                xaxis_title=xlabel,
                yaxis_title=ylabel,
                width=width,
                height=height,
                xaxis={"tickangle": -45},
            )
        else:
            # Bar chart for diversity indices
            fig = go.Figure(
                data=go.Bar(
                    x=row_labels,
                    y=output.diversity_index,
                    marker={
                        "color": output.diversity_index,
                        "colorscale": "Viridis",
                        "showscale": True,
                    },
                    text=[f"{val:.3f}" for val in output.diversity_index],
                    textposition="outside",
                    hovertemplate="Factor: %{x}<br>Diversity: %{y:.3f}<extra></extra>",
                )
            )

            # Set figure size for bar chart
            if figsize is not None:
                width_inches, height_inches = figsize
                width = int(width_inches * 100)
                height = int(height_inches * 100)
            else:
                width = 700
                height = 500

            fig.update_layout(
                title=title,
                xaxis_title=xlabel,
                yaxis_title=ylabel,
                width=width,
                height=height,
                xaxis={"tickangle": -45},
            )

        return fig

    def _plot_sufficiency(
        self,
        output: PlottableSufficiency,
        figsize: tuple[int, int] | None = None,
        class_names: Sequence[str] | None = None,
        show_error_bars: bool = True,
        show_asymptote: bool = True,
        reference_outputs: Sequence[PlottableSufficiency] | PlottableSufficiency | None = None,
    ) -> list[Any]:  # list[go.Figure]
        """
        Plotting function for data sufficiency tasks.

        Parameters
        ----------
        output : PlottableSufficiency
            The sufficiency output object to plot
        figsize : tuple[int, int] or None, default None
            Figure size in pixels (width, height). If None, defaults to 700x500.
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
        list[plotly.graph_objects.Figure]
            List of Plotly figures for each measure
        """
        import numpy as np
        import plotly.graph_objects as go

        # Extrapolation parameters
        projection = calculate_projection(output.steps)

        # Wrap reference (for potential future use)
        _ = reference_outputs  # Currently unused but kept for API compatibility

        figures = []

        for name, measures in output.averaged_measures.items():
            if measures.ndim > 1:
                # Multi-class plotting
                validate_class_names(measures, class_names)

                for i, values in enumerate(measures):
                    class_name = str(i) if class_names is None else class_names[i]

                    fig = go.Figure()

                    # Projection curve
                    proj_values = project_steps(output.params[name][i], projection)
                    fig.add_trace(
                        go.Scatter(
                            x=projection,
                            y=proj_values,
                            mode="lines",
                            name="Potential Model Results",
                            line={"width": 2},
                            hovertemplate="Step: %{x}<br>Value: %{y:.4f}<extra></extra>",
                        )
                    )

                    # Actual measurements
                    error_y = None
                    if show_error_bars and name in output.measures:
                        error = np.std(output.measures[name][:, :, i], axis=0)
                        error_y = {"type": "data", "array": error, "visible": True}

                    fig.add_trace(
                        go.Scatter(
                            x=output.steps,
                            y=values,
                            mode="markers",
                            name="Model Results",
                            marker={"size": 10},
                            error_y=error_y,
                            hovertemplate="Step: %{x}<br>Value: %{y:.4f}<extra></extra>",
                        )
                    )

                    # Add asymptote if requested
                    if show_asymptote:
                        bound = 1 - output.params[name][i][2]
                        fig.add_trace(
                            go.Scatter(
                                x=[projection[0], projection[-1]],
                                y=[bound, bound],
                                mode="lines",
                                name=f"Asymptote: {bound:.4g}",
                                line={"dash": "dash", "width": 2},
                                hovertemplate="Asymptote: %{y:.4f}<extra></extra>",
                            )
                        )

                    # Set figure size
                    if figsize is not None:
                        width_inches, height_inches = figsize
                        width = int(width_inches * 100)
                        height = int(height_inches * 100)
                    else:
                        width = 700
                        height = 500

                    fig.update_layout(
                        title=f"{name} Sufficiency - Class {class_name}",
                        xaxis_title="Steps",
                        yaxis_title=name,
                        xaxis_type="log",
                        width=width,
                        height=height,
                        hovermode="closest",
                    )

                    figures.append(fig)
            else:
                # Single-class plotting
                fig = go.Figure()

                # Projection curve
                proj_values = project_steps(output.params[name], projection)
                fig.add_trace(
                    go.Scatter(
                        x=projection,
                        y=proj_values,
                        mode="lines",
                        name="Potential Model Results",
                        line={"width": 2},
                        hovertemplate="Step: %{x}<br>Value: %{y:.4f}<extra></extra>",
                    )
                )

                # Actual measurements
                error_y = None
                if show_error_bars and name in output.measures:
                    error = np.std(output.measures[name], axis=0)
                    error_y = {"type": "data", "array": error, "visible": True}

                fig.add_trace(
                    go.Scatter(
                        x=output.steps,
                        y=measures,
                        mode="markers",
                        name="Model Results",
                        marker={"size": 10},
                        error_y=error_y,
                        hovertemplate="Step: %{x}<br>Value: %{y:.4f}<extra></extra>",
                    )
                )

                # Add asymptote if requested
                if show_asymptote:
                    bound = 1 - output.params[name][2]
                    fig.add_trace(
                        go.Scatter(
                            x=[projection[0], projection[-1]],
                            y=[bound, bound],
                            mode="lines",
                            name=f"Asymptote: {bound:.4g}",
                            line={"dash": "dash", "width": 2},
                            hovertemplate="Asymptote: %{y:.4f}<extra></extra>",
                        )
                    )

                # Set figure size
                if figsize is not None:
                    width_inches, height_inches = figsize
                    width = int(width_inches * 100)
                    height = int(height_inches * 100)
                else:
                    width = 700
                    height = 500

                fig.update_layout(
                    title=f"{name} Sufficiency",
                    xaxis_title="Steps",
                    yaxis_title=name,
                    xaxis_type="log",
                    width=width,
                    height=height,
                    hovermode="closest",
                )

                figures.append(fig)

        return figures

    def _plot_stats(
        self,
        output: PlottableStats,
        figsize: tuple[int, int] | None = None,
        log: bool = True,
        channel_limit: int | None = None,
        channel_index: int | Iterable[int] | None = None,
    ) -> Any:  # go.Figure
        """
        Plots the statistics as a set of histograms.

        Parameters
        ----------
        output : PlottableBaseStats
            The stats output object to plot
        figsize : tuple[int, int] or None, default None
            Figure size in pixels (width, height). If None, defaults to 300 * cols x 300 * rows.
        log : bool, default True
            If True, plots the histograms on a logarithmic scale.
        channel_limit : int or None, default None
            The maximum number of channels to plot. If None, all channels are plotted.
        channel_index : int, Iterable[int] or None, default None
            The index or indices of the channels to plot. If None, all channels are plotted.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        max_channels, ch_mask = output._get_channels(channel_limit, channel_index)
        factors = output.factors(exclude_constant=True)

        if not factors:
            return go.Figure()

        if max_channels == 1:
            # Single channel histogram
            num_metrics = len(factors)
            rows, cols = calculate_subplot_grid(num_metrics)

            fig = make_subplots(
                rows=rows,
                cols=cols,
                subplot_titles=list(factors.keys()),
            )

            for idx, (metric_name, metric_values) in enumerate(factors.items()):
                row = idx // 3 + 1
                col = idx % 3 + 1

                fig.add_trace(
                    go.Histogram(
                        x=metric_values.flatten(),
                        nbinsx=20,
                        name=metric_name,
                        showlegend=False,
                        hovertemplate="Value: %{x}<br>Count: %{y}<extra></extra>",
                    ),
                    row=row,
                    col=col,
                )

                fig.update_xaxes(title_text="Values", row=row, col=col)
                fig.update_yaxes(
                    title_text="Counts",
                    type="log" if log else "linear",
                    row=row,
                    col=col,
                )

            # Set figure size for single channel
            if figsize is not None:
                width_inches, height_inches = figsize
                width = int(width_inches * 100)
                height = int(height_inches * 100)
            else:
                width = 300 * cols
                height = 300 * rows

            fig.update_layout(height=height, width=width, title="Base Statistics Histograms")

        else:
            # Multi-channel histogram - use shared constant
            data_keys = [key for key in factors if key in CHANNELWISE_METRICS]

            num_metrics = len(data_keys)
            rows, cols = calculate_subplot_grid(num_metrics)

            fig = make_subplots(
                rows=rows,
                cols=cols,
                subplot_titles=data_keys,
            )

            for idx, metric_name in enumerate(data_keys):
                row = idx // 3 + 1
                col = idx % 3 + 1

                # Reshape for channel-wise data
                data = factors[metric_name][ch_mask].reshape(-1, max_channels)

                for ch_idx in range(max_channels):
                    fig.add_trace(
                        go.Histogram(
                            x=data[:, ch_idx],
                            nbinsx=20,
                            name=f"Channel {ch_idx}",
                            opacity=0.7,
                            showlegend=(idx == 0),
                            legendgroup=f"ch{ch_idx}",
                            hovertemplate=f"Channel {ch_idx}<br>Value: %{{x}}<br>Count: %{{y}}<extra></extra>",
                        ),
                        row=row,
                        col=col,
                    )

                fig.update_xaxes(title_text="Values", row=row, col=col)
                fig.update_yaxes(
                    title_text="Counts",
                    type="log" if log else "linear",
                    row=row,
                    col=col,
                )

            # Set figure size for multi-channel
            if figsize is not None:
                width_inches, height_inches = figsize
                width = int(width_inches * 100)
                height = int(height_inches * 100)
            else:
                width = 300 * cols
                height = 300 * rows

            fig.update_layout(
                height=height,
                width=width,
                title="Base Statistics Histograms (Multi-Channel)",
                barmode="overlay",
            )

        return fig

    def _plot_drift_mvdc(
        self,
        output: PlottableDriftMVDC,
        figsize: tuple[int, int] | None = None,
    ) -> Any:  # go.Figure
        """
        Render the roc_auc metric over the train/test data in relation to the threshold.

        Parameters
        ----------
        output : PlottableDriftMVDC
            The drift MVDC output object to plot
        figsize : tuple[int, int] or None, default None
            Figure size in pixels (width, height). If None, defaults to 900x500.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        import plotly.graph_objects as go

        # Use shared helper to prepare drift data
        resdf, trndf, tstdf, driftx, is_sufficient = prepare_drift_data(output)

        if not is_sufficient:
            fig = go.Figure()
            fig.add_annotation(
                text="Insufficient data for drift detection plot",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            return fig

        fig = go.Figure()

        # Threshold lines
        fig.add_trace(
            go.Scatter(
                x=resdf.index,
                y=resdf["domain_classifier_auroc"]["upper_threshold"],
                mode="lines",
                name="Threshold Upper",
                line={"dash": "dash", "color": "red", "width": 2},
                hovertemplate="Index: %{x}<br>Threshold: %{y:.4f}<extra></extra>",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=resdf.index,
                y=resdf["domain_classifier_auroc"]["lower_threshold"],
                mode="lines",
                name="Threshold Lower",
                line={"dash": "dash", "color": "red", "width": 2},
                hovertemplate="Index: %{x}<br>Threshold: %{y:.4f}<extra></extra>",
            )
        )

        # Train data
        fig.add_trace(
            go.Scatter(
                x=trndf.index,
                y=trndf["domain_classifier_auroc"]["value"],
                mode="lines",
                name="Train",
                line={"color": "blue", "width": 2},
                hovertemplate="Index: %{x}<br>ROC AUC: %{y:.4f}<extra></extra>",
            )
        )

        # Test data
        fig.add_trace(
            go.Scatter(
                x=tstdf.index,
                y=tstdf["domain_classifier_auroc"]["value"],
                mode="lines",
                name="Test",
                line={"color": "green", "width": 2},
                hovertemplate="Index: %{x}<br>ROC AUC: %{y:.4f}<extra></extra>",
            )
        )

        # Drift markers
        if len(driftx) > 0:
            fig.add_trace(
                go.Scatter(
                    x=resdf.index.values[driftx],
                    y=resdf["domain_classifier_auroc"]["value"].values[driftx],
                    mode="markers",
                    name="Drift",
                    marker={"symbol": "diamond", "size": 10, "color": "magenta"},
                    hovertemplate="Drift at Index: %{x}<br>ROC AUC: %{y:.4f}<extra></extra>",
                )
            )

        # Set figure size
        if figsize is not None:
            width_inches, height_inches = figsize
            width = int(width_inches * 100)
            height = int(height_inches * 100)
        else:
            width = 900
            height = 500

        fig.update_layout(
            title="Domain Classifier, Drift Detection",
            xaxis_title="Chunk Index",
            yaxis_title="ROC AUC",
            yaxis={"range": [0, 1.1]},
            width=width,
            height=height,
            hovermode="closest",
        )

        return fig

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
        Plot a grid of images from a dataset using Plotly.

        Parameters
        ----------
        dataset : Dataset
            MAITE-compatible dataset containing images
        indices : Sequence[int]
            Indices of images to plot from the dataset
        images_per_row : int, default 3
            Number of images to display per row
        figsize : tuple[int, int] or None, default None
            Figure size in pixels (width, height). If None, defaults to 1000x1000.
        show_labels : bool, default False
            Whether to display labels extracted from targets
        show_metadata : bool, default False
            Whether to display metadata from the dataset items
        additional_metadata : Sequence[dict[str, Any]] or None, default None
            Additional metadata to display for each image (must match length of indices)

        Returns
        -------
        plotly.graph_objects.Figure

        Raises
        ------
        ValueError
            If additional_metadata length doesn't match indices length
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Validate additional_metadata length
        if additional_metadata is not None and len(additional_metadata) != len(indices):
            raise ValueError(
                f"additional_metadata length ({len(additional_metadata)}) must match indices length ({len(indices)})"
            )

        num_images = len(indices)
        num_rows = (num_images + images_per_row - 1) // images_per_row

        # Get index2label mapping if available
        index2label = dataset.metadata.get("index2label") if hasattr(dataset, "metadata") else None

        # Build subplot titles
        subplot_titles = []
        for i, idx in enumerate(indices):
            # Get dataset item and parse it
            datum = dataset[idx]
            image, target, metadata = parse_dataset_item(datum)

            # Merge with additional metadata if provided
            if additional_metadata is not None:
                metadata = merge_metadata(metadata, additional_metadata[i])

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
            if title_parts:
                subplot_titles.append("<br>".join(title_parts))
            else:
                subplot_titles.append("")

        # Create subplots with proper spacing
        fig = make_subplots(
            rows=num_rows,
            cols=images_per_row,
            subplot_titles=subplot_titles,
            horizontal_spacing=0.02,
            vertical_spacing=0.1 if subplot_titles and any(subplot_titles) else 0.02,
        )

        for i, idx in enumerate(indices):
            row = i // images_per_row + 1
            col = i % images_per_row + 1

            # Get image from dataset and parse it
            datum = dataset[idx]
            image, target, _ = parse_dataset_item(datum)

            # Convert to HWC format
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

            img_height, img_width = image_uint8.shape[:2]

            # Add image as a trace - using Image trace which auto-fills the subplot
            fig.add_trace(
                go.Image(z=image_uint8),
                row=row,
                col=col,
            )

            # Set the axis ranges to match the image dimensions for proper aspect ratio
            xaxis_name = "xaxis" if i == 0 else f"xaxis{i + 1}"
            yaxis_name = "yaxis" if i == 0 else f"yaxis{i + 1}"
            # For scaleanchor, use the axis reference format (e.g., "y", "y2", "y3")
            yaxis_ref = "y" if i == 0 else f"y{i + 1}"

            fig.update_layout(
                {
                    xaxis_name: {
                        "range": [0, img_width],
                        "showticklabels": False,
                        "showgrid": False,
                        "zeroline": False,
                        "scaleanchor": yaxis_ref,
                        "scaleratio": 1,
                    },
                    yaxis_name: {
                        "range": [img_height, 0],  # Inverted for correct image orientation
                        "showticklabels": False,
                        "showgrid": False,
                        "zeroline": False,
                    },
                }
            )

        # Set figure size
        if figsize is not None:
            width_inches, height_inches = figsize
            width = int(width_inches * 100)
            height = int(height_inches * 100)
        else:
            width = 1000
            height = 1000

        fig.update_layout(
            width=width,
            height=height,
            showlegend=False,
            margin={"l": 0, "r": 0, "t": 50, "b": 0},
            title="Image Grid",
        )

        return fig
