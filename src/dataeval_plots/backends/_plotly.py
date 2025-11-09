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
    image_to_base64_png,
    image_to_hwc,
    normalize_image_to_uint8,
    prepare_balance_data,
    prepare_coverage_images,
    prepare_diversity_data,
    prepare_drift_data,
    project_steps,
    validate_class_names,
)
from dataeval_plots.protocols import (
    Indexable,
    PlottableBalance,
    PlottableBaseStats,
    PlottableCoverage,
    PlottableDiversity,
    PlottableDriftMVDC,
    PlottableSufficiency,
)


class PlotlyBackend(BasePlottingBackend):
    """Plotly implementation of plotting backend with interactive visualizations."""

    def _plot_coverage(
        self,
        output: PlottableCoverage,
        images: Indexable | None = None,  # Images | Dataset
        top_k: int = 6,
    ) -> Any:  # go.Figure
        """
        Plot the top k images together for visualization.

        Parameters
        ----------
        output : PlottableCoverage
            The coverage output object to plot
        images : Images or Dataset
            Original images (not embeddings) in (N, C, H, W) or (N, H, W) format
        top_k : int, default 6
            Number of images to plot (plotting assumes groups of 3)

        Returns
        -------
        plotly.graph_objects.Figure
        """
        from plotly.subplots import make_subplots

        # Use shared helper to prepare and validate images
        selected_images, num_images, rows, cols = prepare_coverage_images(output, images, top_k)

        # Create subplots
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[f"Image {i}" for i in range(num_images)],
        )

        for idx, img in enumerate(selected_images):
            img_np = image_to_hwc(img)

            # Normalize and convert to base64 using shared helpers
            img_np = normalize_image_to_uint8(img_np)
            img_str = image_to_base64_png(img_np)

            row = idx // 3 + 1
            col = idx % 3 + 1

            # Add image to subplot
            fig.add_layout_image(
                {
                    "source": img_str,
                    "xref": f"x{idx + 1}" if idx > 0 else "x",
                    "yref": f"y{idx + 1}" if idx > 0 else "y",
                    "x": 0,
                    "y": 1,
                    "sizex": 1,
                    "sizey": 1,
                    "sizing": "stretch",
                    "layer": "below",
                },
                row=row,
                col=col,
            )

            # Hide axes for this subplot
            fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=row, col=col)
            fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=row, col=col)

        fig.update_layout(
            title=f"Top {num_images} Uncovered Images",
            showlegend=False,
            height=300 * rows,
            width=300 * cols,
        )

        return fig

    def _plot_balance(
        self,
        output: PlottableBalance,
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

        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            width=600,
            height=600,
            xaxis={"tickangle": -45},
            yaxis={"autorange": "reversed"},  # Reverse y-axis to show first row at top
        )

        return fig

    def _plot_diversity(
        self,
        output: PlottableDiversity,
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
                    colorbar={"title": f"Normalized {method_name} Index"},
                    hovertemplate="Row: %{y}<br>Col: %{x}<br>Value: %{z:.2f}<extra></extra>",
                )
            )

            fig.update_layout(
                title=title,
                xaxis_title=xlabel,
                yaxis_title=ylabel,
                width=600,
                height=600,
                xaxis={"tickangle": -45},
            )
        else:
            # Bar chart for diversity indices
            fig = go.Figure(
                data=go.Bar(
                    x=row_labels,
                    y=output.diversity_index,
                    marker={"color": output.diversity_index, "colorscale": "Viridis", "showscale": True},
                    text=[f"{val:.3f}" for val in output.diversity_index],
                    textposition="outside",
                    hovertemplate="Factor: %{x}<br>Diversity: %{y:.3f}<extra></extra>",
                )
            )

            fig.update_layout(
                title=title,
                xaxis_title=xlabel,
                yaxis_title=ylabel,
                width=700,
                height=500,
                xaxis={"tickangle": -45},
            )

        return fig

    def _plot_sufficiency(
        self,
        output: PlottableSufficiency,
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

                    fig.update_layout(
                        title=f"{name} Sufficiency - Class {class_name}",
                        xaxis_title="Steps",
                        yaxis_title=name,
                        xaxis_type="log",
                        width=700,
                        height=500,
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

                fig.update_layout(
                    title=f"{name} Sufficiency",
                    xaxis_title="Steps",
                    yaxis_title=name,
                    xaxis_type="log",
                    width=700,
                    height=500,
                    hovermode="closest",
                )

                figures.append(fig)

        return figures

    def _plot_base_stats(
        self,
        output: PlottableBaseStats,
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
                fig.update_yaxes(title_text="Counts", type="log" if log else "linear", row=row, col=col)

            fig.update_layout(height=300 * rows, width=300 * cols, title="Base Statistics Histograms")

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
                fig.update_yaxes(title_text="Counts", type="log" if log else "linear", row=row, col=col)

            fig.update_layout(
                height=300 * rows,
                width=300 * cols,
                title="Base Statistics Histograms (Multi-Channel)",
                barmode="overlay",
            )

        return fig

    def _plot_drift_mvdc(
        self,
        output: PlottableDriftMVDC,
    ) -> Any:  # go.Figure
        """
        Render the roc_auc metric over the train/test data in relation to the threshold.

        Parameters
        ----------
        output : PlottableDriftMVDC
            The drift MVDC output object to plot

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

        fig.update_layout(
            title="Domain Classifier, Drift Detection",
            xaxis_title="Chunk Index",
            yaxis_title="ROC AUC",
            yaxis={"range": [0, 1.1]},
            width=900,
            height=500,
            hovermode="closest",
        )

        return fig
