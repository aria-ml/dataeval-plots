"""Plotly plotting backend."""

from __future__ import annotations

import textwrap
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

import numpy as np
import polars as pl
from numpy.typing import NDArray

from dataeval_plots.backends import _feasibility_layout as layout
from dataeval_plots.backends._base import BasePlottingBackend
from dataeval_plots.backends._shared import (
    CHANNELWISE_METRICS,
    FEASIBILITY_COLORS,
    FEASIBILITY_CONFLICT_TEXT,
    FeasibilityBand,
    calculate_projection,
    calculate_subplot_grid,
    feasibility_zones,
    format_label_from_target,
    prepare_balance_data,
    prepare_diversity_data,
    prepare_drift_data,
    prepare_feasibility_data,
    process_dataset_item_for_display,
    project_steps,
    validate_class_names,
)
from dataeval_plots.protocols import (
    Dataset,
    PlottableBalance,
    PlottableDiversity,
    PlottableDriftMVDC,
    PlottableFeasibility,
    PlottableStats,
    PlottableSufficiency,
)

# Feasibility: the shared layout's inches, drawn at layout.PX pixels per inch


def _annotate(fig: Any, n: str, x: float, y: float, text: str, size: float = layout.LABEL_PT, **kwargs: Any) -> None:
    """Text on band ``n``'s axes (data coordinates unless ``xref`` or ``yref`` say otherwise)."""
    kwargs = {"xref": f"x{n}", "yref": f"y{n}", "showarrow": False, "font": {}, **kwargs}
    kwargs["font"] = {"size": layout.px(size), "color": FEASIBILITY_COLORS["ink"], **kwargs["font"]}
    fig.add_annotation(x=x, y=y, text=text.replace("\n", "<br>"), **kwargs)


def _side_note(
    fig: Any,
    n: str,
    x: float,
    y: float,
    text: str,
    prefer_left: bool,
    x_range: tuple[float, float],
    ax_w: float,
    **kwargs: Any,
) -> None:
    """Label beside its mark, top at ``y``, on the preferred side unless that leaves the axes."""
    left, pad = layout.goes_left(x, text, prefer_left, x_range, ax_w), layout.PAD * layout.PX
    _annotate(
        fig, n, x, y, text, xanchor="right" if left else "left", yanchor="top", xshift=-pad if left else pad, **kwargs
    )


def _rule(fig: Any, n: str, x: tuple[float, float], y: tuple[float, float], opacity: float = 1.0, **line: Any) -> None:
    """A line on band ``n``'s axes, from (x[0], y[0]) to (x[1], y[1])."""
    fig.add_shape(
        type="line", x0=x[0], x1=x[1], y0=y[0], y1=y[1], xref=f"x{n}", yref=f"y{n}", opacity=opacity, line=line
    )


def _markers(fig: Any, n: str, x: list[float], y: list[float], hover: str | None = None, **kwargs: Any) -> None:
    """Markers on band ``n``'s axes, unclipped so a mark at the axes' edge stays whole."""
    skip = {"hoverinfo": "skip"} if hover is None else {"hovertemplate": f"{hover}<extra></extra>"}
    fig.add_scatter(x=x, y=y, mode="markers", xaxis=f"x{n}", yaxis=f"y{n}", cliponaxis=False, **skip, **kwargs)


def _plotly_requirements(fig: Any, n: str, placed: list[layout.Placed], y0: float, y_label: float) -> None:
    """Dashed requirement lines from ``y0`` up to their labels' level, labels on top."""
    ink, ys = FEASIBILITY_COLORS["ink"], []
    for x, text, ha, level in placed:
        y = y_label + level * layout.LEVEL
        ys.append(y)
        _rule(fig, n, (x, x), (y0, y), color=ink, width=layout.px(1.2), dash="4px,3px")
        shift = layout.PAD * layout.PX if ha == "right" else -layout.PAD * layout.PX
        _annotate(fig, n, x, y + 0.06, text, xanchor=ha, yanchor="bottom", xshift=shift)
    _markers(fig, n, [p[0] for p in placed], ys, marker={"symbol": "triangle-down", "size": 9, "color": ink})


def _plotly_band(fig: Any, n: str, band: FeasibilityBand, ax_w: float, placed: list[layout.Placed]) -> None:
    """Zones, boundaries and requirement markers of one band."""
    colors = FEASIBILITY_COLORS
    lo, hi = band.x_range
    per_in = (hi - lo) / ax_w
    y0, y1 = layout.BOUNDARY, layout.BOUNDARY + layout.BAR
    for kind, start, end, label in feasibility_zones(band):
        fig.add_shape(
            type="rect",
            x0=start,
            x1=end,
            y0=y0,
            y1=y1,
            xref=f"x{n}",
            yref=f"y{n}",
            fillcolor=colors[f"{kind}_fill"],
            line={"color": "white", "width": layout.px(1.5)},
            layer="below",
        )
        fit = layout.zone_label(band, start, end, label, per_in)
        if fit:
            _annotate(fig, n, fit[0], y0 + layout.BAR / 2, fit[1])
    if band.conflict:
        line = {"color": colors["warning"], "width": layout.px(2), "dash": "dash"}
        fig.add_shape(type="rect", x0=lo, x1=hi, y0=y0, y1=y1, xref=f"x{n}", yref=f"y{n}", line=line)
        conflict = f"<b>{FEASIBILITY_CONFLICT_TEXT}</b>"
        _annotate(fig, n, (lo + hi) / 2, y0 + layout.BAR / 2, conflict, size=11, bgcolor="white", borderpad=1)
    for x, text, kind, left in band.boundary_labels:
        color = colors["demonstrated" if kind == "lower" else "ruled_out"]
        _rule(fig, n, (x, x), (y0 - 0.06, y1 + 0.06), color=color, width=layout.px(2.5))
        _side_note(fig, n, x, y0 - 0.1, text, left, band.x_range, ax_w)
    _plotly_requirements(fig, n, placed, y0, y1 + 0.2)


def _plotly_evidence(
    fig: Any,
    n: str,
    band: FeasibilityBand,
    x_range: tuple[float, float],
    ax_w: float,
    placed: list[layout.Placed],
) -> None:
    """One row per source: a tick at its bound, a dot at its observed score, both labeled."""
    colors, muted = FEASIBILITY_COLORS, FEASIBILITY_COLORS["ink_muted"]
    pad = layout.PAD * layout.PX
    rows_top, ys = layout.evidence_rows(band)
    _plotly_requirements(fig, n, placed, 0, rows_top + 0.12)
    aside = {"xref": f"x{n} domain", "xanchor": "left", "xshift": pad, "font": {"color": muted}}  # no marks
    aside |= {"bgcolor": "white", "borderpad": 1}
    if band.evidence_note:
        _annotate(fig, n, 0, rows_top - layout.ROW / 2, band.evidence_note, **aside)
    for y, row in zip(ys, band.evidence):
        # Bound labeled below, observed above, so the two never collide; a white backing keeps them legible over
        # the requirement lines
        ink = {"font": {"color": muted if row.faint else colors["ink"]}, "bgcolor": "white", "borderpad": 1}
        source = textwrap.fill(row.label, layout.SOURCE_WRAP)
        _annotate(
            fig, n, 0, y, source, xref=f"x{n} domain", xanchor="right", align="right", xshift=-pad, font=ink["font"]
        )
        if row.bound is None:
            _annotate(fig, n, 0, y, row.bound_label, **aside)
            continue
        color = colors["demonstrated" if row.bound_type == "lower" else "ruled_out"]
        opacity = 0.35 if row.faint else 1.0
        # A tick marker, not a line shape, which the axes would clip at their edge
        tick = {"symbol": "line-ns", "size": 0.24 * layout.PX, "line": {"color": color, "width": layout.px(2.5)}}
        _markers(fig, n, [row.bound], [y], f"{row.label}: {row.bound_label}", marker=tick, opacity=opacity)
        _side_note(fig, n, row.bound, y - 0.14, row.bound_label, row.bound_left, x_range, ax_w, **ink)
        if row.observed is not None and row.observed_label is not None:
            dot = {"size": 11, "color": color}
            _rule(fig, n, (row.bound, row.observed), (y, y), opacity, color=color, width=layout.px(3))
            hover = f"{row.label}: {row.observed_label}"
            _markers(fig, n, [row.observed], [y], hover, marker=dot, opacity=opacity)
            _side_note(fig, n, row.observed, y + 0.29, row.observed_label, not row.bound_left, x_range, ax_w, **ink)


class PlotlyBackend(BasePlottingBackend):
    """Plotly implementation of plotting backend with interactive visualizations."""

    def _plot_balance(
        self,
        output: PlottableBalance,
        figsize: tuple[float, float] | None = None,
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
        figsize : tuple[float, float] or None, default None
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
                colorbar={
                    "title": {"text": "Normalized Mutual Information", "side": "right"},
                    "outlinewidth": 0,
                },
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
        figsize: tuple[float, float] | None = None,
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
        figsize : tuple[float, float] or None, default None
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
                        "title": {"text": f"Normalized {method_name} Index", "side": "right"},
                        "outlinewidth": 0,
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
            # DataFrame-based: get diversity values from factors DataFrame
            diversity_values = output.factors["diversity_value"].to_list()

            fig = go.Figure(
                data=go.Bar(
                    x=row_labels,
                    y=diversity_values,
                    marker={
                        "color": diversity_values,
                        "colorscale": "Viridis",
                        "showscale": True,
                    },
                    text=[f"{val:.3f}" for val in diversity_values],
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
        figsize: tuple[float, float] | None = None,
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
        figsize : tuple[float, float] or None, default None
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
        figsize: tuple[float, float] | None = None,
        log: bool = True,
        channel_limit: int | None = None,
        channel_index: int | Iterable[int] | None = None,
    ) -> Any:  # go.Figure
        """
        Plots the statistics as a set of histograms.

        Parameters
        ----------
        output : PlottableStats
            The stats output object to plot
        figsize : tuple[float, float] or None, default None
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
        figsize: tuple[float, float] | None = None,
    ) -> Any:  # go.Figure
        """
        Render the roc_auc metric over the train/test data in relation to the threshold.

        Parameters
        ----------
        output : PlottableDriftMVDC
            The drift MVDC output object to plot
        figsize : tuple[float, float] or None, default None
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

        # Create index arrays for plotting
        indices = np.arange(len(resdf))
        trn_indices = resdf.with_row_index().filter(pl.col("chunk_period") == "reference")["index"].to_numpy()
        tst_indices = resdf.with_row_index().filter(pl.col("chunk_period") == "analysis")["index"].to_numpy()

        fig = go.Figure()

        # Threshold lines
        fig.add_trace(
            go.Scatter(
                x=indices,
                y=resdf["domain_classifier_auroc_upper_threshold"].to_numpy(),
                mode="lines",
                name="Threshold Upper",
                line={"dash": "dash", "color": "red", "width": 2},
                hovertemplate="Index: %{x}<br>Threshold: %{y:.4f}<extra></extra>",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=indices,
                y=resdf["domain_classifier_auroc_lower_threshold"].to_numpy(),
                mode="lines",
                name="Threshold Lower",
                line={"dash": "dash", "color": "red", "width": 2},
                hovertemplate="Index: %{x}<br>Threshold: %{y:.4f}<extra></extra>",
            )
        )

        # Train data
        fig.add_trace(
            go.Scatter(
                x=trn_indices,
                y=trndf["domain_classifier_auroc_value"].to_numpy(),
                mode="lines",
                name="Train",
                line={"color": "blue", "width": 2},
                hovertemplate="Index: %{x}<br>ROC AUC: %{y:.4f}<extra></extra>",
            )
        )

        # Test data
        fig.add_trace(
            go.Scatter(
                x=tst_indices,
                y=tstdf["domain_classifier_auroc_value"].to_numpy(),
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
                    x=driftx,
                    y=resdf["domain_classifier_auroc_value"].to_numpy()[driftx],
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
        indices: Sequence[int] | None = None,
        images_per_row: int = 3,
        figsize: tuple[float, float] | None = None,
        show_labels: bool = False,
        show_metadata: bool | Sequence[str] = False,
        additional_metadata: Sequence[dict[str, Any]] | None = None,
    ) -> Any:
        """
        Plot a grid of images from a dataset using Plotly.

        Parameters
        ----------
        dataset : Dataset
            MAITE-compatible dataset containing images
        indices : Sequence[int] or None, default None
            Indices of images to plot from the dataset. None plots every item.
        images_per_row : int, default 3
            Number of images to display per row
        figsize : tuple[float, float] or None, default None
            Figure size in pixels (width, height). If None, defaults to 1000x1000.
        show_labels : bool, default False
            Whether to display labels extracted from targets
        show_metadata : bool or Sequence[str], default False
            Whether to display metadata from the dataset items. A sequence of keys
            displays only those, in the order given.
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

        indices = range(len(dataset)) if indices is None else indices

        # Validate additional_metadata length
        if additional_metadata is not None and len(additional_metadata) != len(indices):
            raise ValueError(
                f"additional_metadata length ({len(additional_metadata)}) must match indices length ({len(indices)})"
            )

        num_images = len(indices)
        num_rows = (num_images + images_per_row - 1) // images_per_row

        # Get index2label mapping if available
        index2label = dataset.metadata.get("index2label") if hasattr(dataset, "metadata") else None

        # Process all images first to build titles and collect processed images
        processed_images = []
        subplot_titles = []

        for i, idx in enumerate(indices):
            datum = dataset[idx]
            add_meta = additional_metadata[i] if additional_metadata is not None else None
            processed_image, target, metadata = process_dataset_item_for_display(
                datum,
                additional_metadata=add_meta,
                index2label=index2label,
            )
            processed_images.append(processed_image)

            # Build title from labels and metadata
            title_parts = []

            if show_labels and target is not None:
                label_str = format_label_from_target(target, index2label)
                if label_str:
                    title_parts.append(label_str)

            if show_metadata and metadata:
                # Every key, or just the named ones, as key: value pairs
                keys = metadata if isinstance(show_metadata, bool) else [k for k in show_metadata if k in metadata]
                title_parts.extend(f"{k}: {metadata[k]}" for k in keys)

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

        # Calculate actual dimensions from processed images
        max_width_per_col = [0] * images_per_row
        max_height_per_row = [0] * num_rows

        for i, processed_image in enumerate(processed_images):
            row_idx = i // images_per_row
            col_idx = i % images_per_row

            img_height, img_width = processed_image.shape[:2]
            max_width_per_col[col_idx] = max(max_width_per_col[col_idx], img_width)
            max_height_per_row[row_idx] = max(max_height_per_row[row_idx], img_height)

            row = row_idx + 1
            col = col_idx + 1

            # Add image as a trace - using Image trace which auto-fills the subplot
            fig.add_trace(
                go.Image(z=processed_image),
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
            # Auto-detect based on actual processed image dimensions with slim borders
            padding_factor = 0.05
            # Sum the maximum widths across columns
            width = sum(max_width_per_col)
            # Sum the maximum heights across rows, adding padding to each row
            height = sum(int(h * (1 + 2 * padding_factor)) for h in max_height_per_row) + 50  # Add 50 for title space

        fig.update_layout(width=width, height=height, showlegend=False, margin={"l": 0, "r": 0, "t": 50, "b": 0})

        return fig

    def _add_scatter_traces(
        self,
        fig: Any,
        embeddings: NDArray[Any],
        labels: NDArray[Any] | None,
        label_names: Mapping[int, str] | None,
        dimensions: Literal[2, 3],
        *,
        row: int | None = None,
        col: int | None = None,
        legend_added: set[str] | None = None,
    ) -> None:
        """Add scatter traces for one subplot to the given figure."""
        import plotly.graph_objects as go

        hover_tpl = "%{fullData.name}<extra></extra>"
        marker_size = 3 if dimensions == 3 else 5
        ScatterCls = go.Scatter3d if dimensions == 3 else go.Scatter

        rc_kwargs: dict[str, int] = {}
        if row is not None and col is not None:
            rc_kwargs = {"row": row, "col": col}

        if labels is not None:
            unique_labels = np.unique(labels)
            for label in unique_labels:
                mask = labels == label
                name = label_names[int(label)] if label_names and int(label) in label_names else str(label)
                show_legend = True
                legend_kwargs: dict[str, Any] = {}
                if legend_added is not None:
                    show_legend = name not in legend_added
                    legend_added.add(name)
                    legend_kwargs = {"legendgroup": name, "showlegend": show_legend}

                coords: dict[str, Any] = {"x": embeddings[mask, 0], "y": embeddings[mask, 1]}
                if dimensions == 3:
                    coords["z"] = embeddings[mask, 2]

                fig.add_trace(
                    ScatterCls(
                        **coords,
                        mode="markers",
                        marker={"size": marker_size, "opacity": 0.7},
                        name=name,
                        hovertemplate=hover_tpl,
                        **legend_kwargs,
                    ),
                    **rc_kwargs,
                )
        else:
            coords = {"x": embeddings[:, 0], "y": embeddings[:, 1]}
            if dimensions == 3:
                coords["z"] = embeddings[:, 2]

            fig.add_trace(
                ScatterCls(
                    **coords,
                    mode="markers",
                    marker={"size": marker_size, "opacity": 0.7},
                    name="embeddings",
                    showlegend=False,
                    hoverinfo="skip",
                ),
                **rc_kwargs,
            )

    @staticmethod
    def _clean_projection_axes(fig: Any, dimensions: Literal[2, 3], n_scenes: int = 1) -> None:
        """Hide tick labels on projection axes."""
        if dimensions == 3:
            _clean = {"showspikes": False, "showticklabels": False, "title": ""}
            for i in range(n_scenes):
                scene_key = "scene" if i == 0 else f"scene{i + 1}"
                fig.update_layout(**{scene_key: {"xaxis": _clean, "yaxis": _clean, "zaxis": _clean}})
        else:
            fig.update_xaxes(showticklabels=False, ticks="", title_text="")
            fig.update_yaxes(showticklabels=False, ticks="", title_text="")

    def project(
        self,
        embeddings: NDArray[Any],
        labels: NDArray[Any] | None = None,
        label_names: Mapping[int, str] | None = None,
        method: str = "pca",
        dimensions: Literal[2, 3] = 2,
        figsize: tuple[float, float] | None = None,
        title: str | None = None,
    ) -> Any:  # go.Figure
        """
        Plot projected embeddings as an interactive scatter plot.

        Parameters
        ----------
        embeddings : NDArray
            Reduced embeddings with shape ``(N, 2)`` or ``(N, 3)``.
        labels : NDArray or None, default None
            Class labels for coloring points, shape ``(N,)``.
        label_names : dict[int, str] or None, default None
            Mapping from integer labels to display names.
        method : str, default "pca"
            Name of the reduction method used (for title/display).
        dimensions : {2, 3}, default 2
            Number of dimensions in the embeddings.
        figsize : tuple[float, float] or None, default None
            Figure size in inches (width, height).
        title : str or None, default None
            Plot title. If None, auto-generated from method name.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        import plotly.graph_objects as go

        if title is None:
            title = f"{method.upper()} Projection"

        if figsize is not None:
            width = int(figsize[0] * 100)
            height = int(figsize[1] * 100)
        else:
            width = 800
            height = 600

        fig = go.Figure()
        self._add_scatter_traces(fig, embeddings, labels, label_names, dimensions)
        self._clean_projection_axes(fig, dimensions)

        fig.update_layout(
            title=title,
            width=width,
            height=height,
            hovermode="closest",
            legend={"itemsizing": "constant", "itemclick": "toggleothers", "itemdoubleclick": "toggle"},
        )

        return fig

    def project_grid(
        self,
        embeddings_list: Sequence[NDArray[Any]],
        methods: Sequence[str],
        labels: NDArray[Any] | None = None,
        label_names: Mapping[int, str] | None = None,
        dimensions: Literal[2, 3] = 2,
        figsize: tuple[float, float] | None = None,
        title: str | None = None,
    ) -> Any:  # go.Figure
        """
        Plot a grid of projected embeddings comparing multiple reduction methods.

        Parameters
        ----------
        embeddings_list : list[NDArray]
            List of reduced embeddings, one per method.
        methods : list[str]
            Names of the reduction methods.
        labels : NDArray or None, default None
            Class labels for coloring points, shape ``(N,)``.
        label_names : dict[int, str] or None, default None
            Mapping from integer labels to display names.
        dimensions : {2, 3}, default 2
            Number of dimensions in the embeddings.
        figsize : tuple[float, float] or None, default None
            Figure size in inches (width, height) for the entire grid.
        title : str or None, default None
            Overall title for the grid figure.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        from plotly.subplots import make_subplots

        from dataeval_plots.backends._shared import calculate_subplot_grid

        n = len(methods)
        aspect = figsize[0] / figsize[1] if figsize else None
        rows, cols = calculate_subplot_grid(n, aspect_ratio=aspect)

        if figsize is not None:
            width = int(figsize[0] * 100)
            height = int(figsize[1] * 100)
        else:
            width = cols * 400
            height = rows * 350

        subplot_titles = [m.upper() for m in methods]

        if dimensions == 3:
            specs = [[{"type": "scene"} for _ in range(cols)] for _ in range(rows)]
            fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles, specs=specs)
        else:
            fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)

        legend_added: set[str] = set()

        for idx, emb in enumerate(embeddings_list):
            row = idx // cols + 1
            col = idx % cols + 1
            self._add_scatter_traces(
                fig, emb, labels, label_names, dimensions, row=row, col=col, legend_added=legend_added
            )

        fig.update_layout(
            title=title or "Projection Comparison",
            width=width,
            height=height,
            hovermode="closest",
            legend={"itemsizing": "constant", "itemclick": "toggleothers", "itemdoubleclick": "toggle"},
        )

        self._clean_projection_axes(fig, dimensions, n_scenes=n)

        return fig

    def _plot_feasibility(
        self,
        output: PlottableFeasibility,
        figsize: tuple[float, float] | None = None,
        plot_evidence: bool = False,
    ) -> Any:  # go.Figure
        """
        Plot each quantity's requirements as a band of demonstrated, unknown and ruled-out zones.

        Parameters
        ----------
        output : PlottableFeasibility
            The feasibility output object to plot
        figsize : tuple[float, float] or None, default None
            Figure size in inches (width, height), drawn at 100 pixels per inch. If None, 9 inches wide and as tall
            as the bands need. An explicit height stretches or squeezes the bands to fit it.
        plot_evidence : bool, default False
            Whether to plot each source's observed score and confidence bound instead of the bands

        Returns
        -------
        plotly.graph_objects.Figure
        """
        import plotly.graph_objects as go

        ink, muted = FEASIBILITY_COLORS["ink"], FEASIBILITY_COLORS["ink_muted"]
        width = figsize[0] if figsize else 9.0
        gutter, ax_w, height, blocks = layout.blocks(prepare_feasibility_data(output), plot_evidence, width)

        # One pair of axes per band, laid out top-down in paper coordinates; y in inches, so offsets are exact
        fig = go.Figure()
        top = height
        for i, (band, x_range, xlabel, ax_h, placed, texts) in enumerate(blocks):
            n = "" if i == 0 else str(i + 1)
            top -= layout.TITLE + ax_h
            fig.update_layout(
                {
                    f"xaxis{n}": {
                        "domain": [gutter / width, 1 - layout.MARGIN / width],
                        "anchor": f"y{n}",
                        "range": list(x_range),
                        "tickformat": ".0%",
                        "title": {
                            "text": f"<b>{xlabel}</b>",
                            "font": {"size": layout.px(11), "color": ink},
                            "standoff": 8,
                        },
                        "tickfont": {"size": layout.px(layout.LABEL_PT), "color": muted},
                        "ticks": "outside",
                        "ticklen": 4,
                        "tickcolor": muted,
                        "showline": True,
                        "linecolor": muted,
                        "showgrid": plot_evidence,
                        "gridcolor": "#e4e3df",
                        "zeroline": False,
                    },
                    f"yaxis{n}": {
                        "domain": [top / height, (top + ax_h) / height],
                        "anchor": f"x{n}",
                        "range": [0, ax_h],
                        "visible": False,
                        "fixedrange": True,
                    },
                }
            )
            _annotate(
                fig,
                n,
                0,
                1,
                band.subtitle,
                xref=f"x{n} domain",
                yref=f"y{n} domain",
                xanchor="left",
                yanchor="bottom",
                yshift=6,
                font={"color": muted},
            )
            if plot_evidence:
                _plotly_evidence(fig, n, band, x_range, ax_w, placed)
            else:
                _plotly_band(fig, n, band, ax_w, placed)
            top -= layout.XAXIS
            for text, size, color in texts:
                _annotate(
                    fig,
                    n,
                    layout.MARGIN / width,
                    top / height,
                    text,
                    size,
                    xref="paper",
                    yref="paper",
                    xanchor="left",
                    yanchor="top",
                    align="left",
                    font={"color": color},
                )
                top -= layout.text_height(text)
            top -= layout.GAP

        fig.update_layout(
            width=width * layout.PX,
            height=(figsize[1] if figsize else height) * layout.PX,
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            template="plotly_white",
            showlegend=False,
        )
        return fig
