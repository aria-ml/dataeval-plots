"""Altair plotting backend."""

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
    draw_bounding_boxes,
    extract_boxes_and_labels,
    feasibility_zones,
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
    prepare_feasibility_data,
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

# Feasibility: one layer per mark, positioned in the band's data units (y in inches) like the matplotlib backend.
# DejaVu Sans, as matplotlib, where installed: the layout is measured in it and it has every glyph used.
_FONT = "DejaVu Sans, sans-serif"


def _mark(layers: list[Any], axes: tuple[Any, Any], mark: str, row: dict[str, Any], **props: Any) -> None:
    """Draw ``mark`` once at ``row``'s x and y on the band's ``axes``, with its x2, y2 and text if given."""
    import altair as alt

    x, y = axes
    channels = {k: "text:N" if k == "text" else k for k in ("x2", "y2", "text") if k in row}
    layers.append(getattr(alt.Chart(alt.Data(values=[row])), f"mark_{mark}")(**props).encode(x=x, y=y, **channels))


def _text(
    layers: list[Any],
    axes: tuple[Any, Any],
    x: float,
    y: float,
    text: str,
    size: float = layout.LABEL_PT,
    backing: float | None = None,
    **props: Any,
) -> None:
    """Text on the band's axes, centered unless ``align`` or ``baseline`` say otherwise; given the x units per inch
    as ``backing``, a white box behind it keeps it legible over lines."""
    lines, line_h = text.count("\n") + 1, layout.px(size) * 1.25
    align, baseline = props.setdefault("align", "center"), props.setdefault("baseline", "middle")
    if backing is not None:
        bold = 1.1 if props.get("fontWeight") == "bold" else 1.0  # measured at regular weight
        w, h = layout.text_width(text, size) * bold + 0.04, lines * line_h / layout.PX
        x0 = x - {"left": 0.02, "center": w / 2, "right": w - 0.02}[align] * backing
        y0 = y - {"top": h, "middle": h / 2, "bottom": 0}[baseline]
        _mark(layers, axes, "rect", {"x": x0, "x2": x0 + w * backing, "y": y0, "y2": y0 + h}, color="white")
    # Vega puts the first line, not the block, at a middle baseline
    dy = -(lines - 1) * line_h / 2 if baseline == "middle" else 0
    props = {"color": FEASIBILITY_COLORS["ink"], "fontSize": layout.px(size), "lineHeight": line_h, "dy": dy, **props}
    _mark(layers, axes, "text", {"x": x, "y": y, "text": text}, lineBreak="\n", font=_FONT, **props)


def _side(x: float, text: str, prefer_left: bool, x_range: tuple[float, float], ax_w: float) -> tuple[float, str]:
    """Where a label beside its mark starts and how it aligns: the preferred side, unless that leaves the axes."""
    pad = layout.PAD * (x_range[1] - x_range[0]) / ax_w
    return (x - pad, "right") if layout.goes_left(x, text, prefer_left, x_range, ax_w) else (x + pad, "left")


def _altair_requirements(
    layers: list[Any],
    axes: tuple[Any, Any],
    placed: list[layout.Placed],
    per_in: float,
    y0: float,
    y_label: float,
) -> None:
    """Dashed requirement lines from ``y0`` up to their labels' level, labels on top."""
    ink = FEASIBILITY_COLORS["ink"]
    for x, text, ha, level in placed:
        y = y_label + level * layout.LEVEL
        _mark(
            layers, axes, "rule", {"x": x, "y": y0, "y2": y}, color=ink, strokeWidth=layout.px(1.2), strokeDash=[4, 3]
        )
        _mark(
            layers, axes, "point", {"x": x, "y": y}, shape="triangle-down", filled=True, size=60, color=ink, opacity=1
        )
        dx = layout.PAD * per_in if ha == "right" else -layout.PAD * per_in
        _text(layers, axes, x + dx, y + 0.06, text, align=ha, baseline="bottom")


def _altair_band(
    layers: list[Any],
    axes: tuple[Any, Any],
    band: FeasibilityBand,
    ax_w: float,
    placed: list[layout.Placed],
) -> None:
    """Zones, boundaries and requirement markers of one band."""
    colors = FEASIBILITY_COLORS
    lo, hi = band.x_range
    per_in = (hi - lo) / ax_w
    y0, y1 = layout.BOUNDARY, layout.BOUNDARY + layout.BAR
    for kind, start, end, label in feasibility_zones(band):
        zone = {"x": start, "x2": end, "y": y0, "y2": y1}
        _mark(layers, axes, "rect", zone, color=colors[f"{kind}_fill"], stroke="white", strokeWidth=layout.px(1.5))
        fit = layout.zone_label(band, start, end, label, per_in)
        if fit:
            _text(layers, axes, fit[0], y0 + layout.BAR / 2, fit[1])
    if band.conflict:
        dashed = {"stroke": colors["warning"], "strokeWidth": layout.px(2), "strokeDash": [6, 4]}
        _mark(layers, axes, "rect", {"x": lo, "x2": hi, "y": y0, "y2": y1}, fillOpacity=0, **dashed)
    for x, text, kind, left in band.boundary_labels:
        color = colors["demonstrated" if kind == "lower" else "ruled_out"]
        _mark(layers, axes, "rule", {"x": x, "y": y0 - 0.06, "y2": y1 + 0.06}, color=color, strokeWidth=layout.px(2.5))
        at, align = _side(x, text, left, band.x_range, ax_w)
        _text(layers, axes, at, y0 - 0.1, text, align=align, baseline="top")
    _altair_requirements(layers, axes, placed, per_in, y0, y1 + 0.2)
    if band.conflict:  # over the requirement and boundary lines
        center = ((lo + hi) / 2, y0 + layout.BAR / 2)
        _text(layers, axes, *center, FEASIBILITY_CONFLICT_TEXT, 11, per_in, fontWeight="bold")


def _altair_evidence(
    layers: list[Any],
    axes: tuple[Any, Any],
    band: FeasibilityBand,
    x_range: tuple[float, float],
    ax_w: float,
    placed: list[layout.Placed],
) -> None:
    """One row per source: a tick at its bound, a dot at its observed score, both labeled."""
    colors, muted = FEASIBILITY_COLORS, FEASIBILITY_COLORS["ink_muted"]
    per_in = (x_range[1] - x_range[0]) / ax_w
    rows_top, ys = layout.evidence_rows(band)
    _altair_requirements(layers, axes, placed, per_in, 0, rows_top + 0.12)
    at = x_range[0] + layout.PAD * per_in  # text in place of marks
    if band.evidence_note:
        _text(
            layers, axes, at, rows_top - layout.ROW / 2, band.evidence_note, backing=per_in, align="left", color=muted
        )
    for y, row in zip(ys, band.evidence):
        # Bound labeled below, observed above, so the two never collide; a white backing keeps them legible over
        # the requirement lines
        ink = {"color": muted if row.faint else colors["ink"], "backing": per_in}
        source = textwrap.fill(row.label, layout.SOURCE_WRAP)
        _text(layers, axes, x_range[0] - 0.1 * per_in, y, source, align="right", color=ink["color"])
        if row.bound is None:
            _text(layers, axes, at, y, row.bound_label, backing=per_in, align="left", color=muted)
            continue
        color = colors["demonstrated" if row.bound_type == "lower" else "ruled_out"]
        mark = {"color": color, "opacity": 0.35 if row.faint else 1.0}
        _mark(layers, axes, "rule", {"x": row.bound, "y": y - 0.12, "y2": y + 0.12}, strokeWidth=layout.px(2.5), **mark)
        side, align = _side(row.bound, row.bound_label, row.bound_left, x_range, ax_w)
        _text(layers, axes, side, y - 0.14, row.bound_label, align=align, baseline="top", **ink)
        if row.observed is not None and row.observed_label is not None:
            _mark(layers, axes, "rule", {"x": row.bound, "x2": row.observed, "y": y}, strokeWidth=layout.px(3), **mark)
            _mark(layers, axes, "circle", {"x": row.observed, "y": y}, size=100, **mark)
            side, align = _side(row.observed, row.observed_label, not row.bound_left, x_range, ax_w)
            _text(layers, axes, side, y + 0.29, row.observed_label, align=align, baseline="top", **ink)


class AltairBackend(BasePlottingBackend):
    """Altair implementation of plotting backend."""

    def _plot_balance(
        self,
        output: PlottableBalance,
        figsize: tuple[float, float] | None = None,
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
        figsize : tuple[float, float] | None, default None
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
        figsize: tuple[float, float] | None = None,
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
        figsize : tuple[float, float] | None, default None
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
        figsize: tuple[float, float] | None = None,
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
        figsize : tuple[float, float] | None, default None
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
        figsize: tuple[float, float] | None = None,
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
        figsize : tuple[float, float] | None, default None
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
        figsize: tuple[float, float] | None = None,
    ) -> Any:  # alt.Chart
        """
        Render the roc_auc metric over the train/test data in relation to the threshold.

        Parameters
        ----------
        output : PlottableDriftMVDC
            The drift MVDC output object to plot
        figsize : tuple[float, float] | None, default None
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

        # Convert Polars DataFrame to format needed for plotting
        # Create index column and map period values
        # Convert via dict to avoid pyarrow dependency
        plot_df = (
            resdf.with_row_index("index")
            .with_columns(
                [
                    pl.when(pl.col("chunk_period") == "reference")
                    .then(pl.lit("train"))
                    .otherwise(pl.lit("test"))
                    .alias("period")
                ]
            )
            .select(
                [
                    "index",
                    pl.col("domain_classifier_auroc_value").alias("value"),
                    pl.col("domain_classifier_auroc_upper_threshold").alias("upper_threshold"),
                    pl.col("domain_classifier_auroc_lower_threshold").alias("lower_threshold"),
                    "period",
                    pl.col("domain_classifier_auroc_alert").alias("alert"),
                ]
            )
        )
        # Convert to pandas via dict to avoid pyarrow dependency
        df = pd.DataFrame(plot_df.to_dict(as_series=False))

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
        indices: Sequence[int] | None = None,
        images_per_row: int = 3,
        figsize: tuple[float, float] | None = None,
        show_labels: bool = False,
        show_metadata: bool | Sequence[str] = False,
        additional_metadata: Sequence[dict[str, Any]] | None = None,
    ) -> Any:
        """
        Plot a grid of images from a dataset using Altair.

        Parameters
        ----------
        dataset : Dataset
            MAITE-compatible dataset containing images
        indices : Sequence[int] or None, default None
            Indices of images to plot from the dataset. None plots every item.
        images_per_row : int, default 3
            Number of images to display per row
        figsize : tuple[float, float] or None, default None
            Figure size in inches (width, height)
        show_labels : bool, default False
            Whether to display labels extracted from targets
        show_metadata : bool or Sequence[str], default False
            Whether to display metadata from the dataset items. A sequence of keys
            displays only those, in the order given.
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
                # Every key, or just the named ones, as key: value pairs
                keys = metadata if isinstance(show_metadata, bool) else [k for k in show_metadata if k in metadata]
                title_parts.extend(f"{k}: {metadata[k]}" for k in keys)

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

    def _build_scatter_chart(
        self,
        embeddings: NDArray[Any],
        labels: NDArray[Any] | None,
        label_names: Mapping[int, str] | None,
        width: int,
        height: int,
        title: str,
        point_size: int = 30,
    ) -> Any:  # alt.Chart
        """Build a single Altair scatter chart for 2D embeddings."""
        import altair as alt
        import pandas as pd

        _no_axis = alt.Axis(labels=False, ticks=False, title="")
        data: dict[str, Any] = {
            "x": embeddings[:, 0],
            "y": embeddings[:, 1],
        }

        if labels is not None:
            if label_names:
                data["class"] = [label_names.get(int(lbl), str(lbl)) for lbl in labels]
            else:
                data["class"] = [str(lbl) for lbl in labels]

        df = pd.DataFrame(data)

        if labels is not None:
            chart = (
                alt.Chart(df)
                .mark_circle(size=point_size, opacity=0.7)
                .encode(
                    x=alt.X("x:Q", axis=_no_axis),
                    y=alt.Y("y:Q", axis=_no_axis),
                    color=alt.Color("class:N", title="Class"),
                    tooltip=["class:N"],
                )
            )
        else:
            chart = (
                alt.Chart(df)
                .mark_circle(size=point_size, opacity=0.7)
                .encode(
                    x=alt.X("x:Q", axis=_no_axis),
                    y=alt.Y("y:Q", axis=_no_axis),
                )
            )

        return chart.properties(width=width, height=height, title=title)

    def project(
        self,
        embeddings: NDArray[Any],
        labels: NDArray[Any] | None = None,
        label_names: Mapping[int, str] | None = None,
        method: str = "pca",
        dimensions: Literal[2, 3] = 2,
        figsize: tuple[float, float] | None = None,
        title: str | None = None,
    ) -> Any:  # alt.Chart
        """
        Plot projected embeddings as a scatter chart.

        Parameters
        ----------
        embeddings : NDArray
            Reduced embeddings with shape ``(N, 2)``.
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
        alt.Chart

        Raises
        ------
        NotImplementedError
            If dimensions is 3 (Altair does not support 3D scatter plots).
        """
        if dimensions == 3:
            raise NotImplementedError(
                "Altair does not support 3D scatter plots. Use dimensions=2 or a different backend."
            )

        if title is None:
            title = f"{method.upper()} Projection"

        if figsize is not None:
            width = int(figsize[0] * 100)
            height = int(figsize[1] * 100)
        else:
            width = 600
            height = 500

        return self._build_scatter_chart(embeddings, labels, label_names, width, height, title)

    def project_grid(
        self,
        embeddings_list: Sequence[NDArray[Any]],
        methods: Sequence[str],
        labels: NDArray[Any] | None = None,
        label_names: Mapping[int, str] | None = None,
        dimensions: Literal[2, 3] = 2,
        figsize: tuple[float, float] | None = None,
        title: str | None = None,
    ) -> Any:  # alt.VConcatChart | alt.HConcatChart | alt.Chart
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
        alt.VConcatChart or alt.HConcatChart or alt.Chart

        Raises
        ------
        NotImplementedError
            If dimensions is 3.
        """
        if dimensions == 3:
            raise NotImplementedError(
                "Altair does not support 3D scatter plots. Use dimensions=2 or a different backend."
            )

        import altair as alt

        from dataeval_plots.backends._shared import calculate_subplot_grid

        n = len(methods)
        aspect = figsize[0] / figsize[1] if figsize else None
        rows, cols = calculate_subplot_grid(n, aspect_ratio=aspect)

        if figsize is not None:
            chart_width = int(figsize[0] * 100) // cols
            chart_height = int(figsize[1] * 100) // rows
        else:
            chart_width = 250
            chart_height = 220

        charts = [
            self._build_scatter_chart(emb, labels, label_names, chart_width, chart_height, m.upper(), point_size=20)
            for emb, m in zip(embeddings_list, methods)
        ]

        # Arrange in grid rows
        grid_rows = []
        for i in range(0, len(charts), cols):
            row_charts = charts[i : i + cols]
            grid_rows.append(alt.hconcat(*row_charts) if len(row_charts) > 1 else row_charts[0])

        grid = alt.vconcat(*grid_rows) if len(grid_rows) > 1 else grid_rows[0]

        if title:
            grid = grid.properties(title=title)

        return grid

    def _plot_feasibility(
        self,
        output: PlottableFeasibility,
        figsize: tuple[float, float] | None = None,
        plot_evidence: bool = False,
    ) -> Any:  # alt.VConcatChart
        """
        Plot each quantity's requirements as a band of demonstrated, unknown and ruled-out zones.

        Parameters
        ----------
        output : PlottableFeasibility
            The feasibility output object to plot
        figsize : tuple[float, float] or None, default None
            Figure size in inches (width, height), drawn at 100 pixels per inch. If None, 9 inches wide and as tall
            as the bands need. An explicit height stretches or squeezes the band views to meet it; captions keep
            their height.
        plot_evidence : bool, default False
            Whether to plot each source's observed score and confidence bound instead of the bands

        Returns
        -------
        alt.VConcatChart
            One band chart and its caption per quantity, stacked
        """
        import altair as alt

        ink, muted = FEASIBILITY_COLORS["ink"], FEASIBILITY_COLORS["ink_muted"]
        width = figsize[0] if figsize else 9.0
        gutter, ax_w, height, blocks = layout.blocks(prepare_feasibility_data(output), plot_evidence, width)
        # Only the band views stretch, so the rest keeps its height and the total meets the requested one
        views = sum(block[3] for block in blocks)
        stretch = max(figsize[1] - (height - views), 0.1) / views if figsize else 1.0
        view = alt.ViewBackground(stroke=None)  # per chart, so the result stays composable

        charts = []
        for band, x_range, xlabel, ax_h, placed, texts in blocks:
            axis = alt.Axis(
                format=".0%",
                tickCount=max(2, round(ax_w / 1.3)),  # about matplotlib's tick density
                tickMinStep=0.01,
                title=xlabel,
                titleFont=_FONT,
                titleFontSize=layout.px(11),
                titleFontWeight="bold",
                titleColor=ink,
                labelFont=_FONT,
                labelFontSize=layout.px(layout.LABEL_PT),
                labelColor=muted,
                tickColor=muted,
                domainColor=muted,
                grid=plot_evidence,
                gridColor="#e4e3df",
            )
            axes = (
                alt.X("x:Q", scale=alt.Scale(domain=list(x_range), nice=False, zero=False), axis=axis),
                alt.Y("y:Q", scale=alt.Scale(domain=[0, ax_h], nice=False, zero=False), axis=None),
            )
            layers: list[Any] = []
            if plot_evidence:
                _altair_evidence(layers, axes, band, x_range, ax_w, placed)
            else:
                _altair_band(layers, axes, band, ax_w, placed)
            title = alt.TitleParams(
                band.subtitle,
                anchor="start",
                frame="group",
                font=_FONT,
                fontSize=layout.px(10),
                fontWeight="normal",
                color=muted,
            )
            size = {"width": ax_w * layout.PX, "height": ax_h * layout.PX * stretch}
            charts.append(alt.layer(*layers).properties(title=title, view=view, **size))

            # Below the axes, from the figure margin: the evidence legend, the caption paragraph, then the notes
            lines, top = [], 0.0
            for text, points, color in texts:
                style = {
                    "font": _FONT,
                    "fontSize": layout.px(points),
                    "color": color,
                    "lineHeight": layout.LINE * layout.PX,
                }
                x, y = alt.value(-(gutter - layout.MARGIN) * layout.PX), alt.value(top * layout.PX)
                lines.append(
                    alt.Chart(alt.Data(values=[{"text": text}]))
                    .mark_text(align="left", baseline="top", lineBreak="\n", **style)
                    .encode(x=x, y=y, text="text:N")
                )
                top += layout.text_height(text)
            charts.append(alt.layer(*lines).properties(width=ax_w * layout.PX, height=top * layout.PX, view=view))

        return alt.vconcat(*charts, spacing=10)
