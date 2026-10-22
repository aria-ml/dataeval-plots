"""Matplotlib plotting backend."""

from __future__ import annotations

import textwrap
import warnings
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
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
    normalize_reference_outputs,
    plot_drift_on_axis,
    prepare_balance_data,
    prepare_diversity_data,
    prepare_drift_data,
    prepare_feasibility_data,
    project_steps,
    validate_class_names,
)
from dataeval_plots.protocols import (
    PlottableBalance,
    PlottableDiversity,
    PlottableDriftMVDC,
    PlottableFeasibility,
    PlottableStats,
    PlottableSufficiency,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

_BACKING = {"facecolor": "white", "edgecolor": "none", "pad": 1}  # behind feasibility labels that cross lines


def _side_text(
    ax: Axes,
    x: float,
    y: float,
    text: str,
    prefer_left: bool,
    x_range: tuple[float, float],
    ax_w: float,
    **kwargs: Any,
) -> None:
    """Label beside its mark, top at ``y``, on the preferred side unless that leaves the axes."""
    pad = layout.PAD * (x_range[1] - x_range[0]) / ax_w
    left = layout.goes_left(x, text, prefer_left, x_range, ax_w)
    ax.text(
        x - pad if left else x + pad,
        y,
        text,
        ha="right" if left else "left",
        va="top",
        fontsize=layout.LABEL_PT,
        **kwargs,
    )


def _draw_requirements(ax: Axes, placed: list[layout.Placed], per_in: float, y0: float, y_label: float) -> None:
    """Dashed requirement lines from ``y0`` up to their labels' level, labels on top."""
    ink = FEASIBILITY_COLORS["ink"]
    for x, text, ha, level in placed:
        y = y_label + level * layout.LEVEL
        ax.plot([x, x], [y0, y], color=ink, linestyle=(0, (4, 3)), linewidth=1.2)
        ax.plot([x], [y], marker="v", color=ink, markersize=6)
        dx = layout.PAD * per_in if ha == "right" else -layout.PAD * per_in
        ax.text(x + dx, y + 0.06, text, ha=ha, va="bottom", fontsize=layout.LABEL_PT, color=ink)


def _draw_band(ax: Axes, band: FeasibilityBand, ax_w: float, placed: list[layout.Placed]) -> None:
    """Zones, boundaries and requirement markers of one band."""
    from matplotlib.patches import Rectangle

    colors, ink = FEASIBILITY_COLORS, FEASIBILITY_COLORS["ink"]
    lo, hi = band.x_range
    per_in = (hi - lo) / ax_w
    y0, y1 = layout.BOUNDARY, layout.BOUNDARY + layout.BAR
    ax.set_yticks([])
    for kind, start, end, label in feasibility_zones(band):
        fill = colors[f"{kind}_fill"]
        ax.add_patch(Rectangle((start, y0), end - start, layout.BAR, facecolor=fill, edgecolor="white", lw=1.5))
        fit = layout.zone_label(band, start, end, label, per_in)
        if fit:
            ax.text(fit[0], y0 + layout.BAR / 2, fit[1], ha="center", va="center", fontsize=layout.LABEL_PT, color=ink)
    if band.conflict:
        outline = Rectangle(
            (lo, y0), hi - lo, layout.BAR, fill=False, edgecolor=colors["warning"], linestyle="--", lw=2
        )
        ax.add_patch(outline)
        ax.text(
            (lo + hi) / 2,
            y0 + layout.BAR / 2,
            FEASIBILITY_CONFLICT_TEXT,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color=ink,
            bbox=_BACKING,
        )
    for x, text, kind, left in band.boundary_labels:
        ax.plot(
            [x, x], [y0 - 0.06, y1 + 0.06], color=colors["demonstrated" if kind == "lower" else "ruled_out"], lw=2.5
        )
        _side_text(ax, x, y0 - 0.1, text, left, band.x_range, ax_w, color=ink)
    _draw_requirements(ax, placed, per_in, y0, y1 + 0.2)


def _draw_evidence(
    ax: Axes,
    band: FeasibilityBand,
    x_range: tuple[float, float],
    ax_w: float,
    placed: list[layout.Placed],
) -> None:
    """One row per source: a tick at its bound, a dot at its observed score, both labeled."""
    colors, muted = FEASIBILITY_COLORS, FEASIBILITY_COLORS["ink_muted"]
    lo, hi = x_range
    per_in = (hi - lo) / ax_w
    rows_top, ys = layout.evidence_rows(band)
    ax.grid(axis="x", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    _draw_requirements(ax, placed, per_in, 0, rows_top + 0.12)
    aside = {"ha": "left", "va": "center", "fontsize": layout.LABEL_PT, "color": muted, "bbox": _BACKING}  # no marks
    if band.evidence_note:
        ax.text(lo + layout.PAD * per_in, rows_top - layout.ROW / 2, band.evidence_note, **aside)
    for y, row in zip(ys, band.evidence):
        if row.bound is None:
            ax.text(lo + layout.PAD * per_in, y, row.bound_label, **aside)
            continue
        color = colors["demonstrated" if row.bound_type == "lower" else "ruled_out"]
        mark = {"color": color, "alpha": 0.35 if row.faint else 1.0}
        mark["clip_on"] = False  # a bound at the axis edge stays whole
        # Bound labeled below, observed above, so the two never collide; a white backing keeps them legible over
        # the requirement lines
        ink = {"color": muted if row.faint else colors["ink"], "bbox": _BACKING}
        ax.plot([row.bound] * 2, [y - 0.12, y + 0.12], linewidth=2.5, **mark)
        _side_text(ax, row.bound, y - 0.14, row.bound_label, row.bound_left, x_range, ax_w, **ink)
        if row.observed is not None and row.observed_label is not None:
            ax.plot([row.bound, row.observed], [y, y], linewidth=3, **mark)
            ax.plot([row.observed], [y], marker="o", markersize=8, **mark)
            _side_text(ax, row.observed, y + 0.29, row.observed_label, not row.bound_left, x_range, ax_w, **ink)
    ax.set_yticks(ys, labels=[textwrap.fill(r.label, layout.SOURCE_WRAP) for r in band.evidence])
    ax.tick_params(axis="y", length=0)
    for tick, row in zip(ax.get_yticklabels(), band.evidence):
        tick.set_color(muted if row.faint else colors["ink"])


class MatplotlibBackend(BasePlottingBackend):
    """Matplotlib implementation of plotting backend."""

    def heatmap(
        self,
        data: list[Any] | NDArray[Any],
        row_labels: list[str] | NDArray[Any],
        col_labels: list[str] | NDArray[Any],
        xlabel: str = "",
        ylabel: str = "",
        cbarlabel: str = "",
        figsize: tuple[float, float] | None = None,
    ) -> Figure:
        """
        Plots a formatted heatmap.

        Parameters
        ----------
        data : NDArray
            Array containing numerical values for factors to plot
        row_labels : ArrayLike
            List/Array containing the labels for rows in the histogram
        col_labels : ArrayLike
            List/Array containing the labels for columns in the histogram
        xlabel : str, default ""
            X-axis label
        ylabel : str, default ""
            Y-axis label
        cbarlabel : str, default ""
            Label for the colorbar

        Returns
        -------
        matplotlib.figure.Figure
            Formatted heatmap
        """
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter

        np_data = np.asarray(data)
        rows: list[str] = [str(n) for n in np.asarray(row_labels)]
        cols: list[str] = [str(n) for n in np.asarray(col_labels)]

        if figsize is None:
            figsize = (10, 10)
        fig, ax = plt.subplots(figsize=figsize)

        # Plot the heatmap
        im = ax.imshow(np_data, vmin=0, vmax=1.0)

        # Create colorbar
        cbar = fig.colorbar(im, shrink=0.5)
        cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_ticklabels(["0.0", "0.25", "0.5", "0.75", "1.0"])
        cbar.set_label(cbarlabel, loc="center")

        # Show all ticks and label them with the respective list entries.
        ax.set_xticks(np.arange(np_data.shape[1]), labels=cols)
        ax.set_yticks(np.arange(np_data.shape[0]), labels=rows)

        ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        light_gray = "0.9"
        # Turn spines on and create light gray easily visible grid.
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(light_gray)

        xticks = np.arange(np_data.shape[1] + 1) - 0.5
        yticks = np.arange(np_data.shape[0] + 1) - 0.5
        ax.set_xticks(xticks, minor=True)
        ax.set_yticks(yticks, minor=True)
        ax.grid(which="minor", color=light_gray, linestyle="-", linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)

        valfmt = FuncFormatter(self.format_text)

        # Normalize the threshold to the images color range.
        threshold = im.norm(1.0) / 2.0

        # Set default alignment to center, but allow it to be
        # overwritten by textkw.
        kw = {"horizontalalignment": "center", "verticalalignment": "center"}

        # Loop over the data and create a `Text` for each "pixel".
        # Change the text's color depending on the data.
        textcolors = ("white", "black")
        texts = []
        for i in range(np_data.shape[0]):
            for j in range(np_data.shape[1]):
                kw.update(color=textcolors[int(im.norm(np_data[i, j]) > threshold)])
                text = im.axes.text(j, i, valfmt(np_data[i, j], None), **kw)  # type: ignore
                texts.append(text)

        fig.tight_layout()
        return fig

    # Function to define how the text is displayed in the heatmap
    def format_text(self, *args: str) -> str:
        """
        Helper function to format text for heatmap().

        Parameters
        ----------
        *args : tuple[str, str]
            Text to be formatted. Second element is ignored, but is a
            mandatory pass-through argument as per matplotlib.ticker.FuncFormatter

        Returns
        -------
        str
            Formatted text
        """
        x = args[0]
        return f"{x:.2f}".replace("0.00", "0").replace("0.", ".").replace("nan", "")

    def histogram_plot(
        self,
        data_dict: Mapping[str, Any],
        log: bool = True,
        xlabel: str = "values",
        ylabel: str = "counts",
        figsize: tuple[float, float] | None = None,
    ) -> Figure:
        """
        Plots a formatted histogram.

        Parameters
        ----------
        data_dict : dict
            Dictionary containing the metrics and their value arrays
        log : bool, default True
            If True, plots the histogram on a semi-log scale (y axis)
        xlabel : str, default "values"
            X-axis label
        ylabel : str, default "counts"
            Y-axis label

        Returns
        -------
        matplotlib.figure.Figure
            Formatted plot of histograms
        """
        import matplotlib.pyplot as plt

        num_metrics = len(data_dict)
        rows, cols = calculate_subplot_grid(num_metrics)
        if figsize is None:
            figsize = (cols * 3 + 1, rows * 3)
        fig, axs = plt.subplots(rows, 3, figsize=figsize)
        axs_flat = np.asarray(axs).flatten()
        for ax, metric in zip(
            axs_flat,
            data_dict,
        ):
            # Plot the histogram for the chosen metric
            ax.hist(data_dict[metric].astype(np.float64), bins=20, log=log)

            # Add labels to the histogram
            ax.set_title(metric)
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)

        for ax in axs_flat[num_metrics:]:
            ax.axis("off")
            ax.set_visible(False)

        fig.tight_layout()
        return fig

    def channel_histogram_plot(
        self,
        data_dict: Mapping[str, Any],
        log: bool = True,
        max_channels: int = 3,
        ch_mask: Sequence[bool] | None = None,
        xlabel: str = "values",
        ylabel: str = "counts",
        figsize: tuple[float, float] | None = None,
    ) -> Figure:
        """
        Plots a formatted channel-wise histogram.

        Parameters
        ----------
        data_dict : dict
            Dictionary containing the metrics and their value arrays
        log : bool, default True
            If True, plots the histogram on a semi-log scale (y axis)
        max_channels : int, default 3
            Maximum number of channels to plot
        ch_mask : Sequence[bool] | None, default None
            Boolean mask for selecting channels
        xlabel : str, default "values"
            X-axis label
        ylabel : str, default "counts"
            Y-axis label

        Returns
        -------
        matplotlib.figure.Figure
            Formatted plot of histograms
        """
        import matplotlib.pyplot as plt

        # Use shared constant for channelwise metrics
        data_keys = [key for key in data_dict if key in CHANNELWISE_METRICS]
        label_kwargs = {"label": [f"Channel {i}" for i in range(max_channels)]}

        num_metrics = len(data_keys)
        rows, cols = calculate_subplot_grid(num_metrics)
        if figsize is None:
            figsize = (cols * 3 + 1, rows * 3)
        fig, axs = plt.subplots(rows, 3, figsize=figsize)
        axs_flat = np.asarray(axs).flatten()
        for ax, metric in zip(
            axs_flat,
            data_keys,
        ):
            # Plot the histogram for the chosen metric
            data = data_dict[metric][ch_mask].reshape(-1, max_channels)
            ax.hist(
                data.astype(np.float64),
                bins=20,
                density=True,
                log=log,
                **label_kwargs,
            )
            # Only plot the labels once for channels
            if label_kwargs:
                ax.legend()
                label_kwargs = {}

            # Add labels to the histogram
            ax.set_title(metric)
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)

        for ax in axs_flat[num_metrics:]:
            ax.axis("off")
            ax.set_visible(False)

        fig.tight_layout()
        return fig

    def _plot_balance(
        self,
        output: PlottableBalance,
        figsize: tuple[float, float] | None = None,
        row_labels: Sequence[Any] | NDArray[Any] | None = None,
        col_labels: Sequence[Any] | NDArray[Any] | None = None,
        plot_classwise: bool = False,
    ) -> Figure:
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
        matplotlib.figure.Figure
        """
        import numpy as np

        # Use shared helper to prepare data
        data, row_labels, col_labels, xlabel, ylabel, title = prepare_balance_data(
            output, row_labels, col_labels, plot_classwise
        )

        return self.heatmap(
            data,
            np.asarray(row_labels),
            np.asarray(col_labels),
            xlabel=xlabel,
            ylabel=ylabel,
            cbarlabel="Normalized Mutual Information",
            figsize=figsize,
        )

    def _plot_diversity(
        self,
        output: PlottableDiversity,
        figsize: tuple[float, float] | None = None,
        row_labels: Sequence[Any] | NDArray[Any] | None = None,
        col_labels: Sequence[Any] | NDArray[Any] | None = None,
        plot_classwise: bool = False,
    ) -> Figure:
        """
        Plot a heatmap of diversity information.

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
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        # Use shared helper to prepare data
        data, row_labels, col_labels, xlabel, ylabel, title, method_name = prepare_diversity_data(
            output, row_labels, col_labels, plot_classwise
        )

        if plot_classwise:
            fig = self.heatmap(
                data,
                np.asarray(row_labels),
                np.asarray(col_labels),
                xlabel=xlabel,
                ylabel=ylabel,
                cbarlabel=f"Normalized {method_name} Index",
                figsize=figsize,
            )
        else:
            # Bar chart for diversity indices
            if figsize is None:
                figsize = (8, 8)
            fig, ax = plt.subplots(figsize=figsize)

            # DataFrame-based: get diversity values from factors DataFrame
            diversity_values = output.factors["diversity_value"].to_list()
            ax.bar(row_labels, diversity_values)

            ax.set_xlabel(xlabel)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            fig.tight_layout()

        return fig

    def _plot_measure(
        self,
        name: str,
        steps: NDArray[Any],
        averaged_measure: NDArray[Any],
        measures: NDArray[Any] | None,
        params: NDArray[Any],
        projection: NDArray[Any],
        show_error_bars: bool,
        show_asymptote: bool,
        ax: Axes,
    ) -> None:
        ax.set_title(f"{name} Sufficiency")
        ax.set_xlabel("Steps")
        projection_curve = ax.plot(
            projection,
            project_steps(params, projection),
            linestyle="solid",
            label=f"Potential Model Results ({name})",
            linewidth=2,
            zorder=2,
        )
        projection_color = projection_curve[0].get_color()
        # Calculate error bars
        # Plot measure over each step with associated error
        if show_error_bars:
            if measures is None:
                warnings.warn(
                    "Error bars cannot be plotted without full, unaveraged data",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                error = np.std(measures, axis=0)
                ax.errorbar(
                    steps,
                    averaged_measure,
                    ecolor=projection_color,
                    color=projection_color,
                    yerr=error,
                    capsize=7,
                    capthick=1.5,
                    elinewidth=1.5,
                    fmt="o",
                    label=f"Model Results ({name})",
                    markersize=5,
                    zorder=3,
                )
        else:
            ax.scatter(steps, averaged_measure, color=projection_color, label=f"Model Results ({name})", zorder=3)
        # Plot asymptote
        if show_asymptote:
            bound = 1 - params[2]
            ax.axhline(
                y=bound, linestyle="dashed", color=projection_color, label=f"Asymptote: {bound:.4g} ({name})", zorder=1
            )

    def _plot_single_class(
        self,
        name: str,
        steps: NDArray[Any],
        averaged_measure: NDArray[Any],
        measures: NDArray[Any] | None,
        params: NDArray[Any],
        projection: NDArray[Any],
        show_error_bars: bool,
        show_asymptote: bool,
        plots: list[Figure],
        reference_outputs: Sequence[Any],
        figsize: tuple[float, float] | None = None,
    ) -> None:
        from matplotlib import pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_ylabel(f"{name}")
        self._plot_measure(
            name,
            steps,
            averaged_measure,
            measures,
            params,
            projection,
            show_error_bars,
            show_asymptote,
            ax,
        )
        # Plot metric for each provided reference output
        for index, output in enumerate(reference_outputs):
            if name in output.averaged_measures:
                self._plot_measure(
                    f"{name} Output {index + 2}",
                    output.steps,
                    output.averaged_measures[name],
                    output.measures.get(name),
                    output.params[name],
                    projection,
                    show_error_bars,
                    show_asymptote,
                    ax,
                )
        ax.set_xscale("log")
        ax.legend(loc="best")
        plots.append(fig)

    def _plot_multiclass(
        self,
        name: str,
        steps: NDArray[Any],
        averaged_measure: NDArray[Any],
        measures: NDArray[Any] | None,
        params: NDArray[Any],
        projection: NDArray[Any],
        show_error_bars: bool,
        show_asymptote: bool,
        plots: list[Figure],
        reference_outputs: Sequence[Any],
        class_names: Sequence[str] | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> None:
        from matplotlib import pyplot as plt

        validate_class_names(averaged_measure, class_names)
        for i, values in enumerate(averaged_measure):
            # Create a plot for each class
            fig, ax = plt.subplots(figsize=figsize)
            class_name = str(i) if class_names is None else class_names[i]
            ax.set_ylabel(f"{name}")
            self._plot_measure(
                f"{name}_{class_name}",
                steps,
                values,
                None if measures is None else measures[:, :, i],
                params[i],
                projection,
                show_error_bars,
                show_asymptote,
                ax,
            )
            # Iterate through each reference output to plot similar class
            for index, output in enumerate(reference_outputs):
                if (
                    name in output.averaged_measures
                    and output.averaged_measures[name].ndim > 1
                    and i <= len(output.averaged_measures[name])
                ):
                    self._plot_measure(
                        f"{name}_{class_name} Output {index + 2}",
                        output.steps,
                        output.averaged_measures[name][i],
                        output.measures[name][:, :, i] if len(output.measures) else None,
                        output.params[name][i],
                        projection,
                        show_error_bars,
                        show_asymptote,
                        ax,
                    )
            ax.set_xscale("log")
            ax.legend(loc="best")
            plots.append(fig)

    def _plot_sufficiency(
        self,
        output: PlottableSufficiency,
        figsize: tuple[float, float] | None = None,
        class_names: Sequence[str] | None = None,
        show_error_bars: bool = True,
        show_asymptote: bool = True,
        reference_outputs: Sequence[PlottableSufficiency] | PlottableSufficiency | None = None,
    ) -> list[Figure]:
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
        reference_outputs : Sequence[SufficiencyOutput] | SufficiencyOutput, default None
            Singular or multiple SufficiencyOutput objects to include in plots

        Returns
        -------
        list[Figure]
            List of Figures for each measure
        """
        # Extrapolation parameters
        extrapolated = calculate_projection(output.steps)

        # Stores all plots
        plots = []

        # Wrap reference
        reference_outputs = normalize_reference_outputs(reference_outputs)

        # Iterate through measures
        for name, measures in output.averaged_measures.items():
            if measures.ndim > 1:
                self._plot_multiclass(
                    name,
                    output.steps,
                    measures,
                    output.measures.get(name),
                    output.params[name],
                    extrapolated,
                    show_error_bars,
                    show_asymptote,
                    plots,
                    reference_outputs,
                    class_names,
                    figsize,
                )
            else:
                self._plot_single_class(
                    name,
                    output.steps,
                    measures,
                    output.measures.get(name),
                    output.params[name],
                    extrapolated,
                    show_error_bars,
                    show_asymptote,
                    plots,
                    reference_outputs,
                    figsize,
                )
        return plots

    def _plot_stats(
        self,
        output: PlottableStats,
        figsize: tuple[float, float] | None = None,
        log: bool = True,
        channel_limit: int | None = None,
        channel_index: int | Iterable[int] | None = None,
    ) -> Figure:
        """
        Plots the statistics as a set of histograms.

        Parameters
        ----------
        output : PlottableStats
            The stats output object to plot
        log : bool, default True
            If True, plots the histograms on a logarithmic scale.
        channel_limit : int or None, default None
            The maximum number of channels to plot. If None, all channels are plotted.
        channel_index : int, Iterable[int] or None, default None
            The index or indices of the channels to plot. If None, all channels are plotted.

        Returns
        -------
        matplotlib.figure.Figure
        """
        from matplotlib.figure import Figure

        max_channels, ch_mask = output._get_channels(channel_limit, channel_index)
        factors = output.factors(exclude_constant=True)
        if not factors:
            return Figure()
        if max_channels == 1:
            return self.histogram_plot(factors, log, figsize=figsize)
        return self.channel_histogram_plot(factors, log, max_channels, ch_mask, figsize=figsize)

    def _plot_drift_mvdc(
        self,
        output: PlottableDriftMVDC,
        figsize: tuple[float, float] | None = None,
    ) -> Figure:
        """
        Render the roc_auc metric over the train/test data in relation to the threshold.

        Parameters
        ----------
        output : PlottableDriftMVDC
            The drift MVDC output object to plot

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        # Use shared helper to prepare drift data
        resdf, trndf, tstdf, driftx, is_sufficient = prepare_drift_data(output)

        fig, ax = plt.subplots(dpi=300, figsize=figsize)

        if is_sufficient and np.size(driftx) > 2:
            # Use shared plotting helper with matplotlib-specific styling
            plot_drift_on_axis(
                ax,
                resdf,
                trndf,
                tstdf,
                driftx,
                threshold_upper_color="red",
                threshold_lower_color="red",
                train_color="b",
                test_color="g",
                drift_color="m",
                threshold_upper_label="thr_up",
                threshold_lower_label="thr_low",
                train_label="train",
                test_label="test",
                drift_label="drift",
                drift_marker="d",
                drift_markersize=3,
                linewidth=1,
                title_fontsize=8,
                label_fontsize=7,
                tick_fontsize=6,
                legend_fontsize=6,
            )
        return fig

    def _plot_feasibility(
        self,
        output: PlottableFeasibility,
        figsize: tuple[float, float] | None = None,
        plot_evidence: bool = False,
    ) -> Figure:
        """
        Plot each quantity's requirements as a band of demonstrated, unknown and ruled-out zones.

        Parameters
        ----------
        output : PlottableFeasibility
            The feasibility output object to plot
        figsize : tuple[float, float] or None, default None
            Figure size in inches (width, height). If None, 9 inches wide and as tall as the bands need. An
            explicit figsize overrides the computed layout height, so the bands stretch or squeeze to fit it.
        plot_evidence : bool, default False
            Whether to plot each source's observed score and confidence bound instead of the bands

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter

        ink, muted = FEASIBILITY_COLORS["ink"], FEASIBILITY_COLORS["ink_muted"]
        width = figsize[0] if figsize else 9.0
        gutter, ax_w, height, blocks = layout.blocks(prepare_feasibility_data(output), plot_evidence, width)

        fig = plt.figure(figsize=(width, height))
        top = height
        for band, x_range, xlabel, ax_h, placed, texts in blocks:
            top -= layout.TITLE + ax_h
            ax = fig.add_axes((gutter / width, top / height, ax_w / width, ax_h / height))
            ax.set_xlim(*x_range)
            ax.set_ylim(0, ax_h)  # y in inches, so offsets above are exact
            ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
            ax.tick_params(colors=muted, labelsize=layout.LABEL_PT, length=3)
            ax.set_xlabel(xlabel, fontsize=11, fontweight="bold", color=ink)
            ax.set_title(band.subtitle, loc="left", fontsize=layout.LABEL_PT, color=muted, pad=6)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(muted)
            if plot_evidence:
                _draw_evidence(ax, band, x_range, ax_w, placed)
            else:
                _draw_band(ax, band, ax_w, placed)
            top -= layout.XAXIS
            for text, size, color in texts:
                fig.text(
                    layout.MARGIN / width, top / height, text, va="top", fontsize=size, color=color, linespacing=1.4
                )
                top -= layout.text_height(text)
            top -= layout.GAP
        if figsize:
            fig.set_size_inches(figsize)
        return fig
