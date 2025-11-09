"""Matplotlib plotting backend."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from dataeval_plots.backends._base import BasePlottingBackend
from dataeval_plots.backends._shared import (
    CHANNELWISE_METRICS,
    calculate_projection,
    calculate_subplot_grid,
    image_to_hwc,
    normalize_reference_outputs,
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

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


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

        fig, ax = plt.subplots(figsize=(10, 10))

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
        fig, axs = plt.subplots(rows, 3, figsize=(cols * 3 + 1, rows * 3))
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
        fig, axs = plt.subplots(rows, 3, figsize=(cols * 3 + 1, rows * 3))
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

    def _plot_coverage(
        self,
        output: PlottableCoverage,
        images: Indexable | None = None,  # Images | Dataset
        top_k: int = 6,
    ) -> Figure:
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
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt
        import numpy as np

        # Use shared helper to prepare and validate images
        selected_images, num_images, rows, cols = prepare_coverage_images(output, images, top_k)

        fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))

        # Flatten axes using numpy array explicitly for compatibility
        axs_flat = np.asarray(axs).flatten()

        for image, ax in zip(selected_images, axs_flat):
            ax.imshow(image_to_hwc(image))
            ax.axis("off")

        for ax in axs_flat[num_images:]:
            ax.axis("off")

        fig.tight_layout()
        return fig

    def _plot_balance(
        self,
        output: PlottableBalance,
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
        )

    def _plot_diversity(
        self,
        output: PlottableDiversity,
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
            )
        else:
            # Bar chart for diversity indices
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.bar(row_labels, output.diversity_index)
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
    ) -> None:
        from matplotlib import pyplot as plt

        fig, ax = plt.subplots()
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
    ) -> None:
        from matplotlib import pyplot as plt

        validate_class_names(averaged_measure, class_names)
        for i, values in enumerate(averaged_measure):
            # Create a plot for each class
            fig, ax = plt.subplots()
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
                )
        return plots

    def _plot_base_stats(
        self,
        output: PlottableBaseStats,
        log: bool = True,
        channel_limit: int | None = None,
        channel_index: int | Iterable[int] | None = None,
    ) -> Figure:
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
        matplotlib.figure.Figure
        """
        from matplotlib.figure import Figure

        max_channels, ch_mask = output._get_channels(channel_limit, channel_index)
        factors = output.factors(exclude_constant=True)
        if not factors:
            return Figure()
        if max_channels == 1:
            return self.histogram_plot(factors, log)
        return self.channel_histogram_plot(factors, log, max_channels, ch_mask)

    def _plot_drift_mvdc(
        self,
        output: PlottableDriftMVDC,
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
        import numpy as np

        # Use shared helper to prepare drift data
        resdf, trndf, tstdf, driftx, is_sufficient = prepare_drift_data(output)

        fig, ax = plt.subplots(dpi=300)
        xticks = np.arange(resdf.shape[0])

        if is_sufficient and np.size(driftx) > 2:
            ax.plot(resdf.index, resdf["domain_classifier_auroc"]["upper_threshold"], "r--", label="thr_up")
            ax.plot(resdf.index, resdf["domain_classifier_auroc"]["lower_threshold"], "r--", label="thr_low")
            ax.plot(trndf.index, trndf["domain_classifier_auroc"]["value"], "b", label="train")
            ax.plot(tstdf.index, tstdf["domain_classifier_auroc"]["value"], "g", label="test")
            ax.plot(
                resdf.index.values[driftx],  # type: ignore
                resdf["domain_classifier_auroc"]["value"].values[driftx],  # type: ignore
                "dm",
                markersize=3,
                label="drift",
            )
            ax.set_xticks(xticks)
            ax.tick_params(axis="x", labelsize=6)
            ax.tick_params(axis="y", labelsize=6)
            ax.legend(loc="lower left", fontsize=6)
            ax.set_title("Domain Classifier, Drift Detection", fontsize=8)
            ax.set_ylabel("ROC AUC", fontsize=7)
            ax.set_xlabel("Chunk Index", fontsize=7)
            ax.set_ylim((0.0, 1.1))
        return fig
