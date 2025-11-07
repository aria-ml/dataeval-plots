"""Seaborn plotting backend."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

from numpy.typing import NDArray

from dataeval_plots.backends._base import BasePlottingBackend
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
    from matplotlib.figure import Figure


class SeabornBackend(BasePlottingBackend):
    """Seaborn implementation of plotting backend with enhanced styling."""

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
        import seaborn as sns

        if images is None:
            raise ValueError("images parameter is required for coverage plotting")

        if np.max(output.uncovered_indices) > len(images):
            raise ValueError(
                f"Uncovered indices {output.uncovered_indices} specify images "
                f"unavailable in the provided number of images {len(images)}."
            )

        # Set seaborn style
        sns.set_style("white")

        # Determine which images to plot
        selected_indices = output.uncovered_indices[:top_k]
        num_images = min(top_k, len(selected_indices))

        rows = int(np.ceil(num_images / 3))
        cols = min(3, num_images)
        fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))

        # Flatten axes using numpy array explicitly for compatibility
        axs_flat = np.asarray(axs).flatten()

        for image, ax in zip(images[:num_images], axs_flat):
            ax.imshow(self.image_to_hwc(image))
            ax.axis("off")
            # Add seaborn-style border
            sns.despine(ax=ax, left=True, bottom=True)

        for ax in axs_flat[num_images:]:
            ax.axis("off")

        fig.suptitle(f"Top {num_images} Uncovered Images", fontsize=14, y=1.02)
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
        Plot a heatmap of balance information using Seaborn.

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
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns

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

        # Create DataFrame for seaborn
        df = pd.DataFrame(data, index=row_labels, columns=col_labels)  # type: ignore[arg-type]

        # Create figure with seaborn style
        fig, ax = plt.subplots(figsize=(10, 10))

        # Create heatmap with seaborn
        sns.heatmap(
            df,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            vmin=0,
            vmax=1,
            cbar_kws={"label": "Normalized Mutual Information"},
            linewidths=0.5,
            linecolor="lightgray",
            ax=ax,
        )

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, pad=20)

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        fig.tight_layout()
        return fig

    def _plot_diversity(
        self,
        output: PlottableDiversity,
        row_labels: Sequence[Any] | NDArray[Any] | None = None,
        col_labels: Sequence[Any] | NDArray[Any] | None = None,
        plot_classwise: bool = False,
    ) -> Figure:
        """
        Plot a heatmap or bar chart of diversity information using Seaborn.

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
        from dataclasses import asdict

        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns

        if plot_classwise:
            if row_labels is None:
                row_labels = output.class_names
            if col_labels is None:
                col_labels = output.factor_names

            data = output.classwise
            method = asdict(output.meta())["arguments"]["method"].title()

            # Create DataFrame for seaborn
            df = pd.DataFrame(data, index=row_labels, columns=col_labels)  # type: ignore[arg-type]

            fig, ax = plt.subplots(figsize=(10, 10))

            sns.heatmap(
                df,
                annot=True,
                fmt=".2f",
                cmap="viridis",
                vmin=0,
                vmax=1,
                cbar_kws={"label": f"Normalized {method} Index"},
                linewidths=0.5,
                linecolor="lightgray",
                ax=ax,
            )

            ax.set_xlabel("Factors", fontsize=12)
            ax.set_ylabel("Class", fontsize=12)
            ax.set_title("Classwise Diversity", fontsize=14, pad=20)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        else:
            # Bar chart for diversity indices
            heat_labels = ["class_labels"] + list(output.factor_names)
            df = pd.DataFrame({"factor": heat_labels, "diversity": output.diversity_index})

            fig, ax = plt.subplots(figsize=(10, 8))

            # Use seaborn barplot
            sns.barplot(data=df, x="factor", y="diversity", palette="viridis", ax=ax)

            ax.set_xlabel("Factors", fontsize=12)
            ax.set_ylabel("Diversity Index", fontsize=12)
            ax.set_title("Diversity Index by Factor", fontsize=14, pad=20)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            sns.despine(ax=ax)

        fig.tight_layout()
        return fig

    def _plot_sufficiency(
        self,
        output: PlottableSufficiency,
        class_names: Sequence[str] | None = None,
        show_error_bars: bool = True,
        show_asymptote: bool = True,
        reference_outputs: Sequence[PlottableSufficiency] | PlottableSufficiency | None = None,
    ) -> list[Figure]:
        """
        Plotting function for data sufficiency tasks with Seaborn styling.

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
        list[Figure]
            List of Figures for each measure
        """
        import seaborn as sns

        # Set seaborn style for all sufficiency plots
        sns.set_style("whitegrid")
        sns.set_palette("husl")

        from dataeval_plots.backends._matplotlib import MatplotlibBackend

        figures = MatplotlibBackend()._plot_sufficiency(
            output,
            class_names=class_names,
            show_error_bars=show_error_bars,
            show_asymptote=show_asymptote,
            reference_outputs=reference_outputs,
        )

        # Enhance each figure with seaborn styling
        for fig in figures:
            for ax in fig.axes:
                sns.despine(ax=ax, left=False, bottom=False)

        return figures

    def _plot_base_stats(
        self,
        output: PlottableBaseStats,
        log: bool = True,
        channel_limit: int | None = None,
        channel_index: int | Iterable[int] | None = None,
    ) -> Figure:
        """
        Plots the statistics as a set of histograms using Seaborn.

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
        import math

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        from matplotlib.figure import Figure

        # Set seaborn style
        sns.set_style("whitegrid")

        max_channels, ch_mask = output._get_channels(channel_limit, channel_index)
        factors = output.factors(exclude_constant=True)

        if not factors:
            return Figure()

        if max_channels == 1:
            # Single channel histogram
            num_metrics = len(factors)
            rows = math.ceil(num_metrics / 3)
            cols = min(num_metrics, 3)
            fig, axs = plt.subplots(rows, 3, figsize=(cols * 3 + 1, rows * 3))
            axs_flat = np.asarray(axs).flatten()

            for ax, (metric_name, metric_values) in zip(axs_flat, factors.items()):
                # Use seaborn histplot
                sns.histplot(
                    metric_values.flatten(),
                    bins=20,
                    log_scale=(False, log),
                    ax=ax,
                    kde=False,
                    color=sns.color_palette("husl")[0],
                )
                ax.set_title(metric_name, fontsize=10)
                ax.set_ylabel("Counts", fontsize=9)
                ax.set_xlabel("Values", fontsize=9)
                sns.despine(ax=ax)

            for ax in axs_flat[num_metrics:]:
                ax.axis("off")
                ax.set_visible(False)

        else:
            # Multi-channel histogram
            channelwise_metrics = [
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
            data_keys = [key for key in factors if key in channelwise_metrics]

            num_metrics = len(data_keys)
            rows = math.ceil(num_metrics / 3)
            cols = min(num_metrics, 3)
            fig, axs = plt.subplots(rows, 3, figsize=(cols * 3 + 1, rows * 3))
            axs_flat = np.asarray(axs).flatten()

            for ax, metric_name in zip(axs_flat, data_keys):
                # Reshape for channel-wise data
                data = factors[metric_name][ch_mask].reshape(-1, max_channels)

                # Create DataFrame for seaborn
                plot_data = []
                for ch_idx in range(max_channels):
                    for val in data[:, ch_idx]:
                        plot_data.append({"value": val, "channel": f"Channel {ch_idx}"})

                df = pd.DataFrame(plot_data)

                # Use seaborn histplot with hue
                sns.histplot(
                    data=df,
                    x="value",
                    hue="channel",
                    bins=20,
                    log_scale=(False, log),
                    ax=ax,
                    kde=False,
                    stat="density",
                    common_norm=False,
                    alpha=0.6,
                )
                ax.set_title(metric_name, fontsize=10)
                ax.set_ylabel("Density", fontsize=9)
                ax.set_xlabel("Values", fontsize=9)
                sns.despine(ax=ax)

            for ax in axs_flat[num_metrics:]:
                ax.axis("off")
                ax.set_visible(False)

        fig.tight_layout()
        return fig

    def _plot_drift_mvdc(
        self,
        output: PlottableDriftMVDC,
    ) -> Figure:
        """
        Render the roc_auc metric over the train/test data using Seaborn styling.

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
        import seaborn as sns

        # Set seaborn style
        sns.set_style("whitegrid")

        fig, ax = plt.subplots(dpi=300, figsize=(10, 6))
        resdf = output.to_dataframe()

        if resdf.shape[0] < 3:
            ax.text(
                0.5,
                0.5,
                "Insufficient data for drift detection plot",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return fig

        xticks = np.arange(resdf.shape[0])
        trndf = resdf[resdf["chunk"]["period"] == "reference"]
        tstdf = resdf[resdf["chunk"]["period"] == "analysis"]

        # Get local indices for drift markers
        driftx = np.where(resdf["domain_classifier_auroc"]["alert"].values)  # type: ignore

        if np.size(driftx) > 2:
            # Use seaborn color palette
            colors = sns.color_palette("husl", 4)

            ax.plot(
                resdf.index,
                resdf["domain_classifier_auroc"]["upper_threshold"],
                "--",
                color="red",
                label="Threshold Upper",
                linewidth=2,
            )
            ax.plot(
                resdf.index,
                resdf["domain_classifier_auroc"]["lower_threshold"],
                "--",
                color="red",
                label="Threshold Lower",
                linewidth=2,
            )
            ax.plot(
                trndf.index,
                trndf["domain_classifier_auroc"]["value"],
                "-",
                color=colors[0],
                label="Train",
                linewidth=2,
            )
            ax.plot(
                tstdf.index,
                tstdf["domain_classifier_auroc"]["value"],
                "-",
                color=colors[1],
                label="Test",
                linewidth=2,
            )
            ax.plot(
                resdf.index.values[driftx],  # type: ignore
                resdf["domain_classifier_auroc"]["value"].values[driftx],  # type: ignore
                "D",
                color="magenta",
                markersize=6,
                label="Drift",
            )

            ax.set_xticks(xticks)
            ax.tick_params(axis="x", labelsize=8)
            ax.tick_params(axis="y", labelsize=8)
            ax.legend(loc="lower left", fontsize=8, frameon=True)
            ax.set_title("Domain Classifier, Drift Detection", fontsize=12, pad=15)
            ax.set_ylabel("ROC AUC", fontsize=10)
            ax.set_xlabel("Chunk Index", fontsize=10)
            ax.set_ylim((0.0, 1.1))
            sns.despine(ax=ax)

        fig.tight_layout()
        return fig
