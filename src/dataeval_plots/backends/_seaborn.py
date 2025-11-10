"""Seaborn plotting backend."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

from numpy.typing import NDArray

from dataeval_plots.backends._base import BasePlottingBackend
from dataeval_plots.backends._shared import (
    CHANNELWISE_METRICS,
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
)
from dataeval_plots.protocols import (
    Dataset,
    PlottableBalance,
    PlottableDiversity,
    PlottableDriftMVDC,
    PlottableStats,
    PlottableSufficiency,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure


class SeabornBackend(BasePlottingBackend):
    """Seaborn implementation of plotting backend with enhanced styling."""

    def _plot_balance(
        self,
        output: PlottableBalance,
        figsize: tuple[int, int] | None = None,
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
        import pandas as pd
        import seaborn as sns

        # Use shared helper to prepare data
        data, row_labels, col_labels, xlabel, ylabel, title = prepare_balance_data(
            output, row_labels, col_labels, plot_classwise
        )

        # Create DataFrame for seaborn
        df = pd.DataFrame(data, index=row_labels, columns=col_labels)  # type: ignore[arg-type]

        # Create figure with seaborn style
        if figsize is None:
            figsize = (10, 10)
        fig, ax = plt.subplots(figsize=figsize)

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
        figsize: tuple[int, int] | None = None,
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
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns

        # Use shared helper to prepare data
        data, row_labels, col_labels, xlabel, ylabel, title, method_name = prepare_diversity_data(
            output, row_labels, col_labels, plot_classwise
        )

        if plot_classwise:
            # Create DataFrame for seaborn
            df = pd.DataFrame(data, index=row_labels, columns=col_labels)  # type: ignore[arg-type]

            if figsize is None:
                figsize = (10, 10)
            fig, ax = plt.subplots(figsize=figsize)

            sns.heatmap(
                df,
                annot=True,
                fmt=".2f",
                cmap="viridis",
                vmin=0,
                vmax=1,
                cbar_kws={"label": f"Normalized {method_name} Index"},
                linewidths=0.5,
                linecolor="lightgray",
                ax=ax,
            )

            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(title, fontsize=14, pad=20)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        else:
            # Bar chart for diversity indices
            df = pd.DataFrame({"factor": row_labels, "diversity": output.diversity_index})

            if figsize is None:
                figsize = (10, 8)
            fig, ax = plt.subplots(figsize=figsize)

            # Use seaborn barplot
            sns.barplot(data=df, x="factor", y="diversity", hue="factor", palette="viridis", legend=False, ax=ax)

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
        figsize: tuple[int, int] | None = None,
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
            figsize=figsize,
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

    def _plot_stats(
        self,
        output: PlottableStats,
        figsize: tuple[int, int] | None = None,
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
            if figsize is None:
                figsize = (cols * 3 + 1, rows * 3)
            fig, axs = plt.subplots(rows, 3, figsize=figsize)
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
            # Multi-channel histogram - use shared constant
            data_keys = [key for key in factors if key in CHANNELWISE_METRICS]

            num_metrics = len(data_keys)
            rows = math.ceil(num_metrics / 3)
            cols = min(num_metrics, 3)
            if figsize is None:
                figsize = (cols * 3 + 1, rows * 3)
            fig, axs = plt.subplots(rows, 3, figsize=figsize)
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
        figsize: tuple[int, int] | None = None,
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

        # Use shared helper to prepare drift data
        resdf, trndf, tstdf, driftx, is_sufficient = prepare_drift_data(output)

        if figsize is None:
            figsize = (10, 6)
        fig, ax = plt.subplots(dpi=300, figsize=figsize)

        if not is_sufficient:
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

        if is_sufficient and np.size(driftx) > 2:
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

    def _plot_image_grid(
        self,
        dataset: Dataset,
        indices: Sequence[int],
        images_per_row: int = 3,
        figsize: tuple[int, int] | None = None,
        show_labels: bool = False,
        show_metadata: bool = False,
        additional_metadata: Sequence[dict[str, Any]] | None = None,
    ) -> Figure:
        """
        Plot a grid of images from a dataset with Seaborn styling.

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
        matplotlib.figure.Figure

        Raises
        ------
        ValueError
            If additional_metadata length doesn't match indices length
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns

        # Validate additional_metadata length
        if additional_metadata is not None and len(additional_metadata) != len(indices):
            raise ValueError(
                f"additional_metadata length ({len(additional_metadata)}) must match indices length ({len(indices)})"
            )

        # Set seaborn style for consistent appearance
        sns.set_style("whitegrid")

        num_images = len(indices)
        num_rows = (num_images + images_per_row - 1) // images_per_row

        if figsize is None:
            figsize = (10, 10)
        fig, axes = plt.subplots(num_rows, images_per_row, figsize=figsize)

        # Flatten axes array for easier iteration
        axes_flat = np.asarray(axes).flatten()

        # Get index2label mapping if available
        index2label = dataset.metadata.get("index2label") if hasattr(dataset, "metadata") else None

        for i, ax in enumerate(axes_flat):
            if i >= num_images:
                ax.set_visible(False)
                continue

            # Get dataset item and parse it
            datum = dataset[indices[i]]
            image, target, metadata = parse_dataset_item(datum)

            # Merge with additional metadata if provided
            if additional_metadata is not None:
                metadata = merge_metadata(metadata, additional_metadata[i])

            # Convert image to HWC format
            image_hwc = image_to_hwc(image)

            # Check if we have object detection targets and should draw boxes
            boxes, labels, scores = extract_boxes_and_labels(target)
            if boxes is not None and len(boxes) > 0:
                # Ensure image is uint8 for OpenCV
                image_uint8 = normalize_image_to_uint8(image_hwc)
                # Draw bounding boxes
                image_with_boxes = draw_bounding_boxes(
                    image_uint8,
                    boxes,
                    labels,
                    scores,
                    index2label,
                )
                ax.imshow(image_with_boxes)
            else:
                ax.imshow(image_hwc)

            ax.axis("off")

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

            # Set title if we have any parts
            if title_parts:
                ax.set_title("\n".join(title_parts), fontsize=8, pad=3)

        plt.tight_layout()
        return fig
