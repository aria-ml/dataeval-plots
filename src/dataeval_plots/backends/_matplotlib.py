"""Matplotlib plotting backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from dataeval.outputs import Output


class MatplotlibBackend:
    """Matplotlib implementation of plotting backend."""

    def plot(self, output: Output, **kwargs: Any) -> Figure | list[Figure]:
        """
        Route to appropriate plot method based on output type.

        Parameters
        ----------
        output : Output
            DataEval output object
        **kwargs
            Plotting parameters

        Returns
        -------
        Figure or list[Figure]
            Matplotlib figure object(s)

        Raises
        ------
        NotImplementedError
            If plotting not implemented for output type
        """
        # Import all output types
        from dataeval.outputs import (
            BalanceOutput,
            BaseStatsOutput,
            CoverageOutput,
            DiversityOutput,
            DriftMVDCOutput,
            SufficiencyOutput,
        )

        # Route to appropriate plotting function
        if isinstance(output, CoverageOutput):
            return self._plot_coverage(output, **kwargs)
        elif isinstance(output, BalanceOutput):
            return self._plot_balance(output, **kwargs)
        elif isinstance(output, DiversityOutput):
            return self._plot_diversity(output, **kwargs)
        elif isinstance(output, SufficiencyOutput):
            return self._plot_sufficiency(output, **kwargs)
        elif isinstance(output, BaseStatsOutput):
            return self._plot_base_stats(output, **kwargs)
        elif isinstance(output, DriftMVDCOutput):
            return self._plot_drift_mvdc(output, **kwargs)
        else:
            raise NotImplementedError(f"Plotting not implemented for {type(output).__name__}")

    def _plot_coverage(
        self,
        output: Any,  # CoverageOutput
        images: Any = None,  # Images | Dataset
        top_k: int = 6,
    ) -> Figure:
        """
        Plot the top k images together for visualization.

        Parameters
        ----------
        output : CoverageOutput
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

        from dataeval.data._images import Images
        from dataeval.protocols import Dataset
        from dataeval.utils._array import as_numpy, channels_first_to_last

        if images is None:
            raise ValueError("images parameter is required for coverage plotting")

        images_obj = Images(images) if isinstance(images, Dataset) else images
        if np.max(output.uncovered_indices) > len(images_obj):
            raise ValueError(
                f"Uncovered indices {output.uncovered_indices} specify images "
                f"unavailable in the provided number of images {len(images_obj)}."
            )

        # Determine which images to plot
        selected_indices = output.uncovered_indices[:top_k]

        # Plot the images
        num_images = min(top_k, len(selected_indices))

        rows = int(np.ceil(num_images / 3))
        cols = min(3, num_images)
        fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))

        # Flatten axes using numpy array explicitly for compatibility
        axs_flat = np.asarray(axs).flatten()

        for image, ax in zip(images_obj[:num_images], axs_flat):
            image = channels_first_to_last(as_numpy(image))
            ax.imshow(image)
            ax.axis("off")

        for ax in axs_flat[num_images:]:
            ax.axis("off")

        fig.tight_layout()
        return fig

    def _plot_balance(
        self,
        output: Any,  # BalanceOutput
        row_labels: Any = None,  # Sequence[Any] | NDArray[Any] | None
        col_labels: Any = None,  # Sequence[Any] | NDArray[Any] | None
        plot_classwise: bool = False,
    ) -> Figure:
        """
        Plot a heatmap of balance information.

        Parameters
        ----------
        output : BalanceOutput
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

        from dataeval_plots._utils import heatmap

        if plot_classwise:
            if row_labels is None:
                row_labels = output.class_names
            if col_labels is None:
                col_labels = output.factor_names

            fig = heatmap(
                output.classwise,
                row_labels,
                col_labels,
                xlabel="Factors",
                ylabel="Class",
                cbarlabel="Normalized Mutual Information",
            )
        else:
            # Combine balance and factors results
            data = np.concatenate(
                [
                    output.balance[np.newaxis, 1:],
                    output.factors,
                ],
                axis=0,
            )
            # Create a mask for the upper triangle of the symmetrical array, ignoring the diagonal
            mask = np.triu(data + 1, k=0) < 1
            # Finalize the data for the plot, last row is last factor x last factor so it gets dropped
            heat_data = np.where(mask, np.nan, data)[:-1]
            # Creating label array for heat map axes
            heat_labels = output.factor_names

            if row_labels is None:
                row_labels = heat_labels[:-1]
            if col_labels is None:
                col_labels = heat_labels[1:]

            fig = heatmap(heat_data, row_labels, col_labels, cbarlabel="Normalized Mutual Information")

        return fig

    def _plot_diversity(
        self,
        output: Any,  # DiversityOutput
        row_labels: Any = None,  # Sequence[Any] | NDArray[Any] | None
        col_labels: Any = None,  # Sequence[Any] | NDArray[Any] | None
        plot_classwise: bool = False,
    ) -> Figure:
        """
        Plot a heatmap of diversity information.

        Parameters
        ----------
        output : DiversityOutput
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

        from dataeval_plots._utils import heatmap

        if plot_classwise:
            if row_labels is None:
                row_labels = output.class_names
            if col_labels is None:
                col_labels = output.factor_names

            fig = heatmap(
                output.classwise,
                row_labels,
                col_labels,
                xlabel="Factors",
                ylabel="Class",
                cbarlabel=f"Normalized {asdict(output.meta())['arguments']['method'].title()} Index",
            )

        else:
            # Creating label array for heat map axes
            fig, ax = plt.subplots(figsize=(8, 8))
            heat_labels = ["class_labels"] + list(output.factor_names)
            ax.bar(heat_labels, output.diversity_index)
            ax.set_xlabel("Factors")
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            fig.tight_layout()

        return fig

    def _plot_sufficiency(
        self,
        output: Any,  # SufficiencyOutput
        class_names: Any = None,  # Sequence[str] | None
        show_error_bars: bool = True,
        show_asymptote: bool = True,
        reference_outputs: Any = None,  # Sequence[SufficiencyOutput] | SufficiencyOutput | None
    ) -> list[Figure]:
        """
        Plotting function for data sufficiency tasks.

        Parameters
        ----------
        output : SufficiencyOutput
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
        from dataeval_plots._sufficiency import plot_sufficiency

        return plot_sufficiency(
            output,
            class_names=class_names,
            show_error_bars=show_error_bars,
            show_asymptote=show_asymptote,
            reference_outputs=reference_outputs,
        )

    def _plot_base_stats(
        self,
        output: Any,  # BaseStatsOutput
        log: bool = True,
        channel_limit: int | None = None,
        channel_index: Any = None,  # int | Iterable[int] | None
    ) -> Figure:
        """
        Plots the statistics as a set of histograms.

        Parameters
        ----------
        output : BaseStatsOutput
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

        from dataeval_plots._utils import channel_histogram_plot, histogram_plot

        max_channels, ch_mask = output._get_channels(channel_limit, channel_index)
        factors = output.factors(exclude_constant=True)
        if not factors:
            return Figure()
        if max_channels == 1:
            return histogram_plot(factors, log)
        return channel_histogram_plot(factors, log, max_channels, ch_mask)

    def _plot_drift_mvdc(
        self,
        output: Any,  # DriftMVDCOutput
    ) -> Figure:
        """
        Render the roc_auc metric over the train/test data in relation to the threshold.

        Parameters
        ----------
        output : DriftMVDCOutput
            The drift MVDC output object to plot

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(dpi=300)
        resdf = output.to_dataframe()
        xticks = np.arange(resdf.shape[0])
        trndf = resdf[resdf["chunk"]["period"] == "reference"]
        tstdf = resdf[resdf["chunk"]["period"] == "analysis"]
        # Get local indices for drift markers
        driftx = np.where(resdf["domain_classifier_auroc"]["alert"].values)  # type: ignore
        if np.size(driftx) > 2:
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
