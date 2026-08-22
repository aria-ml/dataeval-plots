"""Base class and protocol for plotting backends."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast, overload

import numpy as np
from numpy.typing import NDArray

from dataeval_plots.protocols import (
    Dataset,
    PlottableBalance,
    PlottableDiversity,
    PlottableDriftMVDC,
    PlottableStats,
    PlottableSufficiency,
    PlottableType,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure


class PlottingBackend(Protocol):
    """Protocol that all plotting backends must implement."""

    @overload
    def plot(
        self,
        output: PlottableBalance,
        *,
        figsize: tuple[float, float] | None = None,
        row_labels: Sequence[Any] | NDArray[Any] | None = None,
        col_labels: Sequence[Any] | NDArray[Any] | None = None,
        plot_classwise: bool = False,
    ) -> Any: ...

    @overload
    def plot(
        self,
        output: PlottableDiversity,
        *,
        figsize: tuple[float, float] | None = None,
        row_labels: Sequence[Any] | NDArray[Any] | None = None,
        col_labels: Sequence[Any] | NDArray[Any] | None = None,
        plot_classwise: bool = False,
    ) -> Any: ...

    @overload
    def plot(
        self,
        output: PlottableSufficiency,
        *,
        figsize: tuple[float, float] | None = None,
        class_names: Sequence[str] | None = None,
        show_error_bars: bool = True,
        show_asymptote: bool = True,
        reference_outputs: Sequence[PlottableSufficiency] | PlottableSufficiency | None = None,
    ) -> Any: ...

    @overload
    def plot(
        self,
        output: PlottableStats,
        *,
        figsize: tuple[float, float] | None = None,
        log: bool = True,
        channel_limit: int | None = None,
        channel_index: int | Iterable[int] | None = None,
    ) -> Any: ...

    @overload
    def plot(
        self,
        output: PlottableDriftMVDC,
        *,
        figsize: tuple[float, float] | None = None,
    ) -> Any: ...

    @overload
    def plot(
        self,
        output: Dataset,
        *,
        figsize: tuple[float, float] | None = None,
        indices: Sequence[int],
        images_per_row: int = 3,
        show_labels: bool = False,
        show_metadata: bool = False,
        additional_metadata: Sequence[dict[str, Any]] | None = None,
    ) -> Any: ...

    @overload
    def plot(self, output: PlottableType, *, figsize: tuple[float, float] | None = None, **kwargs: Any) -> Any: ...

    def plot(self, output: PlottableType, *, figsize: tuple[float, float] | None = None, **kwargs: Any) -> Any:
        """
        Plot output using this backend.

        Parameters
        ----------
        output : Plottable
            DataEval output to visualize (must implement Plottable protocol)
        figsize : tuple[float, float] or None, default None
            Figure size in inches (width, height). If None, uses backend defaults.
        **kwargs
            Backend-specific parameters

        Returns
        -------
        Figure
            Backend-specific figure object
        """
        ...

    def project(
        self,
        embeddings: NDArray[Any],
        labels: NDArray[Any] | None = None,
        label_names: Mapping[int, str] | None = None,
        method: str = "pca",
        dimensions: Literal[2, 3] = 2,
        figsize: tuple[float, float] | None = None,
        title: str | None = None,
    ) -> Any:
        """
        Plot projected embeddings as a scatter plot.

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
        Any
            Backend-specific figure object.
        """
        ...

    def project_grid(
        self,
        embeddings_list: Sequence[NDArray[Any]],
        methods: Sequence[str],
        labels: NDArray[Any] | None = None,
        label_names: Mapping[int, str] | None = None,
        dimensions: Literal[2, 3] = 2,
        figsize: tuple[float, float] | None = None,
        title: str | None = None,
    ) -> Any:
        """
        Plot a grid of projected embeddings comparing multiple reduction methods.

        Parameters
        ----------
        embeddings_list : list[NDArray]
            List of reduced embeddings, one per method, each with shape
            ``(N, 2)`` or ``(N, 3)``.
        methods : list[str]
            Names of the reduction methods (one per embeddings entry).
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
        Any
            Backend-specific figure object.
        """
        ...


class BasePlottingBackend(PlottingBackend, ABC):
    """Abstract base class for plotting backends with common routing logic.

    This class provides the routing logic based on plot_type() and delegates
    to abstract methods that subclasses must implement.
    """

    def plot(self, output: PlottableType, *, figsize: tuple[float, float] | None = None, **kwargs: Any) -> Any:
        """
        Route to appropriate plot method based on output plot_type.

        Parameters
        ----------
        output : Plottable
            DataEval output object implementing Plottable protocol
        figsize : tuple[float, float] or None, default None
            Figure size in inches (width, height). If None, uses backend defaults.
        **kwargs
            Plotting parameters

        Returns
        -------
        Any
            Backend-specific figure object(s)

        Raises
        ------
        NotImplementedError
            If plotting not implemented for output type
        """
        if isinstance(output, Dataset):
            return self._plot_image_grid(cast(Dataset, output), figsize=figsize, **kwargs)

        plot_type = output.plot_type if isinstance(output.plot_type, str) else output.plot_type()

        if plot_type == "balance":
            return self._plot_balance(cast(PlottableBalance, output), figsize=figsize, **kwargs)
        if plot_type == "diversity":
            return self._plot_diversity(cast(PlottableDiversity, output), figsize=figsize, **kwargs)
        if plot_type == "sufficiency":
            return self._plot_sufficiency(cast(PlottableSufficiency, output), figsize=figsize, **kwargs)
        if plot_type == "drift_mvdc":
            return self._plot_drift_mvdc(cast(PlottableDriftMVDC, output), figsize=figsize, **kwargs)
        if plot_type == "stats":
            return self._plot_stats(cast(PlottableStats, output), figsize=figsize, **kwargs)

        raise NotImplementedError(f"Plotting not implemented for plot_type '{plot_type}'")

    @abstractmethod
    def _plot_balance(
        self,
        output: PlottableBalance,
        figsize: tuple[float, float] | None = None,
        row_labels: Sequence[Any] | Any | None = None,
        col_labels: Sequence[Any] | Any | None = None,
        plot_classwise: bool = False,
    ) -> Any:
        """Plot balance output."""
        ...

    @abstractmethod
    def _plot_diversity(
        self,
        output: PlottableDiversity,
        figsize: tuple[float, float] | None = None,
        row_labels: Sequence[Any] | Any | None = None,
        col_labels: Sequence[Any] | Any | None = None,
        plot_classwise: bool = False,
    ) -> Any:
        """Plot diversity output."""
        ...

    @abstractmethod
    def _plot_sufficiency(
        self,
        output: PlottableSufficiency,
        figsize: tuple[float, float] | None = None,
        class_names: Sequence[str] | None = None,
        show_error_bars: bool = True,
        show_asymptote: bool = True,
        reference_outputs: Sequence[PlottableSufficiency] | PlottableSufficiency | None = None,
    ) -> Any:
        """Plot sufficiency output."""
        ...

    @abstractmethod
    def _plot_stats(
        self,
        output: PlottableStats,
        figsize: tuple[float, float] | None = None,
        log: bool = True,
        channel_limit: int | None = None,
        channel_index: int | Iterable[int] | None = None,
    ) -> Any:
        """Plot base stats output."""
        ...

    @abstractmethod
    def _plot_drift_mvdc(
        self,
        output: PlottableDriftMVDC,
        figsize: tuple[float, float] | None = None,
    ) -> Any:
        """Plot drift MVDC output."""
        ...

    def _plot_image_grid(
        self,
        dataset: Dataset,
        indices: Sequence[int],
        images_per_row: int = 3,
        figsize: tuple[float, float] | None = None,
        show_labels: bool = False,
        show_metadata: bool = False,
        additional_metadata: Sequence[dict[str, Any]] | None = None,
    ) -> Figure:
        """
        Plot a grid of images from a dataset.

        This is a common implementation used by matplotlib and seaborn backends.
        Subclasses can override this method to provide custom styling.

        Parameters
        ----------
        dataset : Dataset
            MAITE-compatible dataset containing images
        indices : Sequence[int]
            Indices of images to plot from the dataset
        images_per_row : int, default 3
            Number of images to display per row
        figsize : tuple[float, float] or None, default None
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

        from dataeval_plots.backends._shared import (
            format_label_from_target,
            process_dataset_item_for_display,
        )

        # Validate additional_metadata length
        if additional_metadata is not None and len(additional_metadata) != len(indices):
            raise ValueError(
                f"additional_metadata length ({len(additional_metadata)}) must match indices length ({len(indices)})"
            )

        num_images = len(indices)
        num_rows = (num_images + images_per_row - 1) // images_per_row

        # Get index2label mapping if available
        index2label = dataset.metadata.get("index2label") if hasattr(dataset, "metadata") else None

        # Auto-detect figsize if not provided
        if figsize is None:
            # Get first image to determine dimensions
            datum = dataset[indices[0]]
            add_meta = additional_metadata[0] if additional_metadata is not None else None
            first_image, _, _ = process_dataset_item_for_display(
                datum,
                additional_metadata=add_meta,
                index2label=index2label,
            )
            img_height, img_width = first_image.shape[:2]

            # Convert to inches (assuming 100 pixels per inch as default DPI)
            # Add slim borders (5% padding on top/bottom)
            padding_factor = 0.05
            single_img_width = img_width / 100
            single_img_height = img_height / 100 * (1 + 2 * padding_factor)
            # Use max to ensure minimum size of 1 inch to avoid singular matrix errors
            figsize = (
                max(1, int(single_img_width * images_per_row)),
                max(1, int(single_img_height * num_rows)),
            )

        fig, axes = plt.subplots(num_rows, images_per_row, figsize=figsize, squeeze=False)

        # Flatten axes array for easier iteration
        axes_flat = np.asarray(axes).flatten()

        for i, ax in enumerate(axes_flat):
            if i >= num_images:
                ax.set_visible(False)
                continue

            # Get dataset item and process it for display
            datum = dataset[indices[i]]
            add_meta = additional_metadata[i] if additional_metadata is not None else None
            processed_image, target, metadata = process_dataset_item_for_display(
                datum,
                additional_metadata=add_meta,
                index2label=index2label,
            )

            ax.imshow(processed_image)
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

    def project(
        self,
        embeddings: NDArray[Any],
        labels: NDArray[Any] | None = None,
        label_names: Mapping[int, str] | None = None,
        method: str = "pca",
        dimensions: Literal[2, 3] = 2,
        figsize: tuple[float, float] | None = None,
        title: str | None = None,
    ) -> Figure:
        """
        Plot projected embeddings as a 2D or 3D scatter plot.

        This is a default matplotlib-based implementation used by matplotlib
        and seaborn backends. Plotly and Altair backends override this method.

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
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        if title is None:
            title = f"{method.upper()} Projection"

        if dimensions == 3:
            fig = plt.figure(figsize=figsize or (10, 8))
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig, ax = plt.subplots(figsize=figsize or (10, 8))

        self._scatter_on_axis(ax, embeddings, labels, label_names, dimensions, title)
        with warnings.catch_warnings():
            # tight_layout can fail to negotiate margins for 3D axes
            warnings.filterwarnings("ignore", message=".*Tight layout.*", category=UserWarning)
            fig.tight_layout()
        return fig

    def _scatter_on_axis(
        self,
        ax: Any,
        embeddings: NDArray[Any],
        labels: NDArray[Any] | None,
        label_names: Mapping[int, str] | None,
        dimensions: Literal[2, 3],
        subtitle: str,
    ) -> None:
        """Render a single scatter plot on the given axis."""
        if labels is not None:
            unique_labels = np.unique(labels)
            for label in unique_labels:
                mask = labels == label
                name = label_names[int(label)] if label_names and int(label) in label_names else str(label)
                if dimensions == 3:
                    ax.scatter(
                        embeddings[mask, 0], embeddings[mask, 1], embeddings[mask, 2], label=name, alpha=0.7, s=10
                    )
                else:
                    ax.scatter(embeddings[mask, 0], embeddings[mask, 1], label=name, alpha=0.7, s=10)
            ax.legend(fontsize=7, markerscale=0.8)
        else:
            if dimensions == 3:
                ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], alpha=0.7, s=10)
            else:
                ax.scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.7, s=10)

        if dimensions == 3:
            # Use set_*ticklabels (not set_*ticks) to keep axis grid lines
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
        else:
            ax.set_xticks([])
            ax.set_yticks([])
        ax.set_title(subtitle, fontsize=10)

    def project_grid(
        self,
        embeddings_list: Sequence[NDArray[Any]],
        methods: Sequence[str],
        labels: NDArray[Any] | None = None,
        label_names: Mapping[int, str] | None = None,
        dimensions: Literal[2, 3] = 2,
        figsize: tuple[float, float] | None = None,
        title: str | None = None,
    ) -> Figure:
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
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        from dataeval_plots.backends._shared import calculate_subplot_grid

        n = len(methods)
        aspect = figsize[0] / figsize[1] if figsize else None
        rows, cols = calculate_subplot_grid(n, aspect_ratio=aspect)

        if figsize is None:
            figsize = (cols * 5, rows * 4)

        if dimensions == 3:
            fig = plt.figure(figsize=figsize)
            axes = [fig.add_subplot(rows, cols, i + 1, projection="3d") for i in range(n)]
            # Hide unused subplot positions
            for i in range(n, rows * cols):
                ax_empty = fig.add_subplot(rows, cols, i + 1)
                ax_empty.set_visible(False)
        else:
            fig, axes_array = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
            axes = list(np.asarray(axes_array).flatten())
            for ax in axes[n:]:
                ax.set_visible(False)

        for ax, emb, method in zip(axes, embeddings_list, methods):
            self._scatter_on_axis(ax, emb, labels, label_names, dimensions, method.upper())

        if title:
            fig.suptitle(title, fontsize=14)
        with warnings.catch_warnings():
            # tight_layout can fail to negotiate margins for 3D axes
            warnings.filterwarnings("ignore", message=".*Tight layout.*", category=UserWarning)
            fig.tight_layout()
        return fig
