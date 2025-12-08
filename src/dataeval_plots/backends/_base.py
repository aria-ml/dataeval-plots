"""Base class and protocol for plotting backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, Protocol, cast, overload

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
        figsize: tuple[int, int] | None = None,
        row_labels: Sequence[Any] | NDArray[Any] | None = None,
        col_labels: Sequence[Any] | NDArray[Any] | None = None,
        plot_classwise: bool = False,
    ) -> Any: ...

    @overload
    def plot(
        self,
        output: PlottableDiversity,
        *,
        figsize: tuple[int, int] | None = None,
        row_labels: Sequence[Any] | NDArray[Any] | None = None,
        col_labels: Sequence[Any] | NDArray[Any] | None = None,
        plot_classwise: bool = False,
    ) -> Any: ...

    @overload
    def plot(
        self,
        output: PlottableSufficiency,
        *,
        figsize: tuple[int, int] | None = None,
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
        figsize: tuple[int, int] | None = None,
        log: bool = True,
        channel_limit: int | None = None,
        channel_index: int | Iterable[int] | None = None,
    ) -> Any: ...

    @overload
    def plot(
        self,
        output: PlottableDriftMVDC,
        *,
        figsize: tuple[int, int] | None = None,
    ) -> Any: ...

    @overload
    def plot(
        self,
        output: Dataset,
        *,
        figsize: tuple[int, int] | None = None,
        indices: Sequence[int],
        images_per_row: int = 3,
        show_labels: bool = False,
        show_metadata: bool = False,
        additional_metadata: Sequence[dict[str, Any]] | None = None,
    ) -> Any: ...

    @overload
    def plot(self, output: PlottableType, *, figsize: tuple[int, int] | None = None, **kwargs: Any) -> Any: ...

    def plot(self, output: PlottableType, *, figsize: tuple[int, int] | None = None, **kwargs: Any) -> Any:
        """
        Plot output using this backend.

        Parameters
        ----------
        output : Plottable
            DataEval output to visualize (must implement Plottable protocol)
        figsize : tuple[int, int] or None, default None
            Figure size in inches (width, height). If None, uses backend defaults.
        **kwargs
            Backend-specific parameters

        Returns
        -------
        Figure
            Backend-specific figure object
        """
        ...


class BasePlottingBackend(PlottingBackend, ABC):
    """Abstract base class for plotting backends with common routing logic.

    This class provides the routing logic based on plot_type() and delegates
    to abstract methods that subclasses must implement.
    """

    def plot(self, output: PlottableType, *, figsize: tuple[int, int] | None = None, **kwargs: Any) -> Any:
        """
        Route to appropriate plot method based on output plot_type.

        Parameters
        ----------
        output : Plottable
            DataEval output object implementing Plottable protocol
        figsize : tuple[int, int] or None, default None
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
        figsize: tuple[int, int] | None = None,
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
        figsize: tuple[int, int] | None = None,
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
        figsize: tuple[int, int] | None = None,
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
        figsize: tuple[int, int] | None = None,
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
        figsize: tuple[int, int] | None = None,
    ) -> Any:
        """Plot drift MVDC output."""
        ...

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
