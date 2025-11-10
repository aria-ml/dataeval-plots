"""Base class and protocol for plotting backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Any, Protocol, cast, overload

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

        plot_type = output.plot_type()

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

    @abstractmethod
    def _plot_image_grid(
        self,
        dataset: Dataset,
        indices: Sequence[int],
        images_per_row: int = 3,
        figsize: tuple[int, int] | None = None,
        show_labels: bool = False,
        show_metadata: bool = False,
        additional_metadata: Sequence[dict[str, Any]] | None = None,
    ) -> Any:
        """Plot image grid - to be implemented by each backend."""
        ...
