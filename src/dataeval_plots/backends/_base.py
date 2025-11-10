"""Base class and protocol for plotting backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Any, Protocol, cast, overload

from numpy.typing import NDArray

from dataeval_plots.protocols import (
    Dataset,
    Indexable,
    PlottableBalance,
    PlottableBaseStats,
    PlottableCoverage,
    PlottableDiversity,
    PlottableDriftMVDC,
    PlottableSufficiency,
    PlottableType,
)


class PlottingBackend(Protocol):
    """Protocol that all plotting backends must implement."""

    @overload
    def plot(
        self,
        output: PlottableCoverage,
        *,
        dataset: Dataset | None = None,
        top_k: int = 6,
    ) -> Any: ...

    @overload
    def plot(
        self,
        output: PlottableBalance,
        *,
        row_labels: Sequence[Any] | NDArray[Any] | None = None,
        col_labels: Sequence[Any] | NDArray[Any] | None = None,
        plot_classwise: bool = False,
    ) -> Any: ...

    @overload
    def plot(
        self,
        output: PlottableDiversity,
        *,
        row_labels: Sequence[Any] | NDArray[Any] | None = None,
        col_labels: Sequence[Any] | NDArray[Any] | None = None,
        plot_classwise: bool = False,
    ) -> Any: ...

    @overload
    def plot(
        self,
        output: PlottableSufficiency,
        *,
        class_names: Sequence[str] | None = None,
        show_error_bars: bool = True,
        show_asymptote: bool = True,
        reference_outputs: Sequence[PlottableSufficiency] | PlottableSufficiency | None = None,
    ) -> Any: ...

    @overload
    def plot(
        self,
        output: PlottableBaseStats,
        *,
        log: bool = True,
        channel_limit: int | None = None,
        channel_index: int | Iterable[int] | None = None,
    ) -> Any: ...

    @overload
    def plot(
        self,
        output: PlottableDriftMVDC,
    ) -> Any: ...

    @overload
    def plot(
        self,
        output: Dataset,
        *,
        indices: Sequence[int],
        images_per_row: int = 3,
        figsize: tuple[int, int] = (10, 10),
    ) -> Any: ...

    @overload
    def plot(self, output: PlottableType, **kwargs: Any) -> Any: ...

    def plot(self, output: PlottableType, **kwargs: Any) -> Any:
        """
        Plot output using this backend.

        Parameters
        ----------
        output : Plottable
            DataEval output to visualize (must implement Plottable protocol)
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

    def plot(self, output: PlottableType, **kwargs: Any) -> Any:
        """
        Route to appropriate plot method based on output plot_type.

        Parameters
        ----------
        output : Plottable
            DataEval output object implementing Plottable protocol
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
            return self._plot_image_grid(cast(Dataset, output), **kwargs)

        plot_type = output.plot_type()

        if plot_type == "coverage":
            return self._plot_coverage(cast(PlottableCoverage, output), **kwargs)
        if plot_type == "balance":
            return self._plot_balance(cast(PlottableBalance, output), **kwargs)
        if plot_type == "diversity":
            return self._plot_diversity(cast(PlottableDiversity, output), **kwargs)
        if plot_type == "sufficiency":
            return self._plot_sufficiency(cast(PlottableSufficiency, output), **kwargs)
        if plot_type == "base_stats":
            return self._plot_base_stats(cast(PlottableBaseStats, output), **kwargs)
        if plot_type == "drift_mvdc":
            return self._plot_drift_mvdc(cast(PlottableDriftMVDC, output), **kwargs)

        raise NotImplementedError(f"Plotting not implemented for plot_type '{plot_type}'")

    @abstractmethod
    def _plot_coverage(
        self,
        output: PlottableCoverage,
        images: Indexable | None = None,
        top_k: int = 6,
    ) -> Any:
        """Plot coverage output."""
        ...

    @abstractmethod
    def _plot_balance(
        self,
        output: PlottableBalance,
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
        class_names: Sequence[str] | None = None,
        show_error_bars: bool = True,
        show_asymptote: bool = True,
        reference_outputs: Sequence[PlottableSufficiency] | PlottableSufficiency | None = None,
    ) -> Any:
        """Plot sufficiency output."""
        ...

    @abstractmethod
    def _plot_base_stats(
        self,
        output: PlottableBaseStats,
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
    ) -> Any:
        """Plot drift MVDC output."""
        ...

    @abstractmethod
    def _plot_image_grid(
        self,
        dataset: Dataset,
        indices: Sequence[int],
        images_per_row: int,
        figsize: tuple[int, int],
    ) -> Any:
        """Plot image grid - to be implemented by each backend."""
        ...
