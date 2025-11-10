"""Plotting backends for DataEval outputs."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, overload

from numpy.typing import NDArray

from dataeval_plots._registry import (
    get_available_backends,
    get_backend,
    register_backend,
    set_default_backend,
)
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

__all__ = [
    "plot",
    "register_backend",
    "set_default_backend",
    "get_backend",
    "get_available_backends",
]


@overload
def plot(
    output: PlottableCoverage,
    /,
    backend: str | None = None,
    *,
    images: Indexable | None = None,
    top_k: int = 6,
) -> Any: ...


@overload
def plot(
    output: PlottableBalance,
    /,
    backend: str | None = None,
    *,
    row_labels: Sequence[Any] | NDArray[Any] | None = None,
    col_labels: Sequence[Any] | NDArray[Any] | None = None,
    plot_classwise: bool = False,
) -> Any: ...


@overload
def plot(
    output: PlottableDiversity,
    /,
    backend: str | None = None,
    *,
    row_labels: Sequence[Any] | NDArray[Any] | None = None,
    col_labels: Sequence[Any] | NDArray[Any] | None = None,
    plot_classwise: bool = False,
) -> Any: ...


@overload
def plot(
    output: PlottableSufficiency,
    /,
    backend: str | None = None,
    *,
    class_names: Sequence[str] | None = None,
    show_error_bars: bool = True,
    show_asymptote: bool = True,
    reference_outputs: Sequence[PlottableSufficiency] | PlottableSufficiency | None = None,
) -> Any: ...


@overload
def plot(
    output: PlottableBaseStats,
    /,
    backend: str | None = None,
    *,
    log: bool = True,
    channel_limit: int | None = None,
    channel_index: int | Iterable[int] | None = None,
) -> Any: ...


@overload
def plot(
    output: PlottableDriftMVDC,
    /,
    backend: str | None = None,
) -> Any: ...


@overload
def plot(
    output: Dataset,
    /,
    backend: str | None = None,
    *,
    indices: Sequence[int],
    images_per_row: int = 3,
    figsize: tuple[int, int] = (10, 10),
) -> Any: ...


@overload
def plot(
    output: PlottableType,
    /,
    backend: str | None = None,
    **kwargs: Any,
) -> Any: ...


def plot(output: PlottableType, /, backend: str | None = None, **kwargs: Any) -> Any:
    """
    Plot any DataEval output object.

    Parameters
    ----------
    output : Plottable
        DataEval output object to visualize (must implement Plottable protocol)
    backend : str or None, default None
        Plotting backend ('matplotlib', 'seaborn', 'plotly', 'altair').
        If None, uses default backend.
    **kwargs
        Backend-specific plotting parameters

    Returns
    -------
    Figure
        Backend-specific figure object

    Raises
    ------
    ImportError
        If backend dependencies are not installed
    NotImplementedError
        If plotting is not implemented for the given output type

    Examples
    --------
    >>> from dataeval_plots import plot
    >>> from dataeval.metrics.bias import coverage
    >>> result = coverage(embeddings)
    >>> fig = plot(result, images=dataset, top_k=6)
    >>> fig.savefig("coverage.png")

    >>> # Use a different backend
    >>> plot(result, backend="seaborn", images=dataset)

    >>> # Set default backend
    >>> from dataeval_plots import set_default_backend
    >>> set_default_backend("seaborn")
    >>> plot(result, images=dataset)  # Uses seaborn
    """
    plotting_backend = get_backend(backend)
    return plotting_backend.plot(output, **kwargs)
