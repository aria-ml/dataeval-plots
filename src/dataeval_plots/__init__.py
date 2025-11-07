"""Plotting backends for DataEval outputs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dataeval_plots._registry import get_backend, register_backend, set_default_backend

if TYPE_CHECKING:
    from dataeval.outputs import Output

__all__ = ["plot", "register_backend", "set_default_backend", "get_backend"]


def plot(output: Output, /, backend: str | None = None, **kwargs: Any) -> Any:
    """
    Plot any DataEval output object.

    Parameters
    ----------
    output : Output
        DataEval output object to visualize
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
