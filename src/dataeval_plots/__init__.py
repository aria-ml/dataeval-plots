"""Plotting backends for DataEval outputs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dataeval_plots._registry import (
    get_available_backends,
    get_backend,
    register_backend,
    set_default_backend,
)
from dataeval_plots.backends._shared import MethodType
from dataeval_plots.protocols import (
    Dataset,
    PlottableBalance,
    PlottableDiversity,
    PlottableDriftMVDC,
    PlottableStats,
    PlottableSufficiency,
    PlottableType,
)

__all__ = [
    "plot",
    "project",
    "register_backend",
    "set_default_backend",
    "get_backend",
    "get_available_backends",
]


@overload
def plot(
    output: PlottableBalance,
    /,
    figsize: tuple[int, int] | None = None,
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
    figsize: tuple[int, int] | None = None,
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
    figsize: tuple[int, int] | None = None,
    backend: str | None = None,
    *,
    class_names: Sequence[str] | None = None,
    show_error_bars: bool = True,
    show_asymptote: bool = True,
    reference_outputs: Sequence[PlottableSufficiency] | PlottableSufficiency | None = None,
) -> Any: ...


@overload
def plot(
    output: PlottableStats,
    /,
    figsize: tuple[int, int] | None = None,
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
    figsize: tuple[int, int] | None = None,
    backend: str | None = None,
) -> Any: ...


@overload
def plot(
    output: Dataset,
    /,
    figsize: tuple[int, int] | None = None,
    backend: str | None = None,
    *,
    indices: Sequence[int],
    images_per_row: int = 3,
    show_labels: bool = False,
    show_metadata: bool = False,
    additional_metadata: Sequence[dict[str, Any]] | None = None,
) -> Any: ...


@overload
def plot(
    output: PlottableType,
    /,
    figsize: tuple[int, int] | None = None,
    backend: str | None = None,
    **kwargs: Any,
) -> Any: ...


def plot(
    output: PlottableType, /, figsize: tuple[int, int] | None = None, backend: str | None = None, **kwargs: Any
) -> Any:
    """
    Plot any DataEval output object.

    Parameters
    ----------
    output : Plottable
        DataEval output object to visualize (must implement Plottable protocol)
    figsize : tuple[int, int] or None, default None
        Figure size in inches (width, height). If None, uses backend defaults.
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

    >>> # Specify custom figure size
    >>> plot(result, figsize=(12, 8), images=dataset)

    >>> # Use a different backend
    >>> plot(result, backend="seaborn", images=dataset)

    >>> # Set default backend
    >>> from dataeval_plots import set_default_backend
    >>> set_default_backend("seaborn")
    >>> plot(result, images=dataset)  # Uses seaborn
    """
    plotting_backend = get_backend(backend)
    return plotting_backend.plot(output, figsize=figsize, **kwargs)


def project(
    embeddings: ArrayLike,
    *,
    method: MethodType | Sequence[MethodType] | None = "pca",
    dimensions: Literal[2, 3] = 2,
    labels: ArrayLike | None = None,
    label_names: Mapping[int, str] | None = None,
    figsize: tuple[int, int] | None = None,
    backend: str | None = None,
    title: str | None = None,
    perplexity: float = 30.0,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int | None = 0,
) -> Any:
    """
    Plot embeddings projected into 2D or 3D space.

    Reduces high-dimensional embeddings using the specified dimensionality
    reduction method(s) and plots the result as a scatter plot. When multiple
    methods are provided, renders a grid of subplots for comparison.

    Parameters
    ----------
    embeddings : ArrayLike
        High-dimensional embeddings with shape ``(N, D)``. If ``method`` is
        None, must already have shape ``(N, 2)`` or ``(N, 3)``.
    method : str, Sequence[str], or None, default "pca"
        Dimensionality reduction method(s). Pass a list to compare multiple
        methods side-by-side in a grid:

        - ``"pca"``: Principal Component Analysis (fast, linear)
        - ``"tsne"``: t-SNE (nonlinear, preserves local structure)
        - ``"umap"``: UMAP (nonlinear, preserves global + local). Requires ``umap-learn``.
        - ``"isomap"``: Isomap (preserves geodesic distances)
        - ``"mds"``: Multidimensional Scaling (preserves pairwise distances)
        - ``"spectral"``: Spectral Embedding (reveals cluster structure)
        - ``"truncated_svd"``: Truncated SVD (works on sparse data)
        - ``"pacmap"``: PaCMAP (balanced local/global). Requires ``pacmap``.
        - ``"phate"``: PHATE (trajectory structure). Requires ``phate``.
        - None: Skip reduction, plot embeddings as-is (must be 2D or 3D).

    dimensions : {2, 3}, default 2
        Number of dimensions for the projection.
    labels : ArrayLike or None, default None
        Class labels for coloring points, shape ``(N,)``.
    label_names : dict[int, str] or None, default None
        Mapping from integer labels to display names for the legend.
    figsize : tuple[int, int] or None, default None
        Figure size in inches (width, height).
    backend : str or None, default None
        Plotting backend (``"matplotlib"``, ``"seaborn"``, ``"plotly"``,
        ``"altair"``). If None, uses default backend.
    title : str or None, default None
        Plot title. If None, auto-generated from method name(s).
    perplexity : float, default 30.0
        Perplexity parameter for t-SNE. Ignored for other methods.
    n_neighbors : int, default 15
        Number of neighbors for neighbor-based methods (UMAP, Isomap,
        Spectral, PaCMAP, PHATE). Ignored for other methods.
    min_dist : float, default 0.1
        Minimum distance for UMAP. Ignored for other methods.
    random_state : int or None, default 0
        Random seed for reproducibility.

    Returns
    -------
    Any
        Backend-specific figure object.

    Raises
    ------
    ImportError
        If scikit-learn or a required optional package is not installed.
    ValueError
        If ``method`` is None and embeddings don't have 2 or 3 columns.

    Examples
    --------
    >>> from dataeval_plots import project
    >>> fig = project(embeddings, method="tsne", labels=class_labels)

    >>> # Compare multiple methods
    >>> fig = project(embeddings, method=["pca", "tsne", "umap"], labels=y)

    >>> # Pre-reduced embeddings
    >>> fig = project(reduced_2d, method=None)

    >>> # 3D with UMAP
    >>> fig = project(embeddings, method="umap", dimensions=3)
    """
    from dataeval_plots.backends._shared import reduce_embeddings

    embeddings_array = np.asarray(embeddings)
    labels_array = np.asarray(labels) if labels is not None else None
    plotting_backend = get_backend(backend)

    # Multiple methods → grid of subplots
    if isinstance(method, Sequence) and not isinstance(method, str):
        methods = list(method)
        if not methods:
            raise ValueError("method sequence must not be empty")
        if len(methods) == 1:
            # Unwrap single-element list so it takes the single-plot path
            method = methods[0]
        else:
            reduced_list = [
                reduce_embeddings(
                    embeddings_array,
                    method=m,
                    dimensions=dimensions,
                    perplexity=perplexity,
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    random_state=random_state,
                )
                for m in methods
            ]
            return plotting_backend.project_grid(
                reduced_list,
                methods=methods,
                labels=labels_array,
                label_names=label_names,
                dimensions=dimensions,
                figsize=figsize,
                title=title,
            )

    # Single method
    if method is not None:
        embeddings_array = reduce_embeddings(
            embeddings_array,
            method=method,
            dimensions=dimensions,
            perplexity=perplexity,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
        )
    else:
        if embeddings_array.ndim != 2 or embeddings_array.shape[1] not in (2, 3):
            raise ValueError(
                f"When method is None, embeddings must have shape (N, 2) or (N, 3), got {embeddings_array.shape}"
            )
        dimensions = embeddings_array.shape[1]  # type: ignore[assignment]

    return plotting_backend.project(
        embeddings_array,
        labels=labels_array,
        label_names=label_names,
        method=method or "custom",
        dimensions=dimensions,
        figsize=figsize,
        title=title,
    )
