"""Shared helper functions for plotting backends."""

from __future__ import annotations

import contextlib
import inspect
import io
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import polars as pl
from numpy.typing import NDArray

if TYPE_CHECKING:
    from dataeval_plots.protocols import (
        PlottableBalance,
        PlottableDiversity,
        PlottableDriftMVDC,
        PlottableFeasibility,
    )

MethodType = Literal[
    "pca",
    "tsne",
    "umap",
    "isomap",
    "mds",
    "spectral",
    "truncated_svd",
    "pacmap",
    "phate",
]

__all__ = [
    "MethodType",
    "f_out",
    "project_steps",
    "calculate_projection",
    "normalize_reference_outputs",
    "reduce_embeddings",
    "prepare_balance_data",
    "prepare_diversity_data",
    "prepare_drift_data",
    "plot_drift_on_axis",
    "FEASIBILITY_COLORS",
    "FEASIBILITY_CONFLICT_TEXT",
    "FeasibilityBand",
    "FeasibilityEvidenceRow",
    "prepare_feasibility_data",
    "feasibility_zones",
    "feasibility_evidence_axis",
    "normalize_image_to_uint8",
    "image_to_base64_png",
    "image_to_hwc",
    "calculate_subplot_grid",
    "validate_class_names",
    "parse_dataset_item",
    "format_label_from_target",
    "merge_metadata",
    "draw_bounding_boxes",
    "extract_boxes_and_labels",
    "process_dataset_item_for_display",
    "CHANNELWISE_METRICS",
]

# Constants
CHANNELWISE_METRICS = [
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


def reduce_embeddings(
    embeddings: NDArray[Any],
    method: MethodType,
    dimensions: Literal[2, 3] = 2,
    perplexity: float = 30.0,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int | None = 0,
) -> NDArray[Any]:
    """
    Reduce high-dimensional embeddings using the specified method.

    Parameters
    ----------
    embeddings : NDArray
        High-dimensional embeddings with shape ``(N, D)``.
    method : MethodType
        Dimensionality reduction method to use.
    dimensions : {2, 3}, default 2
        Number of output dimensions.
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
    NDArray
        Reduced embeddings with shape ``(N, dimensions)``.

    Raises
    ------
    ImportError
        If scikit-learn or a required optional package is not installed.
    ValueError
        If an unknown method is specified.
    """
    if method in ("pca", "tsne", "isomap", "mds", "spectral", "truncated_svd"):
        try:
            import sklearn  # noqa: F401
        except ImportError:
            raise ImportError(
                "Dimension reduction requires scikit-learn. Install with: pip install dataeval-plots[projection]"
            ) from None

    n_samples = embeddings.shape[0]

    if method == "pca":
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=dimensions, random_state=random_state)
    elif method == "tsne":
        from sklearn.manifold import TSNE

        # Perplexity must be less than the number of samples
        effective_perplexity = min(perplexity, max(1.0, n_samples - 1.0))
        reducer = TSNE(
            n_components=dimensions,
            perplexity=effective_perplexity,
            learning_rate="auto",
            max_iter=1500,
            init="pca",
            random_state=random_state,
        )
    elif method == "umap":
        try:
            import umap
        except ImportError:
            raise ImportError("UMAP requires the 'umap-learn' package. Install with: pip install umap-learn") from None
        reducer = umap.UMAP(
            n_components=dimensions, n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state, n_jobs=1
        )
    elif method == "isomap":
        from sklearn.manifold import Isomap

        reducer = Isomap(n_components=dimensions, n_neighbors=n_neighbors)
    elif method == "mds":
        from sklearn.manifold import MDS

        # scikit-learn 1.8 renamed the non-metric switch from `metric` to `metric_mds`
        # -- `metric` now names the distance metric -- and moved `init` from fit() onto
        # the constructor. The projection extra still allows older releases, where neither
        # new spelling exists.
        #
        # Ask the signature rather than compare version strings: it needs no extra
        # dependency and tests exactly the thing being depended on. Omitting `init` on
        # 1.7.x is not a behaviour change -- smacof already initialises randomly from
        # `random_state` when init is left unset.
        mds_kwargs: dict[str, Any] = {
            "n_components": dimensions,
            "random_state": random_state,
            "normalized_stress": "auto",
            "n_init": 4,
            "max_iter": 300,
        }
        if "metric_mds" in inspect.signature(MDS.__init__).parameters:
            mds_kwargs["metric_mds"] = False
            mds_kwargs["init"] = "random"
        else:
            mds_kwargs["metric"] = False
        reducer = MDS(**mds_kwargs)
    elif method == "spectral":
        from sklearn.manifold import SpectralEmbedding

        reducer = SpectralEmbedding(
            n_components=dimensions, n_neighbors=n_neighbors, affinity="nearest_neighbors", random_state=random_state
        )
    elif method == "truncated_svd":
        from sklearn.decomposition import TruncatedSVD

        reducer = TruncatedSVD(n_components=dimensions, random_state=random_state)
    elif method == "pacmap":
        try:
            import pacmap
        except ImportError:
            raise ImportError("PaCMAP requires the 'pacmap' package. Install with: pip install pacmap") from None
        # PaCMAP prints warnings to stderr during __init__ — suppress them
        with contextlib.redirect_stderr(io.StringIO()):
            reducer = pacmap.PaCMAP(n_components=dimensions, n_neighbors=n_neighbors, random_state=random_state)
    elif method == "phate":
        try:
            import phate
        except ImportError:
            raise ImportError("PHATE requires the 'phate' package. Install with: pip install phate") from None
        reducer = phate.PHATE(n_components=dimensions, knn=n_neighbors, random_state=random_state, verbose=False)
    else:
        raise ValueError(f"Unknown reduction method: {method}")

    return np.asarray(reducer.fit_transform(embeddings)).astype(np.float64)


def f_out(n_i: NDArray[Any], x: NDArray[Any]) -> NDArray[Any]:
    """
    Calculates the line of best fit based on its free parameters.

    Parameters
    ----------
    n_i : NDArray
        Array of sample sizes
    x : NDArray
        Array of inverse power curve coefficients

    Returns
    -------
    NDArray
        Data points for the line of best fit
    """
    return x[0] * n_i ** (-x[1]) + x[2]


def project_steps(params: NDArray[Any], projection: NDArray[Any]) -> NDArray[Any]:
    """
    Projects the measures for each value of X.

    Parameters
    ----------
    params : NDArray
        Inverse power curve coefficients used to calculate projection
    projection : NDArray
        Steps to extrapolate

    Returns
    -------
    NDArray
        Extrapolated measure values at each projection step
    """
    return 1 - f_out(projection, params)


def calculate_projection(steps: NDArray[Any]) -> NDArray[Any]:
    """
    Calculate the projection array for extrapolation.

    Parameters
    ----------
    steps : NDArray
        Array of step values from the output

    Returns
    -------
    NDArray
        Projection array for extrapolation
    """
    last_X = steps[-1]
    geomshape = (0.01 * last_X, last_X * 4, len(steps))
    return np.geomspace(*geomshape).astype(np.int64)


def normalize_reference_outputs(
    reference_outputs: Sequence[Any] | Any | None,
) -> list[Any]:
    """
    Normalize reference outputs to a list.

    Parameters
    ----------
    reference_outputs : Sequence, single object, or None
        Reference outputs to normalize

    Returns
    -------
    list
        List of reference outputs (empty if None provided)
    """
    if reference_outputs is None:
        return []
    if not isinstance(reference_outputs, list | tuple):
        return [reference_outputs]
    return list(reference_outputs)


def prepare_balance_data(
    output: PlottableBalance,
    row_labels: Sequence[Any] | NDArray[Any] | None = None,
    col_labels: Sequence[Any] | NDArray[Any] | None = None,
    plot_classwise: bool = False,
) -> tuple[
    NDArray[Any],
    NDArray[Any] | Sequence[Any],
    NDArray[Any] | Sequence[Any],
    str,
    str,
    str,
]:
    """
    Prepare balance data for plotting across all backends.

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
    tuple
        (data, row_labels, col_labels, xlabel, ylabel, title)
    """
    if plot_classwise:
        # DataFrame-based output
        class_names = output.classwise["class_name"].unique(maintain_order=True).to_list()
        factor_names = output.classwise["factor_name"].unique(maintain_order=True).to_list()

        # Pivot to matrix format
        classwise_pivoted = output.classwise.pivot(on="factor_name", index="class_name", values="mi_value")
        data = classwise_pivoted.select(classwise_pivoted.columns[1:]).to_numpy()

        if row_labels is None:
            row_labels = class_names
        if col_labels is None:
            col_labels = factor_names

        xlabel = "Factors"
        ylabel = "Class"
        title = "Classwise Balance"
    else:
        # Get all factor names from balance DataFrame (includes "class_label" + metadata factors)
        all_factor_names = output.balance["factor_name"].to_list()
        # Metadata factor names only (exclude class_label)
        factor_names = sorted(all_factor_names[1:])

        # Create matrix: first row is balance (class-to-factor MI for metadata factors only),
        # rest is interfactor MI
        balance_row = output.balance["mi_value"].to_numpy()[1:]  # Skip class_label self-MI
        interfactor_matrix = (
            output.factors.pivot(
                on="factor2",
                index="factor1",
                values="mi_value",
                aggregate_function=None,
            )
            .sort("factor1")  # Sort the rows alphabetically
            .select(factor_names)  # Select columns in the exact same order
            .to_numpy()  # Export to pure NumPy
        )

        # Combine: balance row + interfactor matrix
        data = np.concatenate([balance_row[np.newaxis, :], interfactor_matrix], axis=0)

        # Create mask for lower triangle (excluding diagonal)
        # This creates an upper triangular matrix for visualization
        # Shift diagonal down by 1 to account for the class_label row at the top
        mask = np.tril(np.ones_like(data, dtype=bool), k=-1)
        data = np.where(mask, np.nan, data)[:-1, :]

        if row_labels is None:
            row_labels = ["class_label"] + factor_names[:-1]
        if col_labels is None:
            col_labels = factor_names

        xlabel = ""
        ylabel = ""
        title = "Balance Heatmap"

    return data, row_labels, col_labels, xlabel, ylabel, title


def prepare_diversity_data(
    output: PlottableDiversity,
    row_labels: Sequence[Any] | NDArray[Any] | None = None,
    col_labels: Sequence[Any] | NDArray[Any] | None = None,
    plot_classwise: bool = False,
) -> tuple[
    NDArray[Any],
    NDArray[Any] | Sequence[Any],
    NDArray[Any] | Sequence[Any],
    str,
    str,
    str,
    str,
]:
    """
    Prepare diversity data for plotting across all backends.

    Parameters
    ----------
    output : PlottableDiversity
        The diversity output object to plot
    row_labels : ArrayLike or None, default None
        List/Array containing the labels for rows in the histogram
    col_labels : ArrayLike or None, default None
        List/Array containing the labels for columns in the histogram
    plot_classwise : bool, default False
        Whether to plot per-class diversity instead of global diversity

    Returns
    -------
    tuple
        (data, row_labels, col_labels, xlabel, ylabel, title, method_name)
        data is None for non-classwise bar charts
    """
    # Try to get method name from metadata state, fall back to "Diversity"
    try:
        meta = output.meta()
        method_name = getattr(meta, "state", {}).get("method", "Diversity").title()
    except (AttributeError, TypeError):
        method_name = "Diversity"

    if plot_classwise:
        # DataFrame-based output
        class_names = output.classwise["class_name"].unique(maintain_order=True).to_list()
        factor_names = output.classwise["factor_name"].unique(maintain_order=True).to_list()

        # Pivot to matrix format
        classwise_pivoted = output.classwise.pivot(on="factor_name", index="class_name", values="diversity_value")
        data = classwise_pivoted.select(classwise_pivoted.columns[1:]).to_numpy()

        if row_labels is None:
            row_labels = class_names
        if col_labels is None:
            col_labels = factor_names

        xlabel = "Factors"
        ylabel = "Class"
        title = "Classwise Diversity"
    else:
        # DataFrame-based output - bar chart
        factor_names = output.factors["factor_name"].to_list()
        data = np.ndarray(0)  # unused for bar charts
        row_labels = factor_names
        col_labels = []  # unused

        xlabel = "Factors"
        ylabel = "Diversity Index"
        title = "Diversity Index by Factor"

    return data, row_labels, col_labels, xlabel, ylabel, title, method_name


def prepare_drift_data(
    output: PlottableDriftMVDC,
) -> tuple[Any, Any, Any, NDArray[Any], bool]:
    """
    Prepare drift detection data for plotting across all backends.

    Parameters
    ----------
    output : PlottableDriftMVDC
        The drift MVDC output object to plot

    Returns
    -------
    tuple
        (resdf, trndf, tstdf, driftx, is_sufficient)
        resdf: Full results dataframe
        trndf: Training/reference data
        tstdf: Test/analysis data
        driftx: Indices where drift was detected
        is_sufficient: Whether there's enough data to plot (>= 3 rows)
    """
    resdf = output.data()
    is_sufficient = resdf.shape[0] >= 3

    if not is_sufficient:
        return resdf, None, None, np.array([]), False

    # Filter for reference and analysis periods
    trndf = resdf.filter(pl.col("chunk_period") == "reference")
    tstdf = resdf.filter(pl.col("chunk_period") == "analysis")

    # Get drift alert indices
    drift_mask = resdf["domain_classifier_auroc_alert"].to_numpy()
    driftx = np.where(drift_mask)[0]

    return resdf, trndf, tstdf, driftx, True


def plot_drift_on_axis(
    ax: Any,
    resdf: Any,
    trndf: Any,
    tstdf: Any,
    driftx: NDArray[Any],
    threshold_upper_color: str = "red",
    threshold_lower_color: str = "red",
    train_color: Any = "b",
    test_color: Any = "g",
    drift_color: str = "magenta",
    threshold_upper_label: str = "Threshold Upper",
    threshold_lower_label: str = "Threshold Lower",
    train_label: str = "Train",
    test_label: str = "Test",
    drift_label: str = "Drift",
    drift_marker: str = "D",
    drift_markersize: int = 3,
    threshold_linestyle: str = "--",
    train_linestyle: str = "-",
    test_linestyle: str = "-",
    linewidth: int = 2,
    title: str = "Domain Classifier, Drift Detection",
    xlabel: str = "Chunk Index",
    ylabel: str = "ROC AUC",
    title_fontsize: int = 12,
    label_fontsize: int = 10,
    tick_fontsize: int = 8,
    legend_fontsize: int = 8,
    legend_loc: str = "lower left",
) -> None:
    """
    Plot drift detection data on a matplotlib axis.

    This is a shared helper function that plots drift detection data with customizable styling.
    Used by both matplotlib and seaborn backends with different styling parameters.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    resdf : DataFrame
        Full results dataframe
    trndf : DataFrame
        Training/reference data
    tstdf : DataFrame
        Test/analysis data
    driftx : NDArray
        Indices where drift was detected
    threshold_upper_color : str, default "red"
        Color for upper threshold line
    threshold_lower_color : str, default "red"
        Color for lower threshold line
    train_color : Any, default "b"
        Color for training data line (can be string or RGB tuple)
    test_color : Any, default "g"
        Color for test data line (can be string or RGB tuple)
    drift_color : str, default "magenta"
        Color for drift markers
    threshold_upper_label : str, default "Threshold Upper"
        Label for upper threshold
    threshold_lower_label : str, default "Threshold Lower"
        Label for lower threshold
    train_label : str, default "Train"
        Label for training data
    test_label : str, default "Test"
        Label for test data
    drift_label : str, default "Drift"
        Label for drift markers
    drift_marker : str, default "D"
        Marker style for drift points
    drift_markersize : int, default 3
        Size of drift markers
    threshold_linestyle : str, default "--"
        Line style for thresholds
    train_linestyle : str, default "-"
        Line style for training data
    test_linestyle : str, default "-"
        Line style for test data
    linewidth : int, default 2
        Width of lines
    title : str, default "Domain Classifier, Drift Detection"
        Plot title
    xlabel : str, default "Chunk Index"
        X-axis label
    ylabel : str, default "ROC AUC"
        Y-axis label
    title_fontsize : int, default 12
        Font size for title
    label_fontsize : int, default 10
        Font size for axis labels
    tick_fontsize : int, default 8
        Font size for tick labels
    legend_fontsize : int, default 8
        Font size for legend
    legend_loc : str, default "lower left"
        Location of legend
    """
    # Get indices for plotting
    n_rows = len(resdf)
    indices = np.arange(n_rows)
    trn_indices = resdf.with_row_index().filter(pl.col("chunk_period") == "reference")["index"].to_numpy()
    tst_indices = resdf.with_row_index().filter(pl.col("chunk_period") == "analysis")["index"].to_numpy()

    # Plot threshold lines
    ax.plot(
        indices,
        resdf["domain_classifier_auroc_upper_threshold"].to_numpy(),
        threshold_linestyle,
        color=threshold_upper_color,
        label=threshold_upper_label,
        linewidth=linewidth,
    )
    ax.plot(
        indices,
        resdf["domain_classifier_auroc_lower_threshold"].to_numpy(),
        threshold_linestyle,
        color=threshold_lower_color,
        label=threshold_lower_label,
        linewidth=linewidth,
    )

    # Plot train and test data
    ax.plot(
        trn_indices,
        trndf["domain_classifier_auroc_value"].to_numpy(),
        train_linestyle,
        color=train_color,
        label=train_label,
        linewidth=linewidth,
    )
    ax.plot(
        tst_indices,
        tstdf["domain_classifier_auroc_value"].to_numpy(),
        test_linestyle,
        color=test_color,
        label=test_label,
        linewidth=linewidth,
    )

    # Plot drift points
    ax.plot(
        driftx,
        resdf["domain_classifier_auroc_value"].to_numpy()[driftx],
        drift_marker,
        color=drift_color,
        markersize=drift_markersize,
        label=drift_label,
    )

    # Set ticks and labels
    xticks = np.arange(resdf.shape[0])
    ax.set_xticks(xticks)
    ax.tick_params(axis="x", labelsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    ax.legend(loc=legend_loc, fontsize=legend_fontsize, frameon=True)
    ax.set_title(title, fontsize=title_fontsize, pad=15)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylim((0.0, 1.1))


# Colors: demonstrated blue, ruled out orange, unknown light gray, warning amber; text in neutral ink
FEASIBILITY_COLORS = {
    "demonstrated": "#2a78d6",
    "demonstrated_fill": "#cfe1f7",
    "ruled_out": "#eb6834",
    "ruled_out_fill": "#fbd9c9",
    "unknown_fill": "#e9e8e4",
    "warning": "#eda100",
    "ink": "#0b0b0b",
    "ink_muted": "#52514e",
}
FEASIBILITY_CONFLICT_TEXT = "Evidence conflicts: check assumptions"

_SCOPE_RANK = {"partition_restricted": 0, "fixed_representation": 1, "full_input": 2}
_SCOPE_SUBTITLE = {
    "full_input": "Models may use the whole {unit}",
    "fixed_representation": "Models may use only the embeddings",
    "partition_restricted": "Models give one answer per group",
}
_SCOPE_MODELS = {
    "full_input": "models that use the whole {unit}",
    "fixed_representation": "models restricted to the embeddings",
    "partition_restricted": "models that give one answer per group",
}
_SCOPE_ROW = {"fixed_representation": "embeddings only", "partition_restricted": "one answer per group"}
# "ruled out*" never stands alone: its "*" footnote in the same band's caption says it holds only under the recorded
# assumptions, and an infeasible row always has a ceiling, which writes that footnote
_MARKER_WORDS = {
    "demonstrated_feasible": "demonstrated ✓",
    "unresolved": "insufficient evidence ?",
    "infeasible_under_recorded_assumptions": "ruled out* ✗",
    "not_ruled_out": "not ruled out ?",
}
_RULED_OUT = "ruled out*"
_AGGREGATION_WORDS = {None: "", "sequence_mean": "per-sequence mean ", "frame_pooled": "frame-pooled "}
_GROUP_CEILING = "group-based ceiling"
_NO_BOUND = "no bound available"


def _pct(value: float, direction: str | None = None) -> str:
    """
    Format to 0.1% as DataEval prints it.

    A lower bound rounds down and an upper bound up, so rounding never overstates a bound.
    """
    if direction is None:
        return f"{value:.1%}"
    tenths = value * 1000  # the 1e-9 absorbs float error only, such as 0.957 * 1000 = 956.9999999999999
    tenths = math.floor(tenths + 1e-9) if direction == "lower" else math.ceil(tenths - 1e-9)
    return f"{tenths / 10:.1f}%"


def _percent(value: float) -> str:
    """A threshold, confidence or alpha as DataEval prints it: no rounding to whole percents, no trailing zeros."""
    return f"{value * 100:g}%"


@dataclass(frozen=True)
class FeasibilityEvidenceRow:
    """One evidence source's bound on one band, ready to draw.

    ``bound_left`` puts the bound's label left of its tick (outward from the observed score) and the observed label
    right of its dot; false swaps them. A source with no bound for the band has ``bound`` None and is drawn faint,
    its ``bound_label`` saying so.
    """

    source: str
    bound_type: str
    bound: float | None
    observed: float | None
    restricted: bool
    label: str
    bound_label: str
    observed_label: str | None
    bound_left: bool
    faint: bool


@dataclass(frozen=True)
class FeasibilityBand:
    """One quantity's requirements (a target without its threshold), ready to draw."""

    title: str
    subtitle: str
    x_range: tuple[float, float]
    demonstrated: float | None
    ruled_out: float | None
    conflict: bool
    boundary_labels: list[tuple[float, str, str, bool]]  # (bound, text, "lower" or "upper", label left of the line)
    markers: list[tuple[float, str]]  # requirement labels in the band view
    evidence_markers: list[tuple[float, str]]  # requirement labels in the evidence view
    caption: list[str]
    notes: list[str]
    evidence_caption: list[str]  # the evidence view's caption, in place of the caption and notes
    evidence: list[FeasibilityEvidenceRow]
    evidence_legend: str
    evidence_note: str | None  # drawn in the evidence view when no source has a bound for the band


def _sentence_case(text: str) -> str:
    return text[:1].upper() + text[1:]


def _feasibility_range(values: Iterable[float]) -> tuple[float, float]:
    """[min − 0.05, 1.0], floored at 0 and rounded down to 0.05."""
    low = math.floor(round((min(values) - 0.05) / 0.05, 9)) * 0.05
    return round(max(low, 0.0), 2), 1.0


def _feasibility_evidence(
    evidence: list[dict[str, Any]], scope: str, unit: str
) -> tuple[list[FeasibilityEvidenceRow], list[str]]:
    """
    One row per evidence item, at its most favorable bound over the band's thresholds, and the notes.

    A source with no bound for any of the band's thresholds keeps one row, with no bound.
    """
    kept: dict[str, dict[str, Any]] = {}
    for e in evidence:
        sign = 1 if e["bound_type"] == "lower" else -1
        prev = kept.get(e["evidence_id"])
        if prev is None or (
            e["bound"] is not None and (prev["bound"] is None or sign * e["bound"] > sign * prev["bound"])
        ):
            kept[e["evidence_id"]] = e
    rows, notes = [], []
    for e in kept.values():
        # A ceiling counts at its scope and narrower ones; a witness at its scope and broader ones
        info, bound, kind = e["information_scope"], e["bound"], e["bound_type"]
        rank, judged = _SCOPE_RANK[info], _SCOPE_RANK[scope]
        restricted = rank < judged if kind == "upper" else rank > judged
        models = _SCOPE_MODELS[info].format(unit=unit)
        label = _sentence_case(e["source"]) + (f" ({_SCOPE_ROW.get(info, models)})" if restricted else "")
        observed = None if bound is None else e["observed"]  # an observed score alone bounds nothing
        rows.append(
            FeasibilityEvidenceRow(
                source=e["source"],
                bound_type=kind,
                bound=bound,
                observed=observed,
                restricted=restricted,
                label=label,
                bound_label=_NO_BOUND if bound is None else f"{_pct(bound, kind)} {kind} bound",
                observed_label=None if observed is None else f"{_pct(observed)} observed",
                bound_left=kind == "lower",
                faint=restricted or bound is None,
            )
        )
        if restricted and kind == "upper" and bound is not None:
            ceiling = _pct(bound, "upper")
            notes.append(
                f"Group-based ceiling ({ceiling}) applies only to models that give one answer per group."
                if info == "partition_restricted"
                else f"{_sentence_case(e['source'])} ({ceiling}) applies only to {models}."
            )
    return rows, list(dict.fromkeys(notes))


def _deciding(
    row: dict[str, Any], evidence: list[dict[str, Any]]
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """A determinations row's deciding witness and ceiling bounds, found as DataEval's printout finds them."""
    mine = [e for e in evidence if e["target_spec"] == row["target_spec"]]
    witness = next((e for e in mine if e["evidence_id"] == row["witness_source"] and e["bound_type"] == "lower"), None)
    # The deciding ceiling is the bound at the row's ceiling scope; a narrower one can share its value
    ceiling = next(
        (
            e
            for e in mine
            if e["bound_type"] == "upper"
            and e["bound"] is not None
            and row["ceiling"] is not None
            and e["information_scope"] == row["ceiling_scope"]
            and math.isclose(e["bound"], row["ceiling"])
        ),
        None,
    )
    return witness, ceiling


def _unit(rows: list[dict[str, Any]], evidence: list[dict[str, Any]]) -> str | None:
    """The unit of the first row's deciding witness or ceiling, as the printout names it; None without one."""
    for row in rows:
        witness, ceiling = _deciding(row, evidence)
        if witness or ceiling:
            return (witness or ceiling or {})["unit"]
    return None


def _feasibility_title(spec: dict[str, Any]) -> str:
    """The axis title, naming the AP class (or macro) and any aggregation so per-class bands differ."""
    metric = spec["metric"]
    if metric == "ap":
        metric = "macro AP" if spec.get("class_name") is None else f"AP for class {spec['class_name']}"
    return "Required " + _AGGREGATION_WORDS[spec.get("aggregation")] + metric


def _feasibility_band(
    rows: list[dict[str, Any]], evidence: list[dict[str, Any]], alpha: float, noun: str
) -> FeasibilityBand:
    """One band; ``noun`` is the unit the printout's footer names for the whole call."""
    scope = rows[0]["judged_scope"]
    unit = _unit(rows, evidence) or "image"
    # A lower bound of 0% demonstrates nothing and an upper bound of 100% rules nothing out
    witnessed = [r for r in rows if r["witness"] is not None and r["witness"] > 0.0]
    best = max(witnessed, key=lambda r: r["witness"], default=None)
    w = None if best is None else best["witness"]
    ceilinged = [r for r in rows if r["ceiling"] is not None and r["ceiling"] < 1.0]
    deciding = min(ceilinged, key=lambda r: r["ceiling"], default=None)
    c = None if deciding is None else deciding["ceiling"]
    conflict = any(r["determination"] == "evidence_conflict" for r in rows)
    ev_rows, notes = _feasibility_evidence(evidence, scope, unit)

    # Each label outward from the unknown zone; in a conflict the bounds cross, so outward flips
    boundary_labels = []
    if w is not None:
        boundary_labels.append((w, f"{_pct(w, 'lower')} lower confidence bound", "lower", not conflict))
    if c is not None:
        boundary_labels.append((c, f"{_pct(c, 'upper')} upper confidence bound", "upper", conflict))

    markers = []
    for r in sorted(rows, key=lambda r: r["threshold"]):
        words = _MARKER_WORDS.get(r["determination"])
        markers.append((r["threshold"], _percent(r["threshold"]) + (f": {words}" if words else "")))

    conflict_caption = (
        f"Bounds disagree: check the assumptions, such as evaluation {unit}s entering training. "
        f"Even when they hold, the chance that any bound fails is at most {_percent(alpha)}."
    )
    if conflict:
        caption = [conflict_caption]
    else:
        caption = []
        witness = None if best is None else _deciding(best, evidence)[0]
        if witness and witness["observed"] is not None and w is not None:
            on = "the " if witness["n"] is None else f"{witness['n']:,} "
            caption.append(
                f"{_sentence_case(witness['source'])} scored {_pct(witness['observed'])} on {on}evaluation "
                f"{unit}s; given sampling uncertainty, its lower confidence bound is {_pct(w, 'lower')}."
            )
        caption.append(
            f"Bounds apply to new {noun}s from the same population, at {_percent(1 - alpha)} joint confidence."
        )
        # The "*" footnote to every "ruled out*" goes last, as a footnote does
        if deciding is not None and c is not None:
            ceiling = _deciding(deciding, evidence)[1]
            sign_off = ceiling is not None and ceiling["source"] == _GROUP_CEILING
            caption.append(
                "* Ruled out under the recorded assumptions, at the stated confidence"
                + (" (group-based ceiling; validity pending lead-scientist sign-off)" if sign_off else "")
                + "."
            )
    evidence_caption = [
        f"Observed on the evaluation {unit}s; bounds apply to new {unit}s from the same population, "
        f"at {_percent(1 - alpha)} joint confidence."
    ]
    if conflict:
        evidence_caption.append(conflict_caption)

    legend = "● observed · ┃ lower confidence bound"
    if any(r.bound_type == "upper" and r.bound is not None for r in ev_rows):
        legend += " · ┃ upper confidence bound"

    return FeasibilityBand(
        title=_feasibility_title(json.loads(rows[0]["target_spec"])),
        subtitle=_SCOPE_SUBTITLE[scope].format(unit=unit),
        x_range=_feasibility_range([*(t for t, _ in markers), *(b[0] for b in boundary_labels)]),
        demonstrated=None if conflict else w,
        ruled_out=None if conflict else c,
        conflict=conflict,
        boundary_labels=boundary_labels,
        markers=markers,
        evidence_markers=[(t, f"{_percent(t)} required") for t, _ in markers],
        caption=caption,
        notes=notes,
        evidence_caption=evidence_caption,
        evidence=ev_rows,
        evidence_legend=legend,
        evidence_note=None if any(r.bound is not None for r in ev_rows) else "No bound is available yet.",
    )


# These mirror DataEval's performance/_feasibility.py: _SOURCES, _source_name, _unit_noun and FeasibilityOutput._bounds
_SOURCE_NAMES = {
    "replica_ceiling/v1": "repeated labels",
    "builtin_witness/v1:knn": "built-in k-NN",
    "builtin_witness/v1:probe": "built-in linear probe",
    "user_witness/v1": "your model",
}
_EVIDENCE_SCHEMA = {
    "target_spec": pl.String,
    "metric": pl.String,
    "threshold": pl.Float64,
    "source": pl.String,
    "evidence_id": pl.String,
    "information_scope": pl.String,
    "unit": pl.String,
    "bound_type": pl.String,
    "observed": pl.Float64,
    "bound": pl.Float64,
    "n": pl.Int64,
    "state": pl.String,
}


def _evidence_rows(output: PlottableFeasibility) -> pl.DataFrame:
    """One row per endpoint on a target in ``determinations``, in their order, then the order of ``evidence``."""
    targets = {r["target_spec"]: r for r in output.determinations.iter_rows(named=True)}
    rows = []
    for spec, target in targets.items():
        for eid, ev in output.evidence.items():
            for e in ev.endpoints:
                if e.target_spec != spec:
                    continue
                stats = ev.details["endpoints"][spec] or {}
                rows.append(
                    {
                        "target_spec": spec,
                        "metric": target["metric"],
                        "threshold": target["threshold"],
                        "source": _GROUP_CEILING
                        if ev.kind.startswith("local_panel_ceiling/v1")
                        else _SOURCE_NAMES.get(eid, eid),
                        "evidence_id": eid,
                        "information_scope": ev.information_scope,
                        "unit": "sequence" if ev.unit == "group" else "image",
                        "bound_type": e.direction,
                        # A ceiling's mean is panel agreement, not an observed score
                        "observed": stats.get("mean") if e.direction == "lower" else None,
                        "bound": e.value,
                        "n": stats.get("n"),
                        "state": e.state,
                    }
                )
    return pl.DataFrame(rows, schema=_EVIDENCE_SCHEMA)


def prepare_feasibility_data(output: PlottableFeasibility) -> list[FeasibilityBand]:
    """
    Prepare feasibility data for plotting across all backends.

    Determinations rows whose target is equal apart from its threshold share one band, so only thresholds of the
    same quantity share zones.

    Parameters
    ----------
    output : PlottableFeasibility
        The feasibility output object to plot

    Returns
    -------
    list[FeasibilityBand]
        One band per quantity, in the order the determinations first name it
    """
    rows = list(output.determinations.iter_rows(named=True))
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        spec = json.loads(row["target_spec"])
        spec.pop("threshold", None)
        groups.setdefault(json.dumps(spec, sort_keys=True), []).append(row)
    evidence = _evidence_rows(output).to_dicts()
    alpha = output.meta().state["alpha"]
    noun = _unit(rows, evidence) or "image"  # the printout's footer noun, for the whole call
    bands = []
    for band in groups.values():
        specs = {r["target_spec"] for r in band}
        bands.append(_feasibility_band(band, [e for e in evidence if e["target_spec"] in specs], alpha, noun))
    return bands


def feasibility_zones(band: FeasibilityBand) -> list[tuple[str, float, float, str]]:
    """
    Split a band into its zones.

    Parameters
    ----------
    band : FeasibilityBand
        The band to split

    Returns
    -------
    list[tuple[str, float, float, str]]
        (kind, start, end, label) per zone, kind one of ``demonstrated``, ``unknown`` or ``ruled_out``; none when
        the evidence conflicts
    """
    if band.conflict:
        return []
    lo, hi = band.x_range
    start = lo if band.demonstrated is None else band.demonstrated
    end = hi if band.ruled_out is None else band.ruled_out
    zones = [("unknown", start, end, "unknown")]
    if band.demonstrated is not None:
        zones.insert(0, ("demonstrated", lo, start, "demonstrated"))
    if band.ruled_out is not None:
        zones.append(("ruled_out", end, hi, _RULED_OUT))
    return [zone for zone in zones if zone[2] > zone[1]]  # a zero-width zone has nothing to show


def feasibility_evidence_axis(band: FeasibilityBand) -> tuple[tuple[float, float], str]:
    """
    Axis range and title for a band's evidence view.

    Parameters
    ----------
    band : FeasibilityBand
        The band whose evidence is drawn

    Returns
    -------
    tuple[tuple[float, float], str]
        The range covering every threshold, bound and observed score, and the metric's name
    """
    values = [t for t, _ in band.markers] + [r.bound for r in band.evidence if r.bound is not None]
    values += [r.observed for r in band.evidence if r.observed is not None]
    return _feasibility_range(values), _sentence_case(band.title.removeprefix("Required "))


def normalize_image_to_uint8(img_np: NDArray[Any]) -> NDArray[Any]:
    """
    Normalize image array to 0-255 uint8 range.

    Parameters
    ----------
    img_np : NDArray
        Image array in HWC format

    Returns
    -------
    NDArray
        Image array in uint8 format (0-255 range)
    """
    if img_np.max() <= 1.0:
        return (img_np * 255).astype(np.uint8)
    return img_np.astype(np.uint8)


def image_to_base64_png(img_np: NDArray[Any]) -> str:
    """
    Convert numpy image array to base64 encoded PNG string.

    Parameters
    ----------
    img_np : NDArray
        Image array in uint8 format

    Returns
    -------
    str
        Base64 encoded PNG data URL string (data:image/png;base64,...)
    """
    import base64
    from io import BytesIO

    from PIL import Image

    # Convert to PIL Image
    pil_img = Image.fromarray(img_np)

    # Convert to base64
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return f"data:image/png;base64,{img_str}"


def image_to_hwc(image: NDArray[Any]) -> NDArray[Any]:
    """Convert image array to Height-Width-Channel (HWC) format.

    Parameters
    ----------
    image : NDArray[Any]
        Input image array

    Returns
    -------
    NDArray[Any]
        Image in HWC format
    """
    image = np.asarray(image)
    if image.ndim == 2:
        return image[:, :, np.newaxis]  # Grayscale to HWC
    if image.shape[0] in {1, 3, 4}:
        return np.transpose(image, (1, 2, 0))  # Channels-first to HWC
    return image  # Assume already HWC


def calculate_subplot_grid(
    num_items: int,
    cols_per_row: int = 3,
    aspect_ratio: float | None = None,
) -> tuple[int, int]:
    """
    Calculate grid layout for subplots.

    When *aspect_ratio* is provided the layout is chosen so that the
    ``cols / rows`` ratio best matches the target, producing a grid that
    fills the available figure shape with minimal wasted space.

    Parameters
    ----------
    num_items : int
        Number of items to plot.
    cols_per_row : int, default 3
        Maximum columns per row (used when *aspect_ratio* is None).
    aspect_ratio : float or None, default None
        Target width / height ratio of the figure.  When given, the
        function picks the ``(rows, cols)`` combination that minimises
        ``abs(cols / rows - aspect_ratio)`` while fitting all items.

    Returns
    -------
    tuple[int, int]
        ``(rows, cols)`` for the subplot grid.
    """
    import math

    if aspect_ratio is None:
        rows = math.ceil(num_items / cols_per_row)
        cols = min(num_items, cols_per_row)
        return rows, cols

    best: tuple[int, int] = (num_items, 1)
    best_diff = float("inf")

    for r in range(1, num_items + 1):
        c = math.ceil(num_items / r)
        diff = abs(c / r - aspect_ratio)
        if diff < best_diff:
            best_diff = diff
            best = (r, c)

    return best


def validate_class_names(measures: NDArray[Any], class_names: Sequence[str] | None) -> None:
    """
    Validate that class names align with measures.

    Parameters
    ----------
    measures : NDArray
        Measures array (multiclass, first dimension is classes)
    class_names : Sequence[str] or None
        List of class names

    Raises
    ------
    IndexError
        If class name count does not align with measures
    """
    if class_names is not None and len(measures) != len(class_names):
        raise IndexError("Class name count does not align with measures")


def parse_dataset_item(datum: Any) -> tuple[Any, Any | None, dict[str, Any]]:
    """
    Parse a dataset item into image, target, and metadata components.

    Parameters
    ----------
    datum : Any
        Dataset item that can be:
        - Just an image (array-like)
        - Tuple of (image,)
        - Tuple of (image, target)
        - Tuple of (image, target, metadata)

    Returns
    -------
    tuple
        (image, target, metadata) where:
        - image: The image array
        - target: Target labels/boxes/etc. or None if not present
        - metadata: Dictionary of metadata (empty dict if not present)
    """
    if isinstance(datum, tuple):
        if len(datum) == 1:
            return datum[0], None, {}
        if len(datum) == 2:
            return datum[0], datum[1], {}
        if len(datum) >= 3:
            # Extract metadata - convert to dict if it's not already
            meta = datum[2] if isinstance(datum[2], dict) else {}
            return datum[0], datum[1], meta

    # Single item - just the image
    return datum, None, {}


def format_label_from_target(
    target: Any,
    index2label: dict[int, str] | None = None,
) -> str | None:
    """
    Format a human-readable label string from various target types.

    Parameters
    ----------
    target : Any
        Target can be:
        - Array of pseudo probabilities or one-hot encoded labels
        - Dict with keys 'boxes', 'labels', 'scores' (object detection)
        - Object with attributes 'boxes', 'labels', 'scores'
        - None
    index2label : dict[int, str] or None
        Mapping from class indices to class names

    Returns
    -------
    str or None
        Formatted label string or None if target is None
    """
    if target is None:
        return None

    # Handle dict or object with boxes/labels/scores (object detection format)
    boxes = None
    labels = None

    if isinstance(target, dict):
        boxes = target.get("boxes")
        labels = target.get("labels")
        target.get("scores")
    elif hasattr(target, "boxes") and hasattr(target, "labels"):
        boxes = getattr(target, "boxes", None)
        labels = getattr(target, "labels", None)
        getattr(target, "scores", None)

    # If we found object detection format
    if boxes is not None and labels is not None:
        boxes_arr = np.asarray(boxes)
        labels_arr = np.asarray(labels)

        if len(boxes_arr) == 0:
            return "No objects"

        # Check if labels are one-hot or probabilities (2D array)
        label_indices = np.argmax(labels_arr, axis=1) if labels_arr.ndim == 2 else labels_arr.astype(int)

        # Count objects per class
        unique_labels, counts = np.unique(label_indices, return_counts=True)

        # Format label string
        label_parts = []
        for lbl, cnt in zip(unique_labels, counts):
            if index2label and lbl in index2label:
                label_parts.append(f"{index2label[lbl]}: {cnt}")
            else:
                label_parts.append(f"Class {lbl}: {cnt}")

        return ", ".join(label_parts)

    # Handle array of probabilities or one-hot encoding (classification)
    target_arr = np.asarray(target)

    if target_arr.ndim == 0:
        # Scalar label
        label_idx = int(target_arr)
        if index2label and label_idx in index2label:
            return index2label[label_idx]
        return f"Class {label_idx}"

    if target_arr.ndim == 1 and len(target_arr) > 0:
        # 1D array - could be probabilities or one-hot
        if target_arr.dtype in (np.float32, np.float64):
            # Check if it's one-hot or probabilities
            if np.allclose(target_arr.sum(), 1.0) and np.max(target_arr) <= 1.0:
                # Probabilities or one-hot
                label_idx = int(np.argmax(target_arr))
                confidence = target_arr[label_idx]

                if index2label and label_idx in index2label:
                    return f"{index2label[label_idx]} ({confidence:.2f})"
                return f"Class {label_idx} ({confidence:.2f})"
        else:
            # Integer array - direct label
            if len(target_arr) == 1:
                label_idx = int(target_arr[0])
                if index2label and label_idx in index2label:
                    return index2label[label_idx]
                return f"Class {label_idx}"

    return None


def merge_metadata(
    base_metadata: dict[str, Any],
    additional_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Merge base metadata with additional metadata.

    Parameters
    ----------
    base_metadata : dict[str, Any]
        Base metadata from the dataset item
    additional_metadata : dict[str, Any] or None
        Additional metadata to merge in

    Returns
    -------
    dict[str, Any]
        Merged metadata dictionary
    """
    if additional_metadata is None:
        return base_metadata

    merged = base_metadata.copy()
    merged.update(additional_metadata)
    return merged


def extract_boxes_and_labels(
    target: Any,
) -> tuple[NDArray[Any] | None, NDArray[Any] | None, NDArray[Any] | None]:
    """
    Extract bounding boxes, labels, and scores from object detection targets.

    Parameters
    ----------
    target : Any
        Target that may contain object detection information:
        - Dict with keys 'boxes', 'labels', 'scores'
        - Object with attributes 'boxes', 'labels', 'scores'

    Returns
    -------
    tuple[NDArray | None, NDArray | None, NDArray | None]
        (boxes, labels, scores) where each is None if not found.
        boxes are in XYXY format.
    """
    boxes = None
    labels = None
    scores = None

    if isinstance(target, dict):
        boxes = target.get("boxes")
        labels = target.get("labels")
        scores = target.get("scores")
    elif hasattr(target, "boxes") and hasattr(target, "labels"):
        boxes = getattr(target, "boxes", None)
        labels = getattr(target, "labels", None)
        scores = getattr(target, "scores", None)

    if boxes is not None:
        boxes = np.asarray(boxes)
    if labels is not None:
        labels = np.asarray(labels)
    if scores is not None:
        scores = np.asarray(scores)

    return boxes, labels, scores


def _get_label_indices(labels: NDArray[Any] | None) -> NDArray[Any] | None:
    """
    Process labels to extract class indices.

    Parameters
    ----------
    labels : NDArray or None
        Label indices or probabilities for each box

    Returns
    -------
    NDArray or None
        Array of label indices, or None if labels is None
    """
    if labels is None:
        return None
    return np.argmax(labels, axis=1) if labels.ndim == 2 else labels.astype(int)


def _build_label_text(
    label_idx: int,
    score_i: Any | None,
    index2label: dict[int, str] | None,
) -> str:
    """
    Build label text for a bounding box.

    Parameters
    ----------
    label_idx : int
        The class index for this box
    score_i : Any or None
        Score for this box (can be scalar or array of pseudo probs)
    index2label : dict[int, str] or None
        Mapping from class indices to class names

    Returns
    -------
    str
        Formatted label text with optional score
    """
    # Get class name or default to "Class {idx}"
    label_text = index2label[label_idx] if index2label and label_idx in index2label else f"Class {label_idx}"

    # Add score if available
    if score_i is not None:
        # Check if score is an array (pseudo probs) or scalar
        score_value = score_i[label_idx] if np.ndim(score_i) > 0 else score_i
        label_text += f" {score_value:.2f}"

    return label_text


def _draw_boxes_opencv(
    image: NDArray[Any],
    boxes: NDArray[Any],
    label_indices: NDArray[Any] | None,
    scores: NDArray[Any] | None,
    index2label: dict[int, str] | None,
    color: tuple[int, int, int],
    thickness: int,
) -> NDArray[Any]:
    """
    Draw bounding boxes using OpenCV.

    Parameters
    ----------
    image : NDArray
        Image array to draw on (will be modified in-place)
    boxes : NDArray
        Bounding boxes in XYXY format
    label_indices : NDArray or None
        Array of label indices
    scores : NDArray or None
        Confidence scores for each box
    index2label : dict[int, str] or None
        Mapping from class indices to class names
    color : tuple[int, int, int]
        RGB color for bounding boxes
    thickness : int
        Line thickness for bounding boxes

    Returns
    -------
    NDArray
        Image with bounding boxes drawn
    """
    import cv2  # type: ignore

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(int)

        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # Build label text
        if label_indices is not None:
            label_idx = label_indices[i]
            score_i = scores[i] if scores is not None and i < len(scores) else None
            label_text = _build_label_text(label_idx, score_i, index2label)

            # Draw label background and text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)

            # Draw background rectangle
            cv2.rectangle(
                image,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1,  # Filled
            )

            # Draw text
            cv2.putText(
                image,
                label_text,
                (x1, y1 - baseline - 2),
                font,
                font_scale,
                (255, 255, 255),  # White text
                font_thickness,
            )

    return image


def _draw_boxes_pil(
    image: NDArray[Any],
    boxes: NDArray[Any],
    label_indices: NDArray[Any] | None,
    scores: NDArray[Any] | None,
    index2label: dict[int, str] | None,
    color: tuple[int, int, int],
    thickness: int,
) -> NDArray[Any]:
    """
    Draw bounding boxes using PIL.

    Parameters
    ----------
    image : NDArray
        Image array to draw on
    boxes : NDArray
        Bounding boxes in XYXY format
    label_indices : NDArray or None
        Array of label indices
    scores : NDArray or None
        Confidence scores for each box
    index2label : dict[int, str] or None
        Mapping from class indices to class names
    color : tuple[int, int, int]
        RGB color for bounding boxes
    thickness : int
        Line thickness for bounding boxes

    Returns
    -------
    NDArray
        Image with bounding boxes drawn
    """
    from PIL import Image, ImageDraw, ImageFont

    # Convert to PIL Image
    pil_img = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_img)

    # Try to load a font, fallback to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except Exception:
        font = ImageFont.load_default()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(int)

        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=tuple(color), width=thickness)

        # Build label text
        if label_indices is not None:
            label_idx = label_indices[i]
            score_i = scores[i] if scores is not None and i < len(scores) else None
            label_text = _build_label_text(label_idx, score_i, index2label)

            # Draw text background
            try:
                bbox = draw.textbbox((x1, y1 - 15), label_text, font=font)
                draw.rectangle(bbox, fill=tuple(color))
                draw.text((x1, y1 - 15), label_text, fill=(255, 255, 255), font=font)
            except Exception:
                # Fallback for older PIL versions
                draw.text((x1, y1 - 15), label_text, fill=tuple(color), font=font)

    # Convert back to numpy
    return np.array(pil_img)


def draw_bounding_boxes(
    image: NDArray[Any],
    boxes: NDArray[Any],
    labels: NDArray[Any] | None = None,
    scores: NDArray[Any] | None = None,
    index2label: dict[int, str] | None = None,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> NDArray[Any]:
    """
    Draw bounding boxes on an image.

    Parameters
    ----------
    image : NDArray
        Image array in HWC format (uint8)
    boxes : NDArray
        Bounding boxes in XYXY format, shape (N, 4)
    labels : NDArray or None, default None
        Label indices or probabilities for each box
    scores : NDArray or None, default None
        Confidence scores for each box
    index2label : dict[int, str] or None, default None
        Mapping from class indices to class names
    color : tuple[int, int, int], default (0, 255, 0)
        RGB color for bounding boxes
    thickness : int, default 2
        Line thickness for bounding boxes

    Returns
    -------
    NDArray
        Image with bounding boxes drawn (copy of input)

    Notes
    -----
    This function requires opencv-python (cv2) to be installed.
    If cv2 is not available, it falls back to using PIL which provides
    basic rectangle drawing but with less features.
    """
    # Make a copy to avoid modifying the original
    img_with_boxes = image.copy()

    if len(boxes) == 0:
        return img_with_boxes

    # Process labels to get class indices
    label_indices = _get_label_indices(labels)

    # Try to use OpenCV, fall back to PIL if not available
    try:
        import cv2  # type: ignore  # noqa: F401

        return _draw_boxes_opencv(img_with_boxes, boxes, label_indices, scores, index2label, color, thickness)
    except ImportError:
        return _draw_boxes_pil(img_with_boxes, boxes, label_indices, scores, index2label, color, thickness)


def process_dataset_item_for_display(
    datum: Any,
    additional_metadata: dict[str, Any] | None = None,
    index2label: dict[int, str] | None = None,
) -> tuple[NDArray[Any], Any | None, dict[str, Any]]:
    """
    Process a dataset item for display in image grids.

    This function consolidates all the common processing steps:
    1. Parse dataset item (image, target, metadata)
    2. Merge with additional metadata if provided
    3. Convert image to HWC format
    4. Extract boxes and labels from target
    5. Normalize image to uint8
    6. Draw bounding boxes if present

    Parameters
    ----------
    datum : Any
        Dataset item to process (can be image, tuple of (image,), (image, target),
        or (image, target, metadata))
    additional_metadata : dict[str, Any] or None, default None
        Additional metadata to merge with item's metadata
    index2label : dict[int, str] or None, default None
        Mapping from class indices to class names for bounding box labels

    Returns
    -------
    tuple[NDArray, Any | None, dict[str, Any]]
        (processed_image, target, merged_metadata) where:
        - processed_image: Image in HWC uint8 format with bounding boxes drawn if applicable
        - target: The original target data
        - merged_metadata: Combined metadata dictionary
    """
    # Step 1: Parse dataset item
    image, target, metadata = parse_dataset_item(datum)

    # Step 2: Merge with additional metadata if provided
    if additional_metadata is not None:
        metadata = merge_metadata(metadata, additional_metadata)

    # Step 3: Convert image to HWC format
    image_hwc = image_to_hwc(image)

    # Step 4: Extract boxes and labels from target
    boxes, labels, scores = extract_boxes_and_labels(target)

    # Step 5: Normalize image to uint8
    image_uint8 = normalize_image_to_uint8(image_hwc)

    # Step 6: Draw bounding boxes if present
    if boxes is not None and len(boxes) > 0:
        image_uint8 = draw_bounding_boxes(
            image_uint8,
            boxes,
            labels,
            scores,
            index2label,
        )

    return image_uint8, target, metadata
