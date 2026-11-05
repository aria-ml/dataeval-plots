"""Protocols for plottable DataEval outputs.

This module defines structural protocols that DataEval outputs must implement
to be plottable. This provides loose coupling between dataeval and dataeval-plots
packages while maintaining type safety.

The protocol hierarchy:
- Plottable: Base protocol with plot_type discrimination
- Type-specific protocols: Define exact attributes needed for each plot type
  - PlottableBalance
  - PlottableDiversity
  - PlottableSufficiency
  - PlottableStats
  - PlottableDriftMVDC
  - PlottableDrift
  - PlottableFeasibilityDeterminations
  - PlottableFeasibilityDecisions
  - PlottableFeasibility (either of the two feasibility shapes)
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from typing import (
    Any,
    Literal,
    NotRequired,
    Protocol,
    Required,
    TypedDict,
    runtime_checkable,
)

import polars as pl
from numpy.typing import NDArray
from typing_extensions import ReadOnly


class ExecutionMetadata(Protocol):
    """Metadata about the execution of a DataEval function.

    This is a minimal stub for type checking. The actual ExecutionMetadata
    is defined in the dataeval package.

    Attributes
    ----------
    name : str
        Name of the function or method
    execution_time : datetime
        Time of execution
    execution_duration : float
        Duration of execution in seconds
    arguments : dict[str, Any]
        Arguments passed to the function or method
    state : dict[str, Any]
        State attributes of the executing class
    version : str
        Version of DataEval
    """

    @property
    def name(self) -> str: ...
    @property
    def execution_time(self) -> datetime: ...
    @property
    def execution_duration(self) -> float: ...
    @property
    def arguments(self) -> Mapping[str, Any]: ...
    @property
    def state(self) -> Mapping[str, Any]: ...
    @property
    def version(self) -> str: ...


class DatasetMetadata(TypedDict, total=False):
    """Metadata for MAITE datasets.

    Attributes
    ----------
    id : int or str
        Dataset identifier (read-only)
    index2label : dict[int, str], optional
        Mapping from class indices to class labels (read-only)
    """

    id: Required[ReadOnly[str]]
    index2label: NotRequired[ReadOnly[dict[int, str]]]


@runtime_checkable
class Dataset(Protocol):
    """Protocol for MAITE-compatible datasets.

    This protocol defines the interface for datasets following the MAITE
    (Modular AI Trustworthy Engineering) specification. Datasets implementing
    this protocol can be used for image grid plotting and other visualization tasks.

    Methods
    -------
    __getitem__(index: int) -> Any
        Retrieve an item from the dataset by integer index
    __len__() -> int
        Return the number of items in the dataset

    Properties
    ----------
    metadata : Mapping[str, Any]
        Dataset metadata including id and optional index2label mapping
    """

    @property
    def metadata(self) -> Mapping[str, Any]: ...
    def __getitem__(self, index: int, /) -> Any: ...
    def __len__(self) -> int: ...


@runtime_checkable
class Plottable(Protocol):
    """Base protocol for all plottable DataEval outputs.

    Any object that wants to be plottable must implement:
    1. A plot_type property/method that returns the plot type identifier
    2. A meta() method that returns execution metadata (optional but recommended)
    """

    @property
    def plot_type(
        self,
    ) -> Literal["balance", "diversity", "sufficiency", "stats", "drift_mvdc", "drift", "feasibility"]:
        """Return the plot type identifier for routing to appropriate plot function.

        Returns
        -------
        str
            One of: 'balance', 'diversity', 'sufficiency', 'stats', 'drift_mvdc', 'drift', 'feasibility'
        """
        ...

    def meta(self) -> ExecutionMetadata:
        """Return execution metadata for the output.

        Returns
        -------
        ExecutionMetadata
            Metadata about the execution of the function that created this output
        """
        ...


@runtime_checkable
class PlottableBalance(Plottable, Protocol):
    """Protocol for balance plot outputs.

    Required attributes:
    - balance: Global class-to-factor MI values (DataFrame)
    - factors: Factor correlation data (DataFrame)
    - classwise: Per-class balance data (DataFrame)
    - plot_type -> 'balance'

    DataFrame structures:
    - balance DataFrame columns: factor_name, mi_value
    - factors DataFrame columns: factor1, factor2, mi_value, is_correlated
    - classwise DataFrame columns: class_name, factor_name, mi_value, is_imbalanced
    """

    @property
    def balance(self) -> pl.DataFrame: ...
    @property
    def factors(self) -> pl.DataFrame: ...
    @property
    def classwise(self) -> pl.DataFrame: ...
    @property
    def plot_type(self) -> Literal["balance"]: ...


@runtime_checkable
class PlottableDiversity(Plottable, Protocol):
    """Protocol for diversity plot outputs.

    Required attributes:
    - factors: Factor-level diversity data (DataFrame)
    - classwise: Per-class diversity data (DataFrame)
    - plot_type -> 'diversity'

    DataFrame structures:
    - factors DataFrame columns: factor_name, diversity_value, is_low_diversity
    - classwise DataFrame columns: class_name, factor_name, diversity_value, is_low_diversity
    """

    @property
    def factors(self) -> pl.DataFrame: ...
    @property
    def classwise(self) -> pl.DataFrame: ...
    @property
    def plot_type(self) -> Literal["diversity"]: ...


@runtime_checkable
class PlottableSufficiency(Plottable, Protocol):
    """Protocol for sufficiency plot outputs.

    Required attributes:
    - steps: Array of data size steps
    - averaged_measures: Averaged performance measures across steps
    - measures: Per-run performance measures
    - params: Fitted parameters for sufficiency curves
    - plot_type() -> 'sufficiency'
    """

    @property
    def steps(self) -> NDArray[Any]: ...
    @property
    def averaged_measures(self) -> Mapping[str, NDArray[Any]]: ...
    @property
    def measures(self) -> Mapping[str, NDArray[Any]]: ...
    @property
    def params(self) -> Mapping[str, NDArray[Any]]: ...
    @property
    def plot_type(self) -> Literal["sufficiency"]: ...


@runtime_checkable
class PlottableStats(Plottable, Protocol):
    """Protocol for base statistics plot outputs.

    Required methods:
    - _get_channels(): Get channel information for plotting
    - factors(): Get factor data for histogram plotting
    - plot_type() -> 'stats'
    """

    def _get_channels(
        self,
        channel_limit: int | None = None,
        channel_index: int | Iterable[int] | None = None,
    ) -> tuple[int, Sequence[bool] | None]:
        """Get channel information for plotting.

        Parameters
        ----------
        channel_limit : int or None
            Maximum number of channels to include
        channel_index : int, Iterable[int] or None
            Specific channel indices to include

        Returns
        -------
        tuple[int, Sequence[bool] | None]
            Number of channels and channel mask
        """
        ...

    def factors(self, exclude_constant: bool = True) -> dict[str, NDArray[Any]]:
        """Get factor data for plotting.

        Parameters
        ----------
        exclude_constant : bool
            Whether to exclude constant factors

        Returns
        -------
        dict[str, NDArray]
            Dictionary mapping factor names to their data arrays
        """
        ...

    @property
    def plot_type(self) -> Literal["stats"]: ...


@runtime_checkable
class PlottableDriftMVDC(Plottable, Protocol):
    """Protocol for drift MVDC plot outputs.

    Required methods:
    - data(): Drift results as polars DataFrame
    - plot_type() -> 'drift_mvdc'
    """

    def data(self) -> pl.DataFrame:
        """Drift detection results as a polars DataFrame.

        Returns
        -------
        pl.DataFrame
            DataFrame with drift detection results including chunks,
            metrics, thresholds, and alerts
        """
        ...

    @property
    def plot_type(self) -> Literal["drift_mvdc"]: ...


@runtime_checkable
class PlottableDrift(Plottable, Protocol):
    """Protocol for chunked drift detector outputs.

    Required attributes:
    - details: Per-chunk results as a polars DataFrame
    - metric_name: Name of the drift metric that each chunk reports
    - plot_type -> 'drift'
    """

    @property
    def details(self) -> Any:
        """Per-chunk drift results.

        Returns
        -------
        pl.DataFrame
            One row per chunk with the columns ``index``, ``value``, ``lower_threshold``, ``upper_threshold`` and
            ``drifted``. A detector that was not chunked holds a mapping of statistics instead, which cannot be
            plotted.
        """
        ...

    @property
    def metric_name(self) -> str:
        """Name of the metric that ``value`` reports, such as ``auroc`` or ``mmd2``."""
        ...

    @property
    def plot_type(self) -> Literal["drift"]: ...


class FeasibilityEndpoint(Protocol):
    """
    One bound on one target, as DataEval's ``Endpoint`` records it; only the fields the plot reads.

    Newer outputs also give ``quantity``, ``"metric"`` or ``"false_positive_rate"``; the plot reads it with a default
    of ``"metric"`` so older outputs still plot.
    """

    @property
    def target_spec(self) -> str: ...
    @property
    def value(self) -> float | None: ...
    @property
    def direction(self) -> Literal["lower", "upper"]: ...
    @property
    def state(self) -> str: ...


class FeasibilityEvidence(Protocol):
    """One evidence item, as DataEval's ``Evidence`` records it; only the fields the plot reads."""

    @property
    def kind(self) -> str: ...
    @property
    def information_scope(self) -> str: ...
    @property
    def unit(self) -> str: ...
    @property
    def endpoints(self) -> Sequence[FeasibilityEndpoint]: ...
    @property
    def details(self) -> Mapping[str, Any]: ...


@runtime_checkable
class PlottableFeasibilityDeterminations(Plottable, Protocol):
    """Protocol for feasibility plot outputs in the ``determinations`` and ``evidence`` shape.

    Required attributes:
    - determinations: One row per target (DataFrame)
    - evidence: Every evidence item of the call, by evidence id
    - meta().state["alpha"]: The call's error budget
    - plot_type -> 'feasibility'

    Structures:
    - determinations columns: target_spec, metric, threshold, judged_scope, determination, ceiling,
      ceiling_scope, witness, witness_source
    - each evidence item: kind, information_scope, unit (``item`` or ``group``), endpoints (target_spec, value,
      direction, state) and details["endpoints"][target_spec], which is None or holds the bound's mean and n
    """

    @property
    def determinations(self) -> pl.DataFrame: ...
    @property
    def evidence(self) -> Mapping[str, FeasibilityEvidence]: ...
    @property
    def plot_type(self) -> Literal["feasibility"]: ...


@runtime_checkable
class PlottableFeasibilityDecisions(Plottable, Protocol):
    """Protocol for feasibility plot outputs in the ``decisions`` and ``bounds`` shape.

    Required attributes:
    - decisions: One row per requirement (DataFrame)
    - bounds: One row per source, requirement and quantity (DataFrame)
    - meta().state["alpha"]: The call's error budget
    - plot_type -> 'feasibility'

    Structures:
    - decisions columns: requirement, metric, threshold, judged_scope, class_name, aggregation, ap_functional,
      max_false_positive_rate, decision, limitation, envelope, envelope_scope, witness, witness_source
    - bounds columns: requirement, source, role, quantity, bound_type, information_scope, unit, state, bound,
      observed, n; ``requirement`` joins a bound to its decisions row

    Enum columns are read as strings, with these values:
    - decision: demonstrated_feasible, infeasible_under_recorded_assumptions, not_ruled_out, unresolved
    - limitation: only ``conflict`` is read; with decision ``unresolved`` it marks a conflict between a witness and an
      envelope
    - judged_scope, envelope_scope, information_scope: full_input, fixed_representation, partition_restricted
    - source, witness_source: local_panels, replicas, knn, linear_probe, user_model (any other value shows its text)
    - role: envelope, witness. bound_type: lower, upper
    - quantity: a metric name (accuracy, ap, recall, recall_at_fp, mota) or false_positive_rate
    - state: unavailable, uninformative, informative. unit: image, sequence
    - aggregation: frame_pooled, sequence_mean. ap_functional: population_ap_integral, population_ap_101
    """

    @property
    def decisions(self) -> pl.DataFrame: ...
    @property
    def bounds(self) -> pl.DataFrame: ...
    @property
    def plot_type(self) -> Literal["feasibility"]: ...


PlottableFeasibility = PlottableFeasibilityDeterminations | PlottableFeasibilityDecisions
"""A feasibility output in either shape; DataEval 1.2.0 gives the first, later releases the second."""

# Type alias for all plottable types
PlottableType = (
    Dataset
    | PlottableBalance
    | PlottableDiversity
    | PlottableSufficiency
    | PlottableStats
    | PlottableDriftMVDC
    | PlottableDrift
    | PlottableFeasibility
)
