"""Protocols for plottable DataEval outputs.

This module defines structural protocols that DataEval outputs must implement
to be plottable. This provides loose coupling between dataeval and dataeval-plots
packages while maintaining type safety.

The protocol hierarchy:
- Plottable: Base protocol with plot_type discrimination
- Type-specific protocols: Define exact attributes needed for each plot type
  - PlottableCoverage
  - PlottableBalance
  - PlottableDiversity
  - PlottableSufficiency
  - PlottableBaseStats
  - PlottableDriftMVDC
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

import numpy as np
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


@runtime_checkable
class Indexable(Protocol):
    """Protocol for indexable collection."""

    def __len__(self) -> int: ...
    def __getitem__(self, index: Any) -> Any: ...


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

    Type Parameters
    ----------------
    _T_co : covariant
        The type of items returned by __getitem__

    Methods
    -------
    __getitem__(index: int) -> _T_co
        Retrieve an item from the dataset by integer index
    __len__() -> int
        Return the number of items in the dataset

    Properties
    ----------
    metadata : DatasetMetadata
        Dataset metadata including id and optional index2label mapping
    """

    def __getitem__(self, index: int, /) -> Any: ...
    def __len__(self) -> int: ...
    @property
    def metadata(self) -> DatasetMetadata: ...


@runtime_checkable
class Plottable(Protocol):
    """Base protocol for all plottable DataEval outputs.

    Any object that wants to be plottable must implement:
    1. A plot_type property/method that returns the plot type identifier
    2. A meta() method that returns execution metadata (optional but recommended)
    """

    def plot_type(self) -> str:
        """Return the plot type identifier for routing to appropriate plot function.

        Returns
        -------
        str
            One of: 'coverage', 'balance', 'diversity', 'sufficiency',
            'base_stats', 'drift_mvdc'
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
class PlottableCoverage(Plottable, Protocol):
    """Protocol for coverage plot outputs.

    Required attributes:
    - uncovered_indices: Array of indices for uncovered samples
    - plot_type() -> 'coverage'
    """

    @property
    def uncovered_indices(self) -> NDArray[np.integer[Any]]: ...

    def plot_type(self) -> Literal["coverage"]:
        """Return 'coverage' as the plot type."""
        ...


@runtime_checkable
class PlottableBalance(Plottable, Protocol):
    """Protocol for balance plot outputs.

    Required attributes:
    - class_names: Names of classes
    - factor_names: Names of factors
    - classwise: Per-class balance matrix
    - balance: Overall balance scores
    - factors: Factor correlation matrix
    - plot_type() -> 'balance'
    """

    @property
    def class_names(self) -> Sequence[str]: ...
    @property
    def factor_names(self) -> Sequence[str]: ...
    @property
    def classwise(self) -> NDArray[Any]: ...
    @property
    def balance(self) -> NDArray[Any]: ...
    @property
    def factors(self) -> NDArray[Any]: ...

    def plot_type(self) -> Literal["balance"]:
        """Return 'balance' as the plot type."""
        ...


@runtime_checkable
class PlottableDiversity(Plottable, Protocol):
    """Protocol for diversity plot outputs.

    Required attributes:
    - class_names: Names of classes
    - factor_names: Names of factors
    - classwise: Per-class diversity matrix
    - diversity_index: Overall diversity indices
    - plot_type() -> 'diversity'
    """

    @property
    def class_names(self) -> Sequence[str]: ...
    @property
    def factor_names(self) -> Sequence[str]: ...
    @property
    def classwise(self) -> NDArray[Any]: ...
    @property
    def diversity_index(self) -> NDArray[Any]: ...

    def plot_type(self) -> Literal["diversity"]:
        """Return 'diversity' as the plot type."""
        ...


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

    def plot_type(self) -> Literal["sufficiency"]:
        """Return 'sufficiency' as the plot type."""
        ...


@runtime_checkable
class PlottableBaseStats(Plottable, Protocol):
    """Protocol for base statistics plot outputs.

    Required methods:
    - _get_channels(): Get channel information for plotting
    - factors(): Get factor data for histogram plotting
    - plot_type() -> 'base_stats'
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

    def plot_type(self) -> Literal["base_stats"]:
        """Return 'base_stats' as the plot type."""
        ...


@runtime_checkable
class PlottableDriftMVDC(Plottable, Protocol):
    """Protocol for drift MVDC plot outputs.

    Required methods:
    - to_dataframe(): Convert drift results to pandas DataFrame
    - plot_type() -> 'drift_mvdc'
    """

    def to_dataframe(self) -> Any:  # pandas.DataFrame
        """Convert drift detection results to DataFrame.

        Returns
        -------
        pandas.DataFrame
            DataFrame with drift detection results including chunks,
            metrics, thresholds, and alerts
        """
        ...

    def plot_type(self) -> Literal["drift_mvdc"]:
        """Return 'drift_mvdc' as the plot type."""
        ...


# Type alias for all plottable types
PlottableType = (
    Dataset
    | PlottableCoverage
    | PlottableBalance
    | PlottableDiversity
    | PlottableSufficiency
    | PlottableBaseStats
    | PlottableDriftMVDC
)
