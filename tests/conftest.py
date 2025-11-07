"""Shared test fixtures for dataeval-plots tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pytest
from numpy.typing import NDArray


@dataclass
class MockExecutionMetadata:
    """Mock execution metadata."""

    arguments: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.arguments[key]


@dataclass
class MockPlottableCoverage:
    """Mock coverage output for testing."""

    uncovered_indices: NDArray[np.int64]

    def plot_type(self) -> Literal["coverage"]:
        return "coverage"

    def meta(self) -> MockExecutionMetadata:
        return MockExecutionMetadata(arguments={})


@dataclass
class MockPlottableBalance:
    """Mock balance output for testing."""

    class_names: list[str]
    factor_names: list[str]
    classwise: NDArray[np.float64]
    balance: NDArray[np.float64]
    factors: NDArray[np.float64]

    def plot_type(self) -> Literal["balance"]:
        return "balance"

    def meta(self) -> MockExecutionMetadata:
        return MockExecutionMetadata(arguments={})


@dataclass
class MockPlottableDiversity:
    """Mock diversity output for testing."""

    class_names: list[str]
    factor_names: list[str]
    classwise: NDArray[np.float64]
    diversity_index: NDArray[np.float64]

    def plot_type(self) -> Literal["diversity"]:
        return "diversity"

    def meta(self) -> MockExecutionMetadata:
        return MockExecutionMetadata(arguments={"method": "shannon"})


@dataclass
class MockPlottableSufficiency:
    """Mock sufficiency output for testing."""

    steps: NDArray[np.uint32]
    averaged_measures: dict[str, NDArray[np.float64]]
    measures: dict[str, NDArray[np.float64]]
    params: dict[str, NDArray[np.float64]]

    def plot_type(self) -> Literal["sufficiency"]:
        return "sufficiency"

    def meta(self) -> MockExecutionMetadata:
        return MockExecutionMetadata(arguments={})


@dataclass
class MockPlottableBaseStats:
    """Mock base stats output for testing."""

    _factors: dict[str, NDArray[np.float64]]
    _n_channels: int
    _channel_mask: list[bool] | None

    def _get_channels(
        self,
        channel_limit: int | None = None,
        channel_index: int | list[int] | None = None,
    ) -> tuple[int, list[bool] | None]:
        return self._n_channels, self._channel_mask

    def factors(self, exclude_constant: bool = True) -> dict[str, NDArray[np.float64]]:
        return self._factors

    def plot_type(self) -> Literal["base_stats"]:
        return "base_stats"

    def meta(self) -> MockExecutionMetadata:
        return MockExecutionMetadata(arguments={})


@dataclass
class MockPlottableDriftMVDC:
    """Mock drift MVDC output for testing."""

    _df: Any  # pandas.DataFrame

    def to_dataframe(self) -> Any:
        return self._df

    def plot_type(self) -> Literal["drift_mvdc"]:
        return "drift_mvdc"

    def meta(self) -> MockExecutionMetadata:
        return MockExecutionMetadata(arguments={})


# Fixtures for common mock data


@pytest.fixture
def mock_coverage() -> MockPlottableCoverage:
    """Create mock coverage output."""
    return MockPlottableCoverage(
        uncovered_indices=np.array([0, 5, 10, 15, 20], dtype=np.int64),
    )


@pytest.fixture
def mock_balance() -> MockPlottableBalance:
    """Create mock balance output."""
    n_classes = 3
    n_factors = 5

    # Working backwards from the expected result:
    # - Final display is (n_factors-1) x (n_factors-1) with labels factor_names[:-1] x factor_names[1:]
    # - Before [:-1] slice: (n_factors) x (n_factors-1)
    # - This comes from concat of balance[np.newaxis, 1:] and factors
    # - So balance[1:] must have (n_factors-1) elements -> balance has n_factors elements
    # - And factors must be (n_factors-1) x (n_factors-1)
    # - The last row that gets dropped by [:-1] is from the last row of factors
    return MockPlottableBalance(
        class_names=["class_0", "class_1", "class_2"],
        factor_names=["factor_0", "factor_1", "factor_2", "factor_3", "factor_4"],
        classwise=np.random.rand(n_classes, n_factors),
        balance=np.random.rand(n_factors),  # n_factors elements, balance[1:] gives n_factors-1
        factors=np.random.rand(n_factors - 1, n_factors - 1),  # (4, 4)
    )


@pytest.fixture
def mock_diversity() -> MockPlottableDiversity:
    """Create mock diversity output."""
    n_classes = 3
    n_factors = 4

    # diversity_index needs one extra element for "class_labels"
    return MockPlottableDiversity(
        class_names=["class_0", "class_1", "class_2"],
        factor_names=["factor_0", "factor_1", "factor_2", "factor_3"],
        classwise=np.random.rand(n_classes, n_factors),
        diversity_index=np.random.rand(n_factors + 1),
    )


@pytest.fixture
def mock_sufficiency_single_class() -> MockPlottableSufficiency:
    """Create mock sufficiency output with single class."""
    steps = np.array([10, 50, 100, 500, 1000], dtype=np.uint32)
    n_steps = len(steps)
    n_runs = 5

    return MockPlottableSufficiency(
        steps=steps,
        averaged_measures={
            "accuracy": np.array([0.5, 0.65, 0.75, 0.85, 0.90]),
            "f1": np.array([0.45, 0.60, 0.70, 0.80, 0.88]),
        },
        measures={
            "accuracy": np.random.rand(n_runs, n_steps),
            "f1": np.random.rand(n_runs, n_steps),
        },
        params={
            "accuracy": np.array([0.5, 0.5, 0.1]),
            "f1": np.array([0.55, 0.5, 0.12]),
        },
    )


@pytest.fixture
def mock_sufficiency_multi_class() -> MockPlottableSufficiency:
    """Create mock sufficiency output with multiple classes."""
    steps = np.array([10, 50, 100, 500, 1000], dtype=np.uint32)
    n_steps = len(steps)
    n_runs = 5
    n_classes = 3

    return MockPlottableSufficiency(
        steps=steps,
        averaged_measures={
            "accuracy": np.random.rand(n_classes, n_steps),
            "f1": np.random.rand(n_classes, n_steps),
        },
        measures={
            "accuracy": np.random.rand(n_runs, n_steps, n_classes),
            "f1": np.random.rand(n_runs, n_steps, n_classes),
        },
        params={
            "accuracy": np.random.rand(n_classes, 3),
            "f1": np.random.rand(n_classes, 3),
        },
    )


@pytest.fixture
def mock_base_stats_single_channel() -> MockPlottableBaseStats:
    """Create mock base stats output with single channel."""
    return MockPlottableBaseStats(
        _factors={
            "factor_0": np.random.rand(100),
            "factor_1": np.random.rand(100),
            "factor_2": np.random.rand(100),
        },
        _n_channels=1,
        _channel_mask=None,
    )


@pytest.fixture
def mock_base_stats_multi_channel() -> MockPlottableBaseStats:
    """Create mock base stats output with multiple channels."""
    n_samples = 100
    n_channels = 3

    # Use channelwise metric names that are recognized by channel_histogram_plot
    return MockPlottableBaseStats(
        _factors={
            "mean": np.random.rand(n_samples, n_channels),
            "std": np.random.rand(n_samples, n_channels),
            "var": np.random.rand(n_samples, n_channels),
        },
        _n_channels=n_channels,
        _channel_mask=None,  # Let the function handle masking internally
    )


@pytest.fixture
def mock_drift_mvdc() -> MockPlottableDriftMVDC:
    """Create mock drift MVDC output."""
    import pandas as pd

    n_points = 50

    # Create a DataFrame with MultiIndex columns to support nested access like df["chunk"]["period"]
    periods = ["reference"] * 25 + ["analysis"] * 25
    values = np.random.rand(n_points)
    alerts = [False] * 40 + [True] * 10

    # Create MultiIndex columns: first level is the metric group, second level is the field
    columns = pd.MultiIndex.from_tuples(
        [
            ("chunk", "period"),
            ("domain_classifier_auroc", "value"),
            ("domain_classifier_auroc", "lower_threshold"),
            ("domain_classifier_auroc", "upper_threshold"),
            ("domain_classifier_auroc", "alert"),
        ]
    )

    data = {
        ("chunk", "period"): periods,
        ("domain_classifier_auroc", "value"): values,
        ("domain_classifier_auroc", "lower_threshold"): [0.4] * n_points,
        ("domain_classifier_auroc", "upper_threshold"): [0.6] * n_points,
        ("domain_classifier_auroc", "alert"): alerts,
    }

    df = pd.DataFrame(data, columns=columns)

    return MockPlottableDriftMVDC(_df=df)
