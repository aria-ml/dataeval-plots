"""Shared test fixtures for dataeval-plots tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import polars as pl
import pytest
from numpy.typing import NDArray


@dataclass
class MockExecutionMetadata:
    """Mock execution metadata."""

    arguments: dict[str, Any]
    state: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.state is None:
            self.state = {}

    def __getitem__(self, key: str) -> Any:
        return self.arguments[key]


@dataclass
class MockPlottableBalance:
    """Mock balance output for testing."""

    balance: pl.DataFrame
    factors: pl.DataFrame
    classwise: pl.DataFrame

    @property
    def plot_type(self) -> Literal["balance"]:
        return "balance"

    def meta(self) -> MockExecutionMetadata:
        return MockExecutionMetadata(arguments={})


@dataclass
class MockPlottableDiversity:
    """Mock diversity output for testing."""

    factors: pl.DataFrame
    classwise: pl.DataFrame

    @property
    def plot_type(self) -> Literal["diversity"]:
        return "diversity"

    def meta(self) -> MockExecutionMetadata:
        return MockExecutionMetadata(arguments={}, state={"method": "shannon"})


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
class MockPlottableStats:
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

    def plot_type(self) -> Literal["stats"]:
        return "stats"

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


@dataclass
class MockDataset:
    """Mock MAITE-compatible dataset for testing."""

    images: list[NDArray[np.uint8]]
    dataset_id: str = "mock_dataset"
    index2label: dict[int, str] | None = None
    targets: list[Any] | None = None
    metadatas: list[dict[str, Any]] | None = None

    def __getitem__(self, index: int) -> tuple[NDArray[np.uint8], Any, dict[str, Any]]:
        """Get image at index."""
        image = self.images[index]
        target = self.targets[index] if self.targets else np.array([])
        meta = self.metadatas[index] if self.metadatas else {"id": index}
        return (image, target, meta)

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.images)

    @property
    def metadata(self) -> dict[str, Any]:
        """Get dataset metadata."""
        result: dict[str, Any] = {"id": self.dataset_id}
        if self.index2label is not None:
            result["index2label"] = self.index2label
        return result


# Fixtures for common mock data


@pytest.fixture
def mock_balance() -> MockPlottableBalance:
    """Create mock balance output."""
    n_factors = 5
    class_names = ["class_0", "class_1", "class_2"]
    factor_names = ["factor_0", "factor_1", "factor_2", "factor_3", "factor_4"]

    # Create balance DataFrame: factor_name, mi_value
    # Include "class_label" plus all factors
    balance_df = pl.DataFrame(
        {
            "factor_name": ["class_label"] + factor_names,
            "mi_value": np.random.rand(n_factors + 1),
        }
    )

    # Create factors DataFrame: factor1, factor2, mi_value, is_correlated
    # Generate all pairwise combinations
    factor_pairs = []
    for i, f1 in enumerate(factor_names):
        for j, f2 in enumerate(factor_names):
            if i < j:  # Only upper triangle
                factor_pairs.append(
                    {
                        "factor1": f1,
                        "factor2": f2,
                        "mi_value": np.random.rand(),
                        "is_correlated": np.random.rand() > 0.5,
                    }
                )
    factors_df = pl.DataFrame(factor_pairs)

    # Create classwise DataFrame: class_name, factor_name, mi_value, is_imbalanced
    classwise_data = []
    for class_name in class_names:
        for factor_name in factor_names:
            classwise_data.append(
                {
                    "class_name": class_name,
                    "factor_name": factor_name,
                    "mi_value": np.random.rand(),
                    "is_imbalanced": np.random.rand() > 0.5,
                }
            )
    classwise_df = pl.DataFrame(classwise_data)

    return MockPlottableBalance(
        balance=balance_df,
        factors=factors_df,
        classwise=classwise_df,
    )


@pytest.fixture
def mock_diversity() -> MockPlottableDiversity:
    """Create mock diversity output."""
    n_factors = 4
    class_names = ["class_0", "class_1", "class_2"]
    factor_names = ["factor_0", "factor_1", "factor_2", "factor_3"]

    # Create factors DataFrame: factor_name, diversity_value, is_low_diversity
    factors_df = pl.DataFrame(
        {
            "factor_name": factor_names,
            "diversity_value": np.random.rand(n_factors),
            "is_low_diversity": [np.random.rand() > 0.5 for _ in range(n_factors)],
        }
    )

    # Create classwise DataFrame: class_name, factor_name, diversity_value, is_low_diversity
    classwise_data = []
    for class_name in class_names:
        for factor_name in factor_names:
            classwise_data.append(
                {
                    "class_name": class_name,
                    "factor_name": factor_name,
                    "diversity_value": np.random.rand(),
                    "is_low_diversity": np.random.rand() > 0.5,
                }
            )
    classwise_df = pl.DataFrame(classwise_data)

    return MockPlottableDiversity(
        factors=factors_df,
        classwise=classwise_df,
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
def mock_stats_single_channel() -> MockPlottableStats:
    """Create mock base stats output with single channel."""
    return MockPlottableStats(
        _factors={
            "factor_0": np.random.rand(100),
            "factor_1": np.random.rand(100),
            "factor_2": np.random.rand(100),
        },
        _n_channels=1,
        _channel_mask=None,
    )


@pytest.fixture
def mock_stats_multi_channel() -> MockPlottableStats:
    """Create mock base stats output with multiple channels."""
    n_samples = 100
    n_channels = 3

    # Use channelwise metric names that are recognized by channel_histogram_plot
    return MockPlottableStats(
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


@pytest.fixture
def mock_dataset() -> MockDataset:
    """Create mock dataset with sample images."""
    # Create some random RGB images (channels-first format: C, H, W)
    np.random.seed(42)
    images = [np.random.randint(0, 256, (3, 32, 32), dtype=np.uint8) for _ in range(9)]

    return MockDataset(
        images=images,
        dataset_id="test_dataset",
        index2label={0: "cat", 1: "dog", 2: "bird"},
    )
