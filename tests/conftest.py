"""Shared test fixtures for dataeval-plots tests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import polars as pl
import pytest
from numpy.typing import NDArray


@pytest.fixture(autouse=True)
def _close_figures():
    """Close the figures a test opens, so they don't pile up across the suite."""
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


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

    _df: Any  # polars.DataFrame

    def data(self) -> Any:
        """Return the data as a polars DataFrame."""
        return self._df

    def plot_type(self) -> Literal["drift_mvdc"]:
        return "drift_mvdc"

    def meta(self) -> MockExecutionMetadata:
        return MockExecutionMetadata(arguments={})


@dataclass
class MockPlottableFeasibility:
    """Mock feasibility output for testing."""

    determinations: pl.DataFrame
    evidence_table: pl.DataFrame
    alpha: float = 0.05

    def aggregate_by_evidence(self) -> pl.DataFrame:
        return self.evidence_table

    @property
    def plot_type(self) -> Literal["feasibility"]:
        return "feasibility"

    def meta(self) -> MockExecutionMetadata:
        return MockExecutionMetadata(arguments={}, state={"alpha": self.alpha})


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
    # Note: balance DataFrame is NOT sorted; sorting happens in prepare_balance_data
    balance_df = pl.DataFrame(
        {
            "factor_name": ["class_label"] + factor_names,
            "mi_value": np.random.rand(n_factors + 1),
        }
    )

    # Create factors DataFrame: factor1, factor2, mi_value, is_correlated
    # Generate all pairwise combinations (full matrix including diagonal)
    factor_pairs = []
    for i, f1 in enumerate(factor_names):
        for j, f2 in enumerate(factor_names):
            # Create full matrix including diagonal (self-MI = 1.0)
            mi_value = 1.0 if i == j else np.random.rand()
            factor_pairs.append(
                {
                    "factor1": f1,
                    "factor2": f2,
                    "mi_value": mi_value,
                    "is_correlated": np.random.rand() > 0.5 if i != j else True,
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
    n_points = 50

    # Create a Polars DataFrame with the expected column names
    periods = ["reference"] * 25 + ["analysis"] * 25
    values = np.random.rand(n_points)
    alerts = [False] * 40 + [True] * 10

    # Create Polars DataFrame with flat column names
    df = pl.DataFrame(
        {
            "chunk_period": periods,
            "domain_classifier_auroc_value": values,
            "domain_classifier_auroc_lower_threshold": [0.4] * n_points,
            "domain_classifier_auroc_upper_threshold": [0.6] * n_points,
            "domain_classifier_auroc_alert": alerts,
        }
    )

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


def _target_spec(threshold: float, scope: str, metric: str, class_name: str | None, aggregation: str | None) -> str:
    """Canonical target JSON as DataEval writes it: sorted keys, nulls kept, versioned."""
    spec = {
        "aggregation": aggregation,
        "ap_functional": "population_ap_integral" if metric == "ap" else None,
        "class_name": class_name,
        "fp_budget": None,
        "iou": None,
        "metric": metric,
        "scope": scope,
        "task": "IC",
        "threshold": threshold,
        "version": 1,
    }
    return json.dumps(spec, sort_keys=True, separators=(",", ":"))


def _target(
    threshold: float,
    determination: str,
    witness: float | None = None,
    ceiling: float | None = None,
    scope: str = "full_input",
    metric: str = "accuracy",
    class_name: str | None = None,
    aggregation: str | None = None,
) -> dict[str, Any]:
    return {
        "threshold": threshold,
        "judged_scope": scope,
        "determination": determination,
        "witness": witness,
        "ceiling": ceiling,
        "metric": metric,
        "spec": (class_name, aggregation),
    }


def _source(
    source: str,
    bound: float | None,
    observed: float | None = None,
    scope: str = "full_input",
    bound_type: str = "lower",
    n: int | None = 5000,
    unit: str = "image",
) -> dict[str, Any]:
    return {
        "source": source,
        "evidence_id": f"{source.replace(' ', '_')}/v1",
        "information_scope": scope,
        "unit": unit,
        "bound_type": bound_type,
        "observed": observed,
        "bound": bound,
        "n": n,
        "state": "unavailable" if bound is None else "informative",
        "alpha": 0.01,
        "basis": "empirical_bernstein",
    }


# Enum dtypes where DataEval's FeasibilityOutput uses them, with its categories
_METRIC = pl.Enum(["accuracy", "ap", "recall_at_fp", "mota"])
_SCOPE = pl.Enum(["full_input", "fixed_representation", "partition_restricted"])
_DETERMINATION = pl.Enum(
    [
        "evidence_conflict",
        "demonstrated_feasible",
        "infeasible_under_recorded_assumptions",
        "not_ruled_out",
        "unresolved",
    ]
)
_DETERMINATION_SCHEMA = {
    "target_spec": pl.String,
    "metric": _METRIC,
    "threshold": pl.Float64,
    "judged_scope": _SCOPE,
    "determination": _DETERMINATION,
    "ceiling": pl.Float64,
    "ceiling_scope": _SCOPE,
    "witness": pl.Float64,
    "witness_source": pl.String,
}
_EVIDENCE_SCHEMA = {
    "target_spec": pl.String,
    "metric": _METRIC,
    "threshold": pl.Float64,
    "source": pl.String,
    "evidence_id": pl.String,
    "information_scope": _SCOPE,
    "unit": pl.String,
    "bound_type": pl.Enum(["lower", "upper"]),
    "observed": pl.Float64,
    "bound": pl.Float64,
    "n": pl.Int64,
    "state": pl.String,
    "alpha": pl.Float64,
    "basis": pl.String,
}


def _feasibility(targets: list[dict[str, Any]], sources: list[dict[str, Any]]) -> MockPlottableFeasibility:
    """Every source bounds every target; the deciding witness and ceiling are looked up by bound."""
    rows, evidence = [], []
    for target in targets:
        target = dict(target)
        spec = _target_spec(target["threshold"], target["judged_scope"], target["metric"], *target.pop("spec"))
        by_bound = {(s["bound_type"], s["bound"]): s for s in sources if s["bound"] is not None}
        witness = by_bound.get(("lower", target["witness"]))
        ceiling = by_bound.get(("upper", target["ceiling"]))
        rows.append(
            {
                "target_spec": spec,
                **target,
                "ceiling_scope": ceiling and ceiling["information_scope"],
                "witness_source": witness and witness["evidence_id"],
            }
        )
        keys = {"target_spec": spec, "metric": target["metric"], "threshold": target["threshold"]}
        evidence += [{**keys, **s} for s in sources]
    return MockPlottableFeasibility(
        determinations=pl.DataFrame(rows, schema=_DETERMINATION_SCHEMA),
        evidence_table=pl.DataFrame(evidence, schema=_EVIDENCE_SCHEMA),
        alpha=0.05,
    )


_HOW_TO_SOURCES = [
    _source("your model", 0.957, observed=0.969),
    _source("built-in k-NN", 0.923, observed=0.940, scope="fixed_representation"),
    _source("built-in linear probe", 0.882, observed=0.900, scope="fixed_representation"),
    _source("group-based ceiling", 0.721, scope="partition_restricted", bound_type="upper", n=400),
]


@pytest.fixture
def mock_feasibility_how_to() -> MockPlottableFeasibility:
    """The how-to: 95% demonstrated by the user's model, 99% unresolved, a narrower group-based ceiling."""
    return _feasibility(
        [_target(0.95, "demonstrated_feasible", witness=0.957), _target(0.99, "unresolved", witness=0.957)],
        _HOW_TO_SOURCES,
    )


@pytest.fixture
def mock_feasibility_ruled_out() -> MockPlottableFeasibility:
    """A repeated-labels ceiling at 93.0% rules out 95%; the witness demonstrates 90%."""
    return _feasibility(
        [
            _target(0.90, "demonstrated_feasible", witness=0.912, ceiling=0.93),
            _target(0.95, "infeasible_under_recorded_assumptions", witness=0.912, ceiling=0.93),
        ],
        [_source("your model", 0.912, observed=0.925), _source("repeated labels", 0.93, bound_type="upper", n=800)],
    )


@pytest.fixture
def mock_feasibility_conflict() -> MockPlottableFeasibility:
    """The witness's lower bound (94.0%) sits above the ceiling's upper bound (92.0%)."""
    return _feasibility(
        [_target(0.95, "evidence_conflict", witness=0.94, ceiling=0.92)],
        [_source("your model", 0.94, observed=0.952), _source("repeated labels", 0.92, bound_type="upper", n=800)],
    )


@pytest.fixture
def mock_feasibility_empty() -> MockPlottableFeasibility:
    """No witness and no ceiling: the requirement stays unresolved."""
    return _feasibility([_target(0.95, "unresolved")], [])


@pytest.fixture
def mock_feasibility_two_bands() -> MockPlottableFeasibility:
    """The same threshold judged for whole-image models and for embedding-only models."""
    return _feasibility(
        [
            _target(0.95, "demonstrated_feasible", witness=0.957),
            _target(0.95, "unresolved", witness=0.923, scope="fixed_representation"),
        ],
        _HOW_TO_SOURCES[:2],
    )


@pytest.fixture
def mock_feasibility_low_threshold() -> MockPlottableFeasibility:
    """A 50% requirement, far below every bound."""
    return _feasibility([_target(0.5, "demonstrated_feasible", witness=0.957)], _HOW_TO_SOURCES[:1])


@pytest.fixture
def mock_feasibility_ceiling_one() -> MockPlottableFeasibility:
    """An uninformative repeated-labels ceiling at exactly 100%: it rules nothing out."""
    return _feasibility(
        [_target(0.95, "demonstrated_feasible", witness=0.957, ceiling=1.0)],
        [_source("your model", 0.957, observed=0.969), _source("repeated labels", 1.0, bound_type="upper", n=800)],
    )


@pytest.fixture
def mock_feasibility_witness_zero() -> MockPlottableFeasibility:
    """A witness whose lower bound is exactly 0%: it demonstrates nothing."""
    return _feasibility(
        [_target(0.95, "unresolved", witness=0.0)],
        [_source("built-in k-NN", 0.0, observed=0.4, scope="fixed_representation", n=12)],
    )


@pytest.fixture
def mock_feasibility_per_class_ap() -> MockPlottableFeasibility:
    """Per-class AP targets: one band per class."""
    return _feasibility(
        [
            _target(0.8, "demonstrated_feasible", witness=0.84, metric="ap", class_name="cat"),
            _target(0.9, "unresolved", witness=0.84, metric="ap", class_name="dog"),
        ],
        [_source("built-in k-NN", 0.84, scope="fixed_representation")],
    )
