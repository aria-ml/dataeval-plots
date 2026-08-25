"""Cross-backend tests targeting remaining coverage gaps.

Covers branches and kwargs not exercised by the per-backend test suites:
- ``figsize`` / optional kwargs on every plot type (all backends)
- classwise diversity plots
- reference outputs for sufficiency
- drift edge cases (insufficient data, no drift detected)
- image-grid label/metadata/bounding-box rendering
- projection with explicit title/figsize
- registry ImportError paths
- ``_shared.py`` helper edge cases (parse/format/normalize/draw)

Backend-specific edge cases live in the per-backend test files
(``test_{matplotlib,seaborn,plotly,altair}_backend.py``).
"""

from __future__ import annotations

import builtins
import importlib.util
import os
import sys
import types
from typing import Any
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
from conftest import (
    MockDataset,
    MockPlottableDriftMVDC,
    MockPlottableStats,
    MockPlottableSufficiency,
)
from numpy.typing import NDArray
from PIL import ImageFont

from dataeval_plots import _registry as registry
from dataeval_plots import get_backend, plot, project
from dataeval_plots.backends._shared import (
    _draw_boxes_pil,
    calculate_projection,
    draw_bounding_boxes,
    format_label_from_target,
    image_to_hwc,
    merge_metadata,
    normalize_image_to_uint8,
    normalize_reference_outputs,
    parse_dataset_item,
    prepare_diversity_data,
    prepare_drift_data,
    reduce_embeddings,
    validate_class_names,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ALL_BACKENDS = ["matplotlib", "seaborn", "plotly", "altair"]


@pytest.fixture(params=ALL_BACKENDS)
def backend(request: pytest.FixtureRequest) -> Any:
    """A plotting backend instance, skipping when optional deps are missing."""
    name = request.param
    if name != "matplotlib":
        pytest.importorskip(name)
    return get_backend(name)


@pytest.fixture
def embeddings() -> NDArray[np.float64]:
    rng = np.random.default_rng(0)
    return rng.standard_normal((60, 12))


@pytest.fixture
def embeddings_3d() -> NDArray[np.float64]:
    rng = np.random.default_rng(1)
    return rng.standard_normal((40, 12))


def _steps() -> NDArray[np.uint32]:
    return np.array([10, 50, 100, 1000], dtype=np.uint32)


def _sufficiency_single() -> MockPlottableSufficiency:
    steps = _steps()
    return MockPlottableSufficiency(
        steps=steps,
        averaged_measures={
            "accuracy": np.array([0.5, 0.65, 0.8, 0.9]),
            "f1": np.array([0.45, 0.6, 0.75, 0.88]),
        },
        measures={
            "accuracy": np.random.rand(5, 4),
            "f1": np.random.rand(5, 4),
        },
        params={
            "accuracy": np.array([0.5, 0.5, 0.1]),
            "f1": np.array([0.55, 0.5, 0.12]),
        },
    )


def _sufficiency_multi() -> MockPlottableSufficiency:
    steps = _steps()
    n_classes = 3
    return MockPlottableSufficiency(
        steps=steps,
        averaged_measures={
            "accuracy": np.random.rand(n_classes, 4),
            "f1": np.random.rand(n_classes, 4),
        },
        measures={
            "accuracy": np.random.rand(5, 4, n_classes),
            "f1": np.random.rand(5, 4, n_classes),
        },
        params={
            "accuracy": np.random.rand(n_classes, 3),
            "f1": np.random.rand(n_classes, 3),
        },
    )


def _stats_single_channel(n_factors: int = 4) -> MockPlottableStats:
    return MockPlottableStats(
        _factors={f"factor_{i}": np.random.rand(50) for i in range(n_factors)},
        _n_channels=1,
        _channel_mask=None,
    )


def _stats_multi_channel(n_factors: int = 4) -> MockPlottableStats:
    names = ["mean", "std", "var", "skew"][:n_factors]
    return MockPlottableStats(
        _factors={name: np.random.rand(50, 3) for name in names},
        _n_channels=3,
        _channel_mask=None,
    )


def _drift_df(n_ref: int = 25, n_test: int = 25, n_alerts: int = 10) -> pl.DataFrame:
    n = n_ref + n_test
    n_alerts = min(n_alerts, n)
    return pl.DataFrame(
        {
            "chunk_period": ["reference"] * n_ref + ["analysis"] * n_test,
            "domain_classifier_auroc_value": np.random.rand(n),
            "domain_classifier_auroc_lower_threshold": [0.4] * n,
            "domain_classifier_auroc_upper_threshold": [0.6] * n,
            "domain_classifier_auroc_alert": [False] * (n - n_alerts) + [True] * n_alerts,
        }
    )


def _mock_dataset_with_targets() -> MockDataset:
    rng = np.random.default_rng(7)
    images = [rng.integers(0, 256, (3, 16, 16), dtype=np.uint8) for _ in range(4)]
    targets = [
        np.array([0.3, 0.7]),  # probabilities -> has label
        np.array([0.2, 0.3]),  # non-probability -> no label (None)
        np.array([0, 1]),  # integer labels
        None,  # no target -> falls back to empty
    ]
    metadatas = [{"split": "train"}, {"split": "test"}, {"split": "val"}, None]
    return MockDataset(
        images=images,
        dataset_id="gap_dataset",
        index2label={0: "cat", 1: "dog"},
        targets=targets,
        metadatas=metadatas,
    )


def _mock_dataset_with_boxes() -> MockDataset:
    rng = np.random.default_rng(8)
    images = [rng.integers(0, 256, (3, 32, 32), dtype=np.uint8) for _ in range(2)]
    targets = [
        {"boxes": [[5, 5, 20, 20]], "labels": [0], "scores": [0.9]},
        {"boxes": [[10, 10, 30, 30], [2, 2, 12, 28]], "labels": [1, 0], "scores": [0.8, 0.7]},
    ]
    return MockDataset(
        images=images,
        dataset_id="box_dataset",
        index2label={0: "cat", 1: "dog"},
        targets=targets,
    )


# ---------------------------------------------------------------------------
# Cross-backend plot() gap tests
# ---------------------------------------------------------------------------


class TestPlotApiGaps:
    """Top-level ``plot`` / ``project`` API dispatch and edge cases."""

    def test_plot_api_dispatches_to_backend(self, mock_balance: Any) -> None:
        result = plot(mock_balance, backend="matplotlib")
        assert result is not None

    def test_project_empty_method_sequence_raises(self, embeddings: NDArray[np.float64]) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            project(embeddings, method=[])

    def test_project_single_element_method_sequence(self, embeddings: NDArray[np.float64]) -> None:
        result = project(embeddings, method=["pca"])
        assert result is not None

    def test_project_none_method_with_2d_embeddings(self, embeddings_3d: NDArray[np.float64]) -> None:
        reduced = embeddings_3d[:, :2]
        result = project(reduced, method=None)
        assert result is not None


class TestBalanceGaps:
    def test_balance_with_figsize(self, backend: Any, mock_balance: Any) -> None:
        result = backend.plot(mock_balance, figsize=(6, 6))
        assert result is not None

    def test_balance_classwise(self, backend: Any, mock_balance: Any) -> None:
        result = backend.plot(mock_balance, plot_classwise=True)
        assert result is not None


class TestDiversityGaps:
    def test_diversity_with_figsize(self, backend: Any, mock_diversity: Any) -> None:
        result = backend.plot(mock_diversity, figsize=(8, 5))
        assert result is not None

    def test_diversity_classwise(self, backend: Any, mock_diversity: Any) -> None:
        result = backend.plot(mock_diversity, plot_classwise=True)
        assert result is not None

    def test_diversity_classwise_with_figsize(self, backend: Any, mock_diversity: Any) -> None:
        result = backend.plot(mock_diversity, plot_classwise=True, figsize=(9, 6))
        assert result is not None


class TestSufficiencyGaps:
    def test_single_class_with_figsize(self, backend: Any, mock_sufficiency_single_class: Any) -> None:
        result = backend.plot(mock_sufficiency_single_class, figsize=(8, 5))
        assert result is not None

    def test_single_class_with_reference_outputs(self, backend: Any) -> None:
        result = backend.plot(
            _sufficiency_single(),
            reference_outputs=[_sufficiency_single()],
        )
        assert result is not None

    def test_multi_class_with_reference_outputs(self, backend: Any) -> None:
        result = backend.plot(
            _sufficiency_multi(),
            reference_outputs=[_sufficiency_multi()],
        )
        assert result is not None

    def test_multi_class_without_asymptote_or_error_bars(self, backend: Any) -> None:
        result = backend.plot(
            _sufficiency_multi(),
            show_asymptote=False,
            show_error_bars=False,
        )
        assert result is not None

    def test_multi_class_with_figsize(self, backend: Any) -> None:
        result = backend.plot(_sufficiency_multi(), figsize=(10, 6))
        assert result is not None


class TestStatsGaps:
    def test_single_channel_with_figsize(self, backend: Any) -> None:
        result = backend.plot(_stats_single_channel(3), figsize=(9, 5))
        assert result is not None

    def test_single_channel_with_unused_axes(self, backend: Any) -> None:
        # 4 metrics -> 2 rows x 3 cols with 2 unused axes
        result = backend.plot(_stats_single_channel(4))
        assert result is not None

    def test_multi_channel_with_figsize(self, backend: Any) -> None:
        result = backend.plot(_stats_multi_channel(3), figsize=(11, 7))
        assert result is not None

    def test_multi_channel_with_unused_axes(self, backend: Any) -> None:
        # 4 channelwise metrics -> unused axes in the grid
        result = backend.plot(_stats_multi_channel(4))
        assert result is not None


class TestDriftGaps:
    def test_insufficient_data(self, backend: Any) -> None:
        output = MockPlottableDriftMVDC(_df=_drift_df(n_ref=1, n_test=1))
        result = backend.plot(output)
        assert result is not None

    def test_no_drift_detected(self, backend: Any) -> None:
        output = MockPlottableDriftMVDC(_df=_drift_df(n_alerts=0))
        result = backend.plot(output)
        assert result is not None

    def test_with_figsize(self, backend: Any) -> None:
        output = MockPlottableDriftMVDC(_df=_drift_df())
        result = backend.plot(output, figsize=(9, 5))
        assert result is not None


class TestImageGridGaps:
    def test_metadata_length_mismatch_raises(self, backend: Any) -> None:
        dataset = _mock_dataset_with_targets()
        with pytest.raises(ValueError, match="additional_metadata length"):
            backend.plot(
                dataset,
                indices=[0, 1, 2],
                additional_metadata=[{"a": 1}],
            )

    def test_show_labels_and_metadata(self, backend: Any) -> None:
        dataset = _mock_dataset_with_targets()
        result = backend.plot(
            dataset,
            indices=[0, 1, 2, 3],
            images_per_row=2,
            show_labels=True,
            show_metadata=True,
            additional_metadata=[{"x": i} for i in range(4)],
        )
        assert result is not None

    def test_bounding_box_targets(self, backend: Any) -> None:
        pytest.importorskip("cv2")
        dataset = _mock_dataset_with_boxes()
        result = backend.plot(
            dataset,
            indices=[0, 1],
            images_per_row=2,
            show_labels=True,
        )
        assert result is not None


class TestProjectionGaps:
    def test_project_with_labels_no_names(self, backend: Any, embeddings: NDArray[np.float64]) -> None:
        labels = np.repeat([0, 1], 30)
        result = backend.project(embeddings, method="pca", labels=labels)
        assert result is not None

    def test_project_with_figsize_and_title(self, backend: Any, embeddings: NDArray[np.float64]) -> None:
        result = backend.project(
            embeddings,
            method="pca",
            figsize=(7, 4),
            title="custom title",
        )
        assert result is not None

    def test_project_grid_with_figsize(
        self,
        backend: Any,
        embeddings: NDArray[np.float64],
    ) -> None:
        reduced = [reduce_embeddings(embeddings, "pca"), reduce_embeddings(embeddings, "tsne", perplexity=5)]
        result = backend.project_grid(reduced, methods=["pca", "tsne"], figsize=(10, 8))
        assert result is not None


# ---------------------------------------------------------------------------
# Registry error paths
# ---------------------------------------------------------------------------


class TestRegistryErrorPaths:
    def test_discover_backends_import_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real_import = builtins.__import__

        def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name in ("matplotlib", "seaborn", "plotly", "altair"):
                raise ImportError(f"no {name}")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        monkeypatch.setattr(registry, "_AVAILABLE_BACKENDS", None)
        assert registry._discover_available_backends() == set()

    def test_get_backend_seaborn_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real_import = builtins.__import__

        def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "dataeval_plots.backends._seaborn":
                raise ImportError("simulated")
            return real_import(name, *args, **kwargs)

        saved = registry._BACKENDS.pop("seaborn", None)
        try:
            monkeypatch.setattr(builtins, "__import__", fake_import)
            with pytest.raises(ImportError, match="dataeval-plots\\[seaborn\\]"):
                registry.get_backend("seaborn")
        finally:
            if saved is not None:
                registry._BACKENDS["seaborn"] = saved

    def test_get_backend_matplotlib_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real_import = builtins.__import__

        def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "dataeval_plots.backends._matplotlib":
                raise ImportError("simulated")
            return real_import(name, *args, **kwargs)

        saved = registry._BACKENDS.pop("matplotlib", None)
        try:
            monkeypatch.setattr(builtins, "__import__", fake_import)
            with pytest.raises(ImportError, match="pip install dataeval-plots"):
                registry.get_backend("matplotlib")
        finally:
            if saved is not None:
                registry._BACKENDS["matplotlib"] = saved


# ---------------------------------------------------------------------------
# reduce_embeddings optional-method and error paths
# ---------------------------------------------------------------------------


class TestReduceEmbeddingsGaps:
    def test_sklearn_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "sklearn", None)
        with pytest.raises(ImportError, match="scikit-learn"):
            reduce_embeddings(np.random.rand(10, 5), "pca")

    def test_umap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The umap branch passes reducer kwargs through to ``fit_transform``.

        A real UMAP run carries a one-time umap-learn import + numba-JIT cost
        of several seconds, so the default path stubs the ``umap`` module.
        The real-library smoke test is opt-in via ``DATAEVAL_PLOTS_RUN_HEAVY=1``.
        """
        calls: dict[str, Any] = {}

        class FakeUMAP:
            def __init__(self, **kwargs: Any) -> None:
                self.kwargs = kwargs
                calls.update(kwargs)

            def fit_transform(self, X: NDArray[Any]) -> NDArray[Any]:
                return np.zeros((X.shape[0], self.kwargs["n_components"]))

        monkeypatch.setitem(sys.modules, "umap", types.SimpleNamespace(UMAP=FakeUMAP))
        result = reduce_embeddings(
            np.random.default_rng(0).standard_normal((40, 10)),
            "umap",
            n_neighbors=7,
            min_dist=0.3,
        )
        assert result.shape == (40, 2)
        assert calls["n_components"] == 2
        assert calls["n_neighbors"] == 7
        assert calls["min_dist"] == 0.3

    def test_umap_real(self) -> None:
        """Real umap-learn smoke test. Opt-in: ``DATAEVAL_PLOTS_RUN_HEAVY=1``."""
        if os.environ.get("DATAEVAL_PLOTS_RUN_HEAVY") != "1":
            pytest.skip("heavy test; set DATAEVAL_PLOTS_RUN_HEAVY=1 to run")
        if importlib.util.find_spec("umap") is None:
            pytest.skip("umap-learn not installed")
        result = reduce_embeddings(np.random.default_rng(0).standard_normal((40, 10)), "umap")
        assert result.shape == (40, 2)

    def test_pacmap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The pacmap branch passes reducer kwargs through to ``fit_transform``.

        A real PaCMAP run costs ~1s just for the ``pacmap`` import, so the
        default path stubs the module. Real-library smoke test below is
        opt-in via ``DATAEVAL_PLOTS_RUN_HEAVY=1``.
        """
        calls: dict[str, Any] = {}

        class FakePaCMAP:
            def __init__(self, **kwargs: Any) -> None:
                self.kwargs = kwargs
                calls.update(kwargs)

            def fit_transform(self, X: NDArray[Any]) -> NDArray[Any]:
                return np.zeros((X.shape[0], self.kwargs["n_components"]))

        monkeypatch.setitem(sys.modules, "pacmap", types.SimpleNamespace(PaCMAP=FakePaCMAP))
        result = reduce_embeddings(np.random.default_rng(0).standard_normal((30, 5)), "pacmap", n_neighbors=3)
        assert result.shape == (30, 2)
        assert calls["n_components"] == 2
        assert calls["n_neighbors"] == 3

    def test_pacmap_real(self) -> None:
        """Real pacmap smoke test. Opt-in: ``DATAEVAL_PLOTS_RUN_HEAVY=1``."""
        if os.environ.get("DATAEVAL_PLOTS_RUN_HEAVY") != "1":
            pytest.skip("heavy test; set DATAEVAL_PLOTS_RUN_HEAVY=1 to run")
        if importlib.util.find_spec("pacmap") is None:
            pytest.skip("pacmap not installed")
        result = reduce_embeddings(np.random.default_rng(0).standard_normal((30, 5)), "pacmap")
        assert result.shape == (30, 2)

    def test_phate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The phate branch passes reducer kwargs through to ``fit_transform``."""
        calls: dict[str, Any] = {}

        class FakePHATE:
            def __init__(self, **kwargs: Any) -> None:
                self.kwargs = kwargs
                calls.update(kwargs)

            def fit_transform(self, X: NDArray[Any]) -> NDArray[Any]:
                return np.zeros((X.shape[0], self.kwargs["n_components"]))

        monkeypatch.setitem(sys.modules, "phate", types.SimpleNamespace(PHATE=FakePHATE))
        result = reduce_embeddings(np.random.default_rng(0).standard_normal((30, 5)), "phate", n_neighbors=3)
        assert result.shape == (30, 2)
        assert calls["n_components"] == 2
        assert calls["knn"] == 3

    def test_phate_real(self) -> None:
        """Real phate smoke test. Opt-in: ``DATAEVAL_PLOTS_RUN_HEAVY=1``."""
        if os.environ.get("DATAEVAL_PLOTS_RUN_HEAVY") != "1":
            pytest.skip("heavy test; set DATAEVAL_PLOTS_RUN_HEAVY=1 to run")
        if importlib.util.find_spec("phate") is None:
            pytest.skip("phate not installed")
        result = reduce_embeddings(np.random.default_rng(0).standard_normal((30, 5)), "phate")
        assert result.shape == (30, 2)

    def test_mds_legacy_signature(self) -> None:
        """sklearn MDS without ``metric_mds`` support uses the legacy kwargs path."""

        class LegacyMDS:
            def __init__(
                self,
                n_components: int = 2,
                random_state: Any = None,
                normalized_stress: str = "auto",
                n_init: int = 4,
                max_iter: int = 300,
                metric: bool = False,
            ) -> None:
                self._n_components = n_components

            def fit_transform(self, X: NDArray[Any]) -> NDArray[Any]:
                return np.zeros((X.shape[0], self._n_components))

        with patch("sklearn.manifold.MDS", LegacyMDS):
            result = reduce_embeddings(np.random.rand(20, 5), "mds")
        assert result.shape == (20, 2)


# ---------------------------------------------------------------------------
# _shared.py helper edge cases
# ---------------------------------------------------------------------------


class TestPrepareDiversityDataGaps:
    def test_meta_missing_falls_back_to_empty_meta(self, mock_diversity: Any) -> None:
        # Object with .factors but no .meta -> except branch
        class NoMetaDiversity:
            factors = mock_diversity.factors

        data, row_labels, col_labels, xlabel, ylabel, title, method = prepare_diversity_data(NoMetaDiversity())
        assert method == "Diversity"
        assert len(row_labels) > 0
        assert data.shape[0] == 0


class TestPrepareDriftDataGaps:
    def test_insufficient_returns_false(self) -> None:
        output = MockPlottableDriftMVDC(_df=_drift_df(n_ref=1, n_test=1))
        resdf, _, _, _, is_sufficient = prepare_drift_data(output)
        assert is_sufficient is False


class TestImageHelpers:
    def test_normalize_float_image_below_one(self) -> None:
        img = np.random.rand(4, 4, 3).astype(np.float64)
        img[0, 0, 0] = 1.0
        result = normalize_image_to_uint8(img)
        assert result.dtype == np.uint8
        assert result.max() == 255

    def test_image_to_hwc_grayscale_2d(self) -> None:
        gray = np.zeros((8, 6), dtype=np.uint8)
        assert image_to_hwc(gray).shape == (8, 6, 1)

    def test_image_to_hwc_hwc_passthrough(self) -> None:
        hwc = np.zeros((6, 8, 3), dtype=np.uint8)
        assert image_to_hwc(hwc).shape == (6, 8, 3)

    def test_validate_class_names_mismatch(self) -> None:
        with pytest.raises(IndexError, match="Class name count"):
            validate_class_names(np.zeros((3, 5)), ["a", "b"])

    def test_parse_dataset_item_image_only_tuple(self) -> None:
        img = np.zeros((3, 4, 4), dtype=np.uint8)
        image, target, metadata = parse_dataset_item((img,))
        assert image is img
        assert target is None
        assert metadata == {}

    def test_parse_dataset_item_image_and_target_tuple(self) -> None:
        img = np.zeros((3, 4, 4), dtype=np.uint8)
        target = np.array([1, 0])
        image, got_target, metadata = parse_dataset_item((img, target))
        assert image is img
        assert got_target is target
        assert metadata == {}

    def test_parse_dataset_item_bare_image(self) -> None:
        img = np.zeros((3, 4, 4), dtype=np.uint8)
        image, target, metadata = parse_dataset_item(img)
        assert image is img
        assert target is None


class TestFormatLabelFromTargetGaps:
    def test_none_target(self) -> None:
        assert format_label_from_target(None) is None

    def test_object_detection_with_index2label(self) -> None:
        target = types.SimpleNamespace(
            boxes=np.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
            labels=np.array([0, 1]),
            scores=np.array([0.9, 0.8]),
        )
        assert format_label_from_target(target, {0: "cat", 1: "dog"}) == "cat: 1, dog: 1"

    def test_object_detection_label_missing_from_index2label(self) -> None:
        target = types.SimpleNamespace(
            boxes=np.array([[1, 2, 3, 4]]),
            labels=np.array([5]),
            scores=None,
        )
        assert format_label_from_target(target, {0: "cat"}) == "Class 5: 1"

    def test_object_detection_empty_boxes(self) -> None:
        target = types.SimpleNamespace(boxes=np.empty((0, 4)), labels=np.empty(0), scores=None)
        assert format_label_from_target(target) == "No objects"

    def test_scalar_with_index2label(self) -> None:
        assert format_label_from_target(np.int64(1), {0: "cat", 1: "dog"}) == "dog"

    def test_scalar_without_index2label(self) -> None:
        assert format_label_from_target(np.int64(3)) == "Class 3"

    def test_1d_probabilities_with_index2label(self) -> None:
        assert format_label_from_target(np.array([0.2, 0.8]), {1: "dog"}) == "dog (0.80)"

    def test_1d_probabilities_without_index2label(self) -> None:
        assert format_label_from_target(np.array([0.2, 0.8])) == "Class 1 (0.80)"

    def test_1d_int_labels(self) -> None:
        assert format_label_from_target(np.array([0]), {0: "cat"}) == "cat"

    def test_1d_non_probability_float_returns_none(self) -> None:
        assert format_label_from_target(np.array([0.2, 0.3])) is None

    def test_1d_empty_array_returns_none(self) -> None:
        assert format_label_from_target(np.empty(0, dtype=np.float64)) is None


class TestMiscSharedGaps:
    def test_normalize_reference_outputs_none(self) -> None:
        assert normalize_reference_outputs(None) == []

    def test_normalize_reference_outputs_single(self) -> None:
        output = _sufficiency_single()
        assert normalize_reference_outputs(output) == [output]

    def test_normalize_reference_outputs_tuple(self) -> None:
        a, b = _sufficiency_single(), _sufficiency_single()
        assert normalize_reference_outputs((a, b)) == [a, b]

    def test_merge_metadata_none(self) -> None:
        base = {"a": 1}
        assert merge_metadata(base, None) is base

    def test_calculate_projection(self) -> None:
        steps = np.array([10, 100, 1000], dtype=np.uint32)
        result = calculate_projection(steps)
        assert result.shape[0] == 3

    def test_draw_bounding_boxes_empty(self) -> None:
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        result = draw_bounding_boxes(img, np.empty((0, 4)))
        assert result.shape == img.shape

    def test_draw_boxes_pil_font_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real_truetype = ImageFont.truetype

        def _fail_truetype(font: Any, *args: Any, **kwargs: Any) -> Any:
            if "DejaVuSans" in str(font):
                raise OSError("no fonts")
            return real_truetype(font, *args, **kwargs)

        monkeypatch.setattr(ImageFont, "truetype", _fail_truetype)
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        result = _draw_boxes_pil(
            img,
            np.array([[5, 5, 20, 20]]),
            np.array([0]),
            np.array([0.9]),
            {0: "cat"},
            (0, 255, 0),
            2,
        )
        assert result.shape == img.shape

    def test_draw_boxes_pil_textbbox_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from PIL import ImageDraw

        def _fail_textbbox(self: Any, *args: Any, **kwargs: Any) -> Any:
            raise AttributeError("no textbbox")

        monkeypatch.setattr(ImageDraw.ImageDraw, "textbbox", _fail_textbbox)
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        result = _draw_boxes_pil(
            img,
            np.array([[5, 5, 20, 20]]),
            np.array([0]),
            np.array([0.9]),
            {0: "cat"},
            (0, 255, 0),
            2,
        )
        assert result.shape == img.shape

    def test_draw_boxes_pil_no_labels(self) -> None:
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        result = _draw_boxes_pil(
            img,
            np.array([[5, 5, 20, 20]]),
            None,
            None,
            None,
            (0, 255, 0),
            2,
        )
        assert result.shape == img.shape
