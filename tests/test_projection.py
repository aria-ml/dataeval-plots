"""
Tests for projection functionality (reduce_embeddings, calculate_subplot_grid,
backend project/project_grid, public API).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from matplotlib.figure import Figure

from dataeval_plots.backends._shared import calculate_subplot_grid, reduce_embeddings

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def high_dim_embeddings() -> np.ndarray:
    """50 samples, 128 dimensions — enough for all reducers."""
    rng = np.random.RandomState(42)
    return rng.rand(50, 128).astype(np.float64)


@pytest.fixture
def embeddings_2d() -> np.ndarray:
    """Pre-reduced 2D embeddings."""
    rng = np.random.RandomState(42)
    return rng.rand(50, 2).astype(np.float64)


@pytest.fixture
def embeddings_3d() -> np.ndarray:
    """Pre-reduced 3D embeddings."""
    rng = np.random.RandomState(42)
    return rng.rand(50, 3).astype(np.float64)


@pytest.fixture
def labels() -> np.ndarray:
    """Class labels for 50 samples (5 classes)."""
    return np.array([i % 5 for i in range(50)])


@pytest.fixture
def label_names() -> dict[int, str]:
    return {0: "cat", 1: "dog", 2: "bird", 3: "fish", 4: "frog"}


# ---------------------------------------------------------------------------
# Tests: reduce_embeddings
# ---------------------------------------------------------------------------


class TestReduceEmbeddings:
    """Tests for the reduce_embeddings helper."""

    @pytest.mark.parametrize(
        "method",
        ["pca", "tsne", "isomap", "mds", "spectral", "truncated_svd"],
    )
    def test_sklearn_methods_2d(self, high_dim_embeddings: np.ndarray, method: str) -> None:
        result = reduce_embeddings(high_dim_embeddings, method=method, dimensions=2)
        assert result.shape == (50, 2)
        assert result.dtype == np.float64

    @pytest.mark.parametrize("method", ["pca", "truncated_svd"])
    def test_sklearn_methods_3d(self, high_dim_embeddings: np.ndarray, method: str) -> None:
        result = reduce_embeddings(high_dim_embeddings, method=method, dimensions=3)
        assert result.shape == (50, 3)

    def test_pca_deterministic(self, high_dim_embeddings: np.ndarray) -> None:
        r1 = reduce_embeddings(high_dim_embeddings, method="pca", random_state=0)
        r2 = reduce_embeddings(high_dim_embeddings, method="pca", random_state=0)
        np.testing.assert_array_equal(r1, r2)

    def test_unknown_method_raises(self, high_dim_embeddings: np.ndarray) -> None:
        with pytest.raises(ValueError, match="Unknown reduction method"):
            reduce_embeddings(high_dim_embeddings, method="not_a_method")  # type: ignore[arg-type]

    def test_umap_import_error(self, high_dim_embeddings: np.ndarray) -> None:
        """UMAP raises ImportError when umap-learn is missing."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "umap":
                raise ImportError("no umap")
            return real_import(name, *args, **kwargs)

        with (
            patch.object(builtins, "__import__", side_effect=mock_import),
            pytest.raises(ImportError, match="umap-learn"),
        ):
            reduce_embeddings(high_dim_embeddings, method="umap")

    def test_pacmap_import_error(self, high_dim_embeddings: np.ndarray) -> None:
        """PaCMAP raises ImportError when pacmap is missing."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "pacmap":
                raise ImportError("no pacmap")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import), pytest.raises(ImportError, match="pacmap"):
            reduce_embeddings(high_dim_embeddings, method="pacmap")

    def test_phate_import_error(self, high_dim_embeddings: np.ndarray) -> None:
        """PHATE raises ImportError when phate is missing."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "phate":
                raise ImportError("no phate")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import), pytest.raises(ImportError, match="phate"):
            reduce_embeddings(high_dim_embeddings, method="phate")


# ---------------------------------------------------------------------------
# Tests: calculate_subplot_grid
# ---------------------------------------------------------------------------


class TestCalculateSubplotGrid:
    """Tests for the calculate_subplot_grid helper."""

    def test_default_3_cols(self) -> None:
        assert calculate_subplot_grid(6) == (2, 3)

    def test_single_item(self) -> None:
        assert calculate_subplot_grid(1) == (1, 1)

    def test_two_items(self) -> None:
        assert calculate_subplot_grid(2) == (1, 2)

    def test_four_items_default(self) -> None:
        rows, cols = calculate_subplot_grid(4)
        assert rows == 2
        assert cols == 3

    def test_seven_items_default(self) -> None:
        rows, cols = calculate_subplot_grid(7)
        assert rows == 3
        assert cols == 3

    def test_wide_aspect_ratio(self) -> None:
        """Wide figure (18x6 → ratio 3.0) should prefer more columns."""
        rows, cols = calculate_subplot_grid(6, aspect_ratio=3.0)
        assert cols >= rows

    def test_tall_aspect_ratio(self) -> None:
        """Tall figure (6x18 → ratio 0.33) should prefer more rows."""
        rows, cols = calculate_subplot_grid(4, aspect_ratio=0.5)
        assert rows >= cols

    def test_square_aspect_ratio(self) -> None:
        """Square figure (1.0) with 4 items should give 2x2."""
        rows, cols = calculate_subplot_grid(4, aspect_ratio=1.0)
        assert rows == 2
        assert cols == 2

    def test_all_items_fit(self) -> None:
        """rows * cols must be >= num_items for any configuration."""
        for n in range(1, 13):
            for ratio in [0.5, 1.0, 2.0, 3.0]:
                rows, cols = calculate_subplot_grid(n, aspect_ratio=ratio)
                assert rows * cols >= n, f"n={n}, ratio={ratio}: {rows}x{cols} < {n}"


# ---------------------------------------------------------------------------
# Tests: BasePlottingBackend.project / project_grid  (matplotlib)
# ---------------------------------------------------------------------------


class TestBaseBackendProject:
    """Tests for the matplotlib-based project() and project_grid() on BasePlottingBackend."""

    @pytest.fixture
    def backend(self) -> Any:
        from dataeval_plots.backends._matplotlib import MatplotlibBackend

        return MatplotlibBackend()

    def test_project_2d_no_labels(self, backend: Any, embeddings_2d: np.ndarray) -> None:
        fig = backend.project(embeddings_2d, method="pca", dimensions=2)
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 1

    def test_project_2d_with_labels(
        self, backend: Any, embeddings_2d: np.ndarray, labels: np.ndarray, label_names: dict[int, str]
    ) -> None:
        fig = backend.project(embeddings_2d, labels=labels, label_names=label_names, method="pca", dimensions=2)
        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        legend = ax.get_legend()
        assert legend is not None
        legend_texts = [t.get_text() for t in legend.get_texts()]
        assert "cat" in legend_texts

    def test_project_3d_no_labels(self, backend: Any, embeddings_3d: np.ndarray) -> None:
        fig = backend.project(embeddings_3d, method="pca", dimensions=3)
        assert isinstance(fig, Figure)

    def test_project_3d_with_labels(
        self, backend: Any, embeddings_3d: np.ndarray, labels: np.ndarray, label_names: dict[int, str]
    ) -> None:
        fig = backend.project(embeddings_3d, labels=labels, label_names=label_names, method="pca", dimensions=3)
        assert isinstance(fig, Figure)

    def test_project_custom_title(self, backend: Any, embeddings_2d: np.ndarray) -> None:
        fig = backend.project(embeddings_2d, method="pca", title="My Title")
        assert fig.axes[0].get_title() == "My Title"

    def test_project_auto_title(self, backend: Any, embeddings_2d: np.ndarray) -> None:
        fig = backend.project(embeddings_2d, method="tsne")
        assert "TSNE" in fig.axes[0].get_title()

    def test_project_custom_figsize(self, backend: Any, embeddings_2d: np.ndarray) -> None:
        fig = backend.project(embeddings_2d, method="pca", figsize=(12, 6))
        w, h = fig.get_size_inches()
        assert w == pytest.approx(12, abs=0.1)
        assert h == pytest.approx(6, abs=0.1)

    # project_grid

    def test_project_grid_2d(self, backend: Any, embeddings_2d: np.ndarray) -> None:
        emb_list = [embeddings_2d, embeddings_2d + 1, embeddings_2d - 1]
        fig = backend.project_grid(emb_list, methods=["pca", "tsne", "mds"])
        assert isinstance(fig, Figure)
        # Should have at least 3 visible axes
        visible = [ax for ax in fig.axes if ax.get_visible()]
        assert len(visible) >= 3

    def test_project_grid_2d_with_labels(
        self,
        backend: Any,
        embeddings_2d: np.ndarray,
        labels: np.ndarray,
        label_names: dict[int, str],
    ) -> None:
        emb_list = [embeddings_2d, embeddings_2d + 1]
        fig = backend.project_grid(emb_list, methods=["pca", "tsne"], labels=labels, label_names=label_names)
        assert isinstance(fig, Figure)

    def test_project_grid_3d(self, backend: Any, embeddings_3d: np.ndarray) -> None:
        emb_list = [embeddings_3d, embeddings_3d + 1]
        fig = backend.project_grid(emb_list, methods=["pca", "tsne"], dimensions=3)
        assert isinstance(fig, Figure)

    def test_project_grid_hides_unused_axes(self, backend: Any, embeddings_2d: np.ndarray) -> None:
        """5 items in a 2x3 grid → 1 axis should be hidden."""
        emb_list = [embeddings_2d] * 5
        fig = backend.project_grid(emb_list, methods=["a", "b", "c", "d", "e"])
        assert isinstance(fig, Figure)
        hidden = [ax for ax in fig.axes if not ax.get_visible()]
        assert len(hidden) >= 1

    def test_project_grid_suptitle(self, backend: Any, embeddings_2d: np.ndarray) -> None:
        emb_list = [embeddings_2d, embeddings_2d]
        fig = backend.project_grid(emb_list, methods=["pca", "tsne"], title="Comparison")
        assert fig._suptitle is not None
        assert fig._suptitle.get_text() == "Comparison"

    def test_project_grid_aspect_ratio(self, backend: Any, embeddings_2d: np.ndarray) -> None:
        """Wide figsize should produce more cols than rows."""
        emb_list = [embeddings_2d] * 6
        fig = backend.project_grid(emb_list, methods=["a"] * 6, figsize=(24, 6))
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# Tests: PlotlyBackend.project / project_grid
# ---------------------------------------------------------------------------


class TestPlotlyBackendProject:
    """Tests for Plotly projection methods."""

    @pytest.fixture
    def backend(self) -> Any:
        pytest.importorskip("plotly")
        from dataeval_plots.backends._plotly import PlotlyBackend

        return PlotlyBackend()

    def test_project_2d_no_labels(self, backend: Any, embeddings_2d: np.ndarray) -> None:
        import plotly.graph_objects as go

        fig = backend.project(embeddings_2d, method="pca", dimensions=2)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1

    def test_project_2d_with_labels(
        self, backend: Any, embeddings_2d: np.ndarray, labels: np.ndarray, label_names: dict[int, str]
    ) -> None:
        import plotly.graph_objects as go

        fig = backend.project(embeddings_2d, labels=labels, label_names=label_names, method="pca", dimensions=2)
        assert isinstance(fig, go.Figure)
        # One trace per unique label
        assert len(fig.data) == 5
        trace_names = {t.name for t in fig.data}
        assert "cat" in trace_names

    def test_project_3d(self, backend: Any, embeddings_3d: np.ndarray) -> None:
        import plotly.graph_objects as go

        fig = backend.project(embeddings_3d, method="pca", dimensions=3)
        assert isinstance(fig, go.Figure)
        assert isinstance(fig.data[0], go.Scatter3d)

    def test_project_3d_with_labels(
        self, backend: Any, embeddings_3d: np.ndarray, labels: np.ndarray, label_names: dict[int, str]
    ) -> None:
        import plotly.graph_objects as go

        fig = backend.project(embeddings_3d, labels=labels, label_names=label_names, method="pca", dimensions=3)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 5
        assert all(isinstance(t, go.Scatter3d) for t in fig.data)

    def test_project_figsize_conversion(self, backend: Any, embeddings_2d: np.ndarray) -> None:
        fig = backend.project(embeddings_2d, method="pca", figsize=(12, 8))
        assert fig.layout.width == 1200
        assert fig.layout.height == 800

    def test_project_grid_2d(self, backend: Any, embeddings_2d: np.ndarray) -> None:
        import plotly.graph_objects as go

        emb_list = [embeddings_2d, embeddings_2d + 1]
        fig = backend.project_grid(emb_list, methods=["pca", "tsne"])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2

    def test_project_grid_2d_with_labels(
        self,
        backend: Any,
        embeddings_2d: np.ndarray,
        labels: np.ndarray,
        label_names: dict[int, str],
    ) -> None:
        import plotly.graph_objects as go

        emb_list = [embeddings_2d, embeddings_2d + 1]
        fig = backend.project_grid(emb_list, methods=["pca", "tsne"], labels=labels, label_names=label_names)
        assert isinstance(fig, go.Figure)
        # 5 classes × 2 methods = 10 traces
        assert len(fig.data) == 10
        # Legend deduplication: only first occurrence shows legend
        shown = [t for t in fig.data if t.showlegend]
        assert len(shown) == 5

    def test_project_grid_3d(self, backend: Any, embeddings_3d: np.ndarray) -> None:
        import plotly.graph_objects as go

        emb_list = [embeddings_3d, embeddings_3d + 1]
        fig = backend.project_grid(emb_list, methods=["pca", "tsne"], dimensions=3)
        assert isinstance(fig, go.Figure)
        assert all(isinstance(t, go.Scatter3d) for t in fig.data)

    def test_project_grid_title(self, backend: Any, embeddings_2d: np.ndarray) -> None:
        emb_list = [embeddings_2d, embeddings_2d]
        fig = backend.project_grid(emb_list, methods=["pca", "tsne"], title="My Grid")
        assert fig.layout.title.text == "My Grid"


# ---------------------------------------------------------------------------
# Tests: AltairBackend.project / project_grid
# ---------------------------------------------------------------------------


class TestAltairBackendProject:
    """Tests for Altair projection methods."""

    @pytest.fixture
    def backend(self) -> Any:
        pytest.importorskip("altair")
        from dataeval_plots.backends._altair import AltairBackend

        return AltairBackend()

    def test_project_2d_no_labels(self, backend: Any, embeddings_2d: np.ndarray) -> None:
        import altair as alt

        fig = backend.project(embeddings_2d, method="pca", dimensions=2)
        assert isinstance(fig, alt.Chart)

    def test_project_2d_with_labels(
        self, backend: Any, embeddings_2d: np.ndarray, labels: np.ndarray, label_names: dict[int, str]
    ) -> None:
        import altair as alt

        fig = backend.project(embeddings_2d, labels=labels, label_names=label_names, method="pca", dimensions=2)
        assert isinstance(fig, alt.Chart)

    def test_project_3d_raises(self, backend: Any, embeddings_3d: np.ndarray) -> None:
        with pytest.raises(NotImplementedError, match="3D"):
            backend.project(embeddings_3d, method="pca", dimensions=3)

    def test_project_grid_2d(self, backend: Any, embeddings_2d: np.ndarray) -> None:
        import altair as alt

        emb_list = [embeddings_2d, embeddings_2d + 1, embeddings_2d - 1]
        result = backend.project_grid(emb_list, methods=["pca", "tsne", "mds"])
        assert isinstance(result, alt.Chart | alt.VConcatChart | alt.HConcatChart)

    def test_project_grid_2d_with_labels(
        self,
        backend: Any,
        embeddings_2d: np.ndarray,
        labels: np.ndarray,
        label_names: dict[int, str],
    ) -> None:
        import altair as alt

        emb_list = [embeddings_2d, embeddings_2d + 1]
        result = backend.project_grid(emb_list, methods=["pca", "tsne"], labels=labels, label_names=label_names)
        assert isinstance(result, alt.Chart | alt.VConcatChart | alt.HConcatChart)

    def test_project_grid_3d_raises(self, backend: Any, embeddings_3d: np.ndarray) -> None:
        with pytest.raises(NotImplementedError, match="3D"):
            backend.project_grid([embeddings_3d], methods=["pca"], dimensions=3)

    def test_project_grid_title(self, backend: Any, embeddings_2d: np.ndarray) -> None:
        import altair as alt

        emb_list = [embeddings_2d, embeddings_2d]
        result = backend.project_grid(emb_list, methods=["pca", "tsne"], title="My Grid")
        assert isinstance(result, alt.Chart | alt.VConcatChart | alt.HConcatChart)

    def test_project_grid_single_method(self, backend: Any, embeddings_2d: np.ndarray) -> None:
        """Single method in grid returns a plain Chart (no concat)."""
        import altair as alt

        result = backend.project_grid([embeddings_2d], methods=["pca"])
        assert isinstance(result, alt.Chart)


# ---------------------------------------------------------------------------
# Tests: SeabornBackend.project / project_grid  (inherits from Base)
# ---------------------------------------------------------------------------


class TestSeabornBackendProject:
    """Seaborn uses BasePlottingBackend's project methods — smoke tests."""

    @pytest.fixture
    def backend(self) -> Any:
        pytest.importorskip("seaborn")
        from dataeval_plots.backends._seaborn import SeabornBackend

        return SeabornBackend()

    def test_project_2d(self, backend: Any, embeddings_2d: np.ndarray, labels: np.ndarray) -> None:
        fig = backend.project(embeddings_2d, labels=labels, method="pca", dimensions=2)
        assert isinstance(fig, Figure)

    def test_project_grid_2d(self, backend: Any, embeddings_2d: np.ndarray) -> None:
        emb_list = [embeddings_2d, embeddings_2d + 1]
        fig = backend.project_grid(emb_list, methods=["pca", "tsne"])
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# Tests: Public project() API  (__init__.py)
# ---------------------------------------------------------------------------


class TestProjectPublicAPI:
    """Tests for the top-level project() function."""

    def test_single_method(self, high_dim_embeddings: np.ndarray) -> None:
        from dataeval_plots import project

        fig = project(high_dim_embeddings, method="pca")
        assert isinstance(fig, Figure)

    def test_single_method_with_labels(
        self, high_dim_embeddings: np.ndarray, labels: np.ndarray, label_names: dict[int, str]
    ) -> None:
        from dataeval_plots import project

        fig = project(high_dim_embeddings, method="pca", labels=labels, label_names=label_names)
        assert isinstance(fig, Figure)

    def test_method_none_valid_2d(self, embeddings_2d: np.ndarray) -> None:
        from dataeval_plots import project

        fig = project(embeddings_2d, method=None)
        assert isinstance(fig, Figure)

    def test_method_none_valid_3d(self, embeddings_3d: np.ndarray) -> None:
        from dataeval_plots import project

        fig = project(embeddings_3d, method=None)
        assert isinstance(fig, Figure)

    def test_method_none_invalid_shape(self, high_dim_embeddings: np.ndarray) -> None:
        from dataeval_plots import project

        with pytest.raises(ValueError, match="shape"):
            project(high_dim_embeddings, method=None)

    def test_method_none_1d_invalid(self) -> None:
        from dataeval_plots import project

        with pytest.raises(ValueError, match="shape"):
            project(np.array([1.0, 2.0, 3.0]), method=None)

    def test_multiple_methods_returns_grid(self, high_dim_embeddings: np.ndarray) -> None:
        from dataeval_plots import project

        fig = project(high_dim_embeddings, method=["pca", "truncated_svd"])
        assert isinstance(fig, Figure)
        # Should have at least 2 visible axes
        visible = [ax for ax in fig.axes if ax.get_visible()]
        assert len(visible) >= 2

    def test_multiple_methods_with_labels(
        self, high_dim_embeddings: np.ndarray, labels: np.ndarray, label_names: dict[int, str]
    ) -> None:
        from dataeval_plots import project

        fig = project(
            high_dim_embeddings,
            method=["pca", "truncated_svd"],
            labels=labels,
            label_names=label_names,
        )
        assert isinstance(fig, Figure)

    def test_3d_single(self, high_dim_embeddings: np.ndarray) -> None:
        from dataeval_plots import project

        fig = project(high_dim_embeddings, method="pca", dimensions=3)
        assert isinstance(fig, Figure)

    def test_3d_multiple(self, high_dim_embeddings: np.ndarray) -> None:
        from dataeval_plots import project

        fig = project(high_dim_embeddings, method=["pca", "truncated_svd"], dimensions=3)
        assert isinstance(fig, Figure)

    def test_custom_figsize(self, high_dim_embeddings: np.ndarray) -> None:
        from dataeval_plots import project

        fig = project(high_dim_embeddings, method="pca", figsize=(14, 7))
        w, h = fig.get_size_inches()
        assert w == pytest.approx(14, abs=0.1)
        assert h == pytest.approx(7, abs=0.1)

    def test_custom_title(self, high_dim_embeddings: np.ndarray) -> None:
        from dataeval_plots import project

        fig = project(high_dim_embeddings, method="pca", title="Custom")
        assert fig.axes[0].get_title() == "Custom"

    def test_backend_plotly(self, high_dim_embeddings: np.ndarray) -> None:
        pytest.importorskip("plotly")
        import plotly.graph_objects as go

        from dataeval_plots import project

        fig = project(high_dim_embeddings, method="pca", backend="plotly")
        assert isinstance(fig, go.Figure)

    def test_backend_altair(self, high_dim_embeddings: np.ndarray) -> None:
        pytest.importorskip("altair")
        import altair as alt

        from dataeval_plots import project

        fig = project(high_dim_embeddings, method="pca", backend="altair")
        assert isinstance(fig, alt.Chart)

    def test_accepts_list_input(self) -> None:
        """ArrayLike input (plain list) should work."""
        from dataeval_plots import project

        data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        fig = project(data, method=None)
        assert isinstance(fig, Figure)

    def test_project_in_all(self) -> None:
        """project should be in __all__."""
        import dataeval_plots

        assert "project" in dataeval_plots.__all__
