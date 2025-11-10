"""Tests for Plotly backend."""

from __future__ import annotations

import plotly.graph_objects as go
import pytest
from conftest import (
    MockDataset,
    MockPlottableBalance,
    MockPlottableBaseStats,
    MockPlottableCoverage,
    MockPlottableDiversity,
    MockPlottableDriftMVDC,
    MockPlottableSufficiency,
)

from dataeval_plots.backends._plotly import PlotlyBackend

plotly = pytest.importorskip("plotly")


class TestPlotlyBackend:
    """Test suite for Plotly backend."""

    @pytest.fixture
    def backend(self) -> PlotlyBackend:
        """Create Plotly backend instance."""
        return PlotlyBackend()

    def test_plot_coverage(
        self,
        backend: PlotlyBackend,
        mock_coverage: MockPlottableCoverage,
    ) -> None:
        """Test plotting coverage output."""
        # Coverage plotting requires images data, so we expect a ValueError
        with pytest.raises(ValueError, match="images parameter is required"):
            backend.plot(mock_coverage)

    def test_plot_balance_global(
        self,
        backend: PlotlyBackend,
        mock_balance: MockPlottableBalance,
    ) -> None:
        """Test plotting global balance output."""
        result = backend.plot(mock_balance, plot_classwise=False)

        assert isinstance(result, go.Figure)
        assert len(result.data) > 0

    def test_plot_balance_classwise(
        self,
        backend: PlotlyBackend,
        mock_balance: MockPlottableBalance,
    ) -> None:
        """Test plotting classwise balance output."""
        result = backend.plot(mock_balance, plot_classwise=True)

        assert isinstance(result, go.Figure)
        assert len(result.data) > 0

    def test_plot_balance_with_labels(
        self,
        backend: PlotlyBackend,
        mock_balance: MockPlottableBalance,
    ) -> None:
        """Test plotting balance with custom labels."""
        row_labels = ["row_0", "row_1"]
        col_labels = ["col_0", "col_1", "col_2"]

        result = backend.plot(
            mock_balance,
            plot_classwise=False,
            row_labels=row_labels,
            col_labels=col_labels,
        )

        assert isinstance(result, go.Figure)

    def test_plot_diversity(
        self,
        backend: PlotlyBackend,
        mock_diversity: MockPlottableDiversity,
    ) -> None:
        """Test plotting diversity output."""
        result = backend.plot(mock_diversity)

        assert isinstance(result, go.Figure)
        assert len(result.data) > 0

    def test_plot_diversity_with_labels(
        self,
        backend: PlotlyBackend,
        mock_diversity: MockPlottableDiversity,
    ) -> None:
        """Test plotting diversity with custom labels."""
        row_labels = ["class_A", "class_B", "class_C"]
        col_labels = ["factor_X", "factor_Y", "factor_Z", "factor_W"]

        result = backend.plot(
            mock_diversity,
            row_labels=row_labels,
            col_labels=col_labels,
        )

        assert isinstance(result, go.Figure)

    def test_plot_sufficiency_single_class(
        self,
        backend: PlotlyBackend,
        mock_sufficiency_single_class: MockPlottableSufficiency,
    ) -> None:
        """Test plotting sufficiency with single class."""
        result = backend.plot(mock_sufficiency_single_class)

        assert isinstance(result, list)
        assert len(result) == 2  # Two metrics: accuracy and f1
        for fig in result:
            assert isinstance(fig, go.Figure)
            assert len(fig.data) > 0

    def test_plot_sufficiency_multi_class(
        self,
        backend: PlotlyBackend,
        mock_sufficiency_multi_class: MockPlottableSufficiency,
    ) -> None:
        """Test plotting sufficiency with multiple classes."""
        class_names = ["class_A", "class_B", "class_C"]

        result = backend.plot(mock_sufficiency_multi_class, class_names=class_names)

        assert isinstance(result, list)
        # 2 metrics * 3 classes = 6 plots
        assert len(result) == 6
        for fig in result:
            assert isinstance(fig, go.Figure)

    def test_plot_sufficiency_no_error_bars(
        self,
        backend: PlotlyBackend,
        mock_sufficiency_single_class: MockPlottableSufficiency,
    ) -> None:
        """Test plotting sufficiency without error bars."""
        result = backend.plot(mock_sufficiency_single_class, show_error_bars=False)

        assert isinstance(result, list)
        assert len(result) > 0
        for fig in result:
            assert isinstance(fig, go.Figure)

    def test_plot_sufficiency_no_asymptote(
        self,
        backend: PlotlyBackend,
        mock_sufficiency_single_class: MockPlottableSufficiency,
    ) -> None:
        """Test plotting sufficiency without asymptote."""
        result = backend.plot(mock_sufficiency_single_class, show_asymptote=False)

        assert isinstance(result, list)
        assert len(result) > 0
        for fig in result:
            assert isinstance(fig, go.Figure)

    def test_plot_base_stats_single_channel(
        self,
        backend: PlotlyBackend,
        mock_base_stats_single_channel: MockPlottableBaseStats,
    ) -> None:
        """Test plotting base stats with single channel."""
        result = backend.plot(mock_base_stats_single_channel)

        assert isinstance(result, go.Figure)
        assert len(result.data) > 0

    def test_plot_base_stats_multi_channel(
        self,
        backend: PlotlyBackend,
        mock_base_stats_multi_channel: MockPlottableBaseStats,
    ) -> None:
        """Test plotting base stats with multiple channels."""
        result = backend.plot(mock_base_stats_multi_channel)

        assert isinstance(result, go.Figure)
        assert len(result.data) > 0

    def test_plot_base_stats_with_channel_limit(
        self,
        backend: PlotlyBackend,
        mock_base_stats_multi_channel: MockPlottableBaseStats,
    ) -> None:
        """Test plotting base stats with channel limit."""
        result = backend.plot(mock_base_stats_multi_channel, channel_limit=2)

        assert isinstance(result, go.Figure)

    def test_plot_base_stats_log_scale(
        self,
        backend: PlotlyBackend,
        mock_base_stats_single_channel: MockPlottableBaseStats,
    ) -> None:
        """Test plotting base stats with log scale."""
        result = backend.plot(mock_base_stats_single_channel, log=True)

        assert isinstance(result, go.Figure)

    def test_plot_base_stats_empty_factors(
        self,
        backend: PlotlyBackend,
    ) -> None:
        """Test plotting base stats with no factors returns empty figure."""
        mock_empty = MockPlottableBaseStats(
            _factors={},
            _n_channels=1,
            _channel_mask=None,
        )

        result = backend.plot(mock_empty)

        assert isinstance(result, go.Figure)

    def test_plot_drift_mvdc(
        self,
        backend: PlotlyBackend,
        mock_drift_mvdc: MockPlottableDriftMVDC,
    ) -> None:
        """Test plotting drift MVDC output."""
        result = backend.plot(mock_drift_mvdc)

        assert isinstance(result, go.Figure)
        assert len(result.data) > 0

    def test_plot_image_grid(
        self,
        backend: PlotlyBackend,
        mock_dataset: MockDataset,
    ) -> None:
        """Test plotting image grid."""
        indices = [0, 1, 2, 3, 4, 5]
        result = backend.plot(mock_dataset, indices=indices)

        assert isinstance(result, go.Figure)
        assert len(result.data) == 6  # 6 images

    def test_plot_image_grid_custom_layout(
        self,
        backend: PlotlyBackend,
        mock_dataset: MockDataset,
    ) -> None:
        """Test plotting image grid with custom layout."""
        indices = [0, 1, 2, 3]
        result = backend.plot(mock_dataset, indices=indices, images_per_row=2, figsize=(8, 8))

        assert isinstance(result, go.Figure)
        assert len(result.data) == 4  # 4 images

    def test_plot_image_grid_single_image(
        self,
        backend: PlotlyBackend,
        mock_dataset: MockDataset,
    ) -> None:
        """Test plotting image grid with single image."""
        indices = [0]
        result = backend.plot(mock_dataset, indices=indices, images_per_row=3)

        assert isinstance(result, go.Figure)
        assert len(result.data) == 1  # 1 image
