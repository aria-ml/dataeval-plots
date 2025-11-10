"""Tests for Altair backend."""

from __future__ import annotations

import altair as alt
import pytest
from conftest import (
    MockDataset,
    MockPlottableBalance,
    MockPlottableDiversity,
    MockPlottableDriftMVDC,
    MockPlottableStats,
    MockPlottableSufficiency,
)

from dataeval_plots.backends._altair import AltairBackend

altair = pytest.importorskip("altair")


class TestAltairBackend:
    """Test suite for Altair backend."""

    @pytest.fixture
    def backend(self) -> AltairBackend:
        """Create Altair backend instance."""
        return AltairBackend()

    def test_plot_balance_global(
        self,
        backend: AltairBackend,
        mock_balance: MockPlottableBalance,
    ) -> None:
        """Test plotting global balance output."""
        result = backend.plot(mock_balance, plot_classwise=False)

        assert isinstance(result, (alt.Chart, alt.LayerChart))

    def test_plot_balance_classwise(
        self,
        backend: AltairBackend,
        mock_balance: MockPlottableBalance,
    ) -> None:
        """Test plotting classwise balance output."""
        result = backend.plot(mock_balance, plot_classwise=True)

        assert isinstance(result, (alt.Chart, alt.LayerChart))

    def test_plot_balance_with_labels(
        self,
        backend: AltairBackend,
        mock_balance: MockPlottableBalance,
    ) -> None:
        """Test plotting balance with custom labels."""
        # Global balance plot produces (n_factors-1) x (n_factors-1) matrix
        # With 5 factors, that's 4x4
        row_labels = ["row_0", "row_1", "row_2", "row_3"]
        col_labels = ["col_0", "col_1", "col_2", "col_3"]

        result = backend.plot(
            mock_balance,
            plot_classwise=False,
            row_labels=row_labels,
            col_labels=col_labels,
        )

        assert isinstance(result, (alt.Chart, alt.LayerChart))

    def test_plot_diversity(
        self,
        backend: AltairBackend,
        mock_diversity: MockPlottableDiversity,
    ) -> None:
        """Test plotting diversity output."""
        result = backend.plot(mock_diversity)

        assert isinstance(result, alt.Chart)

    def test_plot_diversity_with_labels(
        self,
        backend: AltairBackend,
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

        assert isinstance(result, alt.Chart)

    def test_plot_sufficiency_single_class(
        self,
        backend: AltairBackend,
        mock_sufficiency_single_class: MockPlottableSufficiency,
    ) -> None:
        """Test plotting sufficiency with single class."""
        result = backend.plot(mock_sufficiency_single_class)

        assert isinstance(result, list)
        assert len(result) == 2  # Two metrics: accuracy and f1
        for chart in result:
            assert isinstance(chart, (alt.Chart, alt.LayerChart))

    def test_plot_sufficiency_multi_class(
        self,
        backend: AltairBackend,
        mock_sufficiency_multi_class: MockPlottableSufficiency,
    ) -> None:
        """Test plotting sufficiency with multiple classes."""
        class_names = ["class_A", "class_B", "class_C"]

        result = backend.plot(mock_sufficiency_multi_class, class_names=class_names)

        assert isinstance(result, list)
        # 2 metrics * 3 classes = 6 plots
        assert len(result) == 6
        for chart in result:
            assert isinstance(chart, (alt.Chart, alt.LayerChart))

    def test_plot_sufficiency_no_error_bars(
        self,
        backend: AltairBackend,
        mock_sufficiency_single_class: MockPlottableSufficiency,
    ) -> None:
        """Test plotting sufficiency without error bars."""
        result = backend.plot(mock_sufficiency_single_class, show_error_bars=False)

        assert isinstance(result, list)
        assert len(result) > 0
        for chart in result:
            assert isinstance(chart, (alt.Chart, alt.LayerChart))

    def test_plot_sufficiency_no_asymptote(
        self,
        backend: AltairBackend,
        mock_sufficiency_single_class: MockPlottableSufficiency,
    ) -> None:
        """Test plotting sufficiency without asymptote."""
        result = backend.plot(mock_sufficiency_single_class, show_asymptote=False)

        assert isinstance(result, list)
        assert len(result) > 0
        for chart in result:
            assert isinstance(chart, (alt.Chart, alt.LayerChart))

    def test_plot_stats_single_channel(
        self,
        backend: AltairBackend,
        mock_stats_single_channel: MockPlottableStats,
    ) -> None:
        """Test plotting base stats with single channel."""
        result = backend.plot(mock_stats_single_channel)

        assert isinstance(result, (alt.Chart, alt.HConcatChart))

    def test_plot_stats_multi_channel(
        self,
        backend: AltairBackend,
        mock_stats_multi_channel: MockPlottableStats,
    ) -> None:
        """Test plotting base stats with multiple channels."""
        result = backend.plot(mock_stats_multi_channel)

        assert isinstance(result, (alt.Chart, alt.HConcatChart))

    def test_plot_stats_with_channel_limit(
        self,
        backend: AltairBackend,
        mock_stats_multi_channel: MockPlottableStats,
    ) -> None:
        """Test plotting base stats with channel limit."""
        result = backend.plot(mock_stats_multi_channel, channel_limit=2)

        assert isinstance(result, (alt.Chart, alt.HConcatChart))

    def test_plot_stats_log_scale(
        self,
        backend: AltairBackend,
        mock_stats_single_channel: MockPlottableStats,
    ) -> None:
        """Test plotting base stats with log scale."""
        result = backend.plot(mock_stats_single_channel, log=True)

        assert isinstance(result, (alt.Chart, alt.HConcatChart))

    def test_plot_stats_empty_factors(
        self,
        backend: AltairBackend,
    ) -> None:
        """Test plotting base stats with no factors returns empty chart."""
        mock_empty = MockPlottableStats(
            _factors={},
            _n_channels=1,
            _channel_mask=None,
        )

        result = backend.plot(mock_empty)

        assert isinstance(result, alt.Chart)

    def test_plot_drift_mvdc(
        self,
        backend: AltairBackend,
        mock_drift_mvdc: MockPlottableDriftMVDC,
    ) -> None:
        """Test plotting drift MVDC output."""
        result = backend.plot(mock_drift_mvdc)

        assert isinstance(result, (alt.Chart, alt.LayerChart))

    def test_plot_image_grid(
        self,
        backend: AltairBackend,
        mock_dataset: MockDataset,
    ) -> None:
        """Test plotting image grid."""
        indices = [0, 1, 2, 3, 4, 5]
        result = backend.plot(mock_dataset, indices=indices)

        # Altair returns a VConcatChart for image grids
        assert isinstance(result, (alt.Chart, alt.VConcatChart))

    def test_plot_image_grid_custom_layout(
        self,
        backend: AltairBackend,
        mock_dataset: MockDataset,
    ) -> None:
        """Test plotting image grid with custom layout."""
        indices = [0, 1, 2, 3]
        result = backend.plot(mock_dataset, indices=indices, images_per_row=2, figsize=(8, 8))

        assert isinstance(result, (alt.Chart, alt.VConcatChart))

    def test_plot_image_grid_single_image(
        self,
        backend: AltairBackend,
        mock_dataset: MockDataset,
    ) -> None:
        """Test plotting image grid with single image."""
        indices = [0]
        result = backend.plot(mock_dataset, indices=indices, images_per_row=3)

        assert isinstance(result, (alt.Chart, alt.VConcatChart, alt.HConcatChart))
