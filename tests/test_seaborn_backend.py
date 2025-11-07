"""Tests for Seaborn backend."""

from __future__ import annotations

import pytest
from conftest import (
    MockPlottableBalance,
    MockPlottableBaseStats,
    MockPlottableCoverage,
    MockPlottableDiversity,
    MockPlottableDriftMVDC,
    MockPlottableSufficiency,
)
from matplotlib.figure import Figure

from dataeval_plots.backends._seaborn import SeabornBackend

matplotlib = pytest.importorskip("matplotlib")
seaborn = pytest.importorskip("seaborn")


class TestSeabornBackend:
    """Test suite for Seaborn backend."""

    @pytest.fixture
    def backend(self) -> SeabornBackend:
        """Create Seaborn backend instance."""
        return SeabornBackend()

    def test_plot_coverage(
        self,
        backend: SeabornBackend,
        mock_coverage: MockPlottableCoverage,
    ) -> None:
        """Test plotting coverage output."""
        # Coverage plotting requires images data, so we expect a ValueError
        with pytest.raises(ValueError, match="images parameter is required"):
            backend.plot(mock_coverage)

    def test_plot_balance_global(
        self,
        backend: SeabornBackend,
        mock_balance: MockPlottableBalance,
    ) -> None:
        """Test plotting global balance output."""
        result = backend.plot(mock_balance, plot_classwise=False)

        assert isinstance(result, Figure)
        assert len(result.axes) > 0

    def test_plot_balance_classwise(
        self,
        backend: SeabornBackend,
        mock_balance: MockPlottableBalance,
    ) -> None:
        """Test plotting classwise balance output."""
        result = backend.plot(mock_balance, plot_classwise=True)

        assert isinstance(result, Figure)
        assert len(result.axes) > 0

    def test_plot_balance_with_labels(
        self,
        backend: SeabornBackend,
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

        assert isinstance(result, Figure)

    def test_plot_diversity(
        self,
        backend: SeabornBackend,
        mock_diversity: MockPlottableDiversity,
    ) -> None:
        """Test plotting diversity output."""
        result = backend.plot(mock_diversity)

        assert isinstance(result, Figure)
        assert len(result.axes) > 0

    def test_plot_diversity_with_labels(
        self,
        backend: SeabornBackend,
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

        assert isinstance(result, Figure)

    def test_plot_sufficiency_single_class(
        self,
        backend: SeabornBackend,
        mock_sufficiency_single_class: MockPlottableSufficiency,
    ) -> None:
        """Test plotting sufficiency with single class."""
        result = backend.plot(mock_sufficiency_single_class)

        assert isinstance(result, list)
        assert len(result) == 2  # Two metrics: accuracy and f1
        for fig in result:
            assert isinstance(fig, Figure)
            assert len(fig.axes) > 0

    def test_plot_sufficiency_multi_class(
        self,
        backend: SeabornBackend,
        mock_sufficiency_multi_class: MockPlottableSufficiency,
    ) -> None:
        """Test plotting sufficiency with multiple classes."""
        class_names = ["class_A", "class_B", "class_C"]

        result = backend.plot(mock_sufficiency_multi_class, class_names=class_names)

        assert isinstance(result, list)
        # 2 metrics * 3 classes = 6 plots
        assert len(result) == 6
        for fig in result:
            assert isinstance(fig, Figure)

    def test_plot_sufficiency_no_error_bars(
        self,
        backend: SeabornBackend,
        mock_sufficiency_single_class: MockPlottableSufficiency,
    ) -> None:
        """Test plotting sufficiency without error bars."""
        result = backend.plot(mock_sufficiency_single_class, show_error_bars=False)

        assert isinstance(result, list)
        assert len(result) > 0
        for fig in result:
            assert isinstance(fig, Figure)

    def test_plot_sufficiency_no_asymptote(
        self,
        backend: SeabornBackend,
        mock_sufficiency_single_class: MockPlottableSufficiency,
    ) -> None:
        """Test plotting sufficiency without asymptote."""
        result = backend.plot(mock_sufficiency_single_class, show_asymptote=False)

        assert isinstance(result, list)
        assert len(result) > 0
        for fig in result:
            assert isinstance(fig, Figure)

    def test_plot_base_stats_single_channel(
        self,
        backend: SeabornBackend,
        mock_base_stats_single_channel: MockPlottableBaseStats,
    ) -> None:
        """Test plotting base stats with single channel."""
        result = backend.plot(mock_base_stats_single_channel)

        assert isinstance(result, Figure)
        assert len(result.axes) > 0

    def test_plot_base_stats_multi_channel(
        self,
        backend: SeabornBackend,
        mock_base_stats_multi_channel: MockPlottableBaseStats,
    ) -> None:
        """Test plotting base stats with multiple channels."""
        result = backend.plot(mock_base_stats_multi_channel)

        assert isinstance(result, Figure)
        assert len(result.axes) > 0

    def test_plot_base_stats_with_channel_limit(
        self,
        backend: SeabornBackend,
        mock_base_stats_multi_channel: MockPlottableBaseStats,
    ) -> None:
        """Test plotting base stats with channel limit."""
        result = backend.plot(mock_base_stats_multi_channel, channel_limit=2)

        assert isinstance(result, Figure)

    def test_plot_base_stats_log_scale(
        self,
        backend: SeabornBackend,
        mock_base_stats_single_channel: MockPlottableBaseStats,
    ) -> None:
        """Test plotting base stats with log scale."""
        result = backend.plot(mock_base_stats_single_channel, log=True)

        assert isinstance(result, Figure)

    def test_plot_base_stats_empty_factors(
        self,
        backend: SeabornBackend,
    ) -> None:
        """Test plotting base stats with no factors returns empty figure."""
        mock_empty = MockPlottableBaseStats(
            _factors={},
            _n_channels=1,
            _channel_mask=None,
        )

        result = backend.plot(mock_empty)

        assert isinstance(result, Figure)

    def test_plot_drift_mvdc(
        self,
        backend: SeabornBackend,
        mock_drift_mvdc: MockPlottableDriftMVDC,
    ) -> None:
        """Test plotting drift MVDC output."""
        result = backend.plot(mock_drift_mvdc)

        assert isinstance(result, Figure)
        assert len(result.axes) > 0
