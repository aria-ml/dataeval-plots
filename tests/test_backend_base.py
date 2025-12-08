"""Shared test base class for all backend tests."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pytest
from conftest import (
    MockDataset,
    MockPlottableBalance,
    MockPlottableDiversity,
    MockPlottableDriftMVDC,
    MockPlottableStats,
    MockPlottableSufficiency,
)

from dataeval_plots.backends._base import BasePlottingBackend


class BackendTestBase(ABC):
    """Abstract base class for backend tests.

    Each backend test class should inherit from this class and implement:
    1. backend() fixture - returns the backend instance
    2. validate_* methods - validates the result type for each plot type
    """

    @pytest.fixture
    @abstractmethod
    def backend(self) -> BasePlottingBackend:
        """Create backend instance. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def validate_balance_result(self, result: Any) -> None:
        """Validate the result from plotting balance.

        Should check that result is of the correct type for this backend.
        """
        pass

    @abstractmethod
    def validate_diversity_result(self, result: Any) -> None:
        """Validate the result from plotting diversity.

        Should check that result is of the correct type for this backend.
        """
        pass

    @abstractmethod
    def validate_sufficiency_result(self, result: Any, expected_count: int) -> None:
        """Validate the result from plotting sufficiency.

        Should check that result is a list of the correct type for this backend
        with the expected number of plots.
        """
        pass

    @abstractmethod
    def validate_stats_result(self, result: Any) -> None:
        """Validate the result from plotting stats.

        Should check that result is of the correct type for this backend.
        """
        pass

    @abstractmethod
    def validate_drift_mvdc_result(self, result: Any) -> None:
        """Validate the result from plotting drift MVDC.

        Should check that result is of the correct type for this backend.
        """
        pass

    @abstractmethod
    def validate_image_grid_result(self, result: Any, expected_image_count: int) -> None:
        """Validate the result from plotting image grid.

        Should check that result is of the correct type for this backend
        and contains the expected number of images.
        """
        pass

    # Common test methods that use the validation methods above

    def test_plot_balance_global(
        self,
        backend: BasePlottingBackend,
        mock_balance: MockPlottableBalance,
    ) -> None:
        """Test plotting global balance output."""
        result = backend.plot(mock_balance, plot_classwise=False)
        self.validate_balance_result(result)

    def test_plot_balance_classwise(
        self,
        backend: BasePlottingBackend,
        mock_balance: MockPlottableBalance,
    ) -> None:
        """Test plotting classwise balance output."""
        result = backend.plot(mock_balance, plot_classwise=True)
        self.validate_balance_result(result)

    def test_plot_balance_with_labels(
        self,
        backend: BasePlottingBackend,
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

        self.validate_balance_result(result)

    def test_plot_diversity(
        self,
        backend: BasePlottingBackend,
        mock_diversity: MockPlottableDiversity,
    ) -> None:
        """Test plotting diversity output."""
        result = backend.plot(mock_diversity)
        self.validate_diversity_result(result)

    def test_plot_diversity_with_labels(
        self,
        backend: BasePlottingBackend,
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

        self.validate_diversity_result(result)

    def test_plot_sufficiency_single_class(
        self,
        backend: BasePlottingBackend,
        mock_sufficiency_single_class: MockPlottableSufficiency,
    ) -> None:
        """Test plotting sufficiency with single class."""
        result = backend.plot(mock_sufficiency_single_class)
        # Two metrics: accuracy and f1
        self.validate_sufficiency_result(result, expected_count=2)

    def test_plot_sufficiency_multi_class(
        self,
        backend: BasePlottingBackend,
        mock_sufficiency_multi_class: MockPlottableSufficiency,
    ) -> None:
        """Test plotting sufficiency with multiple classes."""
        class_names = ["class_A", "class_B", "class_C"]

        result = backend.plot(mock_sufficiency_multi_class, class_names=class_names)

        # 2 metrics * 3 classes = 6 plots
        self.validate_sufficiency_result(result, expected_count=6)

    def test_plot_sufficiency_no_error_bars(
        self,
        backend: BasePlottingBackend,
        mock_sufficiency_single_class: MockPlottableSufficiency,
    ) -> None:
        """Test plotting sufficiency without error bars."""
        result = backend.plot(mock_sufficiency_single_class, show_error_bars=False)
        self.validate_sufficiency_result(result, expected_count=2)

    def test_plot_sufficiency_no_asymptote(
        self,
        backend: BasePlottingBackend,
        mock_sufficiency_single_class: MockPlottableSufficiency,
    ) -> None:
        """Test plotting sufficiency without asymptote."""
        result = backend.plot(mock_sufficiency_single_class, show_asymptote=False)
        self.validate_sufficiency_result(result, expected_count=2)

    def test_plot_stats_single_channel(
        self,
        backend: BasePlottingBackend,
        mock_stats_single_channel: MockPlottableStats,
    ) -> None:
        """Test plotting base stats with single channel."""
        result = backend.plot(mock_stats_single_channel)
        self.validate_stats_result(result)

    def test_plot_stats_multi_channel(
        self,
        backend: BasePlottingBackend,
        mock_stats_multi_channel: MockPlottableStats,
    ) -> None:
        """Test plotting base stats with multiple channels."""
        result = backend.plot(mock_stats_multi_channel)
        self.validate_stats_result(result)

    def test_plot_stats_with_channel_limit(
        self,
        backend: BasePlottingBackend,
        mock_stats_multi_channel: MockPlottableStats,
    ) -> None:
        """Test plotting base stats with channel limit."""
        result = backend.plot(mock_stats_multi_channel, channel_limit=2)
        self.validate_stats_result(result)

    def test_plot_stats_log_scale(
        self,
        backend: BasePlottingBackend,
        mock_stats_single_channel: MockPlottableStats,
    ) -> None:
        """Test plotting base stats with log scale."""
        result = backend.plot(mock_stats_single_channel, log=True)
        self.validate_stats_result(result)

    def test_plot_stats_empty_factors(
        self,
        backend: BasePlottingBackend,
    ) -> None:
        """Test plotting base stats with no factors returns empty figure."""
        mock_empty = MockPlottableStats(
            _factors={},
            _n_channels=1,
            _channel_mask=None,
        )

        result = backend.plot(mock_empty)
        self.validate_stats_result(result)

    def test_plot_drift_mvdc(
        self,
        backend: BasePlottingBackend,
        mock_drift_mvdc: MockPlottableDriftMVDC,
    ) -> None:
        """Test plotting drift MVDC output."""
        result = backend.plot(mock_drift_mvdc)
        self.validate_drift_mvdc_result(result)

    def test_plot_image_grid(
        self,
        backend: BasePlottingBackend,
        mock_dataset: MockDataset,
    ) -> None:
        """Test plotting image grid."""
        indices = [0, 1, 2, 3, 4, 5]
        result = backend.plot(mock_dataset, indices=indices)
        self.validate_image_grid_result(result, expected_image_count=6)

    def test_plot_image_grid_custom_layout(
        self,
        backend: BasePlottingBackend,
        mock_dataset: MockDataset,
    ) -> None:
        """Test plotting image grid with custom layout."""
        indices = [0, 1, 2, 3]
        result = backend.plot(mock_dataset, indices=indices, images_per_row=2, figsize=(8, 8))
        self.validate_image_grid_result(result, expected_image_count=4)

    def test_plot_image_grid_single_image(
        self,
        backend: BasePlottingBackend,
        mock_dataset: MockDataset,
    ) -> None:
        """Test plotting image grid with single image."""
        indices = [0]
        result = backend.plot(mock_dataset, indices=indices, images_per_row=3)
        self.validate_image_grid_result(result, expected_image_count=1)
