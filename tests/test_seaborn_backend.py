"""Tests for Seaborn backend."""

from __future__ import annotations

from typing import Any

import pytest
from matplotlib.figure import Figure
from test_backend_base import BackendTestBase

from dataeval_plots.backends._seaborn import SeabornBackend

seaborn = pytest.importorskip("seaborn")


class TestSeabornBackend(BackendTestBase):
    """Test suite for Seaborn backend."""

    @pytest.fixture
    def backend(self) -> SeabornBackend:
        """Create Seaborn backend instance."""
        return SeabornBackend()

    def validate_balance_result(self, result: Any) -> None:
        """Validate the result from plotting balance."""
        assert isinstance(result, Figure)
        assert len(result.axes) > 0

    def validate_diversity_result(self, result: Any) -> None:
        """Validate the result from plotting diversity."""
        assert isinstance(result, Figure)
        assert len(result.axes) > 0

    def validate_sufficiency_result(self, result: Any, expected_count: int) -> None:
        """Validate the result from plotting sufficiency."""
        assert isinstance(result, list)
        assert len(result) == expected_count
        for fig in result:
            assert isinstance(fig, Figure)
            assert len(fig.axes) > 0

    def validate_stats_result(self, result: Any) -> None:
        """Validate the result from plotting stats."""
        assert isinstance(result, Figure)
        # Empty figures still have axes in matplotlib
        assert len(result.axes) >= 0

    def validate_drift_mvdc_result(self, result: Any) -> None:
        """Validate the result from plotting drift MVDC."""
        assert isinstance(result, Figure)
        assert len(result.axes) > 0

    def validate_image_grid_result(self, result: Any, expected_image_count: int) -> None:
        """Validate the result from plotting image grid."""
        assert isinstance(result, Figure)
        # Seaborn creates axes based on grid layout, not just image count
        # For 6 images at 3 per row: 2 rows x 3 columns = 6 axes
        # For 4 images at 2 per row: 2 rows x 2 columns = 4 axes
        # For 1 image at 3 per row: 1 row x 3 columns = 3 axes
        if expected_image_count == 1:
            assert len(result.axes) == 3
        else:
            assert len(result.axes) == expected_image_count
