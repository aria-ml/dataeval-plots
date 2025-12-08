"""Tests for Plotly backend."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import pytest
from test_backend_base import BackendTestBase

from dataeval_plots.backends._plotly import PlotlyBackend


class TestPlotlyBackend(BackendTestBase):
    """Test suite for Plotly backend."""

    @pytest.fixture
    def backend(self) -> PlotlyBackend:
        """Create Plotly backend instance."""
        return PlotlyBackend()

    def validate_balance_result(self, result: Any) -> None:
        """Validate the result from plotting balance."""
        assert isinstance(result, go.Figure)
        assert len(result.data) > 0

    def validate_diversity_result(self, result: Any) -> None:
        """Validate the result from plotting diversity."""
        assert isinstance(result, go.Figure)
        assert len(result.data) > 0

    def validate_sufficiency_result(self, result: Any, expected_count: int) -> None:
        """Validate the result from plotting sufficiency."""
        assert isinstance(result, list)
        assert len(result) == expected_count
        for fig in result:
            assert isinstance(fig, go.Figure)
            assert len(fig.data) > 0

    def validate_stats_result(self, result: Any) -> None:
        """Validate the result from plotting stats."""
        assert isinstance(result, go.Figure)
        # Empty figures may have 0 data traces

    def validate_drift_mvdc_result(self, result: Any) -> None:
        """Validate the result from plotting drift MVDC."""
        assert isinstance(result, go.Figure)
        assert len(result.data) > 0

    def validate_image_grid_result(self, result: Any, expected_image_count: int) -> None:
        """Validate the result from plotting image grid."""
        assert isinstance(result, go.Figure)
        # Plotly creates one data trace per image
        assert len(result.data) == expected_image_count
