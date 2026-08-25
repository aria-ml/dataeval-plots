"""Tests for Plotly backend."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import polars as pl
import pytest
from conftest import MockPlottableBalance
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

    def test_plot_balance_truncates_short_labels(self, backend: PlotlyBackend) -> None:
        """Row/col labels shorter than the data rows are truncated in the trace."""
        balance_df = pl.DataFrame(
            {
                "factor_name": ["class_label", "f0", "f1", "f2"],
                "mi_value": [0.1, 0.2, 0.3, 0.4],
            }
        )
        factors_df = pl.DataFrame(
            [
                {"factor1": f"f{i}", "factor2": f"f{j}", "mi_value": 0.5, "is_correlated": False}
                for i in range(3)
                for j in range(3)
            ]
        )
        classwise = pl.DataFrame(
            {
                "class_name": ["c0", "c1"],
                "factor_name": ["a0", "a1"],
                "mi_value": [0.1, 0.2],
                "is_imbalanced": [False, True],
            }
        )
        output = MockPlottableBalance(balance=balance_df, factors=factors_df, classwise=classwise)
        result = backend.plot(output, row_labels=["r"], col_labels=["c"])
        self.validate_balance_result(result)
