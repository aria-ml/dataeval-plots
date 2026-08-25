"""Tests for Altair backend."""

from __future__ import annotations

from typing import Any

import altair as alt
import numpy as np
import pytest
from conftest import MockPlottableStats
from test_backend_base import BackendTestBase

from dataeval_plots.backends._altair import AltairBackend


class TestAltairBackend(BackendTestBase):
    """Test suite for Altair backend."""

    @pytest.fixture
    def backend(self) -> AltairBackend:
        """Create Altair backend instance."""
        return AltairBackend()

    def validate_balance_result(self, result: Any) -> None:
        """Validate the result from plotting balance."""
        assert isinstance(result, alt.Chart | alt.LayerChart)

    def validate_diversity_result(self, result: Any) -> None:
        """Validate the result from plotting diversity."""
        assert isinstance(result, alt.Chart)

    def validate_sufficiency_result(self, result: Any, expected_count: int) -> None:
        """Validate the result from plotting sufficiency."""
        assert isinstance(result, list)
        assert len(result) == expected_count
        for chart in result:
            assert isinstance(chart, alt.Chart | alt.LayerChart)

    def validate_stats_result(self, result: Any) -> None:
        """Validate the result from plotting stats."""
        assert isinstance(result, alt.Chart | alt.HConcatChart)

    def validate_drift_mvdc_result(self, result: Any) -> None:
        """Validate the result from plotting drift MVDC."""
        assert isinstance(result, alt.Chart | alt.LayerChart)

    def validate_image_grid_result(self, result: Any, expected_image_count: int) -> None:
        """Validate the result from plotting image grid."""
        # Altair returns a VConcatChart for image grids
        assert isinstance(result, alt.Chart | alt.VConcatChart | alt.HConcatChart)

    def test_plot_stats_multi_channel_non_channelwise_metric(
        self,
        backend: AltairBackend,
    ) -> None:
        """Test multi-channel stats with a 2D metric not in CHANNELWISE_METRICS."""
        stats = MockPlottableStats(
            _factors={
                "mean": np.random.rand(50, 2),
                "min": np.random.rand(50, 2),  # not channelwise
            },
            _n_channels=2,
            _channel_mask=None,
        )
        result = backend.plot(stats)
        self.validate_stats_result(result)
