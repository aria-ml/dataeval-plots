"""Tests for Altair backend."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import altair as alt
import numpy as np
import pytest
from conftest import MockPlottableFeasibility, MockPlottableStats
from test_backend_base import ZONE_FILLS, BackendTestBase, assert_draws

from dataeval_plots.backends import _feasibility_layout as layout
from dataeval_plots.backends._altair import AltairBackend


def _strings(spec: Any) -> Iterator[str]:
    """Every string in a chart spec: its data values, titles and labels."""
    if isinstance(spec, str):
        yield spec
    elif isinstance(spec, dict | list):
        for value in spec.values() if isinstance(spec, dict) else spec:
            yield from _strings(value)


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

    def validate_feasibility_result(
        self, result: Any, texts: Sequence[str], zones: int, faint: bool = False, outline: bool = False
    ) -> None:
        """Validate the result from plotting feasibility: zone mark_rect layers, the dashed outline, faint marks."""
        assert isinstance(result, alt.Chart | alt.LayerChart | alt.VConcatChart)
        spec = result.to_dict()
        assert_draws(_strings(spec), texts)
        marks = [layer["mark"] for chart in spec["vconcat"] for layer in chart.get("layer", [])]
        rects = [m for m in marks if m["type"] == "rect"]
        assert sum(m.get("color") in ZONE_FILLS for m in rects) == zones
        assert any(m.get("opacity") == 0.35 for m in marks) == faint
        assert any("strokeDash" in m for m in rects) == outline

    def test_plot_feasibility_figsize_height(
        self, backend: AltairBackend, mock_feasibility_two_bands: MockPlottableFeasibility
    ) -> None:
        """An explicit height stretches the band views until the laid-out height meets it."""
        for height in (5.0, 12.0):
            spec = backend.plot(mock_feasibility_two_bands, figsize=(9, height)).to_dict()
            views = sum(chart["height"] for chart in spec["vconcat"]) / layout.PX  # band views and captions
            fixed = len(spec["vconcat"]) // 2 * (layout.TITLE + layout.XAXIS + layout.GAP)  # subtitle, axis, gap
            assert views + fixed == pytest.approx(height, abs=0.01)

    def test_plot_feasibility_composes(self, backend: AltairBackend, mock_feasibility_how_to) -> None:
        """No chart-level config, so the chart can sit inside another (altair refuses a nested config)."""
        chart = backend.plot(mock_feasibility_how_to)
        alt.hconcat(chart, chart).to_dict()

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
