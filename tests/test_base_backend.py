from __future__ import annotations

from typing import Any

import pytest

from dataeval_plots.backends._base import BasePlottingBackend
from dataeval_plots.protocols import (
    Dataset,
    PlottableBalance,
    PlottableDiversity,
    PlottableDriftMVDC,
    PlottableStats,
    PlottableSufficiency,
)


class MockBackend(BasePlottingBackend):
    """Mock backend for testing base functionality."""

    def _plot_balance(
        self,
        output: PlottableBalance,
        row_labels: Any = None,
        col_labels: Any = None,
        plot_classwise: bool = False,
    ) -> None:
        pass

    def _plot_diversity(
        self,
        output: PlottableDiversity,
        row_labels: Any = None,
        col_labels: Any = None,
    ) -> None:
        pass

    def _plot_sufficiency(
        self,
        output: PlottableSufficiency,
        class_names: list[str] | None = None,
        show_error_bars: bool = True,
        show_asymptote: bool = True,
    ) -> None:
        pass

    def _plot_stats(
        self,
        output: PlottableStats,
        log: bool = False,
        channel_limit: int | None = None,
        channel_index: int | list[int] | None = None,
    ) -> None:
        pass

    def _plot_drift_mvdc(
        self,
        output: PlottableDriftMVDC,
    ) -> None:
        pass

    def _plot_image_grid(
        self,
        dataset: Dataset,
        indices: Any,
        images_per_row: int,
        figsize: tuple[int, int],
    ) -> None:
        pass


class TestBaseBackend:
    """Test suite for base backend."""

    @pytest.fixture
    def backend(self) -> MockBackend:
        """Create mock backend instance."""
        return MockBackend()

    def test_plot_unsupported_type(
        self,
        backend: MockBackend,
    ) -> None:
        """Test plotting with unsupported plot type."""

        class UnsupportedPlottable:
            def plot_type(self) -> str:
                return "unsupported"

        with pytest.raises(NotImplementedError, match="Plotting not implemented for plot_type"):
            backend.plot(UnsupportedPlottable())  # type: ignore[arg-type]
