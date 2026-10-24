"""Shared test base class for all backend tests."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Any

import pytest
from conftest import (
    MockDataset,
    MockPlottableBalance,
    MockPlottableDiversity,
    MockPlottableDriftMVDC,
    MockPlottableFeasibility,
    MockPlottableStats,
    MockPlottableSufficiency,
    _feasibility,
    _source,
    _target,
)

from dataeval_plots.backends._base import BasePlottingBackend
from dataeval_plots.backends._shared import FEASIBILITY_COLORS

ZONE_FILLS = {FEASIBILITY_COLORS[f"{kind}_fill"] for kind in ("demonstrated", "unknown", "ruled_out")}


def assert_draws(drawn: Iterable[str], texts: Sequence[str]) -> None:
    """Assert each of ``texts`` is drawn, folding line breaks and markup to single spaces."""
    folded = {" ".join(re.sub(r"<[^>]+>", " ", t).split()) for t in drawn}
    missing = [t for t in texts if t not in folded]
    assert not missing, f"not drawn: {missing}"


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

    @abstractmethod
    def validate_feasibility_result(
        self, result: Any, texts: Sequence[str], zones: int, faint: bool = False, outline: bool = False
    ) -> None:
        """Validate the result from plotting feasibility.

        Should check that result is of the correct type for this backend; that it draws each of ``texts`` (compared
        with line breaks folded to spaces); that it draws ``zones`` zone fills; whether any evidence mark is drawn
        faint (a restricted source); and whether the dashed conflict outline is drawn.
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
        # Global balance plot produces n_factors x n_factors matrix after masking
        # With 5 factors, that's 5x5 (class_label row + 4 factor rows, and 5 factor columns)
        row_labels = ["row_0", "row_1", "row_2", "row_3", "row_4"]
        col_labels = ["col_0", "col_1", "col_2", "col_3", "col_4"]

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

    def test_plot_feasibility_how_to(
        self,
        backend: BasePlottingBackend,
        mock_feasibility_how_to: MockPlottableFeasibility,
    ) -> None:
        """A demonstrated zone, an unknown zone, both requirements, the caption and the narrower-scope note."""
        result = backend.plot(mock_feasibility_how_to)
        self.validate_feasibility_result(
            result,
            [
                "Models may use the whole image",
                "Required accuracy",
                "demonstrated",
                "unknown",
                "95.7% lower confidence bound",
                "95%: demonstrated ✓",
                "99%: insufficient evidence ?",
                "Your model scored 96.9% on 5,000 evaluation images; given sampling uncertainty, its lower confidence"
                " bound is 95.7%. Bounds apply to new images from the same population, at 95% joint confidence.",
                "Group-based ceiling (72.1%) applies only to models that give one answer per group.",
            ],
            zones=2,
        )

    def test_plot_feasibility_ruled_out(
        self,
        backend: BasePlottingBackend,
        mock_feasibility_ruled_out: MockPlottableFeasibility,
    ) -> None:
        """A ruled-out zone above the ceiling, with its boundary label and caption line."""
        result = backend.plot(mock_feasibility_ruled_out)
        self.validate_feasibility_result(
            result,
            [
                "ruled out*",
                "93.0% upper confidence bound",
                "91.2% lower confidence bound",
                "90%: demonstrated ✓",
                "95%: ruled out* ✗",
                "Your model scored 92.5% on 5,000 evaluation images; given sampling uncertainty, its lower confidence"
                " bound is 91.2%. Bounds apply to new images from the same population, at 95% joint confidence."
                " * Ruled out under the recorded assumptions, at the stated confidence.",
            ],
            zones=3,
        )

    def test_plot_feasibility_recall(
        self,
        backend: BasePlottingBackend,
        mock_feasibility_recall: MockPlottableFeasibility,
    ) -> None:
        """A recall requirement is titled with its false-positive limit and its caption gives the witness's bound."""
        result = backend.plot(mock_feasibility_recall)
        self.validate_feasibility_result(
            result,
            [
                "Required recall for Tank (at most 5% false positives)",
                "62.1% lower confidence bound",
                "85.0% upper confidence bound",
                "60%: demonstrated ✓",
                "Your model scored 70.0% recall on 160 Tank images; given sampling uncertainty, its lower confidence"
                " bound is 62.1%. Its upper confidence bound on false positives is 4.1% (observed 2.6% of 651 other"
                " images). Bounds apply to new images from the same population, at 95% joint confidence."
                " * Ruled out under the recorded assumptions, at the stated confidence.",
            ],
            zones=3,
        )

    def test_plot_feasibility_conflict(
        self,
        backend: BasePlottingBackend,
        mock_feasibility_conflict: MockPlottableFeasibility,
    ) -> None:
        """No zones: the warning text, both clashing bounds and the conflict caption."""
        result = backend.plot(mock_feasibility_conflict)
        self.validate_feasibility_result(
            result,
            [
                "Evidence conflicts: check assumptions",
                "92.0% upper confidence bound",
                "94.0% lower confidence bound",
                "95%",
                "Bounds disagree: check the assumptions, such as evaluation images entering training."
                " Even when they hold, the chance that any bound fails is at most 5%.",
            ],
            zones=0,
            outline=True,
        )

    def test_plot_feasibility_evidence_view(
        self,
        backend: BasePlottingBackend,
        mock_feasibility_how_to: MockPlottableFeasibility,
    ) -> None:
        """One row per source with its observed score and bound, the requirements, the legend line and caption."""
        result = backend.plot(mock_feasibility_how_to, plot_evidence=True)
        self.validate_feasibility_result(
            result,
            [
                "Accuracy",
                "Your model",
                "96.9% observed",
                "95.7% lower bound",
                "Built-in k-NN",
                "72.1% upper bound",
                "95% required",
                "99% required",
                "● observed · ┃ lower confidence bound · ┃ upper confidence bound",
                "Observed on the evaluation images; bounds apply to new images from the same population, at 95% joint"
                " confidence.",
            ],
            zones=0,
            faint=True,
        )

    def test_plot_feasibility_restricted_only_evidence(self, backend: BasePlottingBackend) -> None:
        """Review Focus 5: an evidence view whose only source is a narrower ceiling draws it faint, labeled with its
        scope."""
        output = _feasibility(
            [_target(0.95, "unresolved")],
            [_source("group-based ceiling", 0.721, scope="partition_restricted", bound_type="upper", n=400)],
        )
        result = backend.plot(output, plot_evidence=True)
        self.validate_feasibility_result(
            result,
            [
                "Group-based ceiling (one answer per group)",
                "72.1% upper bound",
                "95% required",
            ],
            zones=0,
            faint=True,
        )

    @pytest.mark.parametrize("plot_evidence", [False, True])
    def test_plot_feasibility_axis_edges(
        self,
        backend: BasePlottingBackend,
        mock_feasibility_low_threshold: MockPlottableFeasibility,
        mock_feasibility_witness_zero: MockPlottableFeasibility,
        mock_feasibility_empty: MockPlottableFeasibility,
        plot_evidence: bool,
    ) -> None:
        """A far threshold, a bound at the axis edge and a band with no bounds all draw their requirements."""
        for output, marker, zones in (
            (mock_feasibility_low_threshold, "50%: demonstrated ✓", 2),
            (mock_feasibility_witness_zero, "95%: insufficient evidence ?", 1),
            (mock_feasibility_empty, "95%: insufficient evidence ?", 1),
        ):
            result = backend.plot(output, plot_evidence=plot_evidence)
            text = marker[:3] + " required" if plot_evidence else marker
            self.validate_feasibility_result(result, [text], zones=0 if plot_evidence else zones)

    def test_plot_feasibility_no_bound_evidence(self, backend: BasePlottingBackend) -> None:
        """A source with no bound is listed as such, and a band with no bound at all says so instead of empty axes."""
        output = _feasibility(
            [_target(0.9, "unresolved")],
            [_source("group-based ceiling", None, scope="partition_restricted", bound_type="upper", n=None)],
        )
        result = backend.plot(output, plot_evidence=True)
        self.validate_feasibility_result(
            result,
            [
                "No bound is available yet.",
                "Group-based ceiling (one answer per group)",
                "no bound available",
                "90% required",
            ],
            zones=0,
        )
