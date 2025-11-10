"""Tests for Matplotlib backend."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import (
    MockDataset,
    MockPlottableBalance,
    MockPlottableBaseStats,
    MockPlottableDiversity,
    MockPlottableDriftMVDC,
    MockPlottableSufficiency,
)
from matplotlib.figure import Figure

from dataeval_plots.backends._matplotlib import MatplotlibBackend

matplotlib = pytest.importorskip("matplotlib")


class TestMatplotlibBackend:
    """Test suite for Matplotlib backend."""

    @pytest.fixture
    def backend(self) -> MatplotlibBackend:
        """Create Matplotlib backend instance."""
        return MatplotlibBackend()

    def test_plot_balance_global(
        self,
        backend: MatplotlibBackend,
        mock_balance: MockPlottableBalance,
    ) -> None:
        """Test plotting global balance output."""
        result = backend.plot(mock_balance, plot_classwise=False)

        assert isinstance(result, Figure)
        assert len(result.axes) > 0

    def test_plot_balance_classwise(
        self,
        backend: MatplotlibBackend,
        mock_balance: MockPlottableBalance,
    ) -> None:
        """Test plotting classwise balance output."""
        result = backend.plot(mock_balance, plot_classwise=True)

        assert isinstance(result, Figure)
        assert len(result.axes) > 0

    def test_plot_balance_with_labels(
        self,
        backend: MatplotlibBackend,
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
        backend: MatplotlibBackend,
        mock_diversity: MockPlottableDiversity,
    ) -> None:
        """Test plotting diversity output."""
        result = backend.plot(mock_diversity)

        assert isinstance(result, Figure)
        assert len(result.axes) > 0

    def test_plot_diversity_with_labels(
        self,
        backend: MatplotlibBackend,
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
        backend: MatplotlibBackend,
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
        backend: MatplotlibBackend,
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
        backend: MatplotlibBackend,
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
        backend: MatplotlibBackend,
        mock_sufficiency_single_class: MockPlottableSufficiency,
    ) -> None:
        """Test plotting sufficiency without asymptote."""
        result = backend.plot(mock_sufficiency_single_class, show_asymptote=False)

        assert isinstance(result, list)
        assert len(result) > 0
        for fig in result:
            assert isinstance(fig, Figure)

    def test_plot_stats_single_channel(
        self,
        backend: MatplotlibBackend,
        mock_stats_single_channel: MockPlottableBaseStats,
    ) -> None:
        """Test plotting base stats with single channel."""
        result = backend.plot(mock_stats_single_channel)

        assert isinstance(result, Figure)
        assert len(result.axes) > 0

    def test_plot_stats_multi_channel(
        self,
        backend: MatplotlibBackend,
        mock_stats_multi_channel: MockPlottableBaseStats,
    ) -> None:
        """Test plotting base stats with multiple channels."""
        result = backend.plot(mock_stats_multi_channel)

        assert isinstance(result, Figure)
        assert len(result.axes) > 0

    def test_plot_stats_with_channel_limit(
        self,
        backend: MatplotlibBackend,
        mock_stats_multi_channel: MockPlottableBaseStats,
    ) -> None:
        """Test plotting base stats with channel limit."""
        result = backend.plot(mock_stats_multi_channel, channel_limit=2)

        assert isinstance(result, Figure)

    def test_plot_stats_log_scale(
        self,
        backend: MatplotlibBackend,
        mock_stats_single_channel: MockPlottableBaseStats,
    ) -> None:
        """Test plotting base stats with log scale."""
        result = backend.plot(mock_stats_single_channel, log=True)

        assert isinstance(result, Figure)

    def test_plot_stats_empty_factors(
        self,
        backend: MatplotlibBackend,
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
        backend: MatplotlibBackend,
        mock_drift_mvdc: MockPlottableDriftMVDC,
    ) -> None:
        """Test plotting drift MVDC output."""
        result = backend.plot(mock_drift_mvdc)

        assert isinstance(result, Figure)
        assert len(result.axes) > 0

    def test_plot_image_grid(
        self,
        backend: MatplotlibBackend,
        mock_dataset: MockDataset,
    ) -> None:
        """Test plotting image grid."""
        indices = [0, 1, 2, 3, 4, 5]
        result = backend._plot_image_grid(mock_dataset, indices)

        assert isinstance(result, Figure)
        assert len(result.axes) == 6  # 2 rows x 3 columns

    def test_plot_image_grid_custom_layout(
        self,
        backend: MatplotlibBackend,
        mock_dataset: MockDataset,
    ) -> None:
        """Test plotting image grid with custom layout."""
        indices = [0, 1, 2, 3]
        result = backend._plot_image_grid(mock_dataset, indices, images_per_row=2, figsize=(8, 8))

        assert isinstance(result, Figure)
        assert len(result.axes) == 4  # 2 rows x 2 columns

    def test_plot_image_grid_single_image(
        self,
        backend: MatplotlibBackend,
        mock_dataset: MockDataset,
    ) -> None:
        """Test plotting image grid with single image."""
        indices = [0]
        result = backend._plot_image_grid(mock_dataset, indices, images_per_row=3)

        assert isinstance(result, Figure)
        # With 1 image and 3 images_per_row, we get 1 row x 3 columns = 3 axes
        assert len(result.axes) == 3

    def test_plot_image_grid_with_labels(
        self,
        backend: MatplotlibBackend,
    ) -> None:
        """Test plotting image grid with labels from targets."""
        # Create dataset with classification targets
        images = [np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8) for _ in range(3)]
        targets = [
            np.array([0.9, 0.1]),  # Probabilities for class 0
            np.array([0.2, 0.8]),  # Probabilities for class 1
            np.array([0.5, 0.5]),  # Equal probabilities
        ]
        dataset = MockDataset(
            images=images,
            targets=targets,
            index2label={0: "cat", 1: "dog"},
        )

        indices = [0, 1, 2]
        result = backend._plot_image_grid(dataset, indices, show_labels=True)

        assert isinstance(result, Figure)
        assert len(result.axes) == 3
        # Check that titles are set (labels are shown)
        assert result.axes[0].get_title() != ""
        assert "cat" in result.axes[0].get_title()
        assert "dog" in result.axes[1].get_title()

    def test_plot_image_grid_with_metadata(
        self,
        backend: MatplotlibBackend,
    ) -> None:
        """Test plotting image grid with metadata."""
        # Create dataset with metadata
        images = [np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8) for _ in range(3)]
        metadatas = [
            {"scene": "outdoor", "time": "day"},
            {"scene": "indoor", "time": "night"},
            {"scene": "outdoor", "time": "night"},
        ]
        dataset = MockDataset(images=images, metadatas=metadatas)

        indices = [0, 1, 2]
        result = backend._plot_image_grid(dataset, indices, show_metadata=True)

        assert isinstance(result, Figure)
        assert len(result.axes) == 3
        # Check that titles contain metadata
        assert "scene" in result.axes[0].get_title()
        assert "outdoor" in result.axes[0].get_title()

    def test_plot_image_grid_with_additional_metadata(
        self,
        backend: MatplotlibBackend,
        mock_dataset: MockDataset,
    ) -> None:
        """Test plotting image grid with additional metadata."""
        indices = [0, 1, 2]
        additional_metadata = [
            {"score": 0.95},
            {"score": 0.87},
            {"score": 0.92},
        ]

        result = backend._plot_image_grid(
            mock_dataset,
            indices,
            show_metadata=True,
            additional_metadata=additional_metadata,
        )

        assert isinstance(result, Figure)
        assert len(result.axes) == 3
        # Check that titles contain additional metadata
        assert "score" in result.axes[0].get_title()

    def test_plot_image_grid_with_object_detection_targets(
        self,
        backend: MatplotlibBackend,
    ) -> None:
        """Test plotting image grid with object detection targets."""
        # Create dataset with object detection targets
        images = [np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8) for _ in range(2)]
        targets = [
            {
                "boxes": np.array([[0, 0, 10, 10], [5, 5, 15, 15]]),
                "labels": np.array([0, 1]),
                "scores": np.array([0.9, 0.8]),
            },
            {
                "boxes": np.array([[2, 2, 12, 12]]),
                "labels": np.array([1]),
                "scores": np.array([0.95]),
            },
        ]
        dataset = MockDataset(
            images=images,
            targets=targets,
            index2label={0: "person", 1: "car"},
        )

        indices = [0, 1]
        result = backend._plot_image_grid(dataset, indices, show_labels=True, images_per_row=2)

        assert isinstance(result, Figure)
        assert len(result.axes) == 2  # 1 row x 2 columns
        # Check that titles show object counts
        assert "person" in result.axes[0].get_title()
        assert "car" in result.axes[0].get_title() or "car" in result.axes[1].get_title()

    def test_plot_image_grid_with_bounding_boxes(
        self,
        backend: MatplotlibBackend,
    ) -> None:
        """Test that bounding boxes are drawn on images with object detection targets."""
        # Create dataset with object detection targets
        images = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(2)]
        targets = [
            {
                "boxes": np.array([[10, 10, 30, 30], [20, 20, 40, 40]]),
                "labels": np.array([0, 1]),
                "scores": np.array([0.9, 0.85]),
            },
            {
                "boxes": np.array([[5, 5, 25, 25]]),
                "labels": np.array([0]),
                "scores": np.array([0.95]),
            },
        ]
        dataset = MockDataset(
            images=images,
            targets=targets,
            index2label={0: "cat", 1: "dog"},
        )

        indices = [0, 1]
        # Draw without labels to test that boxes are still drawn
        result = backend._plot_image_grid(dataset, indices, images_per_row=2)

        assert isinstance(result, Figure)
        assert len(result.axes) == 2
        # Bounding boxes should be drawn even without show_labels=True
