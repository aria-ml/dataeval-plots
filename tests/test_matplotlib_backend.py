"""Tests for Matplotlib backend."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from conftest import MockDataset
from matplotlib.figure import Figure
from test_backend_base import BackendTestBase

from dataeval_plots.backends._matplotlib import MatplotlibBackend


class TestMatplotlibBackend(BackendTestBase):
    """Test suite for Matplotlib backend."""

    @pytest.fixture
    def backend(self) -> MatplotlibBackend:
        """Create Matplotlib backend instance."""
        return MatplotlibBackend()

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
        assert len(result.axes) >= 0

    def validate_drift_mvdc_result(self, result: Any) -> None:
        """Validate the result from plotting drift MVDC."""
        assert isinstance(result, Figure)
        assert len(result.axes) > 0

    def validate_image_grid_result(self, result: Any, expected_image_count: int) -> None:
        """Validate the result from plotting image grid."""
        assert isinstance(result, Figure)
        # Matplotlib creates axes based on grid layout
        # For 6 images at 3 per row: 2 rows x 3 columns = 6 axes
        # For 4 images at 2 per row: 2 rows x 2 columns = 4 axes
        # For 1 image at 3 per row: 1 row x 3 columns = 3 axes
        if expected_image_count == 1:
            assert len(result.axes) == 3
        else:
            assert len(result.axes) == expected_image_count

    # Override test methods that use private _plot_image_grid method
    def test_plot_image_grid(
        self,
        backend: MatplotlibBackend,
        mock_dataset: MockDataset,
    ) -> None:
        """Test plotting image grid."""
        indices = [0, 1, 2, 3, 4, 5]
        result = backend._plot_image_grid(mock_dataset, indices)
        self.validate_image_grid_result(result, expected_image_count=6)

    def test_plot_image_grid_custom_layout(
        self,
        backend: MatplotlibBackend,
        mock_dataset: MockDataset,
    ) -> None:
        """Test plotting image grid with custom layout."""
        indices = [0, 1, 2, 3]
        result = backend._plot_image_grid(mock_dataset, indices, images_per_row=2, figsize=(8, 8))
        self.validate_image_grid_result(result, expected_image_count=4)

    def test_plot_image_grid_single_image(
        self,
        backend: MatplotlibBackend,
        mock_dataset: MockDataset,
    ) -> None:
        """Test plotting image grid with single image."""
        indices = [0]
        result = backend._plot_image_grid(mock_dataset, indices, images_per_row=3)
        self.validate_image_grid_result(result, expected_image_count=1)

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
