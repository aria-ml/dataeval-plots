"""Tests for shared helper functions in _shared module."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from dataeval_plots.backends._shared import draw_bounding_boxes, extract_boxes_and_labels


class TestExtractBoxesAndLabels:
    """Tests for extract_boxes_and_labels function."""

    def test_extract_from_dict_target(self):
        """Test extracting boxes and labels from dictionary target."""
        target = {
            "boxes": np.array([[10, 20, 100, 200], [30, 40, 150, 250]]),
            "labels": np.array([0, 1]),
            "scores": np.array([0.95, 0.87]),
        }

        boxes, labels, scores = extract_boxes_and_labels(target)

        assert boxes is not None
        assert labels is not None
        assert scores is not None
        np.testing.assert_array_equal(boxes, target["boxes"])
        np.testing.assert_array_equal(labels, target["labels"])
        np.testing.assert_array_equal(scores, target["scores"])

    def test_extract_from_dict_target_without_scores(self):
        """Test extracting boxes and labels from dictionary without scores."""
        target = {
            "boxes": np.array([[10, 20, 100, 200]]),
            "labels": np.array([0]),
        }

        boxes, labels, scores = extract_boxes_and_labels(target)

        assert boxes is not None
        assert labels is not None
        assert scores is None
        np.testing.assert_array_equal(boxes, target["boxes"])
        np.testing.assert_array_equal(labels, target["labels"])

    def test_extract_from_object_target(self):
        """Test extracting boxes and labels from object with attributes."""

        class DetectionResult:
            def __init__(self):
                self.boxes = np.array([[10, 20, 100, 200]])
                self.labels = np.array([0])
                self.scores = np.array([0.95])

        target = DetectionResult()
        boxes, labels, scores = extract_boxes_and_labels(target)

        assert boxes is not None
        assert labels is not None
        assert scores is not None
        np.testing.assert_array_equal(boxes, np.array([[10, 20, 100, 200]]))
        np.testing.assert_array_equal(labels, np.array([0]))
        np.testing.assert_array_equal(scores, np.array([0.95]))

    def test_extract_from_none_target(self):
        """Test extracting from None returns all None."""
        boxes, labels, scores = extract_boxes_and_labels(None)

        assert boxes is None
        assert labels is None
        assert scores is None

    def test_extract_from_array_target(self):
        """Test extracting from array target (classification) returns None."""
        target = np.array([0, 1, 0])
        boxes, labels, scores = extract_boxes_and_labels(target)

        assert boxes is None
        assert labels is None
        assert scores is None

    def test_extract_converts_list_to_array(self):
        """Test that lists are converted to numpy arrays."""
        target = {
            "boxes": [[10, 20, 100, 200], [30, 40, 150, 250]],
            "labels": [0, 1],
            "scores": [0.95, 0.87],
        }

        boxes, labels, scores = extract_boxes_and_labels(target)

        assert isinstance(boxes, np.ndarray)
        assert isinstance(labels, np.ndarray)
        assert isinstance(scores, np.ndarray)


class TestDrawBoundingBoxesWithCV2:
    """Tests for draw_bounding_boxes function using OpenCV (cv2)."""

    @pytest.fixture
    def sample_image(self):
        """Create a sample RGB image."""
        return np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)

    @pytest.fixture
    def sample_boxes(self):
        """Create sample bounding boxes in XYXY format."""
        return np.array([[10, 10, 50, 50], [60, 60, 90, 90]])

    @pytest.fixture
    def sample_labels(self):
        """Create sample labels."""
        return np.array([0, 1])

    @pytest.fixture
    def sample_scores(self):
        """Create sample scores."""
        return np.array([0.95, 0.87])

    @pytest.fixture
    def index2label(self):
        """Create sample index to label mapping."""
        return {0: "cat", 1: "dog"}

    def test_draw_boxes_with_cv2(self, sample_image, sample_boxes):
        """Test drawing boxes with OpenCV available."""
        try:
            import cv2  # noqa: F401

            result = draw_bounding_boxes(sample_image, sample_boxes)

            # Result should be same shape as input
            assert result.shape == sample_image.shape
            # Result should be uint8
            assert result.dtype == np.uint8
            # Image should be modified (not identical)
            assert not np.array_equal(result, sample_image)
        except ImportError:
            pytest.skip("OpenCV not available")

    def test_draw_boxes_with_labels(self, sample_image, sample_boxes, sample_labels, index2label):
        """Test drawing boxes with labels using OpenCV."""
        try:
            import cv2  # noqa: F401

            result = draw_bounding_boxes(
                sample_image,
                sample_boxes,
                labels=sample_labels,
                index2label=index2label,
            )

            assert result.shape == sample_image.shape
            assert result.dtype == np.uint8
        except ImportError:
            pytest.skip("OpenCV not available")

    def test_draw_boxes_with_scores(self, sample_image, sample_boxes, sample_labels, sample_scores, index2label):
        """Test drawing boxes with scores using OpenCV."""
        try:
            import cv2  # noqa: F401

            result = draw_bounding_boxes(
                sample_image,
                sample_boxes,
                labels=sample_labels,
                scores=sample_scores,
                index2label=index2label,
            )

            assert result.shape == sample_image.shape
            assert result.dtype == np.uint8
        except ImportError:
            pytest.skip("OpenCV not available")

    def test_draw_boxes_custom_color_thickness(self, sample_image, sample_boxes):
        """Test drawing boxes with custom color and thickness."""
        try:
            import cv2  # noqa: F401

            result = draw_bounding_boxes(
                sample_image,
                sample_boxes,
                color=(255, 0, 0),  # Red
                thickness=3,
            )

            assert result.shape == sample_image.shape
            assert result.dtype == np.uint8
        except ImportError:
            pytest.skip("OpenCV not available")

    def test_draw_empty_boxes(self, sample_image):
        """Test drawing with empty boxes array."""
        try:
            import cv2  # noqa: F401

            empty_boxes = np.array([]).reshape(0, 4)
            result = draw_bounding_boxes(sample_image, empty_boxes)

            # Should return copy of original image
            assert result.shape == sample_image.shape
            np.testing.assert_array_equal(result, sample_image)
        except ImportError:
            pytest.skip("OpenCV not available")


class TestDrawBoundingBoxesWithPIL:
    """Tests for draw_bounding_boxes function using PIL fallback."""

    @pytest.fixture
    def sample_image(self):
        """Create a sample RGB image."""
        return np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)

    @pytest.fixture
    def sample_boxes(self):
        """Create sample bounding boxes in XYXY format."""
        return np.array([[10, 10, 50, 50], [60, 60, 90, 90]])

    @pytest.fixture
    def sample_labels(self):
        """Create sample labels."""
        return np.array([0, 1])

    @pytest.fixture
    def sample_scores(self):
        """Create sample scores."""
        return np.array([0.95, 0.87])

    @pytest.fixture
    def index2label(self):
        """Create sample index to label mapping."""
        return {0: "cat", 1: "dog"}

    def test_draw_boxes_with_pil_fallback(self, sample_image, sample_boxes):
        """Test drawing boxes with PIL when cv2 is not available."""
        # Mock cv2 import to raise ImportError - need to use builtins.__import__
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "cv2":
                raise ImportError("No module named 'cv2'")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            result = draw_bounding_boxes(sample_image, sample_boxes)

            # Result should be same shape as input
            assert result.shape == sample_image.shape
            # Result should be uint8
            assert result.dtype == np.uint8
            # Image should be modified (not identical)
            assert not np.array_equal(result, sample_image)

    def test_draw_boxes_with_labels_pil(self, sample_image, sample_boxes, sample_labels, index2label):
        """Test drawing boxes with labels using PIL fallback."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "cv2":
                raise ImportError("No module named 'cv2'")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            result = draw_bounding_boxes(
                sample_image,
                sample_boxes,
                labels=sample_labels,
                index2label=index2label,
            )

            assert result.shape == sample_image.shape
            assert result.dtype == np.uint8

    def test_draw_boxes_with_scores_pil(self, sample_image, sample_boxes, sample_labels, sample_scores, index2label):
        """Test drawing boxes with scores using PIL fallback."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "cv2":
                raise ImportError("No module named 'cv2'")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            result = draw_bounding_boxes(
                sample_image,
                sample_boxes,
                labels=sample_labels,
                scores=sample_scores,
                index2label=index2label,
            )

            assert result.shape == sample_image.shape
            assert result.dtype == np.uint8

    def test_draw_boxes_custom_color_pil(self, sample_image, sample_boxes):
        """Test drawing boxes with custom color using PIL."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "cv2":
                raise ImportError("No module named 'cv2'")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            result = draw_bounding_boxes(
                sample_image,
                sample_boxes,
                color=(255, 0, 0),  # Red
                thickness=3,
            )

            assert result.shape == sample_image.shape
            assert result.dtype == np.uint8

    def test_draw_empty_boxes_pil(self, sample_image):
        """Test drawing with empty boxes array using PIL."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "cv2":
                raise ImportError("No module named 'cv2'")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            empty_boxes = np.array([]).reshape(0, 4)
            result = draw_bounding_boxes(sample_image, empty_boxes)

            # Should return copy of original image
            assert result.shape == sample_image.shape
            np.testing.assert_array_equal(result, sample_image)

    def test_draw_boxes_labels_only_indices(self, sample_image, sample_boxes, sample_labels):
        """Test drawing boxes with label indices but no index2label mapping."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "cv2":
                raise ImportError("No module named 'cv2'")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            result = draw_bounding_boxes(
                sample_image,
                sample_boxes,
                labels=sample_labels,
                index2label=None,
            )

            assert result.shape == sample_image.shape
            assert result.dtype == np.uint8
