"""Tests for shared helper functions in _shared module."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
from conftest import _feasibility, _source, _target

from dataeval_plots.backends._shared import (
    draw_bounding_boxes,
    extract_boxes_and_labels,
    feasibility_evidence_axis,
    feasibility_zones,
    prepare_feasibility_data,
)


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


class TestPrepareFeasibilityData:
    """Tests for prepare_feasibility_data."""

    def test_how_to(self, mock_feasibility_how_to):
        (band,) = prepare_feasibility_data(mock_feasibility_how_to)
        assert band.title == "Required accuracy"
        assert band.subtitle == "Models may use the whole image"
        assert band.demonstrated == 0.957
        assert band.ruled_out is None
        assert band.conflict is False
        assert band.markers == [
            (0.95, "95% requirement: demonstrated ✓"),
            (0.99, "99% requirement: not enough evidence ?"),
        ]
        assert band.boundary_labels == [(0.957, "95.7% lower confidence bound", "lower", True)]
        assert band.evidence_markers == [(0.95, "95% required"), (0.99, "99% required")]
        assert band.caption[0] == (
            "Your model scored 96.9% on the 5,000 evaluation images. Allowing for sampling uncertainty gives a "
            "lower confidence bound of 95.7%."
        )
        assert band.caption[-1] == (
            "Bounds concern new images drawn from the same population. Confidence is 95% jointly across this "
            "evaluation."
        )
        assert band.notes == [
            "The group-based ceiling (72.1%) does not limit models that distinguish individual images."
        ]

    def test_how_to_evidence_rows(self, mock_feasibility_how_to):
        (band,) = prepare_feasibility_data(mock_feasibility_how_to)
        assert [(r.label, r.bound_type, r.bound, r.observed, r.restricted) for r in band.evidence] == [
            ("Your model", "lower", 0.957, 0.969, False),
            ("Built-in k-NN", "lower", 0.923, 0.940, False),
            ("Built-in linear probe", "lower", 0.882, 0.900, False),
            ("Group-based ceiling (models that give one answer per group)", "upper", 0.721, None, True),
        ]
        # Bound labels outward from the observed score: left of a lower bound, right of an upper one
        assert [(r.bound_label, r.observed_label, r.bound_left) for r in (band.evidence[0], band.evidence[-1])] == [
            ("95.7% lower bound", "96.9% observed", True),
            ("72.1% upper bound", None, False),
        ]

    def test_ruled_out(self, mock_feasibility_ruled_out):
        (band,) = prepare_feasibility_data(mock_feasibility_ruled_out)
        assert band.demonstrated == 0.912
        assert band.ruled_out == 0.93
        assert "Under the recorded assumptions, requirements above 93.0% are ruled out at the stated confidence." in (
            band.caption
        )
        assert (0.95, "95% requirement: ruled out under the recorded assumptions ✗") in band.markers
        assert feasibility_zones(band)[-1][3] == "ruled out under the recorded assumptions"
        assert (0.93, "93.0% upper confidence bound", "upper", False) in band.boundary_labels

    def test_group_based_ceiling_caption_awaits_sign_off(self, mock_feasibility_ruled_out):
        output = mock_feasibility_ruled_out
        output.evidence_table = output.evidence_table.with_columns(
            pl.col("source").replace("repeated labels", "group-based ceiling")
        )
        (band,) = prepare_feasibility_data(output)
        assert (
            "Under the recorded assumptions, requirements above 93.0% are ruled out at the stated confidence"
            " (group-based ceiling; its validity argument awaits lead-scientist sign-off)."
        ) in band.caption

    def test_conflict(self, mock_feasibility_conflict):
        (band,) = prepare_feasibility_data(mock_feasibility_conflict)
        assert band.conflict is True
        assert band.demonstrated is None and band.ruled_out is None
        assert any("at most 5%" in line for line in band.caption)
        # The bounds cross, so each label still points away from the other: lower right, upper left
        assert band.boundary_labels == [
            (0.94, "94.0% lower confidence bound", "lower", False),
            (0.92, "92.0% upper confidence bound", "upper", True),
        ]

    def test_no_witness_no_ceiling(self, mock_feasibility_empty):
        """Review Focus 1: an all-unknown band still carries its markers."""
        (band,) = prepare_feasibility_data(mock_feasibility_empty)
        assert band.demonstrated is None and band.ruled_out is None and band.conflict is False
        assert band.markers == [(0.95, "95% requirement: not enough evidence ?")]
        assert band.boundary_labels == []

    def test_two_bands(self, mock_feasibility_two_bands):
        """Review Focus 2: one band per judged scope, each with its own markers."""
        bands = prepare_feasibility_data(mock_feasibility_two_bands)
        assert len(bands) == 2
        assert [b.subtitle for b in bands] == ["Models may use the whole image", "Models may use only the embeddings"]
        assert [b.demonstrated for b in bands] == [0.957, 0.923]
        assert [b.markers for b in bands] == [
            [(0.95, "95% requirement: demonstrated ✓")],
            [(0.95, "95% requirement: not enough evidence ?")],
        ]
        # The whole-image model does not count for embedding-only models
        assert [r.restricted for r in bands[1].evidence] == [True, False]

    def test_observed_null(self, mock_feasibility_how_to):
        """Review Focus 3: no observed score, no "scored" sentence."""
        output = mock_feasibility_how_to
        output.evidence_table = output.evidence_table.with_columns(pl.lit(None, dtype=pl.Float64).alias("observed"))
        (band,) = prepare_feasibility_data(output)
        assert not any("scored" in line for line in band.caption)
        assert band.caption[-1].startswith("Bounds concern new images")

    def test_low_threshold(self, mock_feasibility_low_threshold):
        """Review Focus 4: the axis extends to a threshold below the bounds."""
        (band,) = prepare_feasibility_data(mock_feasibility_low_threshold)
        assert band.x_range[0] <= 0.5
        assert band.x_range == (0.45, 1.0)

    def test_x_range_rounds_down(self, mock_feasibility_how_to):
        (band,) = prepare_feasibility_data(mock_feasibility_how_to)
        assert band.x_range == (0.9, 1.0)

    def test_ceiling_at_one_rules_nothing_out(self, mock_feasibility_ceiling_one):
        """A 100% upper bound rules nothing out: no zone, no boundary label, no ceiling caption."""
        (band,) = prepare_feasibility_data(mock_feasibility_ceiling_one)
        assert band.ruled_out is None
        assert band.boundary_labels == [(0.957, "95.7% lower confidence bound", "lower", True)]
        assert not any("ruled out" in line for line in band.caption)
        assert [zone[0] for zone in feasibility_zones(band)] == ["demonstrated", "unknown"]

    def test_witness_at_zero_demonstrates_nothing(self, mock_feasibility_witness_zero):
        """A 0% lower bound demonstrates nothing: no zone, no boundary label, no observed caption."""
        (band,) = prepare_feasibility_data(mock_feasibility_witness_zero)
        assert band.demonstrated is None
        assert band.boundary_labels == []
        assert not any("scored" in line for line in band.caption)
        assert band.x_range == (0.9, 1.0)
        assert [zone[0] for zone in feasibility_zones(band)] == ["unknown"]

    def test_zones_drop_zero_width(self, mock_feasibility_ruled_out):
        (band,) = prepare_feasibility_data(mock_feasibility_ruled_out)
        zones = feasibility_zones(replace(band, ruled_out=band.x_range[1]))
        assert [zone[0] for zone in zones] == ["demonstrated", "unknown"]
        assert all(end > start for _, start, end, _ in zones)

    def test_restricted_group_ceiling_at_the_same_bound_gets_no_caveat(self):
        output = _feasibility(
            [_target(0.95, "infeasible_under_recorded_assumptions", witness=0.912, ceiling=0.93)],
            [
                _source("your model", 0.912, observed=0.925),
                _source("group-based ceiling", 0.93, scope="partition_restricted", bound_type="upper", n=400),
                _source("repeated labels", 0.93, bound_type="upper", n=800),
            ],
        )
        (band,) = prepare_feasibility_data(output)
        assert band.caption[1] == (
            "Under the recorded assumptions, requirements above 93.0% are ruled out at the stated confidence."
        )

    def test_observed_without_n(self, mock_feasibility_how_to):
        output = mock_feasibility_how_to
        output.evidence_table = output.evidence_table.with_columns(pl.lit(None, dtype=pl.Int64).alias("n"))
        (band,) = prepare_feasibility_data(output)
        assert band.caption[0] == (
            "Your model scored 96.9% on the evaluation images. Allowing for sampling uncertainty gives a lower "
            "confidence bound of 95.7%."
        )

    def test_unit_falls_back_to_image(self, mock_feasibility_empty):
        (band,) = prepare_feasibility_data(mock_feasibility_empty)
        assert band.subtitle == "Models may use the whole image"
        assert band.caption[-1].startswith("Bounds concern new images")

    def test_narrower_embedding_ceiling_note(self):
        output = _feasibility(
            [_target(0.95, "demonstrated_feasible", witness=0.957)],
            [
                _source("your model", 0.957, observed=0.969),
                _source("group-based ceiling", 0.88, scope="fixed_representation", bound_type="upper", n=400),
            ],
        )
        (band,) = prepare_feasibility_data(output)
        assert band.notes == [
            "The group-based ceiling (88.0%) applies only to models restricted to the embeddings; it does not limit "
            "the models judged here."
        ]

    def test_per_class_ap_titles(self, mock_feasibility_per_class_ap):
        bands = prepare_feasibility_data(mock_feasibility_per_class_ap)
        assert [b.title for b in bands] == ["Required AP for class cat", "Required AP for class dog"]
        assert feasibility_evidence_axis(bands[0])[1] == "AP for class cat"

    def test_macro_ap_and_aggregation_titles(self):
        (macro,) = prepare_feasibility_data(_feasibility([_target(0.5, "unresolved", metric="ap")], []))
        (pooled,) = prepare_feasibility_data(
            _feasibility([_target(0.9, "unresolved", aggregation="sequence_mean")], [])
        )
        assert macro.title == "Required macro AP"
        # The printout's words for the aggregation
        assert pooled.title == "Required per-sequence mean accuracy"
        assert feasibility_evidence_axis(pooled)[1] == "Per-sequence mean accuracy"
        (frames,) = prepare_feasibility_data(_feasibility([_target(0.9, "unresolved", aggregation="frame_pooled")], []))
        assert frames.title == "Required frame-pooled accuracy"

    def test_confidence_and_alpha_keep_their_decimals(self, mock_feasibility_how_to, mock_feasibility_conflict):
        """At alpha 0.025 the captions read 97.5% and 2.5%, as DataEval prints them, never a rounded 98% or 2%."""
        mock_feasibility_how_to.alpha = mock_feasibility_conflict.alpha = 0.025
        (band,) = prepare_feasibility_data(mock_feasibility_how_to)
        assert band.caption[-1].endswith("Confidence is 97.5% jointly across this evaluation.")
        (conflict,) = prepare_feasibility_data(mock_feasibility_conflict)
        assert conflict.caption[0].endswith("the probability that any bound fails is at most 2.5%.")

    def test_bounds_round_away_from_overstating(self):
        """Lower bounds round down and upper bounds up, to 0.1%, everywhere a bound is written."""
        output = _feasibility(
            [_target(0.7, "demonstrated_feasible", witness=0.74672, ceiling=0.975073)],
            [
                _source("your model", 0.74672, observed=0.8),
                _source("repeated labels", 0.975073, bound_type="upper", n=800),
            ],
        )
        (band,) = prepare_feasibility_data(output)
        assert [b[1] for b in band.boundary_labels] == ["74.6% lower confidence bound", "97.6% upper confidence bound"]
        assert feasibility_zones(band)[0][3] == "Requirements through 74.6% are demonstrated"
        assert band.caption[0].endswith("gives a lower confidence bound of 74.6%.")
        assert "requirements above 97.6% are ruled out" in band.caption[1]
        assert [r.bound_label for r in band.evidence] == ["74.6% lower bound", "97.6% upper bound"]
        ap = _feasibility(
            [_target(0.4, "demonstrated_feasible", witness=0.40656, metric="ap")], [_source("k-NN", 0.40656)]
        )
        (ap_band,) = prepare_feasibility_data(ap)
        assert ap_band.boundary_labels[0][1] == "40.6% lower confidence bound"

    def test_threshold_keeps_its_decimals(self):
        (band,) = prepare_feasibility_data(_feasibility([_target(0.955, "unresolved")], []))
        assert band.markers == [(0.955, "95.5% requirement: not enough evidence ?")]
        assert band.evidence_markers == [(0.955, "95.5% required")]

    def test_unit_comes_from_the_deciding_bounds(self):
        """A band without a deciding bound says "image", as the printout does; the footer names the call's unit."""
        output = _feasibility(
            [
                _target(0.6, "demonstrated_feasible", witness=0.65, aggregation="sequence_mean"),
                _target(0.7, "unresolved", aggregation="frame_pooled"),
            ],
            [_source("built-in linear probe", 0.65, observed=0.81, scope="fixed_representation", unit="sequence")],
        )
        sequences, frames = prepare_feasibility_data(output)
        assert sequences.subtitle == "Models may use the whole sequence"
        assert frames.subtitle == "Models may use the whole image"
        assert frames.caption[-1].startswith("Bounds concern new sequences")

    def test_source_without_a_bound(self, mock_feasibility_how_to):
        """A source with no bound is a faint row saying so; a band with no bound at all carries a note."""
        nothing = _feasibility(
            [_target(0.9, "unresolved")],
            [_source("group-based ceiling", None, scope="partition_restricted", bound_type="upper", n=None)],
        )
        (band,) = prepare_feasibility_data(nothing)
        (row,) = band.evidence
        assert (row.bound, row.observed, row.bound_label, row.faint) == (None, None, "no bound available", True)
        assert band.evidence_note == "No bound is available yet."
        assert band.notes == []
        assert band.evidence_legend.endswith("from the same population")
        assert feasibility_evidence_axis(band)[0] == (0.85, 1.0)
        mixed = _feasibility(
            [_target(0.95, "demonstrated_feasible", witness=0.957)],
            [_source("your model", 0.957, observed=0.969), _source("built-in k-NN", None, observed=0.9, n=None)],
        )
        (band,) = prepare_feasibility_data(mixed)
        assert [(r.label, r.bound_label, r.observed_label, r.faint) for r in band.evidence] == [
            ("Your model", "95.7% lower bound", "96.9% observed", False),
            ("Built-in k-NN", "no bound available", None, True),
        ]
        assert band.evidence_note is None

    def test_evidence_legend(self, mock_feasibility_how_to, mock_feasibility_two_bands):
        (band,) = prepare_feasibility_data(mock_feasibility_how_to)
        assert band.evidence_legend == (
            "● observed on the evaluation images · ┃ lower confidence bound for new images from the same population"
            " · ┃ upper confidence bound"
        )
        no_ceiling = prepare_feasibility_data(mock_feasibility_two_bands)[0]
        assert no_ceiling.evidence_legend.endswith("from the same population")
