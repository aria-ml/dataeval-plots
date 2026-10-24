"""Tests for shared helper functions in _shared module."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
from conftest import MockEndpoint, MockPlottableFeasibility, _feasibility, _source, _target

from dataeval_plots import plot
from dataeval_plots.backends import _feasibility_layout as layout
from dataeval_plots.backends._shared import (
    _evidence_rows,
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


def _restat(output: MockPlottableFeasibility, **stats: object) -> MockPlottableFeasibility:
    """Override every endpoint's summary statistics."""
    output.evidence = {
        eid: replace(ev, details={"endpoints": {k: v and {**v, **stats} for k, v in ev.details["endpoints"].items()}})
        for eid, ev in output.evidence.items()
    }
    return output


class TestEvidenceRows:
    """Tests for _evidence_rows, which reads the full output as the old summary table did."""

    def test_plots_without_the_summary_table(self, mock_feasibility_how_to):
        assert not hasattr(mock_feasibility_how_to, "aggregate_by_evidence")
        assert plot(mock_feasibility_how_to, backend="matplotlib") is not None

    def test_rows(self, mock_feasibility_how_to):
        rows = _evidence_rows(mock_feasibility_how_to)
        assert rows.height == 8  # two targets, four sources
        assert rows.row(0, named=True) == {
            "target_spec": mock_feasibility_how_to.determinations["target_spec"][0],
            "metric": "accuracy",
            "threshold": 0.95,
            "source": "your model",
            "evidence_id": "user_witness/v1",
            "information_scope": "full_input",
            "unit": "image",
            "bound_type": "lower",
            "observed": 0.969,
            "bound": 0.957,
            "n": 5000,
            "state": "informative",
            "quantity": "metric",
            "uncertified": False,
            "false_positive_bound": None,
            "false_positive_limit": None,
        }
        assert rows["source"].to_list()[:4] == [
            "your model",
            "built-in k-NN",
            "built-in linear probe",
            "group-based ceiling",
        ]

    def test_ceiling_mean_is_not_observed(self):
        output = _restat(
            _feasibility([_target(0.9, "unresolved")], [_source("repeated labels", 0.93, bound_type="upper")]),
            mean=0.97,
        )
        assert _evidence_rows(output)["observed"].to_list() == [None]

    def test_no_stats_gives_null_observed_and_n(self, mock_feasibility_how_to):
        output = mock_feasibility_how_to
        model = output.evidence["user_witness/v1"]
        output.evidence["user_witness/v1"] = replace(
            model, details={"endpoints": dict.fromkeys(model.details["endpoints"])}
        )
        rows = _evidence_rows(output).filter(pl.col("evidence_id") == "user_witness/v1")
        assert rows["observed"].to_list() == rows["n"].to_list() == [None, None]
        assert rows["bound"].to_list() == [0.957, 0.957]
        (band,) = prepare_feasibility_data(output)
        assert not any("scored" in line for line in band.caption)

    def test_unknown_kind_keeps_its_evidence_id(self):
        output = _feasibility([_target(0.9, "unresolved")], [_source("custom_witness/v1:mine", 0.85)])
        assert _evidence_rows(output)["source"].to_list() == ["custom_witness/v1:mine"]

    def test_group_unit_reads_sequence(self):
        output = _feasibility([_target(0.9, "unresolved")], [_source("your model", 0.85, unit="group")])
        assert _evidence_rows(output)["unit"].to_list() == ["sequence"]

    def test_endpoint_off_the_determinations_is_ignored(self, mock_feasibility_ruled_out):
        output = mock_feasibility_ruled_out
        before = _evidence_rows(output)
        model = output.evidence["user_witness/v1"]
        output.evidence["user_witness/v1"] = replace(
            model,
            endpoints=(*model.endpoints, MockEndpoint("not a target", 0.99, "lower", "informative")),
            details={"endpoints": {**model.details["endpoints"], "not a target": None}},
        )
        assert _evidence_rows(output).equals(before)

    def test_no_evidence(self, mock_feasibility_empty):
        assert _evidence_rows(mock_feasibility_empty).shape == (0, 16)


class TestPrepareFeasibilityData:
    """Tests for prepare_feasibility_data."""

    def test_how_to(self, mock_feasibility_how_to):
        (band,) = prepare_feasibility_data(mock_feasibility_how_to)
        assert band.title == "Required accuracy"
        assert band.subtitle == "Models may use the whole image"
        assert band.demonstrated == 0.957
        assert band.ruled_out is None
        assert band.conflict is False
        assert band.markers == [(0.95, "95%: demonstrated ✓"), (0.99, "99%: insufficient evidence ?")]
        assert band.boundary_labels == [(0.957, "95.7% lower confidence bound", "lower", True)]
        assert band.evidence_markers == [(0.95, "95% required"), (0.99, "99% required")]
        assert band.caption == [
            "Your model scored 96.9% on 5,000 evaluation images; given sampling uncertainty, its lower confidence "
            "bound is 95.7%.",
            "Bounds apply to new images from the same population, at 95% joint confidence.",
        ]
        assert band.notes == ["Group-based ceiling (72.1%) applies only to models that give one answer per group."]

    def test_evidence_caption_is_short_without_notes(self, mock_feasibility_how_to):
        """The evidence view's caption replaces the band's caption and notes; the restricted row's label has its
        scope."""
        (band,) = prepare_feasibility_data(mock_feasibility_how_to)
        assert band.evidence_caption == [
            "Observed on the evaluation images; bounds apply to new images from the same population, at 95% joint "
            "confidence."
        ]
        assert band.evidence[-1].label == "Group-based ceiling (one answer per group)"
        *_, ((*_, texts),) = layout.blocks([band], plot_evidence=True, width=10.0)
        drawn = [" ".join(text.split()) for text, _, _ in texts]
        assert drawn == ["● observed · ┃ lower confidence bound · ┃ upper confidence bound", *band.evidence_caption]

    def test_how_to_evidence_rows(self, mock_feasibility_how_to):
        (band,) = prepare_feasibility_data(mock_feasibility_how_to)
        assert [(r.label, r.bound_type, r.bound, r.observed, r.restricted) for r in band.evidence] == [
            ("Your model", "lower", 0.957, 0.969, False),
            ("Built-in k-NN", "lower", 0.923, 0.940, False),
            ("Built-in linear probe", "lower", 0.882, 0.900, False),
            ("Group-based ceiling (one answer per group)", "upper", 0.721, None, True),
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
        assert band.caption[-1] == "* Ruled out under the recorded assumptions, at the stated confidence."
        assert feasibility_zones(band)[-1][3] == "ruled out*"
        assert (0.93, "93.0% upper confidence bound", "upper", False) in band.boundary_labels

    def test_ruled_out_marker_has_its_footnote(self, mock_feasibility_ruled_out):
        """Every "ruled out*" marker has the "*" footnote, which names the recorded assumptions, in its caption."""
        (band,) = prepare_feasibility_data(mock_feasibility_ruled_out)
        assert band.markers == [(0.9, "90%: demonstrated ✓"), (0.95, "95%: ruled out* ✗")]
        assert any(line.startswith("* Ruled out under the recorded assumptions") for line in band.caption)
        # At the default 9-inch width a plain wrap would end a line on the footnote's "*"
        wrapped = layout.wrap(" ".join(band.caption), layout.CAPTION_PT, 9.0)
        assert "\n* Ruled out" in wrapped

    def test_group_based_ceiling_caption_has_no_sign_off(self, mock_feasibility_ruled_out):
        output = mock_feasibility_ruled_out
        output.evidence["replica_ceiling/v1"] = replace(
            output.evidence["replica_ceiling/v1"], kind="local_panel_ceiling/v1"
        )
        (band,) = prepare_feasibility_data(output)
        assert band.caption[-1] == "* Ruled out under the recorded assumptions, at the stated confidence."

    def test_recall_band(self, mock_feasibility_recall):
        (band,) = prepare_feasibility_data(mock_feasibility_recall)
        assert band.title == "Required recall for Tank (at most 5% false positives)"
        assert band.markers == [(0.6, "60%: demonstrated ✓")]
        assert band.boundary_labels == [
            (0.621, "62.1% lower confidence bound", "lower", True),
            (0.85, "85.0% upper confidence bound", "upper", False),
        ]
        assert band.caption == [
            "Your model scored 70.0% recall on 160 Tank images; given sampling uncertainty, its lower confidence"
            " bound is 62.1%.",
            "Its upper confidence bound on false positives is 4.1% (observed 2.6% of 651 other images).",
            "Bounds apply to new images from the same population, at 95% joint confidence.",
            "* Ruled out under the recorded assumptions, at the stated confidence.",
        ]

    def test_recall_evidence_rows_hold_only_the_recall_bounds(self, mock_feasibility_recall):
        (band,) = prepare_feasibility_data(mock_feasibility_recall)
        # The false-positive upper bound is no ceiling: one row per source, three in all
        assert [(r.source, r.bound_type, r.bound) for r in band.evidence] == [
            ("your model", "lower", 0.621),
            ("built-in k-NN", "lower", 0.66),
            ("repeated labels", "upper", 0.85),
        ]
        # Not counted, yet both its measured bounds stay visible
        knn = band.evidence[1]
        assert knn.observed == 0.72 and knn.faint
        assert knn.bound_label == "66.0% lower bound · not counted: false positives ≤8.3% (limit 5%)"
        assert not band.evidence[0].faint
        assert band.evidence[0].bound_label == "62.1% lower bound"
        assert band.ruled_out == 0.85

    def test_recall_false_positive_bound_is_read_from_its_tagged_stats(self, mock_feasibility_recall):
        rows = _evidence_rows(mock_feasibility_recall)
        fp = rows.filter(pl.col("quantity") == "false_positive_rate").sort("evidence_id")
        assert fp["bound"].to_list() == [0.083, 0.041]
        assert fp["observed"].to_list() == [0.07, 0.026]
        assert fp["n"].to_list() == [651, 651]

    def test_recall_witness_without_a_false_positive_bound_is_not_counted(self, mock_feasibility_recall):
        output = mock_feasibility_recall
        knn = output.evidence["builtin_witness/v1:knn"]
        output.evidence["builtin_witness/v1:knn"] = replace(knn, endpoints=knn.endpoints[:1], details={
            "endpoints": {k: v for k, v in knn.details["endpoints"].items() if not k.endswith("#false_positive_rate")}
        })  # fmt: skip
        (band,) = prepare_feasibility_data(output)
        assert band.evidence[1].bound == 0.66 and band.evidence[1].faint
        assert band.evidence[1].bound_label == "66.0% lower bound · not counted: no false-positive bound"

    def test_missing_false_positive_stats_leave_the_bound_without_an_observed_rate(self, mock_feasibility_recall):
        output = mock_feasibility_recall
        ev = output.evidence["user_witness/v1"]
        output.evidence["user_witness/v1"] = replace(
            ev, details={"endpoints": {k: v for k, v in ev.details["endpoints"].items() if "#" not in k}}
        )
        (band,) = prepare_feasibility_data(output)
        assert "Its upper confidence bound on false positives is 4.1%." in band.caption

    def test_false_positive_sentence_without_an_observed_score(self, mock_feasibility_recall):
        output = mock_feasibility_recall
        ev = output.evidence["user_witness/v1"]
        stats = {k: v | {"mean": None} if "#" not in k else v for k, v in ev.details["endpoints"].items()}
        output.evidence["user_witness/v1"] = replace(ev, details={"endpoints": stats})
        (band,) = prepare_feasibility_data(output)
        assert not any("scored" in line for line in band.caption)
        assert (
            "Its upper confidence bound on false positives is 4.1% (observed 2.6% of 651 other images)." in band.caption
        )

    def test_outputs_without_quantity_still_plot(self, mock_feasibility_ruled_out):
        for eid, ev in mock_feasibility_ruled_out.evidence.items():  # endpoints as an older DataEval gives them
            old = tuple(SimpleNamespace(**{k: v for k, v in vars(e).items() if k != "quantity"}) for e in ev.endpoints)
            mock_feasibility_ruled_out.evidence[eid] = replace(ev, endpoints=old)
        (band,) = prepare_feasibility_data(mock_feasibility_ruled_out)
        assert band.title == "Required accuracy"
        assert band.ruled_out == 0.93

    def test_confidence_rounds_down_and_error_share_up(self, mock_feasibility_ruled_out, mock_feasibility_conflict):
        """alpha = 0.05 / 3 reads as the printout does: never "98.3333%" or "1.66667%", never overstated."""
        mock_feasibility_ruled_out.alpha = mock_feasibility_conflict.alpha = 0.05 / 3
        (band,) = prepare_feasibility_data(mock_feasibility_ruled_out)
        assert "at 98.3% joint confidence." in band.caption[1]
        assert band.evidence_caption[0].endswith("at 98.3% joint confidence.")
        (conflict,) = prepare_feasibility_data(mock_feasibility_conflict)
        assert conflict.caption[0].endswith("at most 1.7%.")

    def test_conflict(self, mock_feasibility_conflict):
        (band,) = prepare_feasibility_data(mock_feasibility_conflict)
        assert band.conflict is True
        assert band.demonstrated is None and band.ruled_out is None
        assert band.caption == [
            "Bounds disagree: check the assumptions, such as evaluation images entering training. Even when they hold,"
            " the chance that any bound fails is at most 5%."
        ]
        assert band.markers == [(0.95, "95%")]
        # The evidence view keeps the conflict caption after its own
        assert band.evidence_caption[1:] == band.caption
        # The bounds cross, so each label still points away from the other: lower right, upper left
        assert band.boundary_labels == [
            (0.94, "94.0% lower confidence bound", "lower", False),
            (0.92, "92.0% upper confidence bound", "upper", True),
        ]

    def test_no_witness_no_ceiling(self, mock_feasibility_empty):
        """Review Focus 1: an all-unknown band still carries its markers."""
        (band,) = prepare_feasibility_data(mock_feasibility_empty)
        assert band.demonstrated is None and band.ruled_out is None and band.conflict is False
        assert band.markers == [(0.95, "95%: insufficient evidence ?")]
        assert band.boundary_labels == []

    def test_two_bands(self, mock_feasibility_two_bands):
        """Review Focus 2: one band per judged scope, each with its own markers."""
        bands = prepare_feasibility_data(mock_feasibility_two_bands)
        assert len(bands) == 2
        assert [b.subtitle for b in bands] == ["Models may use the whole image", "Models may use only the embeddings"]
        assert [b.demonstrated for b in bands] == [0.957, 0.923]
        assert [b.markers for b in bands] == [
            [(0.95, "95%: demonstrated ✓")],
            [(0.95, "95%: insufficient evidence ?")],
        ]
        # The whole-image model does not count for embedding-only models
        assert [r.restricted for r in bands[1].evidence] == [True, False]

    def test_observed_null(self, mock_feasibility_how_to):
        """Review Focus 3: no observed score, no "scored" sentence."""
        output = _restat(mock_feasibility_how_to, mean=None)
        (band,) = prepare_feasibility_data(output)
        assert not any("scored" in line for line in band.caption)
        assert band.caption[-1].startswith("Bounds apply to new images")

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
        assert not any("ruled out" in line.lower() for line in band.caption)
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
        assert band.caption[-1] == "* Ruled out under the recorded assumptions, at the stated confidence."

    def test_observed_without_n(self, mock_feasibility_how_to):
        output = _restat(mock_feasibility_how_to, n=None)
        (band,) = prepare_feasibility_data(output)
        assert band.caption[0] == (
            "Your model scored 96.9% on the evaluation images; given sampling uncertainty, its lower confidence "
            "bound is 95.7%."
        )

    def test_unit_falls_back_to_image(self, mock_feasibility_empty):
        (band,) = prepare_feasibility_data(mock_feasibility_empty)
        assert band.subtitle == "Models may use the whole image"
        assert band.caption[-1].startswith("Bounds apply to new images")

    def test_narrower_embedding_ceiling_note(self):
        output = _feasibility(
            [_target(0.95, "demonstrated_feasible", witness=0.957)],
            [
                _source("your model", 0.957, observed=0.969),
                _source("group-based ceiling", 0.88, scope="fixed_representation", bound_type="upper", n=400),
            ],
        )
        (band,) = prepare_feasibility_data(output)
        assert band.notes == ["Group-based ceiling (88.0%) applies only to models restricted to the embeddings."]
        assert band.evidence[-1].label == "Group-based ceiling (embeddings only)"

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
        assert band.caption[-1].endswith("at 97.5% joint confidence.")
        assert band.evidence_caption[0].endswith("at 97.5% joint confidence.")
        (conflict,) = prepare_feasibility_data(mock_feasibility_conflict)
        assert conflict.caption[0].endswith("the chance that any bound fails is at most 2.5%.")

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
        assert band.caption[0].endswith("its lower confidence bound is 74.6%.")
        assert [r.bound_label for r in band.evidence] == ["74.6% lower bound", "97.6% upper bound"]
        ap = _feasibility(
            [_target(0.4, "demonstrated_feasible", witness=0.40656, metric="ap")], [_source("k-NN", 0.40656)]
        )
        (ap_band,) = prepare_feasibility_data(ap)
        assert ap_band.boundary_labels[0][1] == "40.6% lower confidence bound"

    def test_threshold_keeps_its_decimals(self):
        (band,) = prepare_feasibility_data(_feasibility([_target(0.955, "unresolved")], []))
        assert band.markers == [(0.955, "95.5%: insufficient evidence ?")]
        assert band.evidence_markers == [(0.955, "95.5% required")]

    def test_unit_comes_from_the_deciding_bounds(self):
        """A band without a deciding bound says "image", as the printout does; the footer names the call's unit."""
        output = _feasibility(
            [
                _target(0.6, "demonstrated_feasible", witness=0.65, aggregation="sequence_mean"),
                _target(0.7, "unresolved", aggregation="frame_pooled"),
            ],
            [_source("built-in linear probe", 0.65, observed=0.81, scope="fixed_representation", unit="group")],
        )
        sequences, frames = prepare_feasibility_data(output)
        assert sequences.subtitle == "Models may use the whole sequence"
        assert frames.subtitle == "Models may use the whole image"
        assert frames.caption[-1].startswith("Bounds apply to new sequences")

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
        assert band.evidence_legend == "● observed · ┃ lower confidence bound"
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
        assert band.evidence_legend == "● observed · ┃ lower confidence bound · ┃ upper confidence bound"
        no_ceiling = prepare_feasibility_data(mock_feasibility_two_bands)[0]
        assert no_ceiling.evidence_legend == "● observed · ┃ lower confidence bound"
