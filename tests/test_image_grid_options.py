"""Image grid options: plain arrays, titles, chosen boxes, a shared crop and the box color, in every backend."""

from __future__ import annotations

import base64
import hashlib
import io
from typing import Any

import numpy as np
import pytest
from conftest import MockDataset
from numpy.typing import NDArray

from dataeval_plots import plot

BACKENDS = ["matplotlib", "seaborn", "plotly", "altair"]

# Digests of v0.0.18's grid for `boxed_dataset` with labels and metadata shown, recorded on unmodified code. They pin
# that a call passing none of the new options draws exactly what it drew before.
EXPECTED_DIGESTS = {
    "matplotlib": "4025663d13fc71e2e8361277e207f56714bc447514476116103fcd20730977a5",
    "seaborn": "4025663d13fc71e2e8361277e207f56714bc447514476116103fcd20730977a5",
    "plotly": "c01be232642a1184ebcf12c36c3f1d223eba35835d22937477175302df519f2e",
    "altair": "57f3ded4c222fcdb9920a4fc90119c538b43a4a2ef46546e14b5a80dd6f6b273",
}


def _altair_frame(chart: Any) -> Any:
    import altair as alt

    # A layered chart whose layers share one data frame keeps it at the top level.
    return chart.layer[0].data if chart.data is alt.Undefined else chart.data


def _decode_png(url: str) -> NDArray[Any]:
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))))


def panels(fig: Any) -> list[NDArray[Any]]:
    """The pixels of each panel of an image grid from any backend."""
    if hasattr(fig, "axes") and not hasattr(fig, "layout"):
        return [np.asarray(ax.get_images()[0].get_array()) for ax in fig.axes if ax.get_visible()]
    if hasattr(fig, "layout"):
        return [np.asarray(trace.z) for trace in fig.data]
    return [_decode_png(url) for url in _altair_frame(fig)["image"]]


def panel_titles(fig: Any) -> list[str]:
    """The title of each panel of an image grid from any backend."""
    if hasattr(fig, "axes") and not hasattr(fig, "layout"):
        return [ax.get_title() for ax in fig.axes if ax.get_visible()]
    if hasattr(fig, "layout"):
        return [annotation.text for annotation in fig.layout.annotations]
    return list(_altair_frame(fig)["title"])


def _digest(fig: Any) -> str:
    h = hashlib.sha256()
    for panel in panels(fig):
        h.update(str(panel.shape).encode())
        h.update(np.ascontiguousarray(panel).tobytes())
    for title in panel_titles(fig):
        h.update(title.encode())
    return h.hexdigest()


@pytest.fixture
def boxed_dataset() -> MockDataset:
    """Three images, one channels-first; two carry boxes, labels and scores; every item has metadata."""
    rng = np.random.default_rng(0)
    images = [
        rng.integers(0, 256, (3, 48, 64), dtype=np.uint8),
        rng.integers(0, 256, (48, 64, 3), dtype=np.uint8),
        rng.integers(0, 256, (48, 64, 3), dtype=np.uint8),
    ]
    targets = [
        {
            "boxes": np.array([[4, 4, 20, 20], [30, 10, 60, 40]]),
            "labels": np.array([0, 1]),
            "scores": np.array([0.9, 0.7]),
        },
        {"boxes": np.array([[10, 5, 40, 30]]), "labels": np.array([1]), "scores": np.array([0.8])},
        None,
    ]
    metadatas = [{"scene": "road", "frame": i} for i in range(3)]
    return MockDataset(images=images, targets=targets, metadatas=metadatas, index2label={0: "car", 1: "van"})


@pytest.mark.parametrize("backend", BACKENDS)
def test_default_grid_is_unchanged(backend: str, boxed_dataset: MockDataset) -> None:
    fig = plot(boxed_dataset, backend=backend, show_labels=True, show_metadata=True)
    assert _digest(fig) == EXPECTED_DIGESTS[backend]


def _as_panel(image: NDArray[Any]) -> NDArray[Any]:
    from dataeval_plots.backends._shared import image_to_hwc, normalize_image_to_uint8

    return normalize_image_to_uint8(image_to_hwc(image))


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_list_of_arrays_plots(backend: str) -> None:
    rng = np.random.default_rng(1)
    images = [
        rng.integers(2, 256, (3, 20, 30), dtype=np.uint8),
        rng.integers(2, 256, (20, 30, 3), dtype=np.uint8),
        rng.integers(2, 256, (20, 30), dtype=np.uint8),
    ]
    shown = panels(plot(images, backend=backend))
    assert len(shown) == 3
    for panel, image in zip(shown, images, strict=True):
        # A gray panel may come back as (H, W) or (H, W, 1), depending on the backend
        assert np.array_equal(panel.reshape(-1), _as_panel(image).reshape(-1))


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_4d_array_plots(backend: str) -> None:
    assert len(panels(plot(np.zeros((2, 3, 20, 30), np.uint8), backend=backend))) == 2


@pytest.mark.parametrize(
    "images",
    [
        np.zeros((3, 20, 30)),
        np.zeros((20, 30)),
        (np.zeros((3, 20, 30)), {}, {}),
        [],
        [np.zeros(5)],
        np.zeros((0, 3, 20, 30)),
    ],
    ids=["one 3-D array", "one 2-D array", "a dataset item", "empty", "1-D item", "empty 4-D"],
)
def test_rejected_inputs_say_pass_a_list(images: Any) -> None:
    with pytest.raises(ValueError, match="list of images"):
        plot(images)


GREEN = (0, 255, 0)
BLACK = (0, 0, 0)


@pytest.fixture
def black_boxed() -> MockDataset:
    """Two black RGB images, each with a target box at (2, 2, 10, 10)."""
    target = {"boxes": np.array([[2, 2, 10, 10]]), "labels": np.array([0]), "scores": np.array([1.0])}
    return MockDataset(images=[np.zeros((40, 40, 3), np.uint8)] * 2, targets=[target, target])


@pytest.mark.parametrize("backend", BACKENDS)
def test_titles_come_first(backend: str, boxed_dataset: MockDataset) -> None:
    assert panel_titles(plot(boxed_dataset, backend=backend, titles=["a", "b", "c"])) == ["a", "b", "c"]
    shown = panel_titles(
        plot(boxed_dataset, backend=backend, titles=["a", "b", "c"], show_labels=True, show_metadata=True)
    )
    for title, given in zip(shown, ["a", "b", "c"], strict=True):
        assert title.startswith(given)
        assert "scene: road" in title


@pytest.mark.parametrize("backend", BACKENDS)
def test_boxes_false_leaves_pixels(backend: str, boxed_dataset: MockDataset) -> None:
    shown = panels(plot(boxed_dataset, backend=backend, boxes=False))
    for panel, image in zip(shown, boxed_dataset.images, strict=True):
        assert np.array_equal(panel, _as_panel(image))


@pytest.mark.parametrize("backend", BACKENDS)
def test_given_boxes_replace_the_target(backend: str, black_boxed: MockDataset) -> None:
    first, second = panels(plot(black_boxed, backend=backend, boxes=[[[20, 20, 30, 30]], []]))
    assert tuple(first[25, 20]) == GREEN  # left edge of the given box
    assert tuple(first[20, 25]) == GREEN  # top edge of the given box
    assert tuple(first[6, 2]) == BLACK  # where the target's box would be
    assert not second.any()


def test_given_boxes_on_gray_raise() -> None:
    gray = [np.zeros((40, 40), np.uint8)]
    with pytest.raises(ValueError, match="1-channel"):
        plot(gray, boxes=[[[1, 1, 5, 5]]])
    plot(gray, boxes=[[]])


def test_flat_box_raises() -> None:
    with pytest.raises(ValueError, match=r"panel 0.*\(N, 4\)"):
        plot([np.zeros((40, 40, 3), np.uint8)], boxes=[[1, 1, 5, 5]])


def test_length_mismatch_names_both_counts(boxed_dataset: MockDataset) -> None:
    with pytest.raises(ValueError, match=r"titles.*\b2\b.*\b3\b"):
        plot(boxed_dataset, titles=["a", "b"])
    with pytest.raises(ValueError, match=r"boxes.*\b2\b.*\b3\b"):
        plot(boxed_dataset, boxes=[[], []])


@pytest.mark.parametrize("backend", BACKENDS)
def test_region_crops_every_panel(backend: str) -> None:
    images = [np.zeros((40, 60, 3), np.uint8)] * 2
    first, second = panels(plot(images, backend=backend, region=(10, 5, 50, 35), boxes=[[[20, 10, 30, 20]], []]))
    assert first.shape == second.shape == (30, 40, 3)
    assert tuple(first[5, 10]) == GREEN  # full-image (x=20, y=10), shifted by the crop's (10, 5)
    assert not second.any()


def test_region_that_does_not_fit_raises() -> None:
    with pytest.raises(ValueError, match=r"panel 0.*\b60\b"):
        plot([np.zeros((40, 60, 3), np.uint8)], region=(0, 0, 70, 10))


@pytest.mark.parametrize("backend", BACKENDS)
def test_box_color(backend: str) -> None:
    (panel,) = panels(
        plot([np.zeros((40, 40, 3), np.uint8)], backend=backend, boxes=[[[20, 20, 30, 30]]], box_color="red")
    )
    assert tuple(panel[25, 20]) == (255, 0, 0)
    with pytest.raises(ValueError):
        plot([np.zeros((40, 40, 3), np.uint8)], backend=backend, box_color="nope")


def test_a_list_of_dataset_items_says_pass_a_list() -> None:
    item = (np.zeros((3, 20, 30), np.uint8), {"boxes": np.array([[1, 1, 5, 5]]), "labels": np.array([0])}, {})
    with pytest.raises(ValueError, match="list of images"):
        plot([item, item])


@pytest.mark.parametrize("region", [(0.0, 0, 10, 10), (0, 0, 10)], ids=["float corner", "three values"])
def test_a_malformed_region_names_region(region: Any) -> None:
    with pytest.raises(ValueError, match=r"region.*whole pixels"):
        plot([np.zeros((40, 60, 3), np.uint8)], region=region)
