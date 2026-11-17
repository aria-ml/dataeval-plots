"""Animated GIFs of frame sequences."""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conftest import MockDataset
from numpy.typing import NDArray
from PIL import Image, ImageSequence

from dataeval_plots import GIF, animate
from dataeval_plots.backends._shared import process_dataset_item_for_display

GREEN = (0, 255, 0)


def _solid(value: int, shape: tuple[int, ...] = (3, 60, 80)) -> NDArray[np.uint8]:
    return np.full(shape, value, np.uint8)


def _frames(gif: GIF) -> list[NDArray[Any]]:
    return [np.asarray(frame.convert("RGB")) for frame in ImageSequence.Iterator(Image.open(io.BytesIO(gif.data)))]


def test_gif_properties() -> None:
    gif = animate([_solid(40 * n) for n in range(5)], fps=15)
    assert gif.data[:6] == b"GIF89a"
    assert len(_frames(gif)) == 5
    image = Image.open(io.BytesIO(gif.data))
    assert image.size == (80, 60)  # smaller than max_size, so not scaled up
    assert image.info["duration"] == 70  # round(100 / 15) * 10 ms
    assert image.info["loop"] == 0


def test_scaled_down_to_max_size() -> None:
    assert Image.open(io.BytesIO(animate([_solid(0, (200, 400, 3))], max_size=100).data)).size == (100, 50)


def test_frames_follow_indices() -> None:
    dataset = MockDataset(images=[_solid(0), _solid(100), _solid(200)])
    shown = [frame.mean() for frame in _frames(animate(dataset, indices=[2, 0]))]
    assert shown == pytest.approx([200, 0], abs=2)


@pytest.mark.parametrize("options", [{"fps": 0}, {"fps": 51}, {"max_size": 0}])
def test_out_of_range_options_raise(options: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=next(iter(options))):
        animate([_solid(0)], **options)


@pytest.mark.parametrize("draw", ["opencv", "pillow"])
def test_box_survives_4k(draw: str, monkeypatch: pytest.MonkeyPatch) -> None:
    if draw == "pillow":
        monkeypatch.setitem(sys.modules, "cv2", None)  # import cv2 now raises, so boxes are drawn with Pillow
    frame = np.zeros((2160, 3840, 3), np.uint8)
    panel, _, _ = process_dataset_item_for_display(
        (frame, None, {}), boxes=np.array([[1000.0, 500.0, 3000.0, 1500.0]]), max_size=480
    )
    assert panel.shape == (270, 480, 3)
    left_edge = (panel[70:181, 124:127] == GREEN).all(axis=2)  # rint(1000 * 0.125) = 125
    assert (left_edge.sum(axis=1) >= 2).all()


def test_region_then_scale() -> None:
    frame = np.zeros((100, 400, 3), np.uint8)
    assert Image.open(io.BytesIO(animate([frame], region=(100, 0, 300, 100), max_size=100).data)).size == (100, 50)
    panel, _, _ = process_dataset_item_for_display(
        (frame, None, {}), boxes=np.array([[150.0, 10.0, 250.0, 90.0]]), region=(100, 0, 300, 100), max_size=100
    )
    assert tuple(panel[25, 25]) == GREEN  # left edge at rint((150 - 100) * 0.5) = 25


def test_animate_box_color() -> None:
    (frame,) = _frames(animate([_solid(0, (40, 40, 3))], boxes=[[[20, 20, 30, 30]]], box_color="red"))
    assert tuple(frame[25, 20]) == (255, 0, 0)


def test_mixed_sizes_raise() -> None:
    with pytest.raises(ValueError, match="frame 1"):
        animate([_solid(0), _solid(0, (3, 40, 80))])


def test_gray_and_rgba_become_rgb() -> None:
    gray = np.zeros((60, 80), np.uint8)
    rgba = np.zeros((60, 80, 4), np.uint8)
    first, second = _frames(animate([gray, rgba], boxes=[[[10, 10, 50, 50]], []]))
    assert tuple(first[30, 10]) == GREEN
    assert not second.any()


def test_float_frame_matches_uint8() -> None:
    frame = np.random.default_rng(0).random((60, 80, 3))
    assert animate([frame]).data == animate([(frame * 255).astype(np.uint8)]).data


def test_no_frames_raise() -> None:
    with pytest.raises(ValueError):
        animate([])
    with pytest.raises(ValueError, match="no frames"):
        animate(MockDataset(images=[_solid(0)]), indices=[])


def test_mimebundle_and_save(tmp_path: Path) -> None:
    gif = animate([_solid(0), _solid(255)])
    bundle = gif._repr_mimebundle_()
    assert set(bundle) == {"image/gif", "text/plain"}
    assert base64.b64decode(bundle["image/gif"]) == gif.data
    gif.save(tmp_path / "a.gif")
    assert (tmp_path / "a.gif").read_bytes() == gif.data
