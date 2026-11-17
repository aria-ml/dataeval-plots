"""Animated GIFs of frame sequences."""

from __future__ import annotations

__all__ = []

import base64
import io
import os
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dataeval_plots.backends._shared import (
    as_image_dataset,
    box_rgb,
    check_panel_options,
    image_to_hwc,
    parse_dataset_item,
    process_dataset_item_for_display,
)
from dataeval_plots.protocols import Dataset


class GIF:
    """
    An animated GIF. In Jupyter, it shows itself.

    Attributes
    ----------
    data : bytes
        The GIF file's bytes
    """

    def __init__(self, data: bytes) -> None:
        self.data = data

    def save(self, path: str | os.PathLike[str]) -> None:
        """Write the GIF to a file."""
        with open(path, "wb") as file:
            file.write(self.data)

    def _repr_mimebundle_(self, include: Any = None, exclude: Any = None) -> dict[str, str]:
        return {
            "image/gif": base64.b64encode(self.data).decode("ascii"),
            "text/plain": f"<GIF, {len(self.data):,} bytes>",
        }


def animate(
    frames: Dataset | Sequence[ArrayLike] | NDArray[Any],
    /,
    *,
    indices: Sequence[int] | None = None,
    fps: float = 10,
    max_size: int = 480,
    boxes: bool | Sequence[ArrayLike] = True,
    region: tuple[int, int, int, int] | None = None,
    box_color: str = "lime",
) -> GIF:
    """
    Make a looping animated GIF from a sequence of frames.

    Parameters
    ----------
    frames : Dataset, Sequence[ArrayLike] or NDArray
        A dataset, such as one video's frames, a list of 2-D or 3-D image arrays, or one 4-D array of images
    indices : Sequence[int] or None, default None
        Frames to show, read in this order. None shows every frame.
    fps : float, default 10
        Playback speed in frames per second; it does not choose which frames are shown. GIF delays are whole
        hundredths of a second, so the GIF plays at ``100 / round(100 / fps)``. At most 50, because browsers slow
        shorter delays down.
    max_size : int, default 480
        Longest side of the GIF in pixels. Larger frames are scaled down before their boxes are drawn, so the boxes
        stay visible; smaller frames are not scaled up.
    boxes, region, box_color
        As for an image grid in ``plot``: each target's boxes, none, or each frame's own; one crop for every frame;
        the color of the boxes. Gray frames are shown in RGB, so boxes on them can be seen.

    Returns
    -------
    GIF
        The GIF, with its bytes in ``data``

    Raises
    ------
    ValueError
        If there are no frames, ``fps`` or ``max_size`` is out of range, or the frames differ in size after
        cropping and scaling
    """
    from PIL import Image

    if not 0 < fps <= 50:
        raise ValueError(f"fps must be greater than 0 and at most 50, got {fps}")
    if max_size < 1:
        raise ValueError(f"max_size must be at least 1 pixel, got {max_size}")
    dataset = as_image_dataset(frames) if isinstance(frames, list | tuple | np.ndarray) else cast(Dataset, frames)
    order = range(len(dataset)) if indices is None else indices
    if len(order) == 0:
        raise ValueError("There are no frames to animate; pass at least one frame, or at least one index.")
    panel_boxes = check_panel_options(len(order), None, boxes)
    color = box_rgb(box_color)
    index2label = dataset.metadata.get("index2label")

    images: list[Image.Image] = []
    for i, index in enumerate(order):
        image, target, metadata = parse_dataset_item(dataset[index])
        hwc = image_to_hwc(np.asarray(image))
        rgb = np.repeat(hwc, 3, axis=2) if hwc.shape[2] == 1 else hwc[:, :, :3]
        pixels, _, _ = process_dataset_item_for_display(
            (rgb, target, metadata),
            index2label=index2label,
            boxes=panel_boxes[i],
            region=region,
            box_color=color,
            max_size=max_size,
            panel=i,
        )
        frame = Image.fromarray(pixels)
        if images and frame.size != images[0].size:
            raise ValueError(
                f"frame {i} is {frame.size[0]} x {frame.size[1]} after cropping and scaling, but frame 0 is "
                f"{images[0].size[0]} x {images[0].size[1]}; a GIF has one size, so give frames of one size"
            )
        images.append(frame)

    buffer = io.BytesIO()
    images[0].save(
        buffer, format="GIF", save_all=True, append_images=images[1:], duration=round(100 / fps) * 10, loop=0
    )
    return GIF(buffer.getvalue())
