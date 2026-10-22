"""
Feasibility plot layout shared by every backend.

Sizes are in inches and fonts in points, measured with matplotlib's default font. The plotly and altair backends draw
them at ``PX`` pixels per inch, so labels stack, flip and wrap as they do in matplotlib.
"""

from __future__ import annotations

import textwrap

from dataeval_plots.backends._shared import FEASIBILITY_COLORS, FeasibilityBand, feasibility_evidence_axis

MARGIN = 0.45  # figure side margin
TITLE = 0.4  # subtitle above each axes
XAXIS = 0.62  # ticks and axis title below each axes
BOUNDARY = 0.32  # boundary labels under the bar
BAR = 0.45
LEVEL = 0.24  # one row of requirement labels
ROW = 0.75  # one evidence source: observed label, marks, bound label
LINE = 0.21  # one caption line
PARAGRAPH = 0.06  # after each paragraph below the axes
GAP = 0.25  # between bands
PAD = 0.05  # from a label to its mark
LABEL_PT = 10
CAPTION_PT = 10.5
SOURCE_WRAP = 28  # characters per line of an evidence source's name
PX = 100  # pixels per inch, as the plotly and altair backends read figsize

# A requirement label placed above its line: (threshold, text, "left" or "right" alignment, stacking level)
Placed = tuple[float, str, str, int]
# One band's block: the band, its x range and title, the axes height, its placed requirement labels and the wrapped
# texts below the axes as (text, points, color)
Block = tuple[FeasibilityBand, tuple[float, float], str, float, list[Placed], list[tuple[str, float, str]]]


def px(points: float) -> float:
    """A size in points, in pixels."""
    return points * PX / 72


def text_width(text: str, size: float = LABEL_PT) -> float:
    """Width in inches of the widest line of ``text`` at ``size`` points."""
    from matplotlib.textpath import TextPath

    return max(TextPath((0, 0), line, size=size).get_extents().width for line in text.split("\n")) / 72


def wrap(text: str, size: float, width: float) -> str:
    """Wrap to the figure's text width, never leaving a legend glyph at a line end."""
    glued = text.replace("● ", "●\xa0").replace("┃ ", "┃\xa0")  # textwrap breaks only at ASCII whitespace
    return textwrap.fill(glued, int((width - 2 * MARGIN) * 72 / (size * 0.55))).replace("\xa0", " ")


def place_labels(labels: list[tuple[float, str]], x_range: tuple[float, float], ax_w: float) -> list[Placed]:
    """Right-align each label at its mark (left-align if that leaves the axes) and stack overlapping ones."""
    lo, hi = x_range
    ends: list[float] = []  # right end of the last label on each level
    placed = []
    for x, text in labels:
        x_in, w = (x - lo) / (hi - lo) * ax_w, text_width(text)
        ha, left = ("right", x_in + PAD - w) if x_in + PAD >= w else ("left", x_in - PAD)
        level = next((i for i, end in enumerate(ends) if left > end + 0.15), len(ends))
        ends[level : level + 1] = [left + w]
        placed.append((x, text, ha, level))
    return placed


def goes_left(x: float, text: str, prefer_left: bool, x_range: tuple[float, float], ax_w: float) -> bool:
    """Whether a label beside its mark goes left: the preferred side, unless that leaves the axes."""
    lo, hi = x_range
    x_in, w = (x - lo) / (hi - lo) * ax_w, text_width(text)
    if prefer_left and x_in - PAD < w:
        return False
    if not prefer_left and x_in + PAD + w > ax_w:
        return True
    return prefer_left


def zone_label(band: FeasibilityBand, start: float, end: float, label: str, per_in: float) -> tuple[float, str] | None:
    """
    A zone label's center and wrapping, in the widest stretch clear of requirement lines.

    One line, else two, else none: the zone is left to its boundary label and the caption.
    """
    marks = sorted({start, end, *(x for x, _ in band.markers if start < x < end)})
    a, b = max(zip(marks, marks[1:]), key=lambda ab: ab[1] - ab[0])
    for text in (label, textwrap.fill(label, len(label) // 2 + 1, break_long_words=False)):
        if text_width(text) + 0.15 < (b - a) / per_in:
            return (a + b) / 2, text
    return None


def blocks(bands: list[FeasibilityBand], plot_evidence: bool, width: float) -> tuple[float, float, float, list[Block]]:
    """
    Size every band's block, top-down.

    Returns the gutter left of the axes (shared, so the axes align), the axes width, the total height and the blocks.
    """
    ink, muted = FEASIBILITY_COLORS["ink"], FEASIBILITY_COLORS["ink_muted"]
    sources = [textwrap.fill(r.label, SOURCE_WRAP) for b in bands for r in b.evidence] if plot_evidence else []
    gutter = max([MARGIN, *(text_width(t) + 0.25 for t in sources)])
    ax_w = width - gutter - MARGIN
    out: list[Block] = []
    height = 0.0
    for band in bands:
        if plot_evidence:
            (x_range, xlabel), labels = feasibility_evidence_axis(band), band.evidence_markers
            body = evidence_rows(band)[0] + 0.12
        else:
            x_range, xlabel, labels = band.x_range, band.title, band.markers
            body = BOUNDARY + BAR + 0.2
        placed = place_labels(labels, x_range, ax_w)
        ax_h = body + (max((p[3] for p in placed), default=-1) + 1) * LEVEL + 0.1
        # Below the axes: the evidence legend, the caption paragraph, then the notes, each wrapped
        texts = [(band.evidence_legend, 10.0, muted)] if plot_evidence else []
        texts += [(" ".join(band.caption), CAPTION_PT, ink), *((note, 10.0, muted) for note in band.notes)]
        wrapped = [(wrap(text, size, width), size, color) for text, size, color in texts]
        out.append((band, x_range, xlabel, ax_h, placed, wrapped))
        height += TITLE + ax_h + XAXIS + sum(text_height(text) for text, _, _ in wrapped) + GAP
    return gutter, ax_w, height, out


def text_height(text: str) -> float:
    """Height in inches of a wrapped paragraph below the axes, with the space after it."""
    return (text.count("\n") + 1) * LINE + PARAGRAPH


def evidence_rows(band: FeasibilityBand) -> tuple[float, list[float]]:
    """
    The evidence view's rows, in inches: their total height and each source's center.

    The band's evidence note, when it has one, takes the first row.
    """
    skip = band.evidence_note is not None
    top = (len(band.evidence) + skip) * ROW
    return top, [top - (i + skip + 0.5) * ROW for i in range(len(band.evidence))]
