"""A returned figure is drawn once in a notebook: closed under the inline backend, left open elsewhere."""

from __future__ import annotations

from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from dataeval_plots import plot, project

_INLINE = "module://matplotlib_inline.backend_inline"


@pytest.fixture
def inline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(matplotlib, "get_backend", lambda: _INLINE)


def _is_open(fig: Figure) -> bool:
    return fig.number in plt.get_fignums()  # type: ignore[attr-defined]


@pytest.mark.parametrize("backend", ["matplotlib", "seaborn"])
def test_a_returned_figure_is_closed_under_the_inline_backend(inline: None, mock_balance: Any, backend: str) -> None:
    fig = plot(mock_balance, backend=backend)
    assert isinstance(fig, Figure)
    assert not _is_open(fig)


def test_a_returned_figure_stays_open_under_other_backends(mock_balance: Any) -> None:
    fig = plot(mock_balance, backend="matplotlib")
    assert _is_open(fig)


def test_a_returned_list_stays_open_under_the_inline_backend(inline: None, mock_sufficiency_multi_class: Any) -> None:
    figs = plot(mock_sufficiency_multi_class, backend="matplotlib", class_names=["class_A", "class_B", "class_C"])
    assert isinstance(figs, list)
    assert all(_is_open(f) for f in figs)  # a notebook shows a list as text, so the open figures are its only drawing


def test_a_projection_is_closed_under_the_inline_backend(inline: None) -> None:
    fig = project(np.random.default_rng(0).normal(size=(30, 4)), backend="matplotlib")
    assert not _is_open(fig)


def test_a_projection_grid_is_closed_under_the_inline_backend(inline: None) -> None:
    fig = project(np.random.default_rng(0).normal(size=(30, 4)), method=["pca", "pca"], backend="matplotlib")
    assert not _is_open(fig)
