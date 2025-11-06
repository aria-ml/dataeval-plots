"""Matplotlib plotting utilities for DataEval outputs.

This package provides a registration-based plotting interface using functools.singledispatch
for all DataEval output types.
"""

from __future__ import annotations

from functools import singledispatch
from typing import Any

import matplotlib.figure

__all__ = ["plot"]


@singledispatch
def plot(output: Any, **kwargs: Any) -> matplotlib.figure.Figure:
    """
    Plot DataEval output objects.

    This function uses singledispatch to route to the appropriate plotting function
    based on the type of the output object.

    Parameters
    ----------
    output : DataEval Output object
        Any output object from dataeval.outputs
    **kwargs
        Plotting options passed to specific plot functions

    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib figure object

    Raises
    ------
    NotImplementedError
        If no plot implementation exists for the given output type

    Examples
    --------
    >>> from dataeval.plots import plot
    >>> output = detector.fit_detect(data)
    >>> fig = plot(output)
    >>> fig.savefig("output.png")
    """
    raise NotImplementedError(
        f"No plot implementation for {type(output).__name__}. "
        f"Make sure you have imported the appropriate plotting module."
    )


# Import modules to register their plot implementations
from dataeval.plots import bias, drift, stats, workflows

__all__ = ["plot"]
