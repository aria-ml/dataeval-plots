"""Shared helper functions for plotting backends."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "f_out",
    "project_steps",
    "calculate_projection",
    "normalize_reference_outputs",
]


def f_out(n_i: NDArray[Any], x: NDArray[Any]) -> NDArray[Any]:
    """
    Calculates the line of best fit based on its free parameters.

    Parameters
    ----------
    n_i : NDArray
        Array of sample sizes
    x : NDArray
        Array of inverse power curve coefficients

    Returns
    -------
    NDArray
        Data points for the line of best fit
    """
    return x[0] * n_i ** (-x[1]) + x[2]


def project_steps(params: NDArray[Any], projection: NDArray[Any]) -> NDArray[Any]:
    """
    Projects the measures for each value of X.

    Parameters
    ----------
    params : NDArray
        Inverse power curve coefficients used to calculate projection
    projection : NDArray
        Steps to extrapolate

    Returns
    -------
    NDArray
        Extrapolated measure values at each projection step
    """
    return 1 - f_out(projection, params)


def calculate_projection(steps: NDArray[Any]) -> NDArray[Any]:
    """
    Calculate the projection array for extrapolation.

    Parameters
    ----------
    steps : NDArray
        Array of step values from the output

    Returns
    -------
    NDArray
        Projection array for extrapolation
    """
    last_X = steps[-1]
    geomshape = (0.01 * last_X, last_X * 4, len(steps))
    return np.geomspace(*geomshape).astype(np.int64)


def normalize_reference_outputs(
    reference_outputs: Sequence[Any] | Any | None,
) -> list[Any]:
    """
    Normalize reference outputs to a list.

    Parameters
    ----------
    reference_outputs : Sequence, single object, or None
        Reference outputs to normalize

    Returns
    -------
    list
        List of reference outputs (empty if None provided)
    """
    if reference_outputs is None:
        return []
    if not isinstance(reference_outputs, (list, tuple)):
        return [reference_outputs]
    return list(reference_outputs)
