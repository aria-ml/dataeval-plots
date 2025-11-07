"""Plotting functions for sufficiency workflow outputs."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

__all__ = ["plot_sufficiency"]


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


def _plot_measure(
    name: str,
    steps: NDArray[Any],
    averaged_measure: NDArray[Any],
    measures: NDArray[Any] | None,
    params: NDArray[Any],
    projection: NDArray[Any],
    show_error_bars: bool,
    show_asymptote: bool,
    ax: Axes,
) -> None:
    ax.set_title(f"{name} Sufficiency")
    ax.set_xlabel("Steps")
    projection_curve = ax.plot(
        projection,
        project_steps(params, projection),
        linestyle="solid",
        label=f"Potential Model Results ({name})",
        linewidth=2,
        zorder=2,
    )
    projection_color = projection_curve[0].get_color()
    # Calculate error bars
    # Plot measure over each step with associated error
    if show_error_bars:
        if measures is None:
            warnings.warn(
                "Error bars cannot be plotted without full, unaveraged data",
                UserWarning,
                stacklevel=2,
            )
        else:
            error = np.std(measures, axis=0)
            ax.errorbar(
                steps,
                averaged_measure,
                ecolor=projection_color,
                color=projection_color,
                yerr=error,
                capsize=7,
                capthick=1.5,
                elinewidth=1.5,
                fmt="o",
                label=f"Model Results ({name})",
                markersize=5,
                zorder=3,
            )
    else:
        ax.scatter(steps, averaged_measure, color=projection_color, label=f"Model Results ({name})", zorder=3)
    # Plot asymptote
    if show_asymptote:
        bound = 1 - params[2]
        ax.axhline(
            y=bound, linestyle="dashed", color=projection_color, label=f"Asymptote: {bound:.4g} ({name})", zorder=1
        )


def _plot_single_class(
    name: str,
    steps: NDArray[Any],
    averaged_measure: NDArray[Any],
    measures: NDArray[Any] | None,
    params: NDArray[Any],
    projection: NDArray[Any],
    show_error_bars: bool,
    show_asymptote: bool,
    plots: list[Figure],
    reference_outputs: Sequence[Any],
) -> None:
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots()
    ax.set_ylabel(f"{name}")
    _plot_measure(
        name,
        steps,
        averaged_measure,
        measures,
        params,
        projection,
        show_error_bars,
        show_asymptote,
        ax,
    )
    # Plot metric for each provided reference output
    for index, output in enumerate(reference_outputs):
        if name in output.averaged_measures:
            _plot_measure(
                f"{name} Output {index + 2}",
                output.steps,
                output.averaged_measures[name],
                output.measures.get(name),
                output.params[name],
                projection,
                show_error_bars,
                show_asymptote,
                ax,
            )
    ax.set_xscale("log")
    ax.legend(loc="best")
    plots.append(fig)


def _plot_multiclass(
    name: str,
    steps: NDArray[Any],
    averaged_measure: NDArray[Any],
    measures: NDArray[Any] | None,
    params: NDArray[Any],
    projection: NDArray[Any],
    show_error_bars: bool,
    show_asymptote: bool,
    plots: list[Figure],
    reference_outputs: Sequence[Any],
    class_names: Sequence[str] | None = None,
) -> None:
    from matplotlib import pyplot as plt

    if class_names is not None and len(averaged_measure) != len(class_names):
        raise IndexError("Class name count does not align with measures")
    for i, values in enumerate(averaged_measure):
        # Create a plot for each class
        fig, ax = plt.subplots()
        class_name = str(i) if class_names is None else class_names[i]
        ax.set_ylabel(f"{name}")
        _plot_measure(
            f"{name}_{class_name}",
            steps,
            values,
            None if measures is None else measures[:, :, i],
            params[i],
            projection,
            show_error_bars,
            show_asymptote,
            ax,
        )
        # Iterate through each reference output to plot similar class
        for index, output in enumerate(reference_outputs):
            if (
                name in output.averaged_measures
                and output.averaged_measures[name].ndim > 1
                and i <= len(output.averaged_measures[name])
            ):
                _plot_measure(
                    f"{name}_{class_name} Output {index + 2}",
                    output.steps,
                    output.averaged_measures[name][i],
                    output.measures[name][:, :, i] if len(output.measures) else None,
                    output.params[name][i],
                    projection,
                    show_error_bars,
                    show_asymptote,
                    ax,
                )
        ax.set_xscale("log")
        ax.legend(loc="best")
        plots.append(fig)


def plot_sufficiency(
    output: Any,  # SufficiencyOutput
    class_names: Sequence[str] | None = None,
    show_error_bars: bool = True,
    show_asymptote: bool = True,
    reference_outputs: Sequence[Any] | Any | None = None,
) -> list[Figure]:
    """
    Plotting function for data sufficiency tasks.

    Parameters
    ----------
    output : SufficiencyOutput
        The sufficiency output object to plot
    class_names : Sequence[str] | None, default None
        List of class names
    show_error_bars : bool, default True
        True if error bars should be plotted, False if not
    show_asymptote : bool, default True
        True if asymptote should be plotted, False if not
    reference_outputs : Sequence[SufficiencyOutput] | SufficiencyOutput, default None
        Singular or multiple SufficiencyOutput objects to include in plots

    Returns
    -------
    list[Figure]
        List of Figures for each measure

    Raises
    ------
    ValueError
        If the length of data points in the measures do not match

    Notes
    -----
    When plotting multiple SufficiencyOutput, multi-class metrics will be plotted according to index,
    ensure classes are aligned between SufficiencyOutput classes prior to plotting.
    """
    # Extrapolation parameters
    last_X = output.steps[-1]
    geomshape = (0.01 * last_X, last_X * 4, len(output.steps))
    extrapolated = np.geomspace(*geomshape).astype(np.int64)

    # Stores all plots
    plots = []

    # Wrap reference
    if reference_outputs is None:
        reference_outputs = []
    if not isinstance(reference_outputs, (list, tuple)):
        reference_outputs = [reference_outputs]

    # Iterate through measures
    for name, measures in output.averaged_measures.items():
        if measures.ndim > 1:
            _plot_multiclass(
                name,
                output.steps,
                measures,
                output.measures.get(name),
                output.params[name],
                extrapolated,
                show_error_bars,
                show_asymptote,
                plots,
                reference_outputs,
                class_names,
            )
        else:
            _plot_single_class(
                name,
                output.steps,
                measures,
                output.measures.get(name),
                output.params[name],
                extrapolated,
                show_error_bars,
                show_asymptote,
                plots,
                reference_outputs,
            )
    return plots
