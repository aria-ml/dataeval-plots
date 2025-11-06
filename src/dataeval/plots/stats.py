"""Plotting functions for statistics outputs."""

from __future__ import annotations

from collections.abc import Iterable

from matplotlib.figure import Figure

from dataeval.outputs._stats import BaseStatsOutput
from dataeval.plots import plot
from dataeval.plots.utils import channel_histogram_plot, histogram_plot

__all__ = []


@plot.register
def plot_base_stats_output(
    output: BaseStatsOutput,
    log: bool = True,
    channel_limit: int | None = None,
    channel_index: int | Iterable[int] | None = None,
) -> Figure:
    """
    Plots the statistics as a set of histograms.

    Parameters
    ----------
    output : BaseStatsOutput
        The stats output object to plot
    log : bool, default True
        If True, plots the histograms on a logarithmic scale.
    channel_limit : int or None, default None
        The maximum number of channels to plot. If None, all channels are plotted.
    channel_index : int, Iterable[int] or None, default None
        The index or indices of the channels to plot. If None, all channels are plotted.

    Returns
    -------
    matplotlib.figure.Figure
    """
    max_channels, ch_mask = output._get_channels(channel_limit, channel_index)
    factors = output.factors(exclude_constant=True)
    if not factors:
        return Figure()
    if max_channels == 1:
        return histogram_plot(factors, log)
    return channel_histogram_plot(factors, log, max_channels, ch_mask)
