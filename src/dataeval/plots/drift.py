"""Plotting functions for drift detection outputs."""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from dataeval.outputs._drift import DriftMVDCOutput
from dataeval.plots import plot

__all__ = []


@plot.register
def plot_drift_mvdc_output(output: DriftMVDCOutput) -> Figure:
    """
    Render the roc_auc metric over the train/test data in relation to the threshold.

    Parameters
    ----------
    output : DriftMVDCOutput
        The drift MVDC output object to plot

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(dpi=300)
    resdf = output.to_dataframe()
    xticks = np.arange(resdf.shape[0])
    trndf = resdf[resdf["chunk"]["period"] == "reference"]
    tstdf = resdf[resdf["chunk"]["period"] == "analysis"]
    # Get local indices for drift markers
    driftx = np.where(resdf["domain_classifier_auroc"]["alert"].values)  # type: ignore | dataframe
    if np.size(driftx) > 2:
        ax.plot(resdf.index, resdf["domain_classifier_auroc"]["upper_threshold"], "r--", label="thr_up")
        ax.plot(resdf.index, resdf["domain_classifier_auroc"]["lower_threshold"], "r--", label="thr_low")
        ax.plot(trndf.index, trndf["domain_classifier_auroc"]["value"], "b", label="train")
        ax.plot(tstdf.index, tstdf["domain_classifier_auroc"]["value"], "g", label="test")
        ax.plot(
            resdf.index.values[driftx],  # type: ignore | dataframe
            resdf["domain_classifier_auroc"]["value"].values[driftx],  # type: ignore | dataframe
            "dm",
            markersize=3,
            label="drift",
        )
        ax.set_xticks(xticks)
        ax.tick_params(axis="x", labelsize=6)
        ax.tick_params(axis="y", labelsize=6)
        ax.legend(loc="lower left", fontsize=6)
        ax.set_title("Domain Classifier, Drift Detection", fontsize=8)
        ax.set_ylabel("ROC AUC", fontsize=7)
        ax.set_xlabel("Chunk Index", fontsize=7)
        ax.set_ylim((0.0, 1.1))
    return fig
