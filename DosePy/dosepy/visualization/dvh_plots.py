# DosePy -- Full dose prediction toolkit
#
# Derived from PortPy (https://github.com/PortPy-Project/PortPy), which is
# licensed under Apache 2.0 with the Commons Clause restriction.

"""DVH comparison plots for predicted vs reference dose."""

import os
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from dosepy.evaluation.dvh import compute_dvh

__all__ = ["plot_dvh_comparison", "plot_structure_metric_bars"]


def plot_dvh_comparison(
    doses: Mapping[str, np.ndarray],
    structures: Mapping[str, np.ndarray],
    prescription_gy: float = None,
    dose_scale: str = "Gy",
    n_bins: int = 351,
    ax=None,
    figsize: Sequence[float] = (10, 7),
    save_path: str = None,
    title: str = None,
):
    """Plot cumulative DVHs for several dose volumes on the same structures.

    Args:
        doses: mapping label -> 3-D dose in Gy, e.g.
            ``{"Predicted": pred, "Reference": gt, "Optimized": opt}``.
        structures: mapping structure name -> boolean 3-D mask.
        prescription_gy: required when ``dose_scale='%Rx'``; also draws the
            prescription line in Gy mode.
        dose_scale: ``'Gy'`` or ``'%Rx'``.
        n_bins: number of DVH points.
        ax: optional matplotlib axis.
        figsize: figure size when ``ax`` is ``None``.
        save_path: optional PNG/PDF path.
        title: optional title.

    Returns:
        ``(fig, ax)``.
    """
    if dose_scale not in ("Gy", "%Rx"):
        raise ValueError("dose_scale must be 'Gy' or '%Rx'")
    if dose_scale == "%Rx" and prescription_gy is None:
        raise ValueError("prescription_gy is required for dose_scale='%Rx'")
    if not doses:
        raise ValueError("doses must not be empty")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    struct_colors = plt.get_cmap("tab10")
    linestyles = ["-", "--", ":", "-."]

    for s_idx, (name, mask) in enumerate(structures.items()):
        color = struct_colors(s_idx % 10)
        for d_idx, (dose_label, dose) in enumerate(doses.items()):
            bins, vol = compute_dvh(
                dose,
                mask=mask,
                n_bins=n_bins,
                prescription_gy=prescription_gy,
            )
            x = bins / float(prescription_gy) * 100.0 if dose_scale == "%Rx" else bins
            ax.plot(
                x,
                vol,
                color=color,
                linestyle=linestyles[d_idx % len(linestyles)],
                linewidth=1.6,
                label="{} | {}".format(name, dose_label),
            )

    if prescription_gy is not None:
        if dose_scale == "Gy":
            ax.axvline(float(prescription_gy), color="0.4", linestyle=":", linewidth=1)
        else:
            ax.axvline(100.0, color="0.4", linestyle=":", linewidth=1)

    ax.set_xlabel("Dose (% Rx)" if dose_scale == "%Rx" else "Dose (Gy)", fontsize=12)
    ax.set_ylabel("Volume receiving at least dose (%)", fontsize=12)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 102)
    ax.grid(alpha=0.25)

    # Two legends: structures (colour) and dose curves (linestyle).
    struct_handles = [
        Line2D([0], [0], color=struct_colors(i % 10), lw=2, label=name)
        for i, name in enumerate(structures)
    ]
    struct_legend = ax.legend(
        handles=struct_handles, loc="upper right", fontsize=9, title="Structures"
    )
    ax.add_artist(struct_legend)

    dose_handles = [
        Line2D([0], [0], color="0.2", linestyle=linestyles[i % len(linestyles)], lw=1.6, label=label)
        for i, label in enumerate(doses)
    ]
    ax.legend(handles=dose_handles, loc="lower left", fontsize=9, title="Dose")

    if title:
        ax.set_title(title, fontsize=13)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig, ax


def plot_structure_metric_bars(
    metrics_df,
    metric: str = "MAE",
    structures: Sequence[str] = None,
    ax=None,
    figsize=(9, 4.5),
    color: str = "#4C72B0",
    save_path: str = None,
):
    """Bar chart of one metric from :func:`dose_metrics_by_structure`.

    Args:
        metrics_df: DataFrame returned by
            :func:`dosepy.evaluation.metrics.dose_metrics_by_structure`.
        metric: column to plot (``'MAE'``, ``'RMSE'``, ``'R2'``, ...).
        structures: optional subset/order of structures.
        ax: optional matplotlib axis.
        figsize: figure size.
        color: bar colour.
        save_path: optional output path.

    Returns:
        ``(fig, ax)``.
    """
    df = metrics_df
    if metric not in df.columns:
        raise KeyError("{} not in {}".format(metric, list(df.columns)))

    if structures is not None:
        df = df.set_index("Structure").loc[list(structures)].reset_index()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.bar(df["Structure"], df[metric], color=color)
    ax.set_ylabel(metric)
    ax.set_title("{} by structure".format(metric))
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig, ax
