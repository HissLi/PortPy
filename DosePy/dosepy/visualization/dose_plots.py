# DosePy -- Full dose prediction toolkit
#
# Derived from PortPy (https://github.com/PortPy-Project/PortPy), which is
# licensed under Apache 2.0 with the Commons Clause restriction.

"""Spatial plots for full dose prediction.

These functions work on numpy arrays, so they can be used before a PortPy
``Plan`` exists.  For plan-level figures that need PortPy's dose object, see
:mod:`dosepy.visualization.portpy_plots`.
"""

import os
from collections.abc import Mapping
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from dosepy.preprocess.site_config import get_channel_names, get_oar_label_map

__all__ = ["plot_slice_comparison", "plot_preprocessed_sample", "select_slice"]


def _structure_colors(n: int):
    cmap = plt.get_cmap("tab10")
    return [cmap(i % 10) for i in range(n)]


def select_slice(ct_hu: np.ndarray, structures: Mapping[str, np.ndarray] = None, slice_idx=None) -> int:
    """Choose an informative axial slice index.

    Priority: explicit ``slice_idx`` -> slice with the largest PTV -> slice with
    the largest structure mask -> middle of the non-air CT.
    """
    if slice_idx is not None:
        return int(slice_idx)

    if structures:
        for preferred in ("PTV", "GTV"):
            for name, mask in structures.items():
                if name.upper() == preferred:
                    counts = np.asarray(mask).reshape(np.asarray(mask).shape[0], -1).sum(axis=1)
                    if counts.max() > 0:
                        return int(np.argmax(counts))

        counts = np.zeros(ct_hu.shape[0])
        for mask in structures.values():
            counts += np.asarray(mask).reshape(np.asarray(mask).shape[0], -1).sum(axis=1)
        if counts.max() > 0:
            return int(np.argmax(counts))

    body_slices = np.where(np.any(ct_hu > -500, axis=(1, 2)))[0]
    if body_slices.size:
        return int(body_slices[body_slices.size // 2])
    return ct_hu.shape[0] // 2


def _draw_contours(ax, structures, z, colors):
    handles = []
    for color, (name, mask) in zip(colors, structures.items()):
        mask = np.asarray(mask)
        if mask.ndim != 3 or not np.any(mask):
            continue
        if np.any(mask[z] > 0):
            ax.contour(mask[z] > 0, levels=[0.5], colors=[color], linewidths=1.3)
        handles.append(Line2D([0], [0], color=color, lw=1.5, label=name))
    return handles


def plot_slice_comparison(
    ct_hu: np.ndarray,
    doses: Mapping[str, np.ndarray],
    structures: Mapping[str, np.ndarray] = None,
    slice_idx: int = None,
    prescription_gy: float = None,
    difference: bool = True,
    cmap: str = "jet",
    figsize: Sequence[float] = None,
    save_path: str = None,
):
    """Plot CT + one or more dose volumes + an optional difference map.

    Args:
        ct_hu: CT in HU, shape ``(D, H, W)``.
        doses: mapping label -> 3-D dose in Gy, e.g.
            ``{"Predicted": pred, "Reference": gt}``.  The first entry is used
            as the reference for the difference panel.
        structures: optional mapping name -> boolean 3-D mask, drawn as
            contours on every anatomical panel.
        slice_idx: axial slice; chosen automatically when ``None``.
        prescription_gy: used to fix the dose colour scale at ``1.2*Rx``.
        difference: add a ``first - second`` difference panel when exactly two
            doses are supplied.
        cmap: dose colour map.
        figsize: figure size.
        save_path: optional path for a PNG/PDF copy.

    Returns:
        ``(fig, axes)``.
    """
    ct_hu = np.asarray(ct_hu)
    if not isinstance(doses, Mapping) or len(doses) == 0:
        raise ValueError("doses must be a non-empty dict {label: 3-D array}.")
    for label, dose in doses.items():
        if np.asarray(dose).shape != ct_hu.shape:
            raise ValueError(
                "dose {!r} shape {} != CT shape {}".format(label, np.shape(dose), ct_hu.shape)
            )

    z = select_slice(ct_hu, structures)
    labels = list(doses)
    show_diff = difference and len(labels) == 2
    n_panels = 1 + len(labels) + (1 if show_diff else 0)

    if figsize is None:
        figsize = (5.2 * n_panels, 5.2)
    fig, axes = plt.subplots(1, n_panels, figsize=figsize, squeeze=False)
    axes = axes[0]

    dose_vmax = 1.2 * prescription_gy if prescription_gy is not None else max(
        float(np.max(doses[label])) for label in labels
    )
    dose_vmax = max(dose_vmax, 1e-3)

    colors = _structure_colors(len(structures)) if structures else []

    # CT panel
    ax = axes[0]
    ax.imshow(ct_hu[z], cmap="gray", vmin=-1000, vmax=1000)
    ax.set_title("CT (slice {})".format(z))
    ax.axis("off")
    if structures:
        handles = _draw_contours(ax, structures, z, colors)
        ax.legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.7)

    # Dose panels
    for i, label in enumerate(labels, start=1):
        ax = axes[i]
        im = ax.imshow(np.asarray(doses[label])[z], cmap=cmap, vmin=0, vmax=dose_vmax)
        ax.set_title("{} (Gy)".format(label))
        ax.axis("off")
        if structures:
            _draw_contours(ax, structures, z, colors)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    # Difference panel
    if show_diff:
        diff = np.asarray(doses[labels[0]]) - np.asarray(doses[labels[1]])
        diff_vmax = float(np.max(np.abs(diff)))
        diff_vmax = diff_vmax if diff_vmax > 1e-6 else 1.0
        ax = axes[-1]
        im = ax.imshow(diff[z], cmap="RdBu_r", vmin=-diff_vmax, vmax=diff_vmax)
        ax.set_title("{} - {}".format(labels[0], labels[1]))
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig, axes


def plot_preprocessed_sample(npz_path: str, slice_idx: int = None, site: str = None, figsize=(20, 4.5)):
    """Visualise one preprocessed ``.npz`` file (sanity check of preprocessing).

    Panels: CT, OAR labels, PTV, BODY and DOSE (if present).  This is the
    quickest way to verify that the anatomy-only channel layout and
    normalisation are correct before starting a training run.

    Args:
        npz_path: path to a ``<PatientId>.npz`` file.
        slice_idx: axial slice; chosen from the PTV when ``None``.
        site: optional site name, used only to label the OAR colour bar.
        figsize: figure size.

    Returns:
        ``(fig, axes)``.
    """
    data = np.load(npz_path, allow_pickle=True)
    ct = data["CT"]
    oar = data["OAR"]
    ptv = data["PTV"]
    body = data["BODY"] if "BODY" in data else None
    dose = data["DOSE"] if "DOSE" in data else None

    z = select_slice(ct, {"PTV": ptv}, slice_idx)

    panels = [("CT (normalised HU)", ct, "gray", None, None)]
    panels.append(("OAR labels", oar, "tab20", 0, 5))
    panels.append(("PTV", ptv, "Reds", 0, 1))
    if body is not None:
        panels.append(("BODY", body, "Greys", 0, 1))
    if dose is not None:
        panels.append(("DOSE (Gy)", dose, "jet", 0, None))

    fig, axes = plt.subplots(1, len(panels), figsize=(figsize[0] / 5 * len(panels), figsize[1]))
    if len(panels) == 1:
        axes = [axes]

    for ax, (title, arr, cmap, vmin, vmax) in zip(axes, panels):
        im = ax.imshow(np.asarray(arr)[z], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title("{}\nslice {}".format(title, z), fontsize=10)
        ax.axis("off")
        if title == "OAR labels" and site is not None:
            label_map = get_oar_label_map(site)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, ticks=list(label_map))
            cbar.ax.set_yticklabels([label_map[i] for i in label_map], fontsize=6)
        elif title != "CT (normalised HU)":
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    patient = os.path.basename(npz_path).replace(".npz", "")
    fig.suptitle(
        "{} | channels: {}".format(patient, ", ".join(get_channel_names(site))) if site else patient,
        fontsize=11,
    )
    fig.tight_layout()
    return fig, axes
