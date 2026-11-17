# DosePy -- Full dose prediction toolkit
#
# Derived from PortPy (https://github.com/PortPy-Project/PortPy), which is
# licensed under Apache 2.0 with the Commons Clause restriction.

"""Bundle metrics, DVHs and figures into a per-patient evaluation report."""

import os
from typing import Mapping

import numpy as np

from dosepy.evaluation.dvh import dvh_table
from dosepy.evaluation.metrics import dose_metrics_by_structure
from dosepy.visualization.dose_plots import plot_slice_comparison
from dosepy.visualization.dvh_plots import plot_dvh_comparison

__all__ = ["save_evaluation_report"]


def save_evaluation_report(
    out_dir: str,
    patient_id: str,
    ct_hu: np.ndarray,
    doses: Mapping[str, np.ndarray],
    structures: Mapping[str, np.ndarray],
    prescription_gy: float = None,
    slice_idx: int = None,
    metrics_df=None,
    dvh_df=None,
    score: float = None,
    score_df=None,
) -> dict:
    """Write slices, DVHs, metric tables and a summary for one patient.

    The first two entries of ``doses`` are treated as prediction and reference
    when ``metrics_df`` / ``dvh_df`` are not supplied.

    Args:
        out_dir: output directory for the patient report.
        patient_id: patient id used in file names and the summary.
        ct_hu: CT in HU.
        doses: mapping label -> 3-D dose, e.g.
            ``{"Predicted": pred, "Reference": gt}``.
        structures: mapping structure name -> boolean mask.
        prescription_gy: prescription dose.
        slice_idx: axial slice for the figure; auto-selected when ``None``.
        metrics_df: optional pre-computed table from
            :func:`dosepy.evaluation.metrics.dose_metrics_by_structure`.
        dvh_df: optional pre-computed table from
            :func:`dosepy.evaluation.dvh.dvh_table`.
        score: optional total plan score.
        score_df: optional per-criterion score table.

    Returns:
        Dict with the written paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    labels = list(doses)

    if metrics_df is None or dvh_df is None:
        if len(labels) < 2:
            raise ValueError("Need at least two dose curves to build a metrics table.")
        pred, ref = np.asarray(doses[labels[0]]), np.asarray(doses[labels[1]])
        if metrics_df is None:
            metrics_df = dose_metrics_by_structure(
                pred, ref, structures, prescription_gy=prescription_gy
            )
        if dvh_df is None:
            dvh_df = dvh_table(pred, ref, structures, prescription_gy=prescription_gy)

    paths = {}

    metrics_path = os.path.join(out_dir, "{}_metrics.csv".format(patient_id))
    metrics_df.to_csv(metrics_path, index=False)
    paths["metrics_csv"] = metrics_path

    dvh_path = os.path.join(out_dir, "{}_dvh_metrics.csv".format(patient_id))
    dvh_df.to_csv(dvh_path, index=False)
    paths["dvh_csv"] = dvh_path

    slice_path = os.path.join(out_dir, "{}_slices.png".format(patient_id))
    plot_slice_comparison(
        ct_hu,
        doses,
        structures=structures,
        slice_idx=slice_idx,
        prescription_gy=prescription_gy,
        save_path=slice_path,
    )
    paths["slices_png"] = slice_path

    dvh_fig_path = os.path.join(out_dir, "{}_dvh.png".format(patient_id))
    plot_dvh_comparison(
        doses,
        structures,
        prescription_gy=prescription_gy,
        save_path=dvh_fig_path,
        title="{}: DVH comparison".format(patient_id),
    )
    paths["dvh_png"] = dvh_fig_path

    if score is not None and score_df is not None:
        score_path = os.path.join(out_dir, "{}_scorecard.csv".format(patient_id))
        score_df.to_csv(score_path, index=False)
        paths["scorecard_csv"] = score_path

    summary_path = os.path.join(out_dir, "{}_summary.txt".format(patient_id))
    with open(summary_path, "w") as f:
        f.write("Patient: {}\n".format(patient_id))
        f.write("Dose curves: {}\n".format(", ".join(labels)))
        if prescription_gy is not None:
            f.write("Prescription: {} Gy\n".format(prescription_gy))
        f.write("\nStructure metrics\n")
        f.write(metrics_df.to_string(index=False))
        f.write("\n\nDVH metrics\n")
        f.write(dvh_df.to_string(index=False))
        if score is not None:
            f.write("\n\nTotal plan score: {}\n".format(score))
    paths["summary_txt"] = summary_path

    return paths
