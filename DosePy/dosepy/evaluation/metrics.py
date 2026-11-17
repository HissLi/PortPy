# DosePy -- Full dose prediction toolkit
#
# Derived from PortPy (https://github.com/PortPy-Project/PortPy), which is
# licensed under Apache 2.0 with the Commons Clause restriction.
#
# Copyright 2025, the PortPy Authors
#
# Licensed under the Apache License, Version 2.0 with the Commons Clause restriction.
# You may obtain a copy of the Apache 2.0 License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# ----------------------------------------------------------------------
# Commons Clause Restriction Notice:
# PortPy is licensed under Apache 2.0 with the Commons Clause.
# You may use, modify, and share the code for non-commercial
# academic and research purposes only.
# Commercial use -- including offering PortPy as a service,
# or incorporating it into a commercial product -- requires
# a separate commercial license.
# ----------------------------------------------------------------------

"""Voxel- and structure-level regression metrics for dose prediction.

These metrics answer "how close is the predicted dose to the planner dose?",
which is different from the clinical plan score in
:mod:`dosepy.evaluation.scorecard`.  The same definitions are used inside
``DosePrediction3DModel.calculate_validation_metrics`` so that the number shown
during training and the number computed here agree.

All functions take 3-D numpy arrays in Gy on the same grid.
"""

import numbers
from typing import Dict, Mapping

import numpy as np
import pandas as pd

__all__ = [
    "dose_metrics",
    "dose_metrics_by_structure",
    "summarize_metric_rows",
]


def _validate(pred_dose: np.ndarray, reference_dose: np.ndarray, mask=None):
    pred = np.asarray(pred_dose, dtype=np.float64)
    ref = np.asarray(reference_dose, dtype=np.float64)
    if pred.shape != ref.shape:
        raise ValueError(
            "prediction shape {} does not match reference shape {}".format(pred.shape, ref.shape)
        )
    if mask is None:
        values = np.ones(pred.shape, dtype=bool)
    else:
        values = np.asarray(mask, dtype=bool)
        if values.shape != pred.shape:
            raise ValueError(
                "mask shape {} does not match dose shape {}".format(values.shape, pred.shape)
            )
    return pred, ref, values


def dose_metrics(
    pred_dose: np.ndarray,
    reference_dose: np.ndarray,
    mask=None,
    prescription_gy: float = None,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Compute voxel-wise dose agreement metrics.

    Args:
        pred_dose: predicted 3-D dose in Gy.
        reference_dose: planner / ground-truth 3-D dose in Gy.
        mask: optional boolean region (e.g. BODY).  ``None`` uses all voxels.
        prescription_gy: if given, adds MAE and RMSE as a percentage of the
            prescription (the convention used in dose-prediction papers).
        eps: numerical floor.

    Returns:
        Dict with ``mae``, ``mse``, ``rmse``, ``r2``, ``pearson``, ``nrmse``,
        ``mean_pred``, ``mean_ref`` and ``max_abs_error``, plus
        ``mae_pct_rx`` / ``rmse_pct_rx`` when a prescription is given.
    """
    pred, ref, region = _validate(pred_dose, reference_dose, mask)
    pred = pred[region]
    ref = ref[region]

    if pred.size == 0:
        raise ValueError("No voxels in the evaluation mask.")

    diff = pred - ref
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))

    ref_var = float(np.mean((ref - ref.mean()) ** 2))
    r2 = float(1.0 - mse / (ref_var + eps)) if ref_var > eps else float("nan")

    pred_std = float(pred.std())
    ref_std = float(ref.std())
    if pred_std > eps and ref_std > eps:
        pearson = float(np.corrcoef(pred, ref)[0, 1])
    else:
        pearson = float("nan")

    ref_range = float(ref.max() - ref.min())
    nrmse = float(rmse / (ref_range + eps)) if ref_range > eps else float("nan")

    metrics = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "pearson": pearson,
        "nrmse": nrmse,
        "mean_pred": float(pred.mean()),
        "mean_ref": float(ref.mean()),
        "max_abs_error": float(np.max(np.abs(diff))),
    }

    if prescription_gy is not None:
        rx = float(prescription_gy)
        metrics["mae_pct_rx"] = mae / rx * 100.0
        metrics["rmse_pct_rx"] = rmse / rx * 100.0

    return metrics


def dose_metrics_by_structure(
    pred_dose: np.ndarray,
    reference_dose: np.ndarray,
    structures: Mapping[str, np.ndarray],
    prescription_gy: float = None,
) -> pd.DataFrame:
    """Compute :func:`dose_metrics` for every structure.

    Args:
        pred_dose: predicted 3-D dose in Gy.
        reference_dose: planner 3-D dose in Gy.
        structures: mapping structure name -> boolean mask.
        prescription_gy: optional prescription for the percentage columns.

    Returns:
        DataFrame with columns ``Structure, MAE, RMSE, R2, Pearson, nRMSE,
        MAE(%Rx), RMSE(%Rx), MeanPred, MeanRef, MaxAbsErr``.
    """
    rows = []
    for name, mask in structures.items():
        m = dose_metrics(
            pred_dose,
            reference_dose,
            mask=mask,
            prescription_gy=prescription_gy,
        )
        rows.append(
            {
                "Structure": name,
                "MAE": m["mae"],
                "RMSE": m["rmse"],
                "R2": m["r2"],
                "Pearson": m["pearson"],
                "nRMSE": m["nrmse"],
                "MAE(%Rx)": m.get("mae_pct_rx", np.nan),
                "RMSE(%Rx)": m.get("rmse_pct_rx", np.nan),
                "MeanPred": m["mean_pred"],
                "MeanRef": m["mean_ref"],
                "MaxAbsErr": m["max_abs_error"],
            }
        )

    return pd.DataFrame(rows)


def summarize_metric_rows(rows: list) -> Dict[str, float]:
    """Average a list of metric dicts (e.g. per-patient rows from ``test.py``).

    ``NaN`` values are ignored, so missing R^2/correlation does not poison the
    mean.
    """
    if not rows:
        return {}
    keys = set(rows[0]).intersection(*[set(r) for r in rows[1:]]) if len(rows) > 1 else set(rows[0])
    summary = {}
    for key in keys:
        values = [r[key] for r in rows if isinstance(r.get(key), numbers.Real)]
        if values:
            summary[key] = float(np.nanmean(values))
    return summary
