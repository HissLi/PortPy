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

"""Dose-volume histogram (DVH) computation and reporting.

This replaces the hard-coded ``portpy.ai.statistics.compute_dvh_stats`` script
with library functions.  It works on plain numpy arrays, so a predicted dose and
a planner dose can be compared before (or without) building a PortPy
``Plan`` object.

The DVH convention is the clinical cumulative DVH: for every dose level ``d``,
the y value is the percentage of the structure volume receiving **at least**
``d``.  Voxel volumes are assumed equal, which is true on PortPy's uniform dose
grid.  For a non-uniform grid, resample to a uniform grid first.
"""

from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "compute_dvh",
    "dvh_metrics",
    "dvh_table",
    "DEFAULT_DVH_METRICS",
]

#: Metrics shown by default in :func:`dvh_table`.
DEFAULT_DVH_METRICS = ("D98", "D95", "D50", "D2", "Dmean", "V20Gy")


def _as_values(dose: np.ndarray, mask=None) -> np.ndarray:
    dose = np.asarray(dose, dtype=np.float64)
    if mask is None:
        values = dose.reshape(-1)
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != dose.shape:
            raise ValueError(
                "mask shape {} does not match dose shape {}".format(mask.shape, dose.shape)
            )
        values = dose[mask]
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("No voxels to compute a DVH from (empty mask?).")
    return values


def compute_dvh(
    dose: np.ndarray,
    mask=None,
    n_bins: int = 351,
    dose_max: float = None,
    prescription_gy: float = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a cumulative DVH.

    Args:
        dose: 3-D dose in Gy.
        mask: boolean structure mask; ``None`` uses the whole volume.
        n_bins: number of dose bins.
        dose_max: upper dose axis limit.  Defaults to ``1.2 * prescription_gy``
            when a prescription is given, otherwise ``1.05 * max(dose)``.
        prescription_gy: optional prescription dose.

    Returns:
        ``(dose_bins, volume_percent)`` where ``volume_percent[i]`` is the
        percentage of the structure receiving at least ``dose_bins[i]`` Gy.
    """
    values = _as_values(dose, mask)

    if dose_max is None:
        if prescription_gy is not None:
            dose_max = 1.2 * float(prescription_gy)
        else:
            dose_max = max(float(values.max()) * 1.05, 1.0)

    bins = np.linspace(0.0, float(dose_max), int(n_bins))
    sorted_values = np.sort(np.clip(values, 0.0, None))
    n = sorted_values.size
    idx = np.searchsorted(sorted_values, bins, side="left")
    volume_percent = (n - idx) / n * 100.0
    return bins, volume_percent


def _dose_at_volume(values: np.ndarray, volume_percent: float) -> float:
    """Dose covering ``volume_percent`` % of the structure (e.g. D95)."""
    return float(np.percentile(values, 100.0 - float(volume_percent)))


def dvh_metrics(
    dose: np.ndarray,
    mask=None,
    prescription_gy: float = None,
    d_volume_percents: Sequence[float] = (98, 95, 50, 2),
    v_dose_gy: Iterable[float] = (20,),
) -> Dict[str, float]:
    """Compute common DVH metrics for one structure.

    Args:
        dose: 3-D dose in Gy.
        mask: boolean structure mask.
        prescription_gy: if given, adds ``V<Rx>%`` (volume receiving the full
            prescription).
        d_volume_percents: volume levels for ``D<level>`` metrics (D95, ...).
        v_dose_gy: dose levels for ``V<level>Gy`` metrics.

    Returns:
        Dict such as ``{'D98': ..., 'D95': ..., 'Dmean': ..., 'V20Gy': ...}``.
        ``V`` metrics are percentages, ``D`` metrics are Gy.
    """
    values = _as_values(dose, mask)

    metrics: Dict[str, float] = {}
    for level in d_volume_percents:
        metrics["D{}".format(_format_level(level))] = _dose_at_volume(values, level)

    metrics["Dmean"] = float(np.mean(values))
    metrics["Dmax"] = float(np.max(values))
    metrics["Dmin"] = float(np.min(values))

    for level in v_dose_gy:
        metrics["V{}Gy".format(_format_level(level))] = float(
            np.mean(values >= float(level)) * 100.0
        )

    if prescription_gy is not None:
        metrics["V{}%Rx".format(_format_level(prescription_gy))] = float(
            np.mean(values >= float(prescription_gy)) * 100.0
        )

    return metrics


def _format_level(value: float) -> str:
    """Format 95.0 -> '95', 20.5 -> '20.5' for metric names."""
    value = float(value)
    return str(int(value)) if value.is_integer() else str(value)


def dvh_table(
    pred_dose: np.ndarray,
    reference_dose: np.ndarray,
    structures: Mapping[str, np.ndarray],
    prescription_gy: float = None,
    metrics: Sequence[str] = DEFAULT_DVH_METRICS,
) -> pd.DataFrame:
    """Long-format DVH metric comparison between prediction and reference.

    Args:
        pred_dose: predicted dose on the CT grid, Gy.
        reference_dose: planner (ground-truth) dose on the same grid, Gy.
        structures: mapping structure name -> boolean mask.
        prescription_gy: optional prescription, adds ``V<Rx>%Rx`` if present in
            ``metrics``.
        metrics: metric names to report.

    Returns:
        DataFrame with columns ``Structure, Metric, Predicted, Reference,
        Difference, Difference_pct``.
    """
    rows = []
    v_levels = sorted(
        {
            float(metric[1:-2])
            for metric in metrics
            if metric.startswith("V") and metric.endswith("Gy")
        }
    )
    d_levels = sorted(
        {
            float(metric[1:])
            for metric in metrics
            if metric.startswith("D") and metric[1:].replace(".", "", 1).isdigit()
        }
    )

    for name, mask in structures.items():
        pred_metrics = dvh_metrics(
            pred_dose,
            mask=mask,
            prescription_gy=prescription_gy,
            d_volume_percents=d_levels or (98, 95, 50, 2),
            v_dose_gy=v_levels or (20,),
        )
        ref_metrics = dvh_metrics(
            reference_dose,
            mask=mask,
            prescription_gy=prescription_gy,
            d_volume_percents=d_levels or (98, 95, 50, 2),
            v_dose_gy=v_levels or (20,),
        )

        for metric in metrics:
            if metric not in pred_metrics:
                raise KeyError(
                    "Unknown metric {!r}. Available: {}".format(metric, sorted(pred_metrics))
                )
            pred_value = pred_metrics[metric]
            ref_value = ref_metrics[metric]
            diff = pred_value - ref_value
            rows.append(
                {
                    "Structure": name,
                    "Metric": metric,
                    "Predicted": pred_value,
                    "Reference": ref_value,
                    "Difference": diff,
                    "Difference_pct": (diff / ref_value * 100.0) if ref_value else np.nan,
                }
            )

    return pd.DataFrame(rows)
