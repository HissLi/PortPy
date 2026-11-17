# DosePy -- Full dose prediction toolkit
#
# OpenKBP-specific evaluation helpers.  The underlying voxel/DVH metric code is
# in dosepy/evaluation/metrics.py and dosepy/evaluation/dvh.py.

"""Evaluation helpers for OpenKBP predictions.

The model output and OpenKBP ground truth are 3-D dose arrays in Gy on the
128^3 OpenKBP grid.  This module maps the 7 OAR / 3 PTV channel stacks to named
masks and calls the generic DosePy metric functions.
"""

import numpy as np
import pandas as pd

from dosepy.data.openkbp_io import (
    OPENKBP_OAR_NAMES,
    OPENKBP_PRESCRIPTION_GY,
    OPENKBP_PTV_NAMES,
    convert_openkbp_patient,
)
from dosepy.evaluation.dvh import dvh_table
from dosepy.evaluation.metrics import dose_metrics, dose_metrics_by_structure

__all__ = [
    "openkbp_structure_masks",
    "evaluate_openkbp_dose",
    "evaluate_openkbp_patient",
]


def openkbp_structure_masks(oar: np.ndarray, ptv: np.ndarray, include_empty: bool = False):
    """Convert OpenKBP ``OAR`` / ``PTV`` channel stacks to named masks.

    Args:
        oar: ``(7, 128, 128, 128)`` uint8 array in ``OPENKBP_OAR_NAMES`` order.
        ptv: ``(3, 128, 128, 128)`` uint8 array in ``OPENKBP_PTV_NAMES`` order.
        include_empty: keep structures with no voxels (default: drop them,
            because empty masks cannot be scored).
    """
    masks = {}
    for index, name in enumerate(OPENKBP_OAR_NAMES):
        mask = np.asarray(oar[index]) > 0
        if include_empty or mask.any():
            masks[name] = mask
    for index, name in enumerate(OPENKBP_PTV_NAMES):
        mask = np.asarray(ptv[index]) > 0
        if include_empty or mask.any():
            masks[name] = mask
    return masks


def evaluate_openkbp_dose(
    pred_dose: np.ndarray,
    gt_dose: np.ndarray,
    body: np.ndarray = None,
    oar: np.ndarray = None,
    ptv: np.ndarray = None,
    prescription_gy: float = OPENKBP_PRESCRIPTION_GY,
):
    """Evaluate one OpenKBP prediction against its ground-truth dose.

    Args:
        pred_dose: predicted 3-D dose in Gy.
        gt_dose: OpenKBP ``dose.csv`` ground truth in Gy.
        body: optional ``possible_dose_mask`` (128,128,128).
        oar: optional ``(7,128,128,128)`` OAR stack.
        ptv: optional ``(3,128,128,128)`` PTV stack.
        prescription_gy: prescription dose (OpenKBP: 70 Gy).

    Returns:
        Dict with ``overall`` (dict), ``per_structure`` (DataFrame) and
        ``dvh`` (DataFrame).
    """
    pred = np.asarray(pred_dose, dtype=np.float32)
    gt = np.asarray(gt_dose, dtype=np.float32)
    if pred.shape != gt.shape:
        raise ValueError(
            "prediction shape {} does not match ground truth shape {}".format(
                pred.shape, gt.shape
            )
        )

    body_mask = None
    if body is not None:
        body_mask = np.asarray(body) > 0
        if not body_mask.any():
            body_mask = None

    overall = dose_metrics(
        pred, gt, mask=body_mask, prescription_gy=float(prescription_gy)
    )

    structures = {}
    if oar is not None and ptv is not None:
        structures = openkbp_structure_masks(oar, ptv)
    elif oar is not None or ptv is not None:
        raise ValueError("Pass both OAR and PTV stacks, or neither.")

    if structures:
        per_structure = dose_metrics_by_structure(
            pred, gt, structures, prescription_gy=float(prescription_gy)
        )
        dvh = dvh_table(
            pred, gt, structures, prescription_gy=float(prescription_gy)
        )
    else:
        per_structure = pd.DataFrame()
        dvh = pd.DataFrame()

    return {
        "overall": overall,
        "per_structure": per_structure,
        "dvh": dvh,
    }


def evaluate_openkbp_patient(
    pred_dose: np.ndarray,
    patient_dir: str,
    prescription_gy: float = OPENKBP_PRESCRIPTION_GY,
):
    """Load one OpenKBP patient's masks/GT and call :func:`evaluate_openkbp_dose`."""
    data = convert_openkbp_patient(patient_dir)
    return evaluate_openkbp_dose(
        pred_dose=pred_dose,
        gt_dose=data["DOSE"],
        body=data["BODY"],
        oar=data["OAR"],
        ptv=data["PTV"],
        prescription_gy=prescription_gy,
    )
