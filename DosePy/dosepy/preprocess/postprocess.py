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

"""Post-process a network prediction back to the patient's CT grid.

The network predicts a dose on the fixeds ``128^3`` calculation-box grid.  To
compare it with the planner dose, compute DVHs or score it with a scorecard, it
must be mapped back to the full CT geometry:

1. resize ``128^3`` -> shape of the cropped CT,
2. paste the cropped dose into a zero-filled CT-sized volume,
3. keep the CT origin / spacing / direction.

PortPy shipped this logic as the hard-coded script
``ai/preprocess/pred_dose_to_original_portpy.py``.  The reusable functions are
kept here; the hard-coded patient list and Windows paths are gone.
"""

import os
from typing import Optional

import numpy as np
import SimpleITK as sitk

from dosepy.preprocess.geometry import (
    attach_slices,
    get_crop_settings_calc_box,
    get_ct_image,
    resample_dose,
)

__all__ = [
    "resample_prediction_to_ct_grid",
    "save_prediction_to_nrrd",
    "prediction_to_dose_1d",
    "pred_dose_to_original_portpy",
]


def resample_prediction_to_ct_grid(
    pred_dose: np.ndarray,
    ct,
    start: Optional[list] = None,
    end: Optional[list] = None,
    meta_data: Optional[dict] = None,
) -> np.ndarray:
    """Map a ``128^3`` predicted dose back to the full CT grid.

    Args:
        pred_dose: predicted dose in the calculation-box grid (usually
            ``(128, 128, 128)``), in Gy.
        ct: ``portpy.photon.CT`` object.
        start: optional crop start in z-y-x; computed from ``meta_data`` when
            omitted.
        end: optional crop end in z-y-x; computed from ``meta_data`` when
            omitted.
        meta_data: patient metadata; required when ``start`` / ``end`` are
            omitted.

    Returns:
        Dose array with the same shape as the CT (z, y, x), in Gy.  Voxels
        outside the calculation box are zero.
    """
    if start is None or end is None:
        if meta_data is None:
            raise ValueError("Provide start/end or meta_data.")
        start, end = get_crop_settings_calc_box(ct, meta_data=meta_data)

    ct_arr = ct.ct_dict["ct_hu_3d"][0]
    ct_arr_cropped = ct_arr[start[0]:end[0] + 1, start[1]:end[1], start[2]:end[2]]

    pred_crop = resample_dose(pred_dose, ct_arr_cropped.shape)
    ct_img = get_ct_image(ct)
    pred_full = attach_slices(pred_crop, ct_img, start, end)
    return sitk.GetArrayFromImage(pred_full)


def save_prediction_to_nrrd(pred_dose_3d: np.ndarray, ct, path: str) -> str:
    """Write a CT-grid dose array to ``.nrrd`` with the CT geometry."""
    ct_img = get_ct_image(ct)
    dose_img = sitk.GetImageFromArray(np.asarray(pred_dose_3d, dtype=np.float32))
    dose_img.SetOrigin(ct_img.GetOrigin())
    dose_img.SetSpacing(ct_img.GetSpacing())
    dose_img.SetDirection(ct_img.GetDirection())

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sitk.WriteImage(dose_img, path)
    return path


def prediction_to_dose_1d(pred_dose_3d: np.ndarray, inf_matrix) -> np.ndarray:
    """Convert a CT-grid dose to PortPy's 1-D optimisation-voxel vector."""
    return inf_matrix.dose_3d_to_1d(dose_3d=pred_dose_3d)


def pred_dose_to_original_portpy(
    patient_id: str,
    in_dir: str,
    pred_nrrd: str,
    out_nrrd: Optional[str] = None,
    data=None,
):
    """End-to-end helper used by the legacy scratch script.

    Args:
        patient_id: PortPy patient id.
        in_dir: PortPy raw-data root.
        pred_nrrd: predicted dose in the calculation-box grid (``.nrrd``).
        out_nrrd: optional path for the CT-grid prediction; when given the
            result is written to disk.
        data: optional pre-built ``portpy.photon.DataExplorer``; otherwise one
            is created.

    Returns:
        ``(pred_dose_3d, ct)``.
    """
    import portpy.photon as pp

    if data is None:
        data = pp.DataExplorer(data_dir=in_dir)
    data.patient_id = patient_id
    meta_data = data.load_metadata()
    ct = pp.CT(data)

    pred_img = sitk.ReadImage(pred_nrrd)
    pred_dose = sitk.GetArrayFromImage(pred_img)
    pred_dose_3d = resample_prediction_to_ct_grid(pred_dose, ct, meta_data=meta_data)

    if out_nrrd is not None:
        save_prediction_to_nrrd(pred_dose_3d, ct, out_nrrd)

    return pred_dose_3d, ct
