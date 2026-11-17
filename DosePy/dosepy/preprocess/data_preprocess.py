# DosePy -- Full dose prediction toolkit
#
# Derived from PortPy (https://github.com/PortPy-Project/PortPy), which is
# licensed under Apache 2.0 with the Commons Clause restriction. The original
# copyright notice is retained below.
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

"""Build anatomy-only training / inference tensors for full dose prediction.

For every PortPy patient this module writes one ``<PatientId>.npz`` file with:

======================  ==========================================  ==========
key                     shape / dtype                               meaning
======================  ==========================================  ==========
``CT``                  ``(128,128,128)`` float32                   HU clipped + scaled
``DOSE``                ``(128,128,128)`` float32 (train only)      planner dose, Gy
``OAR``                 ``(128,128,128)`` uint8                     OAR labels 0..4
``PTV``                 ``(128,128,128)`` uint8                     0 / 1
``BODY``                ``(128,128,128)`` uint8                     0 / 1
``PRESCRIPTION_GY``     scalar float32                              prescription, Gy
======================  ==========================================  ==========

There is **no BEAM channel**: the model sees only CT, OAR masks and PTV, so
neither a beam arrangement nor an influence matrix is required to build the
model input.  :class:`~dosepy.preprocess.metadata.AnatomyOnlyDataExplorer`
also avoids loading the patient's ``Beams/`` metadata folder.  The planner dose
is still needed during training and validation, and is read from the DICOM RT
dose.

Typical use::

    data_preprocess(
        in_dir="/data/PortPy",
        out_dir="workspace/preprocessed_lung",
        site="lung",
        protocol_name="Lung_2Gy_30Fx",
        technique_name="imrt",
    )
"""

import argparse
import os
import traceback

import numpy as np
import portpy.photon as pp

from dosepy.preprocess.geometry import (
    MODEL_SHAPE,
    crop_resize_img,
    get_crop_settings_calc_box,
    get_structure_masks,
)
from dosepy.preprocess.metadata import AnatomyOnlyDataExplorer
from dosepy.preprocess.site_config import get_site_config

__all__ = ["data_preprocess", "process_case"]


def process_case(
    ct_portpy,
    meta_data,
    ct,
    oar,
    ptv,
    out_dir,
    case,
    prescription_gy,
    dose=None,
    body=None,
    target_shape=MODEL_SHAPE,
):
    """Crop / resample one case and save it as ``<case>.npz``.

    Args:
        ct_portpy: ``portpy.photon.CT`` object (only used for the calc box).
        meta_data: patient metadata from ``DataExplorer.load_metadata()``.
        ct: CT HU array in z-y-x order.
        oar: integer OAR label map.
        ptv: binary PTV mask.
        out_dir: destination directory.
        case: patient id, also the output file stem.
        prescription_gy: prescription dose in Gy.
        dose: planner dose in Gy, or ``None`` at inference time.  When ``None``
            the ``DOSE`` key is omitted (the ground truth is unknown).
        body: binary BODY mask, optional but recommended for the
            ``mednext_hybrid`` losses and for body-masked evaluation.
        target_shape: network input shape.

    Returns:
        Path of the written ``.npz`` file.
    """
    start, end = get_crop_settings_calc_box(ct_portpy, meta_data=meta_data)

    ct = crop_resize_img(ct, start, end, is_mask=False, target_shape=target_shape)
    oar = crop_resize_img(oar, start, end, is_mask=True, target_shape=target_shape)
    ptv = crop_resize_img(ptv, start, end, is_mask=True, target_shape=target_shape)
    if dose is not None:
        dose = crop_resize_img(dose, start, end, is_mask=False, target_shape=target_shape)
    if body is not None:
        body = crop_resize_img(body, start, end, is_mask=True, target_shape=target_shape)

    if dose is not None:
        dose = np.clip(dose, a_min=0, a_max=1.2 * prescription_gy).astype(np.float32)

    ct = np.clip(ct, a_min=-1000, a_max=3071)
    ct = (ct + 1000) / 4071
    ct = ct.astype(np.float32)

    save_dict = {
        "CT": ct,
        "OAR": oar,
        "PTV": ptv,
        "PRESCRIPTION_GY": np.float32(prescription_gy),
    }
    # Do not store a ``None`` dose as an object array: it would force every
    # reader to use allow_pickle=True.  At inference there is simply no DOSE key.
    if dose is not None:
        save_dict["DOSE"] = dose
    if body is not None:
        save_dict["BODY"] = body.astype(np.uint8)

    filename = os.path.join(out_dir, case + ".npz")
    np.savez(filename, **save_dict)
    return filename


def data_preprocess(
    in_dir,
    out_dir,
    site: str = "lung",
    protocol_name: str = None,
    technique_name: str = "imrt",
    patient_ids: list = None,
    overwrite: bool = False,
    verbose: bool = True,
):
    """Preprocess a cohort of PortPy patients for anatomy-only dose prediction.

    Args:
        in_dir: PortPy raw-data root (the directory that contains the
            ``Lung_Patient_*`` / ``Prostate_Patient_*`` folders).
        out_dir: output directory for the ``.npz`` files.  Created if needed.
            The files are written flat; use
            :func:`dosepy.preprocess.split_preprocessed_dataset` afterwards to
            create the ``train/``, ``val/`` and ``test/`` folders.
        site: ``'lung'`` or ``'prostate'``.  Only cases whose id contains the
            site name are processed.
        protocol_name: PortPy protocol used for the prescription, e.g.
            ``'Lung_2Gy_30Fx'``.  ``None`` uses the site default (60 Gy for
            lung, 70.2 Gy for prostate).
        technique_name: hospital technique name used in the planner dose file
            ``DicomFiles/rt_dose_echo_<technique_name>.dcm``.
        patient_ids: optional list of patient ids to process.  ``None``
            processes every matching case in ``in_dir``.
        overwrite: if ``False`` an existing ``<patient>.npz`` is kept and that
            patient is skipped, so a long preprocessing run can be resumed.
        verbose: print progress.

    Returns:
        List of processed patient ids.

    Notes:
        A case is skipped (not raised) when the planner dose file is missing,
        when the PTV is empty, when the BODY structure is missing, or when any
        PortPy call fails.  The traceback is printed so the cause is visible.

        Unlike PortPy's original preprocessing, this function does **not** call
        ``pp.Beams``, build an influence matrix, or load the ``Beams/`` metadata
        folder.  Only CT, structures and the planner dose are used.
    """
    if not os.path.isabs(in_dir):
        in_dir = os.path.join(os.getcwd(), in_dir)
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(os.getcwd(), out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    disease_struct_names, labels, default_prescription_gy = get_site_config(site)

    cases = [c for c in sorted(os.listdir(in_dir)) if os.path.isdir(os.path.join(in_dir, c))]
    if patient_ids is not None:
        wanted = set(patient_ids)
        missing = wanted - set(cases)
        if missing:
            print("Not found in {}: {}".format(in_dir, sorted(missing)))
        cases = [c for c in cases if c in wanted]

    processed = []
    for idx, case in enumerate(cases):
        if site.lower() not in case.lower():
            continue
        if case == "Lung_Phantom_Patient_1":
            # This patient does not include some of the required structures.
            if verbose:
                print("Skipping phantom case {}".format(case))
            continue

        out_file = os.path.join(out_dir, case + ".npz")
        if os.path.exists(out_file) and not overwrite:
            if verbose:
                print("Skipping {}: {} already exists".format(case, out_file))
            processed.append(case)
            continue

        dose_file_path = os.path.join(
            in_dir, case, "DicomFiles", "rt_dose_echo_{}.dcm".format(technique_name.lower())
        )
        if not os.path.exists(dose_file_path):
            print("Dose file does not exist for case {}. Skipping...".format(case))
            continue

        try:
            if verbose:
                print("Processing case {}: {} of {} ...".format(case, idx + 1, len(cases)))

            data = AnatomyOnlyDataExplorer(data_dir=in_dir)
            data.patient_id = case
            meta_data = data.load_metadata()

            if protocol_name is not None:
                clinical_criteria = pp.ClinicalCriteria(data, protocol_name=protocol_name)
                prescription_gy = clinical_criteria.get_prescription()
            else:
                prescription_gy = default_prescription_gy

            ct = pp.CT(data)
            ct_arr = ct.ct_dict["ct_hu_3d"][0]
            structs = pp.Structures(data)

            planner_dose_3d = pp.convert_dose_rt_dicom_to_portpy(ct=ct, dose_file_name=dose_file_path)

            oar_mask, ptv_mask = get_structure_masks(
                structs=structs,
                ct_shape=ct_arr.shape,
                labels=labels,
                disease_struct_names=disease_struct_names,
                case=case,
            )
            if np.sum(ptv_mask) == 0:
                print("PTV not found for case {}. Skipping this case.".format(case))
                continue

            if "BODY" not in [s.upper() for s in structs.get_structures()]:
                print("BODY structure not found for case {}. Skipping this case.".format(case))
                continue
            body_mask = structs.get_structure_mask_3d("BODY").astype(np.uint8)

            process_case(
                ct_portpy=ct,
                meta_data=meta_data,
                ct=ct_arr,
                dose=planner_dose_3d,
                oar=oar_mask,
                ptv=ptv_mask,
                out_dir=out_dir,
                case=case,
                prescription_gy=prescription_gy,
                body=body_mask,
            )
            processed.append(case)

        except Exception:
            print("Processing of case {} failed".format(case))
            print(traceback.format_exc())

    if verbose:
        print("Done. {} case(s) available in {}".format(len(processed), out_dir))
    return processed


def _parse_args():
    parser = argparse.ArgumentParser(description="Preprocess PortPy data for anatomy-only dose prediction.")
    parser.add_argument("--in_dir", required=True, help="PortPy raw-data root")
    parser.add_argument("--out_dir", required=True, help="output directory for .npz files")
    parser.add_argument("--site", default="lung", choices=["lung", "prostate"])
    parser.add_argument("--protocol_name", default=None)
    parser.add_argument("--technique_name", default="imrt")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    data_preprocess(
        in_dir=args.in_dir,
        out_dir=args.out_dir,
        site=args.site,
        protocol_name=args.protocol_name,
        technique_name=args.technique_name,
        overwrite=args.overwrite,
    )
