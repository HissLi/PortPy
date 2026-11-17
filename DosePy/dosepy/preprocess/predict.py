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

"""One-stop inference: preprocess one patient, run the network, post-process.

This is DosePy's anatomy-only inference entry point.  It deliberately keeps the
exact preprocessing and geometry of the training pipeline (CT normalisation,
OAR/PTV masks and the calculation-box crop), but it does **not** build a BEAM
channel and therefore does not need a beam arrangement, a ``Beams/`` metadata
folder, or an influence matrix.

Typical use::

    pred_dose_3d = predict_using_model(
        patient_id="Lung_Patient_94",
        in_dir="/data/PortPy",
        out_dir="workspace/infer_input",
        model_name="dosepy_lung_unet128",
        checkpoints_dir="workspace/checkpoints",
        results_dir="workspace/results",
        netG="unet_128",
        site="lung",
        protocol_name="Lung_2Gy_30Fx",
        epoch="best_mae",
    )
"""

import os
from glob import glob

import numpy as np
import portpy.photon as pp
import SimpleITK as sitk

from dosepy.preprocess.data_preprocess import process_case
from dosepy.preprocess.metadata import AnatomyOnlyDataExplorer
from dosepy.preprocess.geometry import (
    MODEL_SHAPE,
    get_crop_settings_calc_box,
    get_structure_masks,
)
from dosepy.preprocess.postprocess import resample_prediction_to_ct_grid
from dosepy.preprocess.site_config import get_site_config

__all__ = ["predict_using_model"]


def _prepare_inference_dir(out_dir: str, patient_id: str) -> str:
    """Return a per-patient inference directory and remove stale npz files.

    PortPy used one shared ``out_dir/infer`` folder and never cleaned it.  The
    ``SingleDataset`` loads *every* ``.npz`` in that folder, so a second call
    silently re-ran inference for all previously predicted patients.  A
    per-patient sub-folder avoids that.
    """
    infer_dir = os.path.join(out_dir, "infer", patient_id)
    os.makedirs(infer_dir, exist_ok=True)
    for old_file in glob(os.path.join(infer_dir, "*.npz")):
        os.remove(old_file)
    return infer_dir


def predict_using_model(
    patient_id: str,
    in_dir: str,
    out_dir: str = "./dataset/infer",
    model_name: str = "dosepy_model",
    checkpoints_dir: str = "../checkpoints",
    results_dir: str = "../results",
    netG: str = "unet_128",
    site: str = "lung",
    protocol_name: str = None,
    infer_runner=None,
    epoch: str = "latest",
    save_nrrd: bool = False,
    ngf: int = 64,
    mednext_model_id: str = "B",
    mednext_kernel_size: int = 3,
    mednext_n_channels: int = 32,
    mednext_sites=("lung", "prostate"),
    netG_cfg: str = "",
):
    """Preprocess one patient, run a trained model and return the CT-grid dose.

    Args:
        patient_id: PortPy patient id, e.g. ``'Lung_Patient_94'``.
        in_dir: PortPy raw-data root.
        out_dir: working directory for the preprocessed inference file.  A
            per-patient sub-folder ``<out_dir>/infer/<patient_id>`` is used.
        model_name: experiment name (the checkpoint folder under
            ``checkpoints_dir``).
        checkpoints_dir: root of the training checkpoints.
        results_dir: root of the inference results.  The dose is read from
            ``<results_dir>/<model_name>/test_<epoch>/npz_images/``.
        netG: generator name, must match training (``'unet_128'``,
            ``'unet_256'``, ``'mednext'``, ``'mednext_attn'``,
            ``'mednext_duo'``).
        site: ``'lung'`` or ``'prostate'``.
        protocol_name: PortPy protocol used for the prescription.
        infer_runner: optional :class:`dosepy.infer.InferenceRunner` that keeps
            the model in memory across patients.
        epoch: checkpoint suffix to load: ``'latest'``, ``'best_mae'`` or an
            integer epoch string.
        save_nrrd: additionally save the CT-grid prediction next to the
            inference input.
        ngf: U-Net base filters; must match training.
        mednext_model_id / mednext_kernel_size / mednext_n_channels /
        mednext_sites: MedNeXt options; must match training.
        netG_cfg: optional JSON/dict of generator constructor overrides.

    Returns:
        Predicted dose on the CT grid (z, y, x), in Gy.

    Raises:
        ValueError: if the PTV is missing.
        FileNotFoundError: if the prediction file was not written.
    """
    if infer_runner is not None:
        # The runner owns the checkpoint/epoch/results settings; read them off
        # its option namespace so the prediction path cannot diverge.
        epoch = getattr(infer_runner.opt, "epoch", epoch)
        results_dir = infer_runner.opt.results_dir
        model_name = infer_runner.opt.name

    pred_dir = os.path.join(
        results_dir, model_name, "test_{}".format(epoch), "npz_images"
    )
    infer_dir = _prepare_inference_dir(out_dir, patient_id)

    print("Processing case {}...".format(patient_id))
    data = AnatomyOnlyDataExplorer(data_dir=in_dir)
    data.patient_id = patient_id
    meta_data = data.load_metadata()

    ct = pp.CT(data)
    ct_arr = ct.ct_dict["ct_hu_3d"][0]
    structs = pp.Structures(data)

    if protocol_name is not None:
        clinical_criteria = pp.ClinicalCriteria(data, protocol_name=protocol_name)
        prescription_gy = clinical_criteria.get_prescription()
    else:
        _, _, prescription_gy = get_site_config(site)

    # --- OAR / PTV masks ---------------------------------------------------
    disease_struct_names, labels, _ = get_site_config(site)
    oar_mask, ptv_mask = get_structure_masks(
        structs=structs,
        ct_shape=ct_arr.shape,
        labels=labels,
        disease_struct_names=disease_struct_names,
        case=patient_id,
    )
    if np.sum(ptv_mask) == 0:
        raise ValueError("PTV not found for case {}. Cannot run inference.".format(patient_id))

    body_mask = None
    if "BODY" in [s.upper() for s in structs.get_structures()]:
        body_mask = structs.get_structure_mask_3d("BODY").astype(np.uint8)

    process_case(
        ct_portpy=ct,
        meta_data=meta_data,
        ct=ct_arr,
        oar=oar_mask,
        ptv=ptv_mask,
        out_dir=infer_dir,
        case=patient_id,
        prescription_gy=prescription_gy,
        dose=None,
        body=body_mask,
        target_shape=MODEL_SHAPE,
    )

    # Crop indices are needed again for the inverse mapping.
    start, end = get_crop_settings_calc_box(ct, meta_data=meta_data)

    infer_args = {
        "dataroot": infer_dir,
        "netG": netG,
        "name": model_name,
        "checkpoints_dir": checkpoints_dir,
        "phase": "test",
        "epoch": epoch,
        "model": "test",
        "eval": True,
        "input_nc": 7,
        "output_nc": 1,
        "ngf": ngf,
        "mednext_model_id": mednext_model_id,
        "mednext_kernel_size": mednext_kernel_size,
        "mednext_n_channels": mednext_n_channels,
        "mednext_sites": list(mednext_sites),
        "netG_cfg": netG_cfg,
        "results_dir": results_dir,
        "direction": "AtoB",
        "dataset_mode": "single",
        "norm": "batch",
    }

    if infer_runner is None:
        # Imported lazily so that preprocessing-only users do not need the
        # training/test dependencies (dominate, visdom, ...) at import time.
        from dosepy.test import test

        test(infer_args)
    else:
        infer_runner.run(infer_dir)

    filename = os.path.join(pred_dir, patient_id + "_CT2DOSE.nrrd")
    if not os.path.exists(filename):
        raise FileNotFoundError(
            "Expected prediction at {}. Check model_name/epoch/checkpoints_dir.".format(filename)
        )
    pred_dose = sitk.GetArrayFromImage(sitk.ReadImage(filename))

    pred_dose_3d = resample_prediction_to_ct_grid(pred_dose, ct, start=start, end=end)

    if save_nrrd:
        from dosepy.preprocess.postprocess import save_prediction_to_nrrd

        save_prediction_to_nrrd(
            pred_dose_3d, ct, os.path.join(infer_dir, patient_id + "_pred_dose_ct_grid.nrrd")
        )

    return pred_dose_3d
