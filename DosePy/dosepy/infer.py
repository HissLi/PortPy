# DosePy -- Full dose prediction toolkit
#
# Derived from PortPy (https://github.com/PortPy-Project/PortPy), which is
# licensed under Apache 2.0 with the Commons Clause restriction.

"""Reusable inference runner.

``predict_using_model`` builds and loads the network for every patient.  When
predicting a cohort, use :class:`InferenceRunner` instead: the weights are
loaded once and only the preprocessed ``.npz`` changes.

For OpenKBP, use :func:`predict_openkbp_patient`: it reads one OpenKBP patient
folder, builds the same 11-channel input as training, and saves
``DOSE_PRED`` / ``DOSE_GT`` / masks to an ``.npz`` file.

Example::

    runner = InferenceRunner({
        "checkpoints_dir": "workspace/checkpoints",
        "results_dir": "workspace/results",
        "name": "dosepy_lung_unet128",
        "netG": "unet_128",
        "epoch": "best_mae",
        "gpu_ids": [0],
    })
    for infer_dir in prepared_infer_dirs:
        # each folder contains one <Patient>.npz produced by the preprocessing
        # path used in dosepy.preprocess.predict
        runner.run(infer_dir)
"""

import os

import numpy as np
import torch

from dosepy.data import create_dataset
from dosepy.models import create_model
from dosepy.options import TestOptions, build_opt
from dosepy.utils import html
from dosepy.utils.visualizer import save_images

__all__ = ["InferenceRunner", "build_infer_opt", "predict_openkbp_patient"]

NPY_OUT_FNAMES = ["CT2DOSE"]


def build_infer_opt(args: dict):
    """Build inference options from a dict.

    The dataset mode is forced to ``single`` (no ground truth) and the loader
    to a deterministic batch size of one.
    """
    overrides = dict(args)
    overrides.setdefault("model", "test")
    overrides.setdefault("dataset_mode", "single")
    opt = build_opt(TestOptions, overrides)
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    return opt


class InferenceRunner:
    """Load a trained generator once and run it on many preprocessed folders."""

    def __init__(self, args: dict):
        self.opt = build_infer_opt(args)
        self.model = create_model(self.opt)
        self.model.setup(self.opt)
        if self.opt.eval:
            self.model.eval()

    def run(self, dataroot: str):
        """Run inference on every ``.npz`` under ``dataroot``.

        Returns:
            List of written prediction paths
            (``.../npz_images/<Patient>_CT2DOSE.nrrd``).
        """
        self.opt.dataroot = dataroot
        dataset = create_dataset(self.opt)

        web_dir = os.path.join(
            self.opt.results_dir,
            self.opt.name,
            "{}_{}".format(self.opt.phase, self.opt.epoch),
        )
        if self.opt.load_iter > 0:
            web_dir = "{:s}_iter{:d}".format(web_dir, self.opt.load_iter)

        print("creating web directory", web_dir)
        os.makedirs(os.path.join(web_dir, "npz_images"), exist_ok=True)

        webpage = html.HTML(
            web_dir,
            "Experiment = %s, Phase = %s, Epoch = %s"
            % (self.opt.name, self.opt.phase, self.opt.epoch),
        )

        written = []
        for i, data in enumerate(dataset):
            self.model.set_input(data)
            self.model.test()

            visuals = self.model.get_current_visuals()
            img_path = self.model.get_image_paths()
            if i % 5 == 0:
                print("processing (%04d)-th image... %s" % (i, img_path))

            save_images(
                webpage,
                visuals,
                img_path,
                aspect_ratio=self.opt.aspect_ratio,
                width=self.opt.display_winsize,
                npy_out_fnames=NPY_OUT_FNAMES,
            )

            patient = os.path.splitext(os.path.basename(img_path[0]))[0]
            written.append(os.path.join(web_dir, "npz_images", patient + "_CT2DOSE.nrrd"))

        webpage.save()
        return written


def predict_openkbp_patient(
    patient_dir: str,
    checkpoints_dir: str,
    name: str,
    netG: str = "unet_128",
    ngf: int = 64,
    epoch: str = "best_mae",
    gpu_ids=None,
    norm: str = "batch",
    netG_cfg: str = "",
    mednext_model_id: str = "B",
    mednext_kernel_size: int = 3,
    mednext_n_channels: int = 32,
    mednext_sites=None,
    save_path: str = None,
):
    """Run one OpenKBP patient through a trained DosePy generator.

    This is the OpenKBP inference entry point.  CSV parsing lives in
    :mod:`dosepy.data.openkbp_io`; the 11-channel dataset layout is shared with
    training, but here the input tensor is built directly for a single patient
    (no DataLoader / no ground-truth dataset required).

    Args:
        patient_dir: one OpenKBP patient folder, e.g.
            ``OpenKBP/provided-data/test-pats/pt_241``.
        checkpoints_dir: directory containing ``<name>/<epoch>_net_G.pth``.
        name: experiment name.
        netG: ``unet_128``, ``unet_256``, ``mednext``, ``mednext_attn`` or
            ``mednext_duo``.
        ngf: U-Net base channels.
        epoch: checkpoint selector (``best_mae``, ``latest`` or an integer).
        gpu_ids: list of GPU ids; ``None`` means CPU.
        norm: U-Net normalization.
        netG_cfg: optional JSON string for generator overrides.
        mednext_model_id / mednext_kernel_size / mednext_n_channels /
        mednext_sites: MedNeXt constructor options.
        save_path: optional ``.npz`` output path.

    Returns:
        Dict with ``pred_dose``, ``gt_dose``, ``body``, ``oar``, ``ptv``,
        ``voxel_dims``, ``prescription_gy`` and ``patient_id``.
    """
    from dosepy.data.openkbp_io import convert_openkbp_patient, OPENKBP_INPUT_NC

    overrides = {
        "model": "test",
        "dataset_mode": "single",
        "input_nc": OPENKBP_INPUT_NC,
        "output_nc": 1,
        "netG": netG,
        "ngf": ngf,
        "norm": norm,
        "netG_cfg": netG_cfg,
        "mednext_model_id": mednext_model_id,
        "mednext_kernel_size": mednext_kernel_size,
        "mednext_n_channels": mednext_n_channels,
        "checkpoints_dir": checkpoints_dir,
        "name": name,
        "epoch": epoch,
        "gpu_ids": [] if gpu_ids is None else gpu_ids,
        "eval": True,
    }
    if mednext_sites is not None:
        overrides["mednext_sites"] = mednext_sites

    opt = build_opt(TestOptions, overrides)
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.display_id = -1

    model = create_model(opt)
    model.setup(opt)
    if opt.eval:
        model.eval()

    data = convert_openkbp_patient(patient_dir)
    A = np.concatenate(
        (data["CT"][None, ...], data["OAR"], data["PTV"]), axis=0
    ).astype(np.float32)
    model.set_input({
        "A": torch.from_numpy(A).unsqueeze(0),
        "A_paths": patient_dir,
        "SITE": "openkbp",
    })
    with torch.no_grad():
        model.forward()

    pred_dose = model.fake_Dose[0, 0].detach().cpu().numpy().astype(np.float32)
    # Dose is non-negative; clipping also keeps downstream DVH metrics sane.
    pred_dose = np.clip(pred_dose, 0.0, None)

    result = {
        "patient_id": os.path.basename(os.path.normpath(patient_dir)),
        "pred_dose": pred_dose,
        "gt_dose": data["DOSE"].astype(np.float32),
        "body": data["BODY"].astype(np.uint8),
        "oar": data["OAR"].astype(np.uint8),
        "ptv": data["PTV"].astype(np.uint8),
        "voxel_dims": data["VOXEL_DIMS"].astype(np.float32),
        "prescription_gy": float(data["PRESCRIPTION_GY"]),
    }

    if save_path:
        save_path = os.fspath(save_path)
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        np.savez_compressed(
            save_path,
            DOSE_PRED=result["pred_dose"],
            DOSE_GT=result["gt_dose"],
            BODY=result["body"],
            OAR=result["oar"],
            PTV=result["ptv"],
            VOXEL_DIMS=result["voxel_dims"],
            PRESCRIPTION_GY=np.float32(result["prescription_gy"]),
        )
        result["save_path"] = save_path

    return result
