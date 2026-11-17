# DosePy -- Full dose prediction toolkit
#
# OpenKBP input adapter.  Derived from the public OpenKBP data format
# (https://github.com/OpenKBP/OpenKBP_C3D), not from PortPy.

"""PyTorch dataset for OpenKBP head-and-neck dose prediction.

The CSV parsing, normalization and cache logic lives in
:mod:`dosepy.data.openkbp_io`; this module only turns the converted arrays into
11-channel model inputs:

``[CT, Brainstem, SpinalCord, RightParotid, LeftParotid,
 Mandible, Esophagus, Larynx, PTV56, PTV63, PTV70]``

The body mask (``possible_dose_mask``) is returned separately as ``BODY`` and
is used by the ``mednext_hybrid`` loss and body-masked metrics.  PTV70 is the
last channel, so the existing validation MAE (PTV D97 scaling) works without a
model change.
"""

import os

import numpy as np
import torch

from dosepy.data.base_dataset import BaseDataset
from dosepy.data.openkbp_io import (
    OPENKBP_CHANNEL_NAMES,
    OPENKBP_INPUT_NC,
    OPENKBP_OAR_NAMES,
    OPENKBP_PRESCRIPTION_GY,
    OPENKBP_PTV_NAMES,
    OPENKBP_SHAPE,
    PHASE_DIRS,
    convert_openkbp_patient,
    load_openkbp_patient,
)

__all__ = [
    "OpenKBPDataset",
    "convert_openkbp_patient",
    "OPENKBP_OAR_NAMES",
    "OPENKBP_PTV_NAMES",
    "OPENKBP_CHANNEL_NAMES",
    "OPENKBP_INPUT_NC",
    "OPENKBP_SHAPE",
    "OPENKBP_PRESCRIPTION_GY",
]


class OpenKBPDataset(BaseDataset):
    """Dataset for OpenKBP ``provided-data`` splits.

    ``opt.dataroot`` should point at ``OpenKBP/provided-data`` (the directory
    containing ``train-pats``, ``validation-pats`` and ``test-pats``).  Phase
    mapping is::

        train -> train-pats
        val   -> validation-pats
        test  -> test-pats

    The dataset overrides ``opt.input_nc`` to 11 and ``opt.output_nc`` to 1, so
    a training command only needs ``dataset_mode="openkbp"``.
    """

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.opt.input_nc = OPENKBP_INPUT_NC
        self.opt.output_nc = 1
        self.phase = opt.phase
        phase_dir_name = PHASE_DIRS.get(self.phase, self.phase)
        self.phase_dir_name = phase_dir_name

        if os.path.basename(os.path.normpath(opt.dataroot)) == phase_dir_name:
            self.dir_AB = opt.dataroot
            self.cache_root = os.path.join(opt.dataroot, ".openkbp_cache")
        else:
            self.dir_AB = os.path.join(opt.dataroot, phase_dir_name)
            self.cache_root = os.path.join(opt.dataroot, ".openkbp_cache", phase_dir_name)

        if not os.path.isdir(self.dir_AB):
            raise FileNotFoundError(
                "OpenKBP phase directory not found: {}. Expected dataroot to be "
                "OpenKBP/provided-data.".format(self.dir_AB)
            )

        self.patient_dirs = sorted(
            d for d in (os.path.join(self.dir_AB, name) for name in os.listdir(self.dir_AB))
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "ct.csv"))
        )
        if len(self.patient_dirs) == 0:
            raise RuntimeError("No OpenKBP patient folders found in {}".format(self.dir_AB))
        os.makedirs(self.cache_root, exist_ok=True)
        print("OpenKBP [{}]: {} patients from {}".format(
            self.phase, len(self.patient_dirs), self.dir_AB
        ))

    def _get_patient_data(self, patient_dir: str) -> dict:
        patient_id = os.path.basename(patient_dir)
        cache_path = os.path.join(self.cache_root, patient_id + ".npz")
        return load_openkbp_patient(patient_dir, cache_path=cache_path, use_cache=True)

    def __getitem__(self, index):
        patient_dir = self.patient_dirs[index]
        data = self._get_patient_data(patient_dir)

        # 11-channel input: CT + 7 OAR + 3 PTV.
        A = np.concatenate(
            (data["CT"][None, ...], data["OAR"], data["PTV"]), axis=0
        ).astype(np.float32)

        patient_id = os.path.basename(patient_dir)
        return {
            "A": torch.from_numpy(A),
            "B": torch.from_numpy(data["DOSE"][None, ...].astype(np.float32)),
            "BODY": torch.from_numpy(data["BODY"][None, ...].astype(np.float32)),
            "PRESCRIPTION_GY": torch.tensor(
                float(data["PRESCRIPTION_GY"]), dtype=torch.float32
            ),
            "SITE": "openkbp",
            "A_paths": patient_dir,
            "B_paths": patient_dir,
            "PATIENT_ID": patient_id,
        }

    def __len__(self):
        return len(self.patient_dirs)
