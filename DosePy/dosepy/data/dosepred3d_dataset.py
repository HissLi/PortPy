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

"""Training / validation dataset for full dose prediction.

Expects ``<dataroot>/<phase>/<PatientId>.npz`` files produced by
:func:`dosepy.preprocess.data_preprocess.data_preprocess`, with the keys
``CT``, ``DOSE``, ``OAR``, ``PTV`` and optionally ``BODY`` /
``PRESCRIPTION_GY``.  There is no BEAM channel.
"""

import os.path

import numpy as np
import torch

from dosepy.data.base_dataset import BaseDataset, transform_3d_data
from dosepy.data.image_folder import make_dataset

__all__ = ["DosePred3DDataset"]


class DosePred3DDataset(BaseDataset):
    """3-D CT + structures -> dose dataset (anatomy-only).

    Each item contains:

    - ``A``: 7-channel input tensor,
    - ``B``: 1-channel planner dose,
    - ``A_paths`` / ``B_paths``: the source ``.npz`` path,
    - ``SITE``: ``'lung'`` or ``'prostate'`` parsed from the file name,
    - ``BODY`` (optional): body mask used by the ``mednext_hybrid`` losses,
    - ``PRESCRIPTION_GY`` (optional): prescription used by the validation MAE.
    """

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))
        assert self.opt.load_size >= self.opt.crop_size, "crop_size must be <= load_size"
        self.input_nc = self.opt.output_nc if self.opt.direction == "BtoA" else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == "BtoA" else self.opt.output_nc
        self.phase = self.opt.phase
        self.augment = getattr(self.opt, "augment", False)

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        # allow_pickle=True keeps compatibility with older PortPy npz files that
        # stored DOSE=None as an object array for inference.
        AB = np.load(AB_path, allow_pickle=True)

        A = AB["CT"]
        B = AB["DOSE"]
        OAR = AB["OAR"]
        PTV = AB["PTV"]
        body = AB["BODY"] if "BODY" in AB else None
        prescription_gy = float(AB["PRESCRIPTION_GY"]) if "PRESCRIPTION_GY" in AB else None

        augment = self.augment if self.phase == "train" else False
        A, B, OAR = transform_3d_data(A, B, OAR, PTV, augment=augment)

        A = torch.unsqueeze(A, dim=0)
        B = torch.unsqueeze(B, dim=0)
        A = torch.cat((A, OAR), axis=0)

        out = {"A": A, "B": B, "A_paths": AB_path, "B_paths": AB_path}

        # Treatment site parsed from Lung_Patient_*.npz / Prostate_Patient_*.npz.
        # MedNeXt-Duo uses it to select a decoder; other generators ignore it.
        out["SITE"] = os.path.basename(AB_path).split("_")[0].lower()

        if body is not None:
            body_t = torch.from_numpy(body.astype(np.float32)).unsqueeze(dim=0)
            out["BODY"] = body_t

        if prescription_gy is not None:
            out["PRESCRIPTION_GY"] = torch.tensor(prescription_gy, dtype=torch.float32)

        return out

    def __len__(self):
        return len(self.AB_paths)
