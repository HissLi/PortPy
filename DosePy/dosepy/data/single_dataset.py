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

"""Inference dataset: loads anatomy-only ``.npz`` files without a ground-truth dose.

Used by ``test.py --model test --dataset_mode single`` and by
:class:`dosepy.infer.InferenceRunner`.
"""

import os

import numpy as np
import torch

from dosepy.data.base_dataset import BaseDataset, transform_3d_data_test
from dosepy.data.image_folder import make_dataset

__all__ = ["SingleDataset"]


class SingleDataset(BaseDataset):
    """Load every ``.npz`` under ``opt.dataroot`` as an input-only sample."""

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_A = opt.dataroot
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.input_nc = self.opt.output_nc if self.opt.direction == "BtoA" else self.opt.input_nc
        if len(self.A_paths) == 0:
            raise RuntimeError(
                "No .npz files found in {}. Did preprocessing run?".format(self.dir_A)
            )

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        AB = np.load(A_path, allow_pickle=True)

        A = AB["CT"]
        OAR = AB["OAR"]
        PTV = AB["PTV"]
        body = AB["BODY"] if "BODY" in AB else None
        prescription_gy = float(AB["PRESCRIPTION_GY"]) if "PRESCRIPTION_GY" in AB else None

        A, OAR = transform_3d_data_test(A, OAR, PTV)
        A = torch.unsqueeze(A, dim=0)
        A = torch.cat((A, OAR), axis=0)

        out = {
            "A": A,
            "A_paths": A_path,
            "SITE": os.path.basename(A_path).split("_")[0].lower(),
        }
        if body is not None:
            out["BODY"] = torch.from_numpy(body.astype(np.float32)).unsqueeze(dim=0)
        if prescription_gy is not None:
            out["PRESCRIPTION_GY"] = torch.tensor(prescription_gy, dtype=torch.float32)

        return out

    def __len__(self):
        return len(self.A_paths)
