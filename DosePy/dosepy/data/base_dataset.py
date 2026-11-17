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

"""Dataset base class and 3-D transforms for full dose prediction.

Compared with PortPy's ``base_dataset.py`` this module drops the unused 2-D
pix2pix image transforms (and therefore the PIL / torchvision dependency) and
uses the anatomy-only 7-channel input layout.
"""

import random
import warnings
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.utils.data as data
from scipy.ndimage import affine_transform

warnings.filterwarnings("ignore")

__all__ = [
    "BaseDataset",
    "transform_3d_data",
    "transform_3d_data_test",
    "augment_sample",
    "random_flip_rotation_translation",
]


class BaseDataset(data.Dataset, ABC):
    """Abstract base class for DosePy datasets.

    Subclasses implement ``__len__`` and ``__getitem__``; they store ``opt`` and
    ``root = opt.dataroot`` here.
    """

    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Optionally add dataset-specific options (no-op by default)."""
        return parser

    @abstractmethod
    def __len__(self):
        return 0

    @abstractmethod
    def __getitem__(self, index):
        pass


def random_flip_rotation_translation(shape, max_translation=10, max_rotation=10):
    """Create a random 3-D affine transform (flip + rotation + translation)."""
    flip = np.diag([np.random.choice([-1, 1]) for _ in range(3)])

    angles = np.deg2rad(np.random.uniform(-max_rotation, max_rotation, size=3))
    cx, cy, cz = np.cos(angles)
    sx, sy, sz = np.sin(angles)

    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    rotation = rz @ ry @ rx

    transform_matrix = flip @ rotation
    center = np.array(shape) / 2
    translation = np.random.uniform(-max_translation, max_translation, size=3)
    offset = center - transform_matrix @ center + translation

    return transform_matrix, offset


def augment_sample(ct, dose, masks, ptv_mask):
    """Apply the same random 3-D affine transform to every volume.

    Linear interpolation is used for CT / dose, nearest neighbour for masks.
    Dose outside the original dose support is zeroed after the transform.
    """
    assert ct.shape == masks.shape == dose.shape, "All inputs must have the same shape"

    transform_matrix, offset = random_flip_rotation_translation(ct.shape)

    ct_aug = affine_transform(ct, transform_matrix, offset=offset, order=1, mode="nearest")
    masks_aug = affine_transform(masks, transform_matrix, offset=offset, order=0, mode="nearest")
    ptv_aug = affine_transform(ptv_mask, transform_matrix, offset=offset, order=0, mode="nearest")
    dose_aug = affine_transform(dose, transform_matrix, offset=offset, order=1, mode="nearest")

    dose_mask = dose > 0
    dose_mask_aug = affine_transform(
        dose_mask.astype(np.uint8), transform_matrix, offset=offset, order=0, mode="nearest"
    )
    dose_aug *= dose_mask_aug > 0

    return ct_aug, dose_aug, masks_aug, ptv_aug


def transform_3d_data(ct_arr, dose_arr, oar_arr, ptv_arr, augment=False):
    """Build the 7-channel anatomy-only input tensor and the 1-channel target.

    Channel layout::

        [CT, OAR-label-1, OAR-label-2, OAR-label-3, OAR-label-4,
         OAR-label-5 (unused), PTV]

    Args:
        ct_arr: normalised CT, ``(128,128,128)`` float32.
        dose_arr: planner dose in Gy, ``(128,128,128)`` float32.
        oar_arr: OAR labels 0..4, ``(128,128,128)`` uint8.
        ptv_arr: PTV mask, ``(128,128,128)`` uint8.
        augment: apply the random affine augmentation.

    Returns:
        ``(ct, dose, oar_onehot_with_ptv)`` tensors; the 1-channel CT and
        dose are not yet concatenated, the dataset does that.
    """
    if augment:
        ct_arr, dose_arr, oar_arr, ptv_arr = augment_sample(
            ct_arr, dose_arr, oar_arr, ptv_arr
        )

    ct = torch.from_numpy(ct_arr)
    dose = torch.from_numpy(dose_arr)

    oar = torch.from_numpy(oar_arr).long()
    ptv = torch.from_numpy(ptv_arr).unsqueeze(dim=0)

    oar = torch.nn.functional.one_hot(oar, 6)[..., 1:]  # drop background, keep labels 1..5
    oar = oar.permute(3, 0, 1, 2).to(torch.float)  # channels first
    ptv = ptv.to(torch.float)
    oar = torch.cat((oar, ptv), axis=0)

    return ct, dose, oar


def transform_3d_data_test(ct_arr, oar_arr, ptv_arr):
    """Inference-time version of :func:`transform_3d_data` (no target dose)."""
    ct = torch.from_numpy(ct_arr)

    oar = torch.from_numpy(oar_arr).long()
    ptv = torch.from_numpy(ptv_arr).unsqueeze(dim=0)

    oar = torch.nn.functional.one_hot(oar, 6)[..., 1:]
    oar = oar.permute(3, 0, 1, 2).to(torch.float)
    ptv = ptv.to(torch.float)
    oar = torch.cat((oar, ptv), axis=0)

    return ct, oar
