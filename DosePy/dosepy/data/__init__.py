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

"""Dataset factory.

Dataset modes used by the full-dose pipeline:

``dosepred3d``
    :class:`~dosepy.data.dosepred3d_dataset.DosePred3DDataset` for PortPy
    training, validation and testing (requires a ground-truth ``DOSE``).
``openkbp``
    :class:`~dosepy.data.openkbp_dataset.OpenKBPDataset` for OpenKBP CSV
    folders (11 channels; auto-sets ``input_nc=11``).
``single``
    :class:`~dosepy.data.single_dataset.SingleDataset` for inference (no ground
    truth required).
"""

import importlib

import torch.utils.data

from dosepy.data.base_dataset import BaseDataset
from dosepy.data.dosepred3d_dataset import DosePred3DDataset
from dosepy.data.openkbp_dataset import OpenKBPDataset
from dosepy.data.single_dataset import SingleDataset

__all__ = [
    "BaseDataset",
    "DosePred3DDataset",
    "OpenKBPDataset",
    "SingleDataset",
    "find_dataset_using_name",
    "get_option_setter",
    "create_dataset",
    "CustomDatasetDataLoader",
]


def find_dataset_using_name(dataset_name):
    """Import ``dosepy.data.<dataset_name>_dataset`` and return its class."""
    dataset_filename = "dosepy.data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace("_", "") + "dataset"
    for name, cls in datasetlib.__dict__.items():
        if (
            name.lower() == target_dataset_name.lower()
            and isinstance(cls, type)
            and issubclass(cls, BaseDataset)
        ):
            dataset = cls

    if dataset is None:
        raise NotImplementedError(
            "In {} there should be a subclass of BaseDataset with class name "
            "matching {} (case insensitive).".format(dataset_filename, target_dataset_name)
        )
    return dataset


def get_option_setter(dataset_name):
    """Return the dataset class' static ``modify_commandline_options``."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt):
    """Create the data loader described by ``opt.dataset_mode``."""
    data_loader = CustomDatasetDataLoader(opt)
    return data_loader.load_data()


class CustomDatasetDataLoader:
    """Thin wrapper around ``torch.utils.data.DataLoader``."""

    def __init__(self, opt):
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_mode)
        self.dataset = dataset_class(opt)
        print("dataset [%s] was created" % type(self.dataset).__name__)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads),
        )

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data
