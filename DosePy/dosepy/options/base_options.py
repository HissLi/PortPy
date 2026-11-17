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

"""Shared training / test options.

The only API change relative to PortPy is that :meth:`parse` accepts an
explicit ``argv`` and an ``overrides`` dict::

    opt = TrainOptions().parse(argv=[], overrides={"dataroot": "...", "n_epochs": 5})

PortPy parsed ``sys.argv`` unconditionally, which made notebooks fragile (the
Jupyter kernel's arguments leaked into the parser).  Passing ``argv=[]`` gives
deterministic behaviour while the CLI path (``argv=None``) is unchanged.
"""

import argparse
import os

import torch

import dosepy.models as models
from dosepy.data import get_option_setter
from dosepy.utils import util

__all__ = ["BaseOptions"]


class BaseOptions:
    """Options shared by training and testing."""

    def __init__(self):
        self.initialized = False
        self.isTrain = None
        self.parser = None

    def initialize(self, parser):
        """Define the options shared by training and testing."""
        # basic parameters
        parser.add_argument("--dataroot", required=False, help="path to the preprocessed data root")
        parser.add_argument("--name", type=str, default="experiment_name", help="experiment name")
        parser.add_argument("--gpu_ids", type=str, default="0", help="e.g. 0, 0,1,2, -1 for CPU")
        parser.add_argument("--checkpoints_dir", type=str, default="../checkpoints")
        # model parameters
        parser.add_argument("--model", type=str, default="doseprediction3d",
                            help="doseprediction3d (train/val/test) or test (inference)")
        parser.add_argument("--input_nc", type=int, default=7,
                            help="input channels: PortPy anatomy-only=7; OpenKBP dataset "
                                 "overrides this to 11")
        parser.add_argument("--output_nc", type=int, default=1, help="output channels (dose = 1)")
        parser.add_argument("--ngf", type=int, default=64, help="base filters of the U-Net generators")
        parser.add_argument("--ndf", type=int, default=64, help="unused; kept for CLI compatibility")
        parser.add_argument("--netD", type=str, default="basic", help="unused; no discriminator")
        parser.add_argument("--netG", type=str, default="unet_128",
                            help="unet_128 | unet_256 | mednext | mednext_attn | mednext_duo")
        parser.add_argument("--n_layers_D", type=int, default=3, help="unused")
        parser.add_argument("--norm", type=str, default="instance",
                            help="instance | batch | none (U-Net generators only)")
        parser.add_argument("--init_type", type=str, default="normal",
                            help="normal | xavier | kaiming | orthogonal")
        parser.add_argument("--init_gain", type=float, default=0.02)
        parser.add_argument("--no_dropout", action="store_true", help="disable U-Net dropout")
        # dataset parameters
        parser.add_argument("--dataset_mode", type=str, default="dosepred3d",
                            help="dosepred3d (PortPy with GT), openkbp (OpenKBP CSV), "
                                 "or single (inference)")
        parser.add_argument("--direction", type=str, default="AtoB", help="AtoB or BtoA")
        parser.add_argument("--serial_batches", action="store_true")
        parser.add_argument("--num_threads", default=0, type=int)
        parser.add_argument("--batch_size", type=int, default=1)
        parser.add_argument("--load_size", type=int, default=286)
        parser.add_argument("--crop_size", type=int, default=256)
        parser.add_argument("--max_dataset_size", type=int, default=float("inf"))
        parser.add_argument("--preprocess", type=str, default="resize_and_crop")
        parser.add_argument("--no_flip", action="store_true")
        parser.add_argument("--display_winsize", type=int, default=256)
        # additional parameters
        parser.add_argument("--epoch", type=str, default="latest",
                            help="latest | best_mae | an integer epoch")
        parser.add_argument("--load_iter", type=int, default=0)
        parser.add_argument("--verbose", action="store_true")
        parser.add_argument("--suffix", default="", type=str)
        self.initialized = True
        return parser

    def gather_options(self, argv=None):
        """Build the full parser (base + model + dataset options) and parse it."""
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser = self.initialize(parser)

        opt, _ = parser.parse_known_args(argv)

        model_option_setter = models.get_option_setter(opt.model)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args(argv)

        dataset_option_setter = get_option_setter(opt.dataset_mode)
        parser = dataset_option_setter(parser, self.isTrain)

        self.parser = parser
        return parser.parse_args(argv)

    def print_options(self, opt):
        """Print options and save them to ``<checkpoints_dir>/<name>/<phase>_opt.txt``."""
        message = "----------------- Options ---------------\n"
        for k, v in sorted(vars(opt).items()):
            comment = ""
            default = self.parser.get_default(k)
            if v != default:
                comment = "\t[default: %s]" % str(default)
            message += "{:>25}: {:<30}{}\n".format(str(k), str(v), comment)
        message += "----------------- End -------------------"
        print(message)

        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        with open(os.path.join(expr_dir, "{}_opt.txt".format(opt.phase)), "wt") as opt_file:
            opt_file.write(message)
            opt_file.write("\n")

    def parse(self, argv=None, overrides=None):
        """Parse options and apply ``overrides``.

        Args:
            argv: argument list for argparse.  ``None`` reads ``sys.argv``;
                notebooks should pass ``[]``.
            overrides: dict of option values applied after parsing.  Unknown
                keys raise ``ValueError`` (typo protection).

        Returns:
            ``argparse.Namespace`` with ``isTrain`` and a normalised
            ``gpu_ids`` list.
        """
        opt = self.gather_options(argv)
        opt.isTrain = self.isTrain

        if overrides:
            unknown = sorted(set(overrides) - set(vars(opt)))
            if unknown:
                raise ValueError(
                    "Unknown option(s) {} for {}. Available keys: {}".format(
                        unknown, type(self).__name__, sorted(vars(opt))
                    )
                )
            vars(opt).update(overrides)

        if opt.suffix:
            suffix = "_" + opt.suffix.format(**vars(opt)) if opt.suffix != "" else ""
            opt.name = opt.name + suffix

        # Normalise gpu ids: accept "0,1", [0, 1], 0 or -1.
        if isinstance(opt.gpu_ids, str):
            ids = [int(s) for s in opt.gpu_ids.split(",") if s.strip() != ""]
        elif isinstance(opt.gpu_ids, (list, tuple)):
            ids = [int(i) for i in opt.gpu_ids]
        else:
            ids = [int(opt.gpu_ids)]
        opt.gpu_ids = [i for i in ids if i >= 0]

        if len(opt.gpu_ids) > 0:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "gpu_ids={} was requested but CUDA is not available. "
                    "Pass gpu_ids=-1 (CLI) or gpu_ids=[] (notebook) for CPU.".format(opt.gpu_ids)
                )
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
