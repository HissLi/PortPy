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

"""Test / inference options for full dose prediction."""

from dosepy.options.base_options import BaseOptions

__all__ = ["TestOptions"]


class TestOptions(BaseOptions):
    """Options used by :func:`dosepy.test.test` and inference."""

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        parser.add_argument("--results_dir", type=str, default="./results/",
                            help="predictions, HTML and metrics are written here")
        parser.add_argument("--aspect_ratio", type=float, default=1.0)
        parser.add_argument("--phase", type=str, default="test")
        parser.add_argument("--eval", action="store_true",
                            help="call model.eval() before inference")
        parser.add_argument("--num_test", type=int, default=50)
        parser.set_defaults(model="test")

        # Needed to rebuild the same loss/metrics as training.
        parser.add_argument("--dose_loss", type=str, default="mae",
                            choices=["mae", "mednext_hybrid", "mae_moment"])
        parser.add_argument("--lambda_L1", type=float, default=1.0)
        parser.add_argument("--lambda_wL1", type=float, default=1.0)
        parser.add_argument("--lambda_masked_wL1", type=float, default=1.0)
        parser.add_argument("--lambda_moment", type=float, default=0.1)

        parser.set_defaults(load_size=parser.get_default("crop_size"))

        parser.add_argument("--mednext_model_id", type=str, default="B",
                            choices=["S", "B", "M", "L"])
        parser.add_argument("--mednext_kernel_size", type=int, default=3,
                            choices=[3, 5, 7])
        parser.add_argument("--mednext_n_channels", type=int, default=32)
        parser.add_argument("--mednext_sites", type=str, nargs="+",
                            default=["lung", "prostate"])
        parser.add_argument("--netG_cfg", type=str, default="")
        parser.add_argument("--upkern_pretrained_path", type=str, default="")
        parser.add_argument("--load_upkern", action="store_true")

        self.isTrain = False
        return parser
