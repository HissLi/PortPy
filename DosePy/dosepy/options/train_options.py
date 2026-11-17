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

"""Training options for full dose prediction.

DosePy-specific additions compared with PortPy:

- ``val_phase`` (default ``'val'``) exposes the validation folder name.
- beamlet-prediction-only options (``beamlet_y_max``, ``d1_norm``, ...) were
  removed because they do not affect a total-dose model.
"""

import argparse

from dosepy.options.base_options import BaseOptions

__all__ = ["TrainOptions"]


class TrainOptions(BaseOptions):
    """Options used by :func:`dosepy.train.train`."""

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        # display / logging
        parser.add_argument("--display_freq", type=int, default=400,
                            help="frequency of showing results")
        parser.add_argument("--display_ncols", type=int, default=4)
        parser.add_argument("--display_id", type=int, default=1,
                            help="visdom window id; <=0 disables visdom")
        parser.add_argument("--display_server", type=str, default="http://localhost")
        parser.add_argument("--display_env", type=str, default="main")
        parser.add_argument("--display_port", type=int, default=8097)
        parser.add_argument("--update_html_freq", type=int, default=1000)
        parser.add_argument("--print_freq", type=int, default=100)
        parser.add_argument("--no_html", action="store_true",
                            help="do not save intermediate nrrd/html results")
        parser.add_argument("--plot_loss", dest="plot_loss", action="store_true",
                            default=True,
                            help="draw and refresh the loss curve during training "
                                 "(enabled by default)")
        parser.add_argument("--no_plot_loss", dest="plot_loss", action="store_false",
                            default=argparse.SUPPRESS,
                            help="disable the live loss curve")
        parser.add_argument("--plot_freq", type=int, default=1,
                            help="redraw/save the loss curve every N epochs")
        parser.add_argument("--augment", action="store_true",
                            help="random 3-D flip/rotation/translation augmentation")

        # checkpointing
        parser.add_argument("--save_latest_freq", type=int, default=5000)
        parser.add_argument("--save_epoch_freq", type=int, default=100,
                            help="save a numbered checkpoint every N epochs")
        parser.add_argument("--save_by_iter", action="store_true")
        parser.add_argument("--continue_train", action="store_true")
        parser.add_argument("--epoch_count", type=int, default=1)

        # phases / data
        parser.add_argument("--phase", type=str, default="train")
        parser.add_argument("--val_phase", type=str, default="val",
                            help="sub-folder used for validation")

        # optimisation
        parser.add_argument("--n_epochs", type=int, default=100)
        parser.add_argument("--n_epochs_decay", type=int, default=100)
        parser.add_argument("--beta1", type=float, default=0.5)
        parser.add_argument("--lr", type=float, default=0.0002)
        parser.add_argument("--gan_mode", type=str, default="lsgan",
                            help="unused; no discriminator")
        parser.add_argument("--pool_size", type=int, default=50,
                            help="unused; no discriminator")
        parser.add_argument("--lr_policy", type=str, default="linear",
                            help="linear | step | plateau | cosine")
        parser.add_argument("--lr_decay_iters", type=int, default=50)

        # loss selection
        parser.add_argument("--dose_loss", type=str, default="mae",
                            choices=["mae", "mednext_hybrid", "mae_moment"],
                            help="mae: plain L1; mae_moment: L1 + ROI moments; "
                                 "mednext_hybrid: weighted L1 + body-masked L1 + "
                                 "moments + dual-gradient sharpness")
        parser.add_argument("--lambda_L1", type=float, default=1.0)
        parser.add_argument("--lambda_wL1", type=float, default=1.0)
        parser.add_argument("--lambda_masked_wL1", type=float, default=1.0)
        parser.add_argument("--lambda_moment", type=float, default=0.1)

        # MedNeXt options
        parser.add_argument("--mednext_model_id", type=str, default="B",
                            choices=["S", "B", "M", "L"])
        parser.add_argument("--mednext_kernel_size", type=int, default=3,
                            choices=[3, 5, 7])
        parser.add_argument("--mednext_n_channels", type=int, default=32,
                            help="base filters for mednext_attn / mednext_duo")
        parser.add_argument("--mednext_sites", type=str, nargs="+",
                            default=["lung", "prostate"],
                            help="decoder names for mednext_duo")
        parser.add_argument("--netG_cfg", type=str, default="",
                            help="JSON dict of generator constructor overrides")
        parser.add_argument("--upkern_pretrained_path", type=str, default="")
        parser.add_argument("--load_upkern", action="store_true",
                            help="initialise a kernel-5 MedNeXt from a kernel-3 checkpoint")

        self.isTrain = True
        return parser
