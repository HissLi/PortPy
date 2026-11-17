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

"""Network factory, weight initialisation, schedulers and loss re-exports.

Only the full-dose generators are kept here.  The beamlet-dose generators from
``portpy.ai`` (``beamlet_*``) belong to PortPy.AI Task 1 and are intentionally
out of scope for DosePy.

Available generators
--------------------

===================== ================================================ ==========================
``netG``              model                                            defined in
===================== ================================================ ==========================
``unet_128``          3-D U-Net with 7 down-samplings (default)         ``unet_generator3d.py``
``unet_256``          3-D U-Net with 8 down-samplings                   ``unet_generator3d.py``
``mednext``           MedNeXt dose regressor (GDP-HMM Top-1 style)      ``mednext_dose.py``
``mednext_attn``      MedNeXt with attention-gated skips (Top-2)        ``mednext_attention.py``
``mednext_duo``       MedNeXt with one decoder per treatment site       ``mednext_duo.py``
===================== ================================================ ==========================
"""

import json

import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler

__all__ = [
    "init_weights",
    "get_scheduler",
    "init_net",
    "define_G",
    "print_network",
    "Identity",
    "get_norm_layer",
    "UnetGenerator3d",
    "UnetSkipConnectionBlock3d",
    "MedNeXtDose",
    "DVHLoss",
    "BhattLoss",
    "WeightedL1Loss",
    "MaskedWeightedL1Loss",
    "MomentLoss",
    "DualGradientL2Loss",
    "BerHuLoss",
]


def init_weights(net, init_type="normal", init_gain=0.02):
    """Initialise convolution / linear / batch-norm weights.

    Args:
        net: the network to initialise in place.
        init_type: ``'normal'``, ``'xavier'``, ``'kaiming'`` or ``'orthogonal'``.
        init_gain: scaling factor for normal, xavier and orthogonal.
    """

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm3d") != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print("initialize network with %s" % init_type)
    net.apply(init_func)


def get_scheduler(optimizer, opt):
    """Return the learning-rate scheduler selected by ``opt.lr_policy``.

    ``linear`` keeps the learning rate for ``n_epochs`` and then decays it
    linearly over ``n_epochs_decay``.  ``step``, ``plateau`` and ``cosine`` use
    the corresponding PyTorch schedulers.
    """
    if opt.lr_policy == "linear":

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(
                opt.n_epochs_decay + 1
            )
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == "step":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.75, threshold=1e-4, patience=3
        )
    elif opt.lr_policy == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        raise NotImplementedError(
            "learning rate policy [%s] is not implemented" % opt.lr_policy
        )
    return scheduler


def init_net(net, init_type="normal", init_gain=0.02, gpu_ids=None):
    """Move ``net`` to the first GPU, wrap it in ``DataParallel`` and init it."""
    if gpu_ids is None:
        gpu_ids = []
    if len(gpu_ids) > 0:
        assert torch.cuda.is_available(), "gpu_ids is non-empty but CUDA is not available"
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(
    input_nc,
    output_nc,
    ngf,
    netG,
    norm="batch",
    use_dropout=False,
    init_type="normal",
    init_gain=0.02,
    gpu_ids=None,
    mednext_model_id="B",
    mednext_kernel_size=3,
    mednext_n_channels=32,
    mednext_sites=("lung", "prostate"),
    netG_cfg=None,
):
    """Build and initialise a full-dose generator.

    Args:
        input_nc: input channels (7 for the anatomy-only layout).
        output_nc: output channels (1 for a dose map).
        ngf: base filter count for the U-Net generators.
        netG: one of ``'unet_128'``, ``'unet_256'``, ``'mednext'``,
            ``'mednext_attn'``, ``'mednext_duo'``.
        norm: normalisation for the U-Net generators only.
        use_dropout: dropout in the U-Net bottleneck.
        init_type / init_gain: weight initialisation.
        gpu_ids: list of GPU ids; empty for CPU.
        mednext_model_id: MedNeXt size preset (``'S'``, ``'B'``, ``'M'``, ``'L'``).
        mednext_kernel_size: MedNeXt depthwise kernel size (3, 5 or 7).
        mednext_n_channels: base filters for ``mednext_attn`` / ``mednext_duo``.
        mednext_sites: decoder names for ``mednext_duo``.
        netG_cfg: optional dict of constructor overrides (JSON text is accepted
            for command-line use).

    Returns:
        Initialised network.
    """
    if gpu_ids is None:
        gpu_ids = []
    if isinstance(netG_cfg, str):
        netG_cfg = json.loads(netG_cfg) if netG_cfg.strip() else {}

    if netG in ("unet_128", "unet_256"):
        num_downs = 7 if netG == "unet_128" else 8
        net = UnetGenerator3d(
            input_nc,
            output_nc,
            num_downs,
            ngf,
            norm_layer=get_norm_layer(norm_type=norm),
            use_dropout=use_dropout,
        )
    elif netG == "mednext":
        net = MedNeXtDose(
            in_channels=input_nc,
            out_channels=output_nc,
            model_id=mednext_model_id,
            kernel_size=mednext_kernel_size,
        )
    elif netG == "mednext_attn":
        from dosepy.models.mednext_attention import MedNeXtDoseAttention

        net = MedNeXtDoseAttention(
            in_channels=input_nc,
            out_channels=output_nc,
            n_channels=mednext_n_channels,
            kernel_size=mednext_kernel_size,
        )
    elif netG == "mednext_duo":
        from dosepy.models.mednext_duo import MedNeXtDuo

        net = MedNeXtDuo(
            in_channels=input_nc,
            out_channels=output_nc,
            n_channels=mednext_n_channels,
            kernel_size=mednext_kernel_size,
            sites=mednext_sites,
        )
    else:
        raise NotImplementedError(
            "Generator model name [%s] is not recognized. "
            "Full-dose options are: unet_128, unet_256, mednext, mednext_attn, mednext_duo."
            % netG
        )

    return init_net(net, init_type, init_gain, gpu_ids)


def print_network(net):
    """Print the architecture and parameter count."""
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print("Total number of parameters: %d" % num_params)


##############################################################################
# Architectures and losses are re-exported so callers can keep using
# ``networks.MomentLoss``, ``networks.UnetGenerator3d``, etc.
##############################################################################
from dosepy.models.unet_generator3d import (  # noqa: E402,F401
    Identity,
    UnetGenerator3d,
    UnetSkipConnectionBlock3d,
    get_norm_layer,
)
from dosepy.models.mednext_dose import MedNeXtDose  # noqa: E402,F401
from dosepy.models.losses3d import (  # noqa: E402,F401
    BerHuLoss,
    BhattLoss,
    DVHLoss,
    DualGradientL2Loss,
    MaskedWeightedL1Loss,
    MomentLoss,
    WeightedL1Loss,
)
