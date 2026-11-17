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

"""Model registry for full dose prediction.

Two models are available:

``doseprediction3d``
    Training/validation model.  Computes the selected dose loss against the
    planner dose and exposes validation metrics.
``test``
    Inference-only wrapper.  Loads the same generator, ignores the target dose
    and only produces ``fake_Dose``.

Adding a custom model is the same as in PortPy/pix2pix: create
``dosepy/models/<name>_model.py`` with a class ``<Name>Model`` inheriting
:class:`~dosepy.models.base_model.BaseModel`, then pass ``model=<name>``.
"""

import importlib

from dosepy.models.base_model import BaseModel
from dosepy.models import networks3d  # noqa: F401  (kept importable as dosepy.models.networks3d)
from dosepy.models.doseprediction3d_model import DosePrediction3DModel
from dosepy.models.test_model import TestModel

__all__ = [
    "BaseModel",
    "networks3d",
    "DosePrediction3DModel",
    "TestModel",
    "find_model_using_name",
    "get_option_setter",
    "create_model",
]


def find_model_using_name(model_name):
    """Import ``dosepy.models.<model_name>_model`` and return its model class."""
    model_filename = "dosepy.models." + model_name + "_model"
    modellib = importlib.import_module(model_filename)

    model = None
    target_model_name = model_name.replace("_", "") + "model"
    for name, cls in modellib.__dict__.items():
        if (
            name.lower() == target_model_name.lower()
            and isinstance(cls, type)
            and issubclass(cls, BaseModel)
        ):
            model = cls

    if model is None:
        raise NotImplementedError(
            "In {} there should be a subclass of BaseModel with class name "
            "matching {} (case insensitive).".format(model_filename, target_model_name)
        )
    return model


def get_option_setter(model_name):
    """Return the model class' static ``modify_commandline_options``."""
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_options


def create_model(opt):
    """Create the model described by ``opt.model``."""
    model = find_model_using_name(opt.model)
    instance = model(opt)
    print("model [%s] was created" % type(instance).__name__)
    return instance
