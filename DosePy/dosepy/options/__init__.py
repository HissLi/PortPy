# DosePy -- Full dose prediction toolkit
#
# Derived from PortPy (https://github.com/PortPy-Project/PortPy), which is
# licensed under Apache 2.0 with the Commons Clause restriction.

"""Option classes and the notebook-friendly ``build_opt`` helper."""

from dosepy.options.base_options import BaseOptions
from dosepy.options.test_options import TestOptions
from dosepy.options.train_options import TrainOptions

__all__ = ["BaseOptions", "TrainOptions", "TestOptions", "build_opt"]


def build_opt(options_cls, overrides=None, argv=None):
    """Build an option namespace from a dict.

    Args:
        options_cls: ``TrainOptions`` or ``TestOptions`` (or a subclass).
        overrides: dict of option values.  When given, ``argv`` defaults to
            ``[]`` so the Jupyter kernel's arguments are ignored.
        argv: explicit argparse argument list.  ``None`` reads ``sys.argv``.

    Returns:
        Parsed ``argparse.Namespace``.

    Example:
        >>> from dosepy.options import TrainOptions, build_opt
        >>> opt = build_opt(TrainOptions, {"dataroot": "data", "n_epochs": 2})
    """
    if argv is None and overrides is not None:
        argv = []
    return options_cls().parse(argv=argv, overrides=overrides)
