# DosePy -- Full dose prediction toolkit
#
# Derived from PortPy (https://github.com/PortPy-Project/PortPy), which is
# licensed under Apache 2.0 with the Commons Clause restriction.

"""DosePy: an anatomy-only full (total) dose predictor.

This is a 7-channel variant of PortPy's total-dose pipeline: the network sees
CT, OAR masks and PTV only.  It does not use a BEAM channel, so prediction does
not require a beam arrangement or an influence matrix.  PortPy's official
8-channel checkpoints are therefore **not** compatible.

The package is split into the same stages as the total-dose pipeline::

    dosepy.preprocess    DICOM -> 128^3 npz, splits, inference, post-processing
    dosepy.data          torch Dataset classes and the 7-channel layout
    dosepy.models        U-Net / MedNeXt generators, losses, training model
    dosepy.options       train / test option namespaces
    dosepy.train         training loop
    dosepy.test          test / inference loop
    dosepy.evaluation    voxel metrics, DVH metrics, plan scorecard
    dosepy.visualization slice/DVH/training plots and a report writer
    dosepy.downstream    fluence optimisation from a predicted dose

PortPy (``portpy.photon``) remains a dependency for patient data access
(``DataExplorer``, ``CT``, ``Structures``, ``Beams``, ``InfluenceMatrix``,
``Plan``, ``Optimization``).  All full-dose-prediction-specific code lives in
this package.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
