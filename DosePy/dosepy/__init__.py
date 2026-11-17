# DosePy -- Full dose prediction toolkit
#
# Derived from PortPy (https://github.com/PortPy-Project/PortPy), which is
# licensed under Apache 2.0 with the Commons Clause restriction.

"""DosePy: a full (total) dose prediction toolkit.

Two input paths are supported:

- **PortPy**: anatomy-only 7 channels (CT + 4 OAR + PTV).  It does not use a
  BEAM channel, so prediction does not require a beam arrangement or an
  influence matrix.  PortPy's official 8-channel checkpoints are not compatible.
- **OpenKBP**: 11 channels (CT + 7 OAR + 3 PTV), read directly from OpenKBP
  sparse CSV folders.  No PortPy dependency is needed for this path.

The package is split into the same stages as the total-dose pipeline::

    dosepy.preprocess    DICOM -> 128^3 npz, splits, inference, post-processing
    dosepy.data          torch Dataset classes (PortPy 7-channel / OpenKBP 11-channel)
    dosepy.models        U-Net / MedNeXt generators, losses, training model
    dosepy.options       train / test option namespaces
    dosepy.train         training loop
    dosepy.test          test / inference loop
    dosepy.evaluation    voxel metrics, DVH metrics, plan scorecard
    dosepy.visualization slice/DVH/training plots and a report writer
    dosepy.downstream    fluence optimisation from a predicted dose

The PortPy path still uses ``portpy.photon`` for patient data access
(``DataExplorer``, ``CT``, ``Structures``, ``Beams``, ``InfluenceMatrix``,
``Plan``, ``Optimization``).  The OpenKBP path does not import PortPy.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
