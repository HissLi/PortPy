# DosePy -- Full dose prediction toolkit
#
# Derived from PortPy (https://github.com/PortPy-Project/PortPy), which is
# licensed under Apache 2.0 with the Commons Clause restriction.
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

"""A PortPy data explorer that does not require beam metadata.

PortPy's ``DataExplorer.load_metadata`` always reads the patient's ``Beams/``
folder.  For an anatomy-only predictor that folder is irrelevant, and requiring
it would make "no beam input" only half true.  This subclass overrides
``load_metadata`` so the anatomy files are always read and beam metadata is
loaded **only when present**:

- ``StructureSet_MetaData.json`` (required)
- ``OptimizationVoxels_MetaData.json`` (required)
- ``CT_MetaData.json`` (required)
- ``PlannerBeams.json`` / ``Beams/`` (optional; needed only by downstream tools)

``portpy.photon.CT`` and ``portpy.photon.Structures`` both call
``data.load_metadata()``, so passing this object to them is enough; no changes
to PortPy are required.
"""

import os

import portpy.photon as pp
from natsort import natsorted

__all__ = ["AnatomyOnlyDataExplorer"]


class AnatomyOnlyDataExplorer(pp.DataExplorer):
    """``DataExplorer`` variant that does not require ``Beams/`` metadata.

    It is API-compatible with ``portpy.photon.DataExplorer`` for the fields
    used by ``CT``, ``Structures`` and the calculation-box helpers.

    Example::

        data = AnatomyOnlyDataExplorer(data_dir="/path/to/data")
        data.patient_id = "Lung_Patient_94"
        ct = pp.CT(data)
        structs = pp.Structures(data)
    """

    def load_metadata(self, pat_dir: str = None) -> dict:
        if pat_dir is None:
            pat_dir = os.path.join(self.data_dir, self.patient_id)

        meta_data = {}

        fname = os.path.join(pat_dir, "StructureSet_MetaData.json")
        meta_data["structures"] = self.list_to_dict(self.load_json(fname))

        fname = os.path.join(pat_dir, "OptimizationVoxels_MetaData.json")
        meta_data["opt_voxels"] = self.list_to_dict(self.load_json(fname))

        fname = os.path.join(pat_dir, "CT_MetaData.json")
        if os.path.isfile(fname):
            meta_data["ct"] = self.list_to_dict(self.load_json(fname))

        # Optional beam metadata.  It is deliberately not required: prediction
        # must work on an anatomy-only patient folder.  When present, it lets
        # downstream scorecard / optimization build a Plan.
        fname = os.path.join(pat_dir, "PlannerBeams.json")
        if os.path.isfile(fname):
            meta_data["planner_beam_ids"] = self.list_to_dict(self.load_json(fname))

        beam_folder = os.path.join(pat_dir, "Beams")
        if os.path.isdir(beam_folder):
            beam_jsons = natsorted(
                [f for f in os.listdir(beam_folder) if f.endswith(".json")]
            )
            if beam_jsons:
                meta_data["beams"] = {}
                for beam_json in beam_jsons:
                    json_data = self.load_json(os.path.join(beam_folder, beam_json))
                    for key, value in json_data.items():
                        meta_data["beams"].setdefault(key, []).append(value)

        meta_data["patient_folder_path"] = pat_dir
        return meta_data
