# DosePy -- Full dose prediction toolkit
#
# Derived from PortPy (https://github.com/PortPy-Project/PortPy), which is
# licensed under Apache 2.0 with the Commons Clause restriction.

"""Bundled config files copied from PortPy's ``config_files``.

Contains the lung/prostate scorecards and the frozen train/val/test splits.

Example::

    from dosepy.configs import scorecard_path
    path = scorecard_path("lung")          # SC_Lung(60Gy)_2022MAAS_ExampleV2.json
    path = scorecard_path("prostate")
"""

import json
from pathlib import Path

__all__ = [
    "SCORECARD_DIR",
    "DATA_SPLITS_DIR",
    "scorecard_path",
    "list_scorecards",
    "data_splits_path",
    "load_data_splits",
]

SCORECARD_DIR = Path(__file__).resolve().parent / "scorecard"
DATA_SPLITS_DIR = Path(__file__).resolve().parent / "data_splits"

_SCORECARDS = {
    "lung": "SC_Lung(60Gy)_2022MAAS_ExampleV2.json",
    "prostate": "SC_ProstateSBRT-ARTIA-V1.2(KBPModel).json",
}


def scorecard_path(site: str) -> str:
    """Return the bundled scorecard path for ``'lung'`` or ``'prostate'``."""
    key = str(site).lower()
    if key not in _SCORECARDS:
        raise ValueError("No bundled scorecard for {!r}; available: {}".format(site, sorted(_SCORECARDS)))
    path = SCORECARD_DIR / _SCORECARDS[key]
    if not path.exists():
        raise FileNotFoundError("Bundled scorecard missing: {}".format(path))
    return str(path)


def list_scorecards():
    """Return all ``.json`` scorecard paths bundled with DosePy."""
    return sorted(str(p) for p in SCORECARD_DIR.glob("*.json"))


def data_splits_path(name: str = "default") -> str:
    """Return the bundled PortPy frozen-split JSON path."""
    path = DATA_SPLITS_DIR / "data_splits_{}.json".format(name)
    if not path.exists():
        raise FileNotFoundError("Bundled data split file missing: {}".format(path))
    return str(path)


def load_data_splits(name: str = "default") -> dict:
    """Load the bundled frozen train/val/test splits.

    The file is a copy of PortPy's
    ``config_files/data_splits/data_splits_default.json``.  Bundling it keeps
    DosePy's Task 2 split stable even if the installed PortPy package changes.
    """
    with open(data_splits_path(name), "r") as f:
        return json.load(f)
