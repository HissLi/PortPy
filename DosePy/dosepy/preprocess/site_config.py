# DosePy -- Full dose prediction toolkit
#
# Derived from PortPy (https://github.com/PortPy-Project/PortPy), which is
# licensed under Apache 2.0 with the Commons Clause restriction. The original
# copyright notice is retained below.
#
# Copyright 2025, the PortPy Authors
#
# Licensed under the Apache License, Version 2.0 with the Commons Clause restriction.
# You may obtain a copy of the Apache 2 License at:
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

"""Disease-site configuration for full (total) dose prediction.

DosePy is an **anatomy-only** full-dose predictor: a site definition fixes three
things that must stay identical between preprocessing, training and inference:

1. the OAR structures that are rasterised into the input tensor,
2. the integer label used for every structure in the ``OAR`` array, and
3. the prescription dose used to clip the planner dose.

The channel layout is 7 channels (no BEAM channel)::

    input channel 0 : CT, clipped to [-1000, 3071] HU and scaled to [0, 1]
    input channels 1..5 : one-hot OAR labels 1..5 (label 5 is unused)
    input channel 6 : PTV mask

``transform_3d_data`` implements this layout with
``one_hot(oar, 6)[..., 1:]`` followed by concatenating the PTV channel, so the
6 structure channels are ``[label 1, label 2, label 3, label 4, unused, PTV]``.
Do not change it when re-using a DosePy checkpoint.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

__all__ = [
    "SiteProfile",
    "SITE_PROFILES",
    "get_site_profile",
    "get_site_config",
    "get_channel_names",
    "get_oar_label_map",
    "get_dvh_structure_names",
]


@dataclass(frozen=True)
class SiteProfile:
    """Immutable description of one disease site.

    Attributes:
        site: lower-case site name (``'lung'`` or ``'prostate'``).
        disease_struct_names: structures rasterised by ``get_structure_masks``.
            The last entry is always the PTV.
        labels: integer label for each OAR (PTV is stored in a separate mask).
        prescription_gy: fallback prescription when no protocol is given.
        oar_channels: ``(name, label)`` pairs in the order they appear in the
            one-hot OAR block of the input tensor.
        dvh_structures: structures used by the default DVH / metric tables.
        default_protocol: protocol name understood by PortPy's config files.
    """

    site: str
    disease_struct_names: Tuple[str, ...]
    labels: Dict[str, int]
    prescription_gy: float
    oar_channels: Tuple[Tuple[str, int], ...]
    dvh_structures: Tuple[str, ...]
    default_protocol: str


SITE_PROFILES: Dict[str, SiteProfile] = {
    "lung": SiteProfile(
        site="lung",
        disease_struct_names=("cord", "esophagus", "heart", "lungs_not_gtv", "ptv"),
        labels={"cord": 1, "esophagus": 2, "heart": 3, "lungs_not_gtv": 4, "ptv": 1},
        prescription_gy=60.0,
        oar_channels=(
            ("CORD", 1),
            ("ESOPHAGUS", 2),
            ("HEART", 3),
            ("LUNGS_NOT_GTV", 4),
        ),
        dvh_structures=("PTV", "ESOPHAGUS", "HEART", "CORD", "LUNGS_NOT_GTV"),
        default_protocol="Lung_2Gy_30Fx",
    ),
    "prostate": SiteProfile(
        site="prostate",
        disease_struct_names=("bladder", "rectum", "femur_l", "femur_r", "ptv"),
        labels={"bladder": 1, "rectum": 2, "femur_l": 3, "femur_r": 4, "ptv": 1},
        prescription_gy=70.2,
        oar_channels=(
            ("BLADDER", 1),
            ("RECTUM", 2),
            ("FEMUR_L", 3),
            ("FEMUR_R", 4),
        ),
        dvh_structures=("PTV", "BLADDER", "RECTUM", "FEMUR_L", "FEMUR_R"),
        default_protocol="Prostate_26Fx",
    ),
}


def get_site_profile(site: str) -> SiteProfile:
    """Return the :class:`SiteProfile` for ``site``.

    Args:
        site: ``'lung'`` or ``'prostate'`` (case insensitive).

    Raises:
        ValueError: if the site is not supported.
    """
    key = str(site).lower()
    if key not in SITE_PROFILES:
        raise ValueError(
            "site must be one of {}, got {!r}".format(sorted(SITE_PROFILES), site)
        )
    return SITE_PROFILES[key]


def get_site_config(site: str):
    """Backward-compatible PortPy signature.

    Returns:
        ``(disease_struct_names, labels, default_prescription_gy)`` where
        ``disease_struct_names`` is a list, exactly as
        ``portpy.ai.preprocess.data_preprocess.get_site_config`` returned it.
    """
    profile = get_site_profile(site)
    return list(profile.disease_struct_names), dict(profile.labels), profile.prescription_gy


def get_channel_names(site: str) -> List[str]:
    """Return the 7 input-channel names for ``site``.

    Useful for debugging a network input and for figure titles.
    """
    profile = get_site_profile(site)
    names = ["CT"]
    names += ["OAR[{}]".format(label) for _, label in profile.oar_channels]
    names += ["OAR[unused]", "PTV"]
    return names


def get_oar_label_map(site: str) -> Dict[int, str]:
    """Map OAR integer label -> structure name (``0`` is background)."""
    profile = get_site_profile(site)
    mapping = {0: "BACKGROUND"}
    mapping.update({label: name for name, label in profile.oar_channels})
    return mapping


def get_dvh_structure_names(site: str) -> List[str]:
    """Default structure list for DVH plots and metric tables."""
    return list(get_site_profile(site).dvh_structures)
