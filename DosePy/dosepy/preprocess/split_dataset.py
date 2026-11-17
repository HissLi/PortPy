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

"""Create the ``train/``, ``val/`` and ``test/`` folders used by ``dataroot``.

``data_preprocess`` writes every patient flat into one directory.  The dataset
classes, however, expect::

    <dataroot>/train/Lung_Patient_2.npz
    <dataroot>/val/Lung_Patient_91.npz
    <dataroot>/test/Lung_Patient_81.npz

This module applies PortPy's frozen, patient-level benchmark split, of which a
copy is bundled at ``dosepy/configs/data_splits/data_splits_default.json``, and
copies (default) or moves the files into place.  Using the frozen split is important: PortPy.AI
Task 2 (total dose prediction) deliberately swaps the lung validation and test
sets relative to Task 1 (beamlet dose prediction), and re-using the Task 1
split for Task 2 silently leaks patients.
"""

import json
import os
import shutil
from typing import Dict, List, Mapping, Optional, Sequence

from dosepy.configs import load_data_splits

__all__ = [
    "get_benchmark_split",
    "split_preprocessed_dataset",
    "check_split_integrity",
]

TASK = "total_dose_prediction"


def get_benchmark_split(site: str, task: str = TASK) -> Dict[str, List[str]]:
    """Return PortPy's frozen patient split for ``site``.

    Args:
        site: ``'lung'`` or ``'prostate'``.
        task: PortPy benchmark task.  Keep the default
            ``'total_dose_prediction'`` for this package.

    Returns:
        ``{'train': [...], 'val': [...], 'test': [...]}`` of patient ids.
    """
    config = load_data_splits()
    if task not in config:
        raise ValueError(
            "task must be one of {}, got {!r}".format(
                [k for k in config if isinstance(config[k], dict) and "lung" in config[k]], task
            )
        )
    if site not in config[task]:
        raise ValueError(
            "site must be one of {}, got {!r}".format(
                [k for k in config[task] if not k.startswith("_")], site
            )
        )
    splits = config[task][site]
    return {key: list(splits[key]) for key in ("train", "val", "test")}


def split_preprocessed_dataset(
    out_dir: str,
    site: Optional[str] = None,
    splits: Optional[Mapping[str, Sequence[str]]] = None,
    task: str = TASK,
    mode: str = "copy",
    overwrite: bool = False,
    verbose: bool = True,
) -> Dict[str, List[str]]:
    """Move/copy flat ``.npz`` files into train/val/test sub-folders.

    Args:
        out_dir: directory that contains the flat ``<PatientId>.npz`` files.
        site: used to load the benchmark split when ``splits`` is ``None``.
        splits: optional custom mapping ``{'train': [...], ...}``.  Patient ids
            may be given with or without the ``.npz`` suffix.
        task: benchmark task passed to :func:`get_benchmark_split`.
        mode: ``'copy'`` (default, keeps the flat files), ``'move'``, or
            ``'symlink'``.
        overwrite: replace an existing destination file.
        verbose: print per-split counts.

    Returns:
        Mapping ``split -> [destination paths]`` for the files that were
        present and placed.  A ``split_manifest.json`` is written next to the
        split folders for reproducibility.

    Raises:
        ValueError: for an unknown mode, overlapping patient lists, or when
            neither ``site`` nor ``splits`` is given.
    """
    if mode not in ("copy", "move", "symlink"):
        raise ValueError("mode must be 'copy', 'move' or 'symlink', got {!r}".format(mode))

    if splits is None:
        if site is None:
            raise ValueError("Provide either site=... or splits=...")
        splits = get_benchmark_split(site=site, task=task)

    # Normalise ids and check that a patient is not in two partitions.
    normalised: Dict[str, List[str]] = {}
    seen: Dict[str, str] = {}
    for split_name in ("train", "val", "test"):
        ids = []
        for pid in splits.get(split_name, []):
            pid = str(pid).replace(".npz", "")
            if pid in seen:
                raise ValueError(
                    "Patient {} appears in both {} and {}".format(pid, seen[pid], split_name)
                )
            seen[pid] = split_name
            ids.append(pid)
        normalised[split_name] = ids

    os.makedirs(out_dir, exist_ok=True)
    placed: Dict[str, List[str]] = {}

    for split_name, patient_ids in normalised.items():
        split_dir = os.path.join(out_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        dest_paths: List[str] = []

        for pid in patient_ids:
            src = os.path.join(out_dir, pid + ".npz")
            dst = os.path.join(split_dir, pid + ".npz")

            if not os.path.exists(src):
                if os.path.exists(dst):
                    # Already split in an earlier run.
                    dest_paths.append(dst)
                    continue
                print("Missing preprocessed file for {}: {}".format(pid, src))
                continue

            if os.path.exists(dst) and not overwrite:
                dest_paths.append(dst)
                continue

            if mode == "copy":
                shutil.copy2(src, dst)
            elif mode == "move":
                shutil.move(src, dst)
            else:  # symlink
                if os.path.exists(dst) or os.path.islink(dst):
                    os.remove(dst)
                os.symlink(os.path.abspath(src), dst)
            dest_paths.append(dst)

        placed[split_name] = dest_paths
        if verbose:
            print("{}: {} file(s)".format(split_name, len(dest_paths)))

    manifest = {
        "task": task,
        "site": site,
        "mode": mode,
        "splits": normalised,
    }
    with open(os.path.join(out_dir, "split_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return placed


def check_split_integrity(out_dir: str) -> Dict[str, int]:
    """Count ``.npz`` files per split and check for duplicate patient ids.

    This is a cheap sanity check to run after :func:`split_preprocessed_dataset`.
    """
    counts: Dict[str, int] = {}
    seen = set()
    duplicates = []
    for split_name in ("train", "val", "test"):
        split_dir = os.path.join(out_dir, split_name)
        files = []
        if os.path.isdir(split_dir):
            files = [f for f in os.listdir(split_dir) if f.endswith(".npz")]
        counts[split_name] = len(files)
        for f in files:
            if f in seen:
                duplicates.append(f)
            seen.add(f)
    if duplicates:
        print("Warning: duplicate patient files across splits: {}".format(sorted(set(duplicates))))
    return counts
