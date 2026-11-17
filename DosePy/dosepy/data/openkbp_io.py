# DosePy -- Full dose prediction toolkit
#
# OpenKBP CSV input/output utilities.  This module adapts the public OpenKBP
# sparse-CSV layout; it does not depend on PortPy.

"""Read and cache OpenKBP sparse CSV patient folders.

OpenKBP stores every patient on a fixed ``128 x 128 x 128`` grid:

- ``ct.csv``                  CT values (HU + 1000)
- ``dose.csv``                total dose in Gy
- ``possible_dose_mask.csv``  body / dose-support mask (blank value = 1)
- ``Brainstem.csv`` ...       optional OAR masks (blank value = 1)
- ``PTV56.csv``, ``PTV63.csv``, ``PTV70.csv``
- ``voxel_dimensions.csv``    voxel size in mm

Each CSV has a blank header and two columns: a flattened voxel index and a
value.  This module converts those files into dense arrays and maintains a
``.openkbp_cache`` so repeated epochs do not pay the CSV parsing cost.

The dataset class in :mod:`dosepy.data.openkbp_dataset` only consumes the
converted arrays; all CSV parsing and caching lives here.
"""

import os

import numpy as np
import pandas as pd

__all__ = [
    "OPENKBP_OAR_NAMES",
    "OPENKBP_PTV_NAMES",
    "OPENKBP_CHANNEL_NAMES",
    "OPENKBP_INPUT_NC",
    "OPENKBP_SHAPE",
    "OPENKBP_PRESCRIPTION_GY",
    "PHASE_DIRS",
    "convert_openkbp_patient",
    "load_openkbp_patient",
    "prepare_openkbp_cache",
]

OPENKBP_SHAPE = (128, 128, 128)
OPENKBP_OAR_NAMES = (
    "Brainstem",
    "SpinalCord",
    "RightParotid",
    "LeftParotid",
    "Mandible",
    "Esophagus",
    "Larynx",
)
OPENKBP_PTV_NAMES = ("PTV56", "PTV63", "PTV70")
OPENKBP_CHANNEL_NAMES = ("CT",) + OPENKBP_OAR_NAMES + OPENKBP_PTV_NAMES
OPENKBP_INPUT_NC = len(OPENKBP_CHANNEL_NAMES)  # 11
OPENKBP_PRESCRIPTION_GY = 70.0

# OpenKBP CT values are HU + 1000; normalize to [0, 1].
OPENKBP_CT_MIN = 0.0
OPENKBP_CT_MAX = 4071.0

PHASE_DIRS = {
    "train": "train-pats",
    "val": "validation-pats",
    "validation": "validation-pats",
    "test": "test-pats",
}


def _read_sparse_csv(path: str, shape, mask: bool = False) -> np.ndarray:
    """Read one OpenKBP sparse CSV into a dense z-y-x array."""
    if not os.path.exists(path):
        if not mask:
            raise FileNotFoundError(path)
        return np.zeros(shape, dtype=np.uint8)

    df = pd.read_csv(path, header=0, index_col=0)
    if df.shape[1] == 0:
        return np.zeros(shape, dtype=np.uint8 if mask else np.float32)

    indices = df.index.to_numpy(dtype=np.int64)
    values = df.iloc[:, 0].to_numpy(dtype=np.float64)
    if mask:
        # Empty values mean "this voxel belongs to the mask".
        values = np.where(np.isnan(values), 1.0, values)
        values = (values > 0).astype(np.uint8)
        arr = np.zeros(int(np.prod(shape)), dtype=np.uint8)
    else:
        values = np.nan_to_num(values, nan=0.0)
        arr = np.zeros(int(np.prod(shape)), dtype=np.float32)

    if indices.size and (indices.min() < 0 or indices.max() >= arr.size):
        raise ValueError(
            "{} contains index {}, outside the flat {}^3 grid".format(
                path, int(indices.max()), shape[0]
            )
        )
    arr[indices] = values
    return arr.reshape(shape)


def _read_voxel_dims(patient_dir: str) -> np.ndarray:
    path = os.path.join(patient_dir, "voxel_dimensions.csv")
    if not os.path.exists(path):
        return np.array([3.0, 3.0, 3.0], dtype=np.float32)
    dims = np.loadtxt(path, dtype=np.float32).reshape(-1)
    if dims.size != 3:
        raise ValueError("voxel_dimensions.csv must contain 3 values: {}".format(path))
    return dims.astype(np.float32)


def convert_openkbp_patient(patient_dir: str) -> dict:
    """Read one OpenKBP patient folder into dense arrays.

    Returns:
        Dict with ``CT`` (128,128,128) float32 in [0,1], ``DOSE`` float32 in Gy,
        ``BODY`` uint8, ``OAR`` (7,128,128,128) uint8, ``PTV``
        (3,128,128,128) uint8, ``PRESCRIPTION_GY`` and ``VOXEL_DIMS``.
    """
    ct = _read_sparse_csv(os.path.join(patient_dir, "ct.csv"), OPENKBP_SHAPE)
    dose = _read_sparse_csv(os.path.join(patient_dir, "dose.csv"), OPENKBP_SHAPE)
    body = _read_sparse_csv(
        os.path.join(patient_dir, "possible_dose_mask.csv"), OPENKBP_SHAPE, mask=True
    )

    ct = np.clip(ct, OPENKBP_CT_MIN, OPENKBP_CT_MAX) / OPENKBP_CT_MAX
    dose = np.clip(dose, 0.0, 1.2 * OPENKBP_PRESCRIPTION_GY).astype(np.float32)

    oar = np.stack(
        [_read_sparse_csv(os.path.join(patient_dir, name + ".csv"), OPENKBP_SHAPE, mask=True)
         for name in OPENKBP_OAR_NAMES],
        axis=0,
    ).astype(np.uint8)
    ptv = np.stack(
        [_read_sparse_csv(os.path.join(patient_dir, name + ".csv"), OPENKBP_SHAPE, mask=True)
         for name in OPENKBP_PTV_NAMES],
        axis=0,
    ).astype(np.uint8)

    return {
        "CT": ct.astype(np.float32),
        "DOSE": dose,
        "BODY": body.astype(np.uint8),
        "OAR": oar,
        "PTV": ptv,
        "PRESCRIPTION_GY": np.float32(OPENKBP_PRESCRIPTION_GY),
        "VOXEL_DIMS": _read_voxel_dims(patient_dir),
    }


def _load_cached(path: str) -> dict:
    with np.load(path, allow_pickle=False) as f:
        return {key: f[key] for key in f.files}


def _save_cached(path: str, data: dict):
    """Write atomically so concurrent workers never read a partial file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = "{}.tmp.{}.npz".format(path, os.getpid())
    np.savez_compressed(tmp, **data)
    os.replace(tmp, path)


def load_openkbp_patient(patient_dir: str, cache_path: str = None,
                         use_cache: bool = True) -> dict:
    """Load one patient, preferring a cached ``.npz`` when available."""
    if use_cache and cache_path and os.path.exists(cache_path):
        try:
            return _load_cached(cache_path)
        except Exception:
            # Stale/partial cache: rebuild it below.
            pass

    data = convert_openkbp_patient(patient_dir)
    if use_cache and cache_path:
        try:
            _save_cached(cache_path, data)
        except OSError as exc:
            print("Warning: could not cache {}: {}".format(cache_path, exc))
    return data


def prepare_openkbp_cache(data_root: str, splits=None, overwrite: bool = False,
                          verbose: bool = True) -> dict:
    """Convert every OpenKBP patient in ``splits`` to ``.openkbp_cache``.

    Args:
        data_root: directory that contains ``train-pats``,
            ``validation-pats`` and ``test-pats``.
        splits: split directory names to convert.  Defaults to all three.
        overwrite: rebuild existing cache files.
        verbose: print progress.

    Returns:
        Dict ``{split: number_of_cached_patients}``.
    """
    if splits is None:
        splits = ("train-pats", "validation-pats", "test-pats")

    counts = {}
    for split in splits:
        split_dir = os.path.join(data_root, split)
        cache_dir = os.path.join(data_root, ".openkbp_cache", split)
        if not os.path.isdir(split_dir):
            counts[split] = 0
            if verbose:
                print("OpenKBP split not found: {}".format(split_dir))
            continue

        patients = sorted(
            name for name in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, name))
            and os.path.exists(os.path.join(split_dir, name, "ct.csv"))
        )
        done = 0
        for patient in patients:
            cache_path = os.path.join(cache_dir, patient + ".npz")
            if os.path.exists(cache_path) and not overwrite:
                done += 1
                continue
            patient_dir = os.path.join(split_dir, patient)
            data = convert_openkbp_patient(patient_dir)
            _save_cached(cache_path, data)
            done += 1
            if verbose and (done == 1 or done % 25 == 0 or done == len(patients)):
                print("OpenKBP [{}] cached {}/{}".format(split, done, len(patients)))
        counts[split] = done
    return counts
