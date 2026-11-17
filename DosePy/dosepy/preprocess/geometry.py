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

"""Geometry helpers shared by preprocessing and inference.

PortPy's full-dose pipeline always works on a fixed ``128 x 128 x 128`` grid:

1. find the calculation bounding box of the patient (from PortPy metadata),
2. crop the CT / masks / dose volume to that box,
3. resample the cropped volume to ``128^3`` (linear for intensities, nearest
   for masks).

At inference time the reverse operation is needed: resize the predicted
``128^3`` dose back to the cropped CT grid, then paste it into a full-size CT
grid.  All three steps live here so that preprocessing, inference and
post-processing cannot drift apart (in PortPy these helpers were duplicated in
three modules, and one copy was missing the negative-index clamp).
"""

import os
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
from skimage.transform import resize

__all__ = [
    "MODEL_SHAPE",
    "get_ct_image",
    "get_crop_settings_calc_box",
    "get_crop_settings",
    "crop_resize_img",
    "resample_dose",
    "attach_slices",
    "get_structure_masks",
    "write_image",
]

#: Spatial shape expected by every network in this package.
MODEL_SHAPE: Tuple[int, int, int] = (128, 128, 128)


def get_ct_image(ct) -> sitk.Image:
    """Convert a ``portpy.photon.CT`` object to a SimpleITK image.

    The CT array is stored z-y-x (slices first), while SimpleITK uses x-y-z;
    ``GetImageFromArray`` already performs that reversal, so origin/spacing/
    direction can be copied from the metadata unchanged.
    """
    ct_arr = ct.ct_dict["ct_hu_3d"][0]
    ct_image = sitk.GetImageFromArray(ct_arr)
    ct_image.SetOrigin(ct.ct_dict["origin_xyz_mm"])
    ct_image.SetSpacing(ct.ct_dict["resolution_xyz_mm"])
    ct_image.SetDirection(ct.ct_dict["direction"])
    return ct_image


def get_crop_settings_calc_box(ct, meta_data: dict) -> Tuple[List[int], List[int]]:
    """Return the calculation-box crop as ``(start_zyx, end_zyx)``.

    The calculation box is stored in physical x-y-z mm.  It can extend one or
    two voxels beyond the CT; ``TransformPhysicalPointToIndex`` then returns a
    negative index.  Left unclamped, Python turns that into a negative slice
    (e.g. ``img[511:509]``), which is empty, and the subsequent resize divides
    by a zero-length axis.  Clamp to the CT bounds exactly as PortPy does.

    Note:
        Axis 0 (slices) is sliced inclusively as ``start[0]:end[0] + 1`` by
        :func:`crop_resize_img`; axes 1 and 2 are exclusive.  Therefore the
        upper clamp is ``size - 1`` for axis 0 and ``size`` for axes 1 and 2.
    """
    cal_box_xyz_start = meta_data["opt_voxels"]["cal_box_xyz_start"]
    cal_box_xyz_end = meta_data["opt_voxels"]["cal_box_xyz_end"]
    ct_img = get_ct_image(ct)

    start_xyz = ct_img.TransformPhysicalPointToIndex(cal_box_xyz_start)  # X, Y, Z
    end_xyz = ct_img.TransformPhysicalPointToIndex(cal_box_xyz_end)
    start_zyx = [start_xyz[2], start_xyz[1], start_xyz[0]]
    end_zyx = [end_xyz[2], end_xyz[1], end_xyz[0]]

    size_zyx = ct_img.GetSize()[::-1]  # SimpleITK reports (x, y, z)
    start_zyx = [max(0, int(v)) for v in start_zyx]
    end_zyx = [
        min(int(end_zyx[0]), size_zyx[0] - 1),
        min(int(end_zyx[1]), size_zyx[1]),
        min(int(end_zyx[2]), size_zyx[2]),
    ]
    return start_zyx, end_zyx


def get_crop_settings(oar: np.ndarray) -> Tuple[List[int], List[int]]:
    """Legacy crop heuristic based on the OAR mask extent.

    Kept for users who do not have optimization-voxel metadata.  The main
    pipeline uses :func:`get_crop_settings_calc_box`, which is what PortPy's
    model was trained with.

    Args:
        oar: 3-D integer OAR label map (label 1 = cord, 2 = esophagus).
    """
    oar1 = oar.copy()
    oar1[np.where(oar == 1)] = 0
    oar1[np.where(oar == 2)] = 0

    # For 2-D cropping keep a centred 300 x 300 window.
    center = [0, oar.shape[1] // 2, oar1.shape[2] // 2]
    start = [0, center[1] - 150, center[2] - 150]
    end = [0, center[1] + 150, center[2] + 150]

    depth = oar1.shape[0]
    if depth < 128:
        start[0] = 0
        end[0] = depth
        return start, end

    first_slice = -1
    last_slice = -1
    for i in range(depth):
        if np.any(oar1[i]):
            first_slice = i
            break
    for i in range(depth - 1, -1, -1):
        if np.any(oar1[i]):
            last_slice = i
            break

    expanse = last_slice - first_slice + 1
    if expanse >= 128:
        start[0] = first_slice
        end[0] = last_slice
        return start, end

    slices_needed = 128 - expanse
    end_slices = slices_needed // 2
    beg_slices = slices_needed - end_slices

    end_room_available = depth - last_slice - 1
    beg_room_available = first_slice

    leftover_beg = beg_room_available - beg_slices
    if leftover_beg < 0:
        end_slices += np.abs(leftover_beg)
        first_slice = 0
    else:
        first_slice = first_slice - beg_slices

    leftover_end = end_room_available - end_slices
    if leftover_end < 0:
        first_slice -= np.abs(leftover_end)
        last_slice = depth - 1
    else:
        last_slice = last_slice + end_slices

    if first_slice < 0:
        first_slice = 0

    start[0] = first_slice
    end[0] = last_slice
    return start, end


def crop_resize_img(
    img: np.ndarray,
    start: Sequence[int],
    end: Sequence[int],
    is_mask: bool = False,
    target_shape: Tuple[int, int, int] = MODEL_SHAPE,
) -> np.ndarray:
    """Crop a z-y-x volume to ``[start, end]`` and resize it to ``target_shape``.

    Args:
        img: 3-D array in z-y-x order.
        start: inclusive start indices ``(z, y, x)``.
        end: end indices; axis 0 is inclusive (``end[0] + 1``), axes 1 and 2
            are exclusive.
        is_mask: use nearest-neighbour interpolation and return ``uint8``.
        target_shape: output shape, ``(128, 128, 128)`` by default.

    Returns:
        Resized array with shape ``target_shape`` in z-y-x order.
    """
    img_cropped = img[start[0]:end[0] + 1, start[1]:end[1], start[2]:end[2]]
    img_cropped = np.moveaxis(img_cropped, 0, -1)  # slices last for skimage

    order = 0 if is_mask else 1
    img_resized = resize(
        img_cropped,
        target_shape,
        order=order,
        preserve_range=True,
        anti_aliasing=False,
    ).astype(np.float32)
    if is_mask:
        img_resized = img_resized.astype(np.uint8)

    return np.moveaxis(img_resized, -1, 0)  # slices first again


def resample_dose(dose: np.ndarray, ref_shape: Sequence[int]) -> np.ndarray:
    """Resize a 3-D dose volume to ``ref_shape`` (z-y-x).

    Used to map the network output (``128^3``) back onto the cropped CT grid.
    ``skimage.transform.resize`` expects channels-last, hence the two
    ``moveaxis`` calls.
    """
    dose = np.asarray(dose)
    if tuple(dose.shape) == tuple(ref_shape):
        return dose.astype(np.float32, copy=False)

    dims = tuple(ref_shape)
    dose = np.moveaxis(dose, 0, -1)
    expected_shape = (dims[1], dims[2], dims[0])
    dose = resize(
        dose,
        expected_shape,
        order=1,
        preserve_range=True,
        anti_aliasing=False,
    )
    return np.moveaxis(dose, -1, 0)


def attach_slices(
    pred_dose: np.ndarray,
    ct_img: sitk.Image,
    start: Sequence[int],
    end: Sequence[int],
) -> sitk.Image:
    """Paste a cropped prediction back into a full-size CT grid.

    Voxels outside the calculation box stay at zero, and the resulting image
    inherits the CT geometry (origin, spacing, direction).
    """
    ct_arr = sitk.GetArrayFromImage(ct_img)
    ref_dose_arr = np.zeros_like(ct_arr, dtype=np.float64)
    ref_dose_arr[start[0]:end[0] + 1, start[1]:end[1], start[2]:end[2]] = pred_dose

    dose = sitk.GetImageFromArray(ref_dose_arr)
    dose.SetOrigin(ct_img.GetOrigin())
    dose.SetDirection(ct_img.GetDirection())
    dose.SetSpacing(ct_img.GetSpacing())
    return dose


def get_structure_masks(
    structs,
    ct_shape: Sequence[int],
    labels: dict,
    disease_struct_names: Iterable[str],
    case: str = "",
):
    """Rasterise the site's OAR and PTV structures into two label maps.

    Args:
        structs: ``portpy.photon.Structures`` object.
        ct_shape: shape of the CT array (z, y, x).
        labels: mapping structure name -> integer label.
        disease_struct_names: structures to rasterise; the PTV may be anywhere
            in the iterable, it is always written to a separate mask.
        case: patient id used only in warnings.

    Returns:
        ``(oar_mask, ptv_mask)``, both ``uint8`` arrays of ``ct_shape``.

    Note:
        ``LUNGS_NOT_GTV`` is built from ``LUNG_L | LUNG_R - GTV`` when the
        structure is absent, matching PortPy's preprocessing.
    """
    struct_names = structs.structures_dict["name"]
    struct_names_upper = [s.upper() for s in struct_names]

    oar_mask = np.zeros(ct_shape, np.uint8)
    ptv_mask = np.zeros_like(oar_mask)

    for name in disease_struct_names:
        if name == "lungs_not_gtv":
            if "LUNGS_NOT_GTV" in struct_names_upper:
                ind = struct_names_upper.index("LUNGS_NOT_GTV")
                anatomy_mask = structs.structures_dict["structure_mask_3d"][ind]
            else:
                anatomy_mask = np.zeros(ct_shape, dtype=bool)

                if "LUNG_L" in struct_names_upper:
                    ind_l = struct_names_upper.index("LUNG_L")
                    anatomy_mask = np.logical_or(
                        anatomy_mask,
                        structs.structures_dict["structure_mask_3d"][ind_l] > 0,
                    )
                if "LUNG_R" in struct_names_upper:
                    ind_r = struct_names_upper.index("LUNG_R")
                    anatomy_mask = np.logical_or(
                        anatomy_mask,
                        structs.structures_dict["structure_mask_3d"][ind_r] > 0,
                    )
                if "GTV" in struct_names_upper:
                    ind_gtv = struct_names_upper.index("GTV")
                    gtv_mask = structs.structures_dict["structure_mask_3d"][ind_gtv] > 0
                    anatomy_mask = np.logical_and(anatomy_mask, ~gtv_mask)

            oar_mask[np.where(anatomy_mask > 0)] = labels[name]

        elif name.upper() in struct_names_upper:
            ind = struct_names_upper.index(name.upper())
            anatomy_mask = structs.structures_dict["structure_mask_3d"][ind]

            if name == "ptv":
                ptv_mask[np.where(anatomy_mask > 0)] = labels[name]
            else:
                oar_mask[np.where(anatomy_mask > 0)] = labels[name]
        else:
            print(
                "Structure {} not found for case {}. Leaving it as a zero mask.".format(
                    name, case
                )
            )

    return oar_mask, ptv_mask


def write_image(img_arr: np.ndarray, out_dir: str, case: str, suffix: str, ref_ct: sitk.Image):
    """Write a numpy volume with the geometry of ``ref_ct`` (legacy helper)."""
    img_itk = sitk.GetImageFromArray(img_arr)
    img_itk.SetOrigin(ref_ct.GetOrigin())
    img_itk.SetSpacing(ref_ct.GetSpacing())
    img_itk.SetDirection(ref_ct.GetDirection())
    filename = os.path.join(out_dir, case + suffix)
    sitk.WriteImage(img_itk, filename)
    return filename
