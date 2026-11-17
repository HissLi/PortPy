# DosePy -- Full dose prediction toolkit
#
# Derived from PortPy (https://github.com/PortPy-Project/PortPy), which is
# licensed under Apache 2.0 with the Commons Clause restriction.

"""Turn a predicted dose into a deliverable fluence map.

This is the downstream step of PortPy's ``imrt_dose_prediction`` example::

    min_x  ||A_PTV x - pred_PTV / nfx||^2
         + w_OAR * || max(A_OAR x - pred_OAR / nfx, 0) ||^2
         + w_s   * smoothness(x)
         s.t.   x >= 0

where ``A`` is the dose influence matrix and ``pred`` is the total (all
fractions) predicted dose.  ``nfx`` converts the total dose to a per-fraction
dose, because PortPy's optimisation model works per fraction.

The function requires PortPy (for ``Plan`` / ``Optimization`` /
``InfluenceMatrix``) and CVXPY with a solver (MOSEK or a free solver).
"""

from typing import Mapping, Sequence

import numpy as np

__all__ = ["optimize_fluence_to_prediction", "dose_from_fluence"]


def dose_from_fluence(fluence: np.ndarray, inf_matrix, num_fractions: float) -> np.ndarray:
    """Return the total (all-fraction) 1-D dose of a fluence map."""
    return np.asarray(inf_matrix.A @ np.asarray(fluence)) * float(num_fractions)


def optimize_fluence_to_prediction(
    my_plan,
    pred_dose_1d: np.ndarray,
    num_fractions: float = None,
    ptv_weight: float = 1.0,
    oar_weight: float = 1.0,
    smoothness_weights: Sequence[float] = (0.6, 0.4),
    solver: str = "MOSEK",
    verbose: bool = False,
    **solver_kwargs,
) -> Mapping[str, object]:
    """Optimise beamlet fluence to reproduce a predicted dose.

    Args:
        my_plan: ``portpy.photon.Plan``.  Its influence matrix is used.
        pred_dose_1d: predicted total dose on the plan's optimisation voxels,
            obtained with ``inf_matrix.dose_3d_to_1d(pred_dose_3d)``.
        num_fractions: number of fractions; defaults to
            ``my_plan.get_num_of_fractions()`` (requires clinical criteria).
        ptv_weight: weight of the PTV quadratic term.
        oar_weight: weight of the OAR one-sided quadratic term.
        smoothness_weights: ``(x_weight, y_weight)`` for the fluence
            smoothness terms.
        solver: CVXPY solver, e.g. ``'MOSEK'``, ``'SCS'`` or ``'CLARABEL'``.
        verbose: CVXPY verbosity.
        **solver_kwargs: extra keyword arguments for ``cvxpy.Problem.solve``.

    Returns:
        Dict with ``fluence``, ``sol``, ``dose_1d`` (total dose),
        ``pred_dose_1d`` and ``num_fractions``.

    Raises:
        ValueError: if PTV voxels are missing, shapes mismatch, or the solver
            returns no solution.
    """
    import cvxpy as cp
    import portpy.photon as pp

    inf_matrix = my_plan.inf_matrix
    A = inf_matrix.A
    pred = np.asarray(pred_dose_1d, dtype=np.float64).ravel()

    if pred.shape[0] != A.shape[0]:
        raise ValueError(
            "pred_dose_1d has {} voxels but the influence matrix has {} rows".format(
                pred.shape[0], A.shape[0]
            )
        )

    if num_fractions is None:
        num_fractions = my_plan.get_num_of_fractions()
    num_fractions = float(num_fractions)

    ptv_vox = inf_matrix.get_opt_voxels_idx("PTV")
    if len(ptv_vox) == 0:
        raise ValueError("No PTV optimisation voxels; cannot run fluence optimisation.")

    # No clinical objectives/constraints here: this problem is defined by the
    # predicted dose only.  ``solve`` builds the CVXPY problem lazily.
    opt = pp.Optimization(my_plan, inf_matrix=inf_matrix, opt_params=None)
    x = opt.vars["x"]

    # PTV: match the prediction.
    opt.obj += [
        (ptv_weight / len(ptv_vox))
        * cp.sum_squares(A[ptv_vox, :] @ x - pred[ptv_vox] / num_fractions)
    ]

    # OAR / normal tissue: stay below the prediction (one-sided).
    all_vox = np.arange(A.shape[0])
    oar_vox = all_vox[~np.isin(all_vox, ptv_vox)]
    d_oar = cp.Variable(oar_vox.shape[0], pos=True)
    opt.constraints += [
        A[oar_vox, :] @ x <= pred[oar_vox] / num_fractions + d_oar
    ]
    opt.obj += [(oar_weight / max(len(oar_vox), 1)) * cp.sum_squares(d_oar)]

    # Fluence smoothness.
    qx, qy, num_rows, num_cols = opt.get_smoothness_matrix(inf_matrix.beamlets_dict)
    smooth_x, smooth_y = smoothness_weights
    opt.obj += [
        smooth_x * (1.0 / max(num_cols, 1)) * cp.sum_squares(qx @ x)
        + smooth_y * (1.0 / max(num_rows, 1)) * cp.sum_squares(qy @ x)
    ]

    sol = opt.solve(solver=solver, verbose=verbose, **solver_kwargs)
    fluence = sol["optimal_intensity"]
    if fluence is None:
        raise RuntimeError("Fluence optimisation returned no solution.")

    return {
        "fluence": fluence,
        "sol": sol,
        "dose_1d": dose_from_fluence(fluence, inf_matrix, num_fractions),
        "pred_dose_1d": pred,
        "num_fractions": num_fractions,
    }
