# DosePy -- Full dose prediction toolkit
#
# Derived from PortPy (https://github.com/PortPy-Project/PortPy), which is
# licensed under Apache 2.0 with the Commons Clause restriction.

"""Adapters to PortPy's plan-level visualisation.

PortPy's ``Visualization.plot_dvh`` / ``plot_2d_slice`` / ``plot_fluence_3d``
need a ``portpy.photon.Plan`` (for the influence matrix, structure masks and
prescription).  Those functions are large and tightly coupled to PortPy's plan
model, so DosePy keeps thin adapters instead of duplicating them.  Everything
DosePy-specific (prediction vs reference slices, DVH from raw arrays, metric
bars) lives in :mod:`dosepy.visualization.dose_plots` and
:mod:`dosepy.visualization.dvh_plots` and does **not** need a Plan.
"""

__all__ = ["plot_plan_dvh", "plot_plan_2d_slice", "plot_plan_fluence_3d"]


def plot_plan_dvh(
    my_plan,
    sol: dict = None,
    dose_1d=None,
    struct_names=None,
    **kwargs,
):
    """Plot a DVH with PortPy's plan-aware renderer.

    Args:
        my_plan: ``portpy.photon.Plan``.
        sol: optional solution dict with ``inf_matrix`` /
            ``optimal_intensity``.
        dose_1d: optional 1-D dose in Gy.  Use this to plot a predicted dose.
        struct_names: structures to include.
        **kwargs: forwarded to ``portpy.photon.Visualization.plot_dvh``
            (``style``, ``ax``, ``norm_flag``, ``dose_scale``, ...).

    Returns:
        The matplotlib axis used by PortPy (``plot_dvh`` returns ``ax``).
    """
    import portpy.photon as pp

    return pp.Visualization.plot_dvh(
        my_plan, sol=sol, dose_1d=dose_1d, struct_names=struct_names, **kwargs
    )


def plot_plan_2d_slice(
    my_plan,
    sol: dict = None,
    dose_1d=None,
    slice_num: int = None,
    struct_names=None,
    **kwargs,
):
    """Show a CT slice with dose and structures via PortPy's renderer."""
    import portpy.photon as pp

    return pp.Visualization.plot_2d_slice(
        my_plan=my_plan,
        sol=sol,
        dose_1d=dose_1d,
        slice_num=slice_num,
        struct_names=struct_names,
        **kwargs,
    )


def plot_plan_fluence_3d(beam_id: int, sol: dict = None, **kwargs):
    """3-D fluence plot for one beam via PortPy's renderer."""
    import portpy.photon as pp

    return pp.Visualization.plot_fluence_3d(beam_id=beam_id, sol=sol, **kwargs)
