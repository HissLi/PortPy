# DosePy -- Full dose prediction toolkit
#
# Derived from PortPy (https://github.com/PortPy-Project/PortPy), which is
# licensed under Apache 2.0 with the Commons Clause restriction.

"""Plots for preprocessing checks, dose comparison, DVHs and training curves."""

from dosepy.visualization.dose_plots import (
    plot_preprocessed_sample,
    plot_slice_comparison,
    select_slice,
)
from dosepy.visualization.dvh_plots import (
    plot_dvh_comparison,
    plot_structure_metric_bars,
)
from dosepy.visualization.training_plots import (
    load_training_history,
    plot_patient_metric_distribution,
    plot_training_history,
)
from dosepy.visualization.report import save_evaluation_report
from dosepy.visualization.portpy_plots import (
    plot_plan_2d_slice,
    plot_plan_dvh,
    plot_plan_fluence_3d,
)

__all__ = [
    "select_slice",
    "plot_slice_comparison",
    "plot_preprocessed_sample",
    "plot_dvh_comparison",
    "plot_structure_metric_bars",
    "plot_training_history",
    "load_training_history",
    "plot_patient_metric_distribution",
    "save_evaluation_report",
    "plot_plan_dvh",
    "plot_plan_2d_slice",
    "plot_plan_fluence_3d",
]
