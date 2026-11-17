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

"""Evaluation / inference loop for full dose prediction.

Two modes share this module:

- ``model='doseprediction3d'`` + ``dataset_mode='dosepred3d'``: the test split
  has ground-truth dose, so the loop also computes loss, MAE, RMSE, R^2,
  Pearson correlation and nRMSE, and writes ``patient_metrics.csv``;
- ``model='test'`` + ``dataset_mode='single'``: inference only (used by
  :func:`dosepy.preprocess.predict.predict_using_model`), predicted dose
  volumes are written as ``<Patient>_CT2DOSE.nrrd``.

Both modes write predictions and an HTML index under
``<results_dir>/<name>/<phase>_<epoch>/``; the dose volumes are stored as
``.nrrd`` and can be plotted with :mod:`dosepy.visualization`.
"""

import csv
import os

from dosepy.data import create_dataset
from dosepy.models import create_model
from dosepy.options import TestOptions
from dosepy.utils import html
from dosepy.utils.visualizer import save_images

__all__ = ["test", "main", "run_dose_prediction_test"]

# Output file name used by the inference reader in ``preprocess/predict.py``.
NPY_OUT_FNAMES = ["CT2DOSE"]


def _get_web_dir(opt):
    web_dir = os.path.join(opt.results_dir, opt.name, "{}_{}".format(opt.phase, opt.epoch))
    if opt.load_iter > 0:
        web_dir = "{:s}_iter{:d}".format(web_dir, opt.load_iter)
    return web_dir


def _get_path_name(a_paths):
    if isinstance(a_paths, (list, tuple)):
        return os.path.basename(a_paths[0])
    return os.path.basename(a_paths)


def _calculate_metrics(model):
    """Return loss and regression metrics for the current sample."""
    model.calculate_validation_loss()
    losses = model.get_current_losses()
    case_loss = sum(losses.values())

    if hasattr(model, "calculate_validation_metrics"):
        metrics = model.calculate_validation_metrics()
        val_mae = float(metrics["mae"].item())
        val_rmse = float(metrics["rmse"].item())
        val_r2 = float(metrics["r2"].item())
        val_corr = float(metrics["corr"].item())
        val_nrmse = float(metrics["nrmse"].item())
    else:
        val_mae = float(model.calculate_validation_mae().item())
        val_rmse = val_r2 = val_corr = val_nrmse = float("nan")

    return {
        "loss": float(case_loss),
        "mae": val_mae,
        "rmse": val_rmse,
        "r2": val_r2,
        "corr": val_corr,
        "nrmse": val_nrmse,
    }


def run_dose_prediction_test(opt, dataset, model):
    """Run the test split, save predictions and return a summary dict."""
    web_dir = _get_web_dir(opt)
    os.makedirs(web_dir, exist_ok=True)
    print("creating web directory", web_dir)

    webpage = html.HTML(
        web_dir,
        "Experiment = %s, Phase = %s, Epoch = %s" % (opt.name, opt.phase, opt.epoch),
    )

    rows = []
    total_loss = 0.0
    total_mae = 0.0
    n_eval = 0

    for i, data in enumerate(dataset):
        model.set_input(data)
        model.test()

        has_gt = "B" in data and data["B"] is not None
        if has_gt:
            try:
                metrics = _calculate_metrics(model)
                patient_name = _get_path_name(data["A_paths"])
                patient_name = os.path.splitext(patient_name)[0]
                rows.append({"patient": patient_name, **metrics})
                total_loss += metrics["loss"]
                total_mae += metrics["mae"]
                n_eval += 1
            except Exception as e:  # keep inference outputs even if metrics fail
                print("Skipping metric evaluation for this case: {}".format(e))

        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        if i % 5 == 0:
            print("processing (%04d)-th image... %s" % (i, img_path))

        save_images(
            webpage,
            visuals,
            img_path,
            aspect_ratio=opt.aspect_ratio,
            width=opt.display_winsize,
            npy_out_fnames=NPY_OUT_FNAMES,
        )

    webpage.save()

    summary = {
        "web_dir": web_dir,
        "n_eval": n_eval,
        "avg_loss": total_loss / n_eval if n_eval else float("nan"),
        "avg_mae": total_mae / n_eval if n_eval else float("nan"),
        "patients": sorted(rows, key=lambda r: r["mae"], reverse=True),
    }

    if n_eval > 0:
        print("Avg Test Loss: {}".format(summary["avg_loss"]))
        print("Avg Test MAE: {}".format(summary["avg_mae"]))
        metrics_file = os.path.join(web_dir, "patient_metrics.csv")
        with open(metrics_file, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["patient", "loss", "mae", "rmse", "r2", "corr", "nrmse"],
            )
            writer.writeheader()
            writer.writerows(summary["patients"])
        print("Saved per-patient metrics to {}".format(metrics_file))
    else:
        print("No GT available. Only inference outputs were saved.")

    return summary


def main(opt):
    """Create dataset + model and run the test loop."""
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1

    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)
    if opt.eval:
        model.eval()

    return run_dose_prediction_test(opt, dataset, model)


def test(args=None):
    """Run testing / inference.

    Args:
        args: optional dict of test options.  When ``None`` the CLI arguments
            are parsed.

    Returns:
        Summary dict from :func:`run_dose_prediction_test`.
    """
    if args is None:
        opt = TestOptions().parse()
    else:
        overrides = dict(args)
        # TestOptions defaults to model='test' -> dataset_mode='single'.  When
        # the caller switches back to the supervised model, make the matching
        # dataset the default unless it was set explicitly.
        if overrides.get("model") == "doseprediction3d" and "dataset_mode" not in overrides:
            overrides["dataset_mode"] = "dosepred3d"
        opt = TestOptions().parse(argv=[], overrides=overrides)

    return main(opt)


if __name__ == "__main__":
    test()
