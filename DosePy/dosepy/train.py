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

"""Training loop for full dose prediction.

The loop mirrors PortPy's ``ai/train.py``:

1. train for one epoch with the selected dose loss,
2. evaluate the validation split with the same loss plus the challenge-style
   MAE, R^2, Pearson correlation and nRMSE,
3. step the learning-rate scheduler on the validation MAE,
4. checkpoint ``latest``, numbered epochs and the best validation MAE.

DosePy differences (behaviour is otherwise the same):

- the option dict is parsed with ``argv=[]``, so Jupyter kernel arguments do
  not leak into the parser;
- loss/metric history is written to one CSV file (not several ``.xlsx`` files,
  which required ``openpyxl``);
- every optimizer step is written and flushed to
  ``<checkpoints_dir>/<name>/train_step_loss.csv``;
- a live loss curve is drawn and saved to ``<checkpoints_dir>/<name>/loss_curve.png``
  during training (enabled by default; disable with ``plot_loss=False`` or
  ``--no_plot_loss``);
- :func:`train` returns the history dict, ready for
  :func:`dosepy.visualization.plot_training_history`.
"""

import copy
import csv
import os
import sys
import time

import torch

from dosepy.data import create_dataset
from dosepy.models import create_model
from dosepy.options import TrainOptions
from dosepy.utils.visualizer import Visualizer
from dosepy.visualization.training_plots import LossCurvePlotter

__all__ = ["train", "main"]


class _DualOutput:
    """Tee stdout/stderr to the console and ``console_output.txt``."""

    def __init__(self, original_stdout, log_file):
        self.original_stdout = original_stdout
        self.log_file = log_file

    def write(self, message):
        self.original_stdout.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.original_stdout.flush()
        self.log_file.flush()


def _evaluate_validation(model, dataset_val):
    """Run the validation split and return averaged loss/metrics."""
    total_loss = 0.0
    total_mae = 0.0
    total_r2 = 0.0
    total_corr = 0.0
    total_nrmse = 0.0
    n_samples = 0

    model.eval()
    for data_test in dataset_val:
        with torch.no_grad():
            model.set_input(data_test)
            model.test()
            model.calculate_validation_loss()
            losses = model.get_current_losses()
            mae = model.calculate_validation_mae().item()

            if hasattr(model, "calculate_validation_metrics"):
                metrics = model.calculate_validation_metrics()
                r2 = float(metrics["r2"].item())
                corr = float(metrics["corr"].item())
                nrmse = float(metrics["nrmse"].item())
            else:
                r2 = corr = nrmse = float("nan")

        total_loss += sum(losses.values())
        total_mae += mae
        total_r2 += r2
        total_corr += corr
        total_nrmse += nrmse
        n_samples += 1

    if n_samples == 0:
        raise RuntimeError("The validation split is empty. Check dataroot/val_phase.")

    return {
        "loss": total_loss / n_samples,
        "mae": total_mae / n_samples,
        "r2": total_r2 / n_samples,
        "corr": total_corr / n_samples,
        "nrmse": total_nrmse / n_samples,
    }


def main(opt):
    """Run training and return the per-epoch history dict."""
    torch.cuda.empty_cache()
    run_dir = os.path.join(opt.checkpoints_dir, opt.name)
    os.makedirs(run_dir, exist_ok=True)

    console_log_path = os.path.join(run_dir, "console_output.txt")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_file = open(console_log_path, "a")
    sys.stdout = _DualOutput(original_stdout, log_file)
    sys.stderr = _DualOutput(original_stdout, log_file)

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_mae": [],
        "val_r2": [],
        "val_corr": [],
        "val_nrmse": [],
    }
    loss_plotter = None

    try:
        print("Saving console output to: {}".format(console_log_path))

        dataset = create_dataset(opt)
        dataset_size = len(dataset)
        print("The number of training images = %d" % dataset_size)
        print("shuffle: {}".format(not opt.serial_batches))

        # Validation loader: no shuffling, no augmentation, batch size 1.
        opt_val = copy.deepcopy(opt)
        opt_val.phase = getattr(opt, "val_phase", "val")
        opt_val.num_threads = 0
        opt_val.batch_size = 1
        opt_val.serial_batches = True
        opt_val.no_flip = True
        opt_val.display_id = -1
        opt_val.isTrain = False
        dataset_val = create_dataset(opt_val)
        print("The number of validation images = %d" % len(dataset_val))

        model = create_model(opt)
        model.setup(opt)
        visualizer = Visualizer(opt)

        loss_plotter = LossCurvePlotter(
            out_dir=run_dir,
            enabled=getattr(opt, "plot_loss", True),
            freq=getattr(opt, "plot_freq", 1),
            title=opt.name,
        )

        total_iters = 0
        best_val_mae = float("inf")
        best_val_mae_epoch = -1
        total_time_start = time.time()

        history_path = os.path.join(run_dir, "training_history.csv")
        with open(history_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["epoch", "train_loss", "val_loss", "val_mae", "val_r2", "val_corr", "val_nrmse"]
            )
            f.flush()

            for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
                epoch_start_time = time.time()
                iter_data_time = time.time()
                epoch_iter = 0
                visualizer.reset()
                if hasattr(model, "set_epoch"):
                    model.set_epoch(epoch)

                total_loss_epoch = 0.0
                for data in dataset:
                    iter_start_time = time.time()
                    if total_iters % opt.print_freq == 0:
                        t_data = iter_start_time - iter_data_time

                    total_iters += opt.batch_size
                    epoch_iter += opt.batch_size
                    model.set_input(data)
                    model.optimize_parameters()

                    if total_iters % opt.display_freq == 0:
                        save_result = total_iters % opt.update_html_freq == 0
                        model.compute_visuals()
                        visualizer.display_current_results(
                            model.get_current_visuals(), epoch, save_result
                        )

                    losses = model.get_current_losses()
                    if loss_plotter is not None:
                        # Persist every optimizer step immediately; this is
                        # independent of whether the PNG plot is enabled.
                        loss_plotter.record_step(
                            epoch, epoch_iter, total_iters, losses
                        )
                    total_loss_epoch += sum(losses.values())
                    if total_iters % opt.print_freq == 0:
                        t_comp = (time.time() - iter_start_time) / opt.batch_size
                        visualizer.print_current_losses(
                            epoch, epoch_iter, losses, t_comp, t_data
                        )

                    if total_iters % opt.save_latest_freq == 0:
                        print(
                            "saving the latest model (epoch %d, total_iters %d)"
                            % (epoch, total_iters)
                        )
                        save_suffix = "iter_%d" % total_iters if opt.save_by_iter else "latest"
                        model.save_networks(save_suffix)

                    iter_data_time = time.time()

                if epoch % opt.save_epoch_freq == 0:
                    print(
                        "saving the model at the end of epoch %d, iters %d"
                        % (epoch, total_iters)
                    )
                    model.save_networks("latest")
                    model.save_networks(epoch)

                train_loss = total_loss_epoch / max(len(dataset), 1)
                print(
                    "End of epoch %d / %d \t Time Taken: %d sec"
                    % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time)
                )
                print("Avg Loss at the end of epoch {}: {}".format(epoch, train_loss))

                val = _evaluate_validation(model, dataset_val)
                print(
                    "Avg Val Loss at the end of epoch {}: {}".format(epoch, val["loss"])
                )
                print("Avg Val MAE  at the end of epoch {}: {}".format(epoch, val["mae"]))
                print("Avg Val R2   at the end of epoch {}: {}".format(epoch, val["r2"]))
                print("Avg Val Corr at the end of epoch {}: {}".format(epoch, val["corr"]))
                print("Avg Val nRMSE at the end of epoch {}: {}".format(epoch, val["nrmse"]))

                model.metric = val["mae"]
                model.update_learning_rate()
                model.train()

                if val["mae"] < best_val_mae:
                    best_val_mae = val["mae"]
                    best_val_mae_epoch = epoch
                    print(
                        "New best validation MAE at epoch {}: {:.6f}".format(
                            epoch, best_val_mae
                        )
                    )
                    model.save_networks("best_mae")

                history["epoch"].append(epoch)
                history["train_loss"].append(float(train_loss))
                history["val_loss"].append(float(val["loss"]))
                history["val_mae"].append(float(val["mae"]))
                history["val_r2"].append(float(val["r2"]))
                history["val_corr"].append(float(val["corr"]))
                history["val_nrmse"].append(float(val["nrmse"]))

                writer.writerow(
                    [
                        epoch,
                        train_loss,
                        val["loss"],
                        val["mae"],
                        val["r2"],
                        val["corr"],
                        val["nrmse"],
                    ]
                )
                f.flush()

                if loss_plotter is not None:
                    loss_plotter.update(
                        epoch, train_loss, val["loss"], val["mae"]
                    )

        total_time = time.time() - total_time_start
        print("Total Time for training is {} seconds".format(total_time))
        print(
            "Best validation MAE: {} at epoch {}".format(best_val_mae, best_val_mae_epoch)
        )
        with open(os.path.join(run_dir, "loss_log.txt"), "a") as f:
            print("Total Time for training is {} seconds".format(total_time), file=f)
            print(
                "Best validation MAE: {} at epoch {}".format(
                    best_val_mae, best_val_mae_epoch
                ),
                file=f,
            )

        return history

    finally:
        try:
            if loss_plotter is not None:
                loss_plotter.close()
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_file.close()


def train(args=None):
    """Train a full-dose model.

    Args:
        args: optional dict of training options.  When ``None`` the CLI
            arguments are parsed.

    Returns:
        History dict with the keys ``epoch``, ``train_loss``, ``val_loss``,
        ``val_mae``, ``val_r2``, ``val_corr`` and ``val_nrmse``.
    """
    if args is None:
        opt = TrainOptions().parse()
    else:
        opt = TrainOptions().parse(argv=[], overrides=args)
    return main(opt)


if __name__ == "__main__":
    train()
