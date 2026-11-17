# DosePy -- Full dose prediction toolkit
#
# Derived from PortPy (https://github.com/PortPy-Project/PortPy), which is
# licensed under Apache 2.0 with the Commons Clause restriction.

"""Training-curve and per-patient metric plots."""

import csv
import os
import matplotlib.pyplot as plt

__all__ = ["load_training_history", "plot_training_history", "plot_patient_metric_distribution"]


def load_training_history(csv_path: str) -> dict:
    """Load ``training_history.csv`` written by :func:`dosepy.train.main`.

    Returns:
        Dict of column name -> list of floats/ints.
    """
    history = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    pass
                history.setdefault(key, []).append(value)
    return history


def plot_training_history(history, figsize=(15, 4.2), save_path: str = None):
    """Plot training/validation loss, MAE and correlation metrics.

    Args:
        history: history dict returned by :func:`dosepy.train.train`, or a path
            to ``training_history.csv``.
        figsize: figure size.
        save_path: optional output path.

    Returns:
        ``(fig, axes)``.
    """
    if isinstance(history, str):
        history = load_training_history(history)

    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    ax = axes[0]
    if "train_loss" in history:
        ax.plot(epochs, history["train_loss"], label="train")
    if "val_loss" in history:
        ax.plot(epochs, history["val_loss"], label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Dose loss")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1]
    if "val_mae" in history:
        ax.plot(epochs, history["val_mae"], color="tab:red")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE (Gy)")
    ax.set_title("Validation MAE (best-checkpoint criterion)")
    ax.grid(alpha=0.25)

    ax = axes[2]
    if "val_r2" in history:
        ax.plot(epochs, history["val_r2"], label="R$^2$")
    if "val_corr" in history:
        ax.plot(epochs, history["val_corr"], label="Pearson r")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Validation agreement")
    finite_r2 = [v for v in history.get("val_r2", []) if isinstance(v, (int, float)) and v == v]
    ax.set_ylim(min([0.0] + finite_r2), 1.02)
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig, axes


def plot_patient_metric_distribution(csv_path: str, metric: str = "mae", figsize=(10, 4.5), save_path: str = None):
    """Bar chart of a per-patient metric written by ``test.py``.

    Args:
        csv_path: ``patient_metrics.csv`` from the test run.
        metric: column to plot (``'mae'``, ``'rmse'``, ``'r2'``, ...).
        figsize: figure size.
        save_path: optional output path.

    Returns:
        ``(fig, ax)``.
    """
    rows = []
    with open(csv_path, "r", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    if not rows:
        raise ValueError("No rows in {}".format(csv_path))
    if metric not in rows[0]:
        raise KeyError("{} not in {}".format(metric, list(rows[0])))

    patients = [row["patient"] for row in rows]
    values = [float(row[metric]) for row in rows]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(len(patients)), values, color="#4C72B0")
    ax.set_xticks(range(len(patients)))
    ax.set_xticklabels(patients, rotation=90, fontsize=7)
    ax.set_ylabel(metric)
    ax.set_title("Per-patient {} (sorted worst first)".format(metric))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig, ax
