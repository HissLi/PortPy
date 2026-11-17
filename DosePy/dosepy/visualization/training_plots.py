# DosePy -- Full dose prediction toolkit
#
# Derived from PortPy (https://github.com/PortPy-Project/PortPy), which is
# licensed under Apache 2.0 with the Commons Clause restriction.

"""Training-curve and per-patient metric plots."""

import csv
import os
import matplotlib.pyplot as plt

__all__ = [
    "load_training_history",
    "plot_training_history",
    "plot_patient_metric_distribution",
    "LossCurvePlotter",
]


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


class LossCurvePlotter:
    """Live training-loss figure plus a per-step loss log.

    Every optimizer step is appended to ``<out_dir>/train_step_loss.csv`` and
    flushed immediately, so the full loss trace survives a crash.  A figure is
    refreshed every ``freq`` epochs and saved to ``<out_dir>/loss_curve.png``:

    - left panel: training loss per optimizer step;
    - middle panel: epoch-average training loss and validation loss;
    - right panel: validation MAE.

    The step CSV is written even when plotting is disabled, because it is a log
    rather than a visualisation.

    Args:
        out_dir: run directory.
        enabled: draw/save the PNG.  The step CSV is still written.
        freq: redraw/save the PNG every ``freq`` epochs.  ``close`` always
            saves the final figure, so the complete curve is kept.
        title: figure title (usually the experiment name).
        dpi: output resolution.
    """

    def __init__(self, out_dir: str, enabled: bool = True, freq: int = 1,
                 title: str = "Training", dpi: int = 150):
        self.enabled = bool(enabled)
        self.freq = max(1, int(freq))
        self.out_dir = out_dir
        self.out_path = os.path.join(out_dir, "loss_curve.png")
        self.step_csv_path = os.path.join(out_dir, "train_step_loss.csv")
        self.title = title
        self.dpi = dpi
        self.epochs = []
        self.train_loss = []
        self.val_loss = []
        self.val_mae = []
        self.steps = []
        self.step_losses = []
        self._step_file = None
        self._step_writer = None
        self._step_fields = None
        self.fig = None
        self.ax_step = None
        self.ax_epoch = None
        self.ax_mae = None
        self.interactive = self._is_interactive()
        if self.enabled:
            print("Saving live training loss curve to: {}".format(self.out_path))
        print("Saving per-step training loss to: {}".format(self.step_csv_path))

    @staticmethod
    def _is_interactive() -> bool:
        try:
            import matplotlib
            backend = matplotlib.get_backend().lower()
        except Exception:
            return False
        return backend not in ("agg", "pdf", "ps", "svg", "template", "cairo")

    def _open_step_log(self, losses):
        os.makedirs(self.out_dir, exist_ok=True)
        self._step_fields = ["global_step", "epoch", "step", "loss"] + sorted(losses)
        self._step_file = open(self.step_csv_path, "w", newline="")
        self._step_writer = csv.DictWriter(
            self._step_file, fieldnames=self._step_fields, extrasaction="ignore"
        )
        self._step_writer.writeheader()
        self._step_file.flush()

    def record_step(self, epoch: int, step: int, global_step: int, losses: dict):
        """Append one optimizer step to ``train_step_loss.csv`` immediately."""
        if self._step_writer is None:
            self._open_step_log(losses)

        total_loss = float(sum(float(v) for v in losses.values()))
        row = {
            "global_step": int(global_step),
            "epoch": int(epoch),
            "step": int(step),
            "loss": total_loss,
        }
        for key, value in losses.items():
            row[key] = float(value)
        self._step_writer.writerow(row)
        self._step_file.flush()

        self.steps.append(int(global_step))
        self.step_losses.append(total_loss)

    def _ensure_figure(self):
        if self.fig is None:
            if self.interactive:
                try:
                    plt.ion()
                except Exception:
                    self.interactive = False
            self.fig, (self.ax_step, self.ax_epoch, self.ax_mae) = plt.subplots(
                1, 3, figsize=(15, 4.2)
            )

    def update(self, epoch: int, train_loss: float,
               val_loss: float = None, val_mae: float = None):
        """Record one epoch and refresh the figure when due."""
        if not self.enabled:
            return
        self.epochs.append(int(epoch))
        self.train_loss.append(float(train_loss))
        self.val_loss.append(None if val_loss is None else float(val_loss))
        self.val_mae.append(None if val_mae is None else float(val_mae))

        if epoch % self.freq != 0:
            return
        self._redraw()

    def _redraw(self):
        if not self.enabled or (not self.epochs and not self.steps):
            return
        self._ensure_figure()

        self.ax_step.clear()
        if self.steps:
            self.ax_step.plot(self.steps, self.step_losses,
                              color="tab:blue", linewidth=0.8)
        else:
            self.ax_step.plot(self.epochs, self.train_loss,
                              color="tab:blue", label="train")
        self.ax_step.set_xlabel("Optimizer step")
        self.ax_step.set_ylabel("Loss")
        self.ax_step.set_title("Training loss (per step)")
        self.ax_step.grid(alpha=0.25)
        if not self.steps:
            self.ax_step.legend()

        self.ax_epoch.clear()
        self.ax_epoch.plot(self.epochs, self.train_loss,
                           label="train (epoch mean)", color="tab:blue")
        if any(v is not None for v in self.val_loss):
            self.ax_epoch.plot(self.epochs, self.val_loss,
                               label="val", color="tab:orange")
        self.ax_epoch.set_xlabel("Epoch")
        self.ax_epoch.set_ylabel("Loss")
        self.ax_epoch.set_title("Epoch loss")
        self.ax_epoch.grid(alpha=0.25)
        self.ax_epoch.legend()

        self.ax_mae.clear()
        if any(v is not None for v in self.val_mae):
            self.ax_mae.plot(self.epochs, self.val_mae, color="tab:red")
        self.ax_mae.set_xlabel("Epoch")
        self.ax_mae.set_ylabel("MAE (Gy)")
        self.ax_mae.set_title("Validation MAE")
        self.ax_mae.grid(alpha=0.25)

        self.fig.suptitle(self.title)
        self.fig.tight_layout()
        os.makedirs(self.out_dir, exist_ok=True)
        self.fig.savefig(self.out_path, dpi=self.dpi, bbox_inches="tight")
        if self.interactive:
            try:
                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()
            except Exception:
                self.interactive = False

    def close(self):
        """Save the final figure, close the step log, and release resources."""
        try:
            if self.enabled:
                self._redraw()
        finally:
            if self._step_file is not None:
                self._step_file.close()
                self._step_file = None
                self._step_writer = None
            if self.fig is not None:
                plt.close(self.fig)
                self.fig = None


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
