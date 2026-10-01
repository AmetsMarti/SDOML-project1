"""Figures for the notebooks and the report: training curves and confusion matrices.

Every helper accepts an optional ``save_path`` and a ``show`` flag, so the same
call works interactively and when writing the plots to ``reports/figures``.
"""

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


def training_metrics_figure(metrics: dict, include_accuracy: bool = True) -> Figure:
    """Build loss and optionally accuracy figures for training and validation."""
    epochs = range(1, len(metrics["loss_train"]) + 1)
    columns = 2 if include_accuracy else 1
    fig, axes = plt.subplots(1, columns, figsize=(6 * columns, 4.5), squeeze=False)
    axes = axes[0]

    axes[0].plot(epochs, metrics["loss_train"], marker="o", markersize=3, label="Train loss")
    axes[0].plot(epochs, metrics["loss_val"], marker="o", markersize=3, label="Validation loss")
    axes[0].set_title("Loss")
    axes[0].set_ylabel("Loss")

    if include_accuracy:
        axes[1].plot(epochs, metrics["accuracy_train"], marker="o", markersize=3, label="Train accuracy")
        axes[1].plot(epochs, metrics["accuracy_val"], marker="o", markersize=3, label="Validation accuracy")
        axes[1].set_title("Accuracy")
        axes[1].set_ylabel("Accuracy")

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_xticks(list(epochs))
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    return fig


def loss_evolution_figure(metrics: dict) -> Figure:
    """Build the train/validation loss figure."""
    return training_metrics_figure(metrics, include_accuracy=False)


def confusion_matrix_figure(cm: np.ndarray, epoch: int) -> Figure:
    """Build a single confusion matrix figure.

    Unlike :func:`plot_confusion_matrix_epoch` this returns the figure
    instead of showing it, so it can be handed over to a UI component.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix of shape ``(n_classes, n_classes)``.
    epoch : int
        Epoch number (used in the plot title).

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"Confusion matrix - Epoch {epoch}")
    fig.tight_layout()
    return fig


def plot_loss_accuracy(metrics: dict, save_path=None, show=True) -> None:
    """Plot training and validation loss/accuracy curves.

    Creates a two-panel figure showing loss and accuracy for training and
    validation across epochs.

    Parameters
    ----------
    metrics : dict
        Output from :func:`sdoml_task1.modeling.train.train`.
        Must contain ``"loss_train"``, ``"accuracy_train"``,
        ``"loss_val"``, and ``"accuracy_val"`` keys, each mapping
        to a list of per-epoch values.
    save_path : str or pathlib.Path, optional
        If provided, save the figure to this path.
    show : bool, default=True
        If True, display the figure. If False, close it without displaying.
    """
    fig = training_metrics_figure(metrics)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        fig.show()
    else:
        plt.close(fig)


def plot_confusion_matrices(confusion_matrices: list, save_path=None, show=True) -> None:
    """Plot confusion matrices for every epoch.

    Displays confusion matrices in a grid with 5 columns, one matrix
    per epoch.

    Parameters
    ----------
    confusion_matrices : list of np.ndarray
        Confusion matrices, one per epoch. Each is of shape
        ``(n_classes, n_classes)``.
    save_path : str or pathlib.Path, optional
        If provided, save the figure to this path.
    show : bool, default=True
        If True, display the figure. If False, close it without displaying.

    """
    n = len(confusion_matrices)
    cols = 5
    rows = (n + cols - 1) // cols

    _, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).flatten()

    for i, cm in enumerate(confusion_matrices):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
        disp.plot(ax=axes[i], cmap="Blues", colorbar=False)
        axes[i].set_title(f"Epoch {i + 1}")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrix_epoch(cm: np.ndarray, epoch: int, save_path=None, show=True) -> None:
    """Plot a single confusion matrix.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix of shape ``(n_classes, n_classes)``.
    epoch : int
        Epoch number (used in the plot title).
    save_path : str or pathlib.Path, optional
        If provided, save the figure to this path.
    show : bool, default=True
        If True, display the figure. If False, close it without displaying.
    """
    _, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title(f"Confusion Matrix - Epoch {epoch}")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()
