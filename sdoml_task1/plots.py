
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


def plot_loss_accuracy(metrics: dict, save_path=None) -> None:
    """Plot training and validation loss/accuracy curves.

    Creates a 2x2 subplot grid showing training loss, training accuracy,
    validation loss, and validation accuracy across epochs.

    Parameters
    ----------
    metrics : dict
        Output from :func:`sdoml_task1.modeling.train.train`.
        Must contain ``"loss_train"``, ``"accuracy_train"``,
        ``"loss_val"``, and ``"accuracy_val"`` keys, each mapping
        to a list of per-epoch values.
    save_path : str or pathlib.Path, optional
        If provided, save the figure to this path.
    """
    _, axs = plt.subplots(2, 2, figsize=(12, 8))
    ax1, ax2, ax3, ax4 = axs.flatten()

    ax1.plot(metrics["loss_train"])
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.grid()

    ax2.plot(metrics["accuracy_train"])
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training Accuracy")
    ax2.grid()

    ax3.plot(metrics["loss_val"])
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Loss")
    ax3.set_title("Validation Loss")
    ax3.grid()

    ax4.plot(metrics["accuracy_val"])
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Accuracy")
    ax4.set_title("Validation Accuracy")
    ax4.grid()

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def plot_confusion_matrices(confusion_matrices: list, save_path=None) -> None:
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

    plt.show()


def plot_confusion_matrix_epoch(cm: np.ndarray, epoch: int, save_path=None) -> None:
    """Plot a single confusion matrix.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix of shape ``(n_classes, n_classes)``.
    epoch : int
        Epoch number (used in the plot title).
    save_path : str or pathlib.Path, optional
        If provided, save the figure to this path.
    """
    _, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title(f"Confusion Matrix - Epoch {epoch}")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
