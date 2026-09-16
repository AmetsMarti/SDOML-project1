"""Training loop for classification models.

This module implements the training loop for the digit classification model,
including validation evaluation and confusion matrix computation at each epoch.

Functions
---------
train(model, train_loader, val_loader, epochs, lr, device)
    Train a classifier and return per-epoch metrics.
"""
from sklearn.metrics import confusion_matrix
import torch
from torch import nn


def train(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    epochs: int = 40,
    lr: float = 1e-4,
    device=None,
) -> dict:
    """Train a classifier and return metrics per epoch.

    Uses Adam optimizer and cross-entropy loss. Tracks training and
    validation loss, accuracy, and confusion matrices at each epoch.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train.
    train_loader : torch.utils.data.DataLoader
        Training data loader.
    val_loader : torch.utils.data.DataLoader
        Validation data loader.
    epochs : int, default=40
        Number of training epochs.
    lr : float, default=1e-4
        Learning rate for Adam optimizer.
    device : str or torch.device, optional
        Device to use for training. If ``None``, auto-detects CUDA or CPU.

    Returns
    -------
    dict
        Dictionary containing:

        - ``"loss_train"`` : list of float -- Training loss per epoch.
        - ``"accuracy_train"`` : list of float -- Training accuracy per epoch.
        - ``"loss_val"`` : list of float -- Validation loss per epoch.
        - ``"accuracy_val"`` : list of float -- Validation accuracy per epoch.
        - ``"confusion_matrices"`` : list of np.ndarray -- Confusion matrices per epoch.
        - ``"model"`` : torch.nn.Module -- The trained model.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    loss_train, accuracy_train = [], []
    loss_val, accuracy_val = [], []
    confusion_matrices = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        all_preds, all_labels = [], []

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * xb.size(0)
            preds = out.argmax(1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds)
        confusion_matrices.append(cm)

        loss_train.append(epoch_loss / total)
        accuracy_train.append(correct / total)

        model.eval()
        epoch_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = loss_fn(out, yb)
                epoch_loss += loss.item() * xb.size(0)
                preds = out.argmax(1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        loss_val.append(epoch_loss / total)
        accuracy_val.append(correct / total)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {loss_train[-1]:.4f} | Train Acc: {accuracy_train[-1]:.4f} | "
            f"Val Loss: {loss_val[-1]:.4f} | Val Acc: {accuracy_val[-1]:.4f}"
        )

    return {
        "loss_train": loss_train,
        "accuracy_train": accuracy_train,
        "loss_val": loss_val,
        "accuracy_val": accuracy_val,
        "confusion_matrices": confusion_matrices,
        "model": model,
    }
