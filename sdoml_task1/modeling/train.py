"""Training loop for classification models.

This module implements the training loop for the digit classification model,
with optional L1 penalty, weight decay, early stopping and lr scheduling.
"""
from sklearn.metrics import confusion_matrix
import torch
from torch import nn

import numpy as np

from torch.utils.data import DataLoader

from sdoml_task1.dataset import AudioMNISTFeaturesDataset
from sdoml_task1.modeling.model import Net

from sdoml_task1.modeling.optimization import make_optimizer, make_scheduler

from pathlib import Path
import pickle

def train(model, train_loader, val_loader, epochs=40, lr=1e-4, optimizer_name="Adam", scheduler_name=None, limit_epoch=10, device=None):
    """Train a classifier and return metrics per epoch.

    The regularization is read from ``model.regularization``:
    ``"l1"`` adds an L1 penalty to the loss, ``"weight_decay"`` is passed
    to the optimizer, ``"early_stopping"`` stops training when the
    validation loss has not improved for ``limit_epoch`` epochs, and
    ``"dropout"`` is already handled inside the model.

    Parameters
    ----------
    model : Net
        Model to train.
    train_loader : torch.utils.data.DataLoader
        Training data loader.
    val_loader : torch.utils.data.DataLoader
        Validation data loader.
    epochs : int, default=40
        Maximum number of training epochs.
    lr : float, default=1e-4
        Initial learning rate.
    optimizer_name : str, default="Adam"
        One of :data:`sdoml_task1.modeling.optimization.OPTIMIZERS`.
    scheduler_name : str or None, default=None
        One of :data:`sdoml_task1.modeling.optimization.SCHEDULERS`.
    limit_epoch : int, default=10
        Patience for early stopping.
    device : str or torch.device, optional
        If None, uses CUDA when available, otherwise CPU.

    Returns
    -------
    dict
        ``"loss_train"``, ``"accuracy_train"``, ``"loss_val"``, ``"accuracy_val"``
        (lists of float, one per epoch), ``"confusion_matrices"`` (list of
        validation confusion matrices, one per epoch) and ``"model"``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    if model.regularization == "weight_decay":
        optimizer = make_optimizer(optimizer_name, model.parameters(), lr, weight_decay=model.reg_param)
    else:
        optimizer = make_optimizer(optimizer_name, model.parameters(), lr)

    scheduler = make_scheduler(scheduler_name, optimizer)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    confusion_matrices = []
    best_val_loss = float('inf')
    epoch_counter = 0

    for epoch in range(epochs):
        
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            total_loss += loss.item() * xb.size(0)

            if model.regularization == "l1":
                l1 = 0
                for p in model.parameters():
                    l1 = l1 + p.abs().sum()
                loss = loss + model.reg_param * l1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = out.argmax(1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

        train_losses.append(total_loss / total)
        train_accs.append(correct / total)

        
       
        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = criterion(out, yb)
                total_loss += loss.item() * xb.size(0)

                preds = out.argmax(1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(yb.cpu().numpy())

        val_loss = total_loss / total
        val_losses.append(val_loss)
        val_accs.append(correct / total)
        confusion_matrices.append(confusion_matrix(all_labels, all_preds, labels=list(range(10))))

        if scheduler is not None:
            scheduler.step()

        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_losses[-1]:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_accs[-1]:.4f}')

        
        if model.regularization == "early_stopping":
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epoch_counter = 0
            else:
                epoch_counter += 1
            if epoch_counter >= limit_epoch:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return {
        "loss_train": train_losses,
        "accuracy_train": train_accs,
        "loss_val": val_losses,
        "accuracy_val": val_accs,
        "confusion_matrices": confusion_matrices,
        "model": model,
    }

def save_run(metrics, save_dir):
    """Save a trained model and its training history.

    Creates ``model.pt`` (the model weights) and ``history.pkl``
    (per-epoch metrics and the configuration) inside ``save_dir``.

    Parameters
    ----------
    metrics : dict
        Output of :func:`run_training`.
    save_dir : str or pathlib.Path
        Folder where the files are written. Created if it does not exist.

    Examples
    --------
    >>> metrics = run_training(X_train, y_train, X_test, y_test, epochs=10)
    >>> save_run(metrics, PROJECT_DIR / "models")
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.save(metrics["model"].state_dict(), save_dir / "model.pt")

    history = {
        "total_loss_train": metrics["loss_train"],
        "total_accuracy_train": metrics["accuracy_train"],
        "total_loss_valid": metrics["loss_val"],
        "total_accuracy_valid": metrics["accuracy_val"],
        "all_confusion_matrices": metrics["confusion_matrices"],
        "config": metrics["config"],
    }

    with open(save_dir / "history.pkl", "wb") as f:
        pickle.dump(history, f)

    print(f"Model and history saved in {save_dir}")
    
def run_training(X_train, y_train, X_test, y_test, hidden_sizes=(64,), activation_function="relu", regularization=None, reg_param=0.0, optimizer_name="Adam", scheduler_name=None, epochs=40, lr=1e-4, batch_size=64, limit_epoch=10, seed=42):
    """Build the data loaders and the model, then train it.

    This is the single entry point used by both the training notebook
    and the Gradio demo, so that both run exactly the same code.

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training features ``(n_train, 26)`` and labels ``(n_train,)``.
    X_test, y_test : np.ndarray
        Test features ``(n_test, 26)`` and labels ``(n_test,)``.
    hidden_sizes : list of int, default=(64,)
        Number of neurons in each hidden layer, e.g. ``[64, 32]``.
    activation_function : str, default="relu"
        ``"relu"``, ``"sigmoid"`` or ``"tanh"``.
    regularization : str or None, default=None
        ``"l1"``, ``"weight_decay"``, ``"early_stopping"``, ``"dropout"`` or None.
    reg_param : float, default=0.0
        Strength of the regularization (dropout probability for ``"dropout"``).
    optimizer_name : str, default="Adam"
        One of :data:`sdoml_task1.modeling.optimization.OPTIMIZERS`.
    scheduler_name : str or None, default=None
        One of :data:`sdoml_task1.modeling.optimization.SCHEDULERS`.
    epochs : int, default=40
        Maximum number of training epochs.
    lr : float, default=1e-4
        Initial learning rate.
    batch_size : int, default=64
        Number of samples per batch.
    limit_epoch : int, default=10
        Patience for early stopping.
    seed : int or None, default=42
        Random seed for reproducible results. If None, each run is different.

    Returns
    -------
    dict
        The metrics returned by :func:`train`.

        Contain a key config with several parameters
            - input_dim
            - hidden_sizes
            - activation_function
            - regularization
            - reg_param
            - optimizer_name
            - scheduler_name
            - "epochs
            - lr
            - batch_size
            - seed

    Examples
    --------
    >>> metrics = run_training(X_train, y_train, X_test, y_test, hidden_sizes=[64, 32], activation_function="tanh", epochs=10, lr=1e-3)
    """
    
    np.random.seed(42)
    torch.manual_seed(42)

    train_dataset = AudioMNISTFeaturesDataset(X_train, y_train)
    test_dataset = AudioMNISTFeaturesDataset(X_test, y_test)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    net = Net(input_dim=X_train.shape[1], hidden_sizes=hidden_sizes, activation_function=activation_function, regularization=regularization, reg_param=reg_param)
    metrics = train(net, loader, test_loader, epochs=epochs, lr=lr, optimizer_name=optimizer_name, scheduler_name=scheduler_name, limit_epoch=limit_epoch)

    metrics["config"] = {
        "input_dim": X_train.shape[1],
        "hidden_sizes": list(hidden_sizes),
        "activation_function": activation_function,
        "regularization": regularization,
        "reg_param": reg_param,
        "optimizer_name": optimizer_name,
        "scheduler_name": scheduler_name,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "seed": seed,
    }
    
    return metrics