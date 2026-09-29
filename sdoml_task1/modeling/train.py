"""Training loop for classification models.

This module implements the training loop for the digit classification model,
with optional L1 penalty, weight decay, early stopping and lr scheduling.
"""
from sklearn.metrics import confusion_matrix
import torch
from torch import nn

from sdoml_task1.modeling.optimization import make_optimizer, make_scheduler

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