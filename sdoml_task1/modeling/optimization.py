"""Optimizer and learning-rate scheduler factories."""
import torch.optim as optim

OPTIMIZERS = ["SGD", "SGD momentum", "Adagrad", "RMSprop", "Adadelta", "Adam"]
SCHEDULERS = ["None", "Piecewise constant", "Exponential decay", "Polynomial decay"]
REGULARIZATIONS = ["None", "l1", "weight_decay", "early_stopping", "dropout"]
ACTIVATIONS = ["relu", "sigmoid", "tanh"]


def make_optimizer(name, params, lr, momentum=0.9, weight_decay=0.0):
    """Create a PyTorch optimizer from its name.

    Parameters
    ----------
    name : str
        One of :data:`OPTIMIZERS`.
    params : iterable
        Model parameters, usually ``model.parameters()``.
    lr : float
        Learning rate. Note that Adadelta is designed for lr around 1.0.
    momentum : float, default=0.9
        Momentum, only used by ``"SGD momentum"``.
    weight_decay : float, default=0.0
        L2 penalty passed to the optimizer.

    Returns
    -------
    torch.optim.Optimizer
    """
    if name == "SGD":
        return optim.SGD(params, lr=lr, weight_decay=weight_decay)

    elif name == "SGD momentum":
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    elif name == "Adagrad":
        return optim.Adagrad(params, lr=lr, weight_decay=weight_decay)

    elif name == "RMSprop":
        return optim.RMSprop(params, lr=lr, weight_decay=weight_decay)

    elif name == "Adadelta":
        return optim.Adadelta(params, lr=lr, weight_decay=weight_decay)

    elif name == "Adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)


def polynomial_factor(t):
    """Polynomial decay factor ``(beta * t + 1) ** (-delta)`` used by LambdaLR."""
    beta = 0.1
    delta = 1.0
    return (beta * t + 1) ** (-delta)


def make_scheduler(name, optimizer):
    """Create a learning-rate scheduler from its name.

    Parameters
    ----------
    name : str or None
        One of :data:`SCHEDULERS`. ``None`` or ``"None"`` disables scheduling.
        ``"Piecewise constant"`` divides the lr by 10 at epochs 20 and 40,
        so it has no effect with fewer than 20 epochs.
    optimizer : torch.optim.Optimizer
        Optimizer whose learning rate will be scheduled.

    Returns
    -------
    torch.optim.lr_scheduler.LRScheduler or None
    """
    if name is None or name == "None":
        return None
    elif name == "Piecewise constant":
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)
    elif name == "Exponential decay":
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    elif name == "Polynomial decay":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=polynomial_factor)