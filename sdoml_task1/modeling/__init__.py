"""Modeling package.

This subpackage contains the neural network model, training loop,
and prediction utilities for the AudioMNIST digit classification task.

Modules
-------
model
    Neural network architecture definition.
train
    Training loop with metrics tracking.
predict
    Inference utilities for trained models.

Examples
--------
>>> from sdoml_task1.modeling import Net, train
>>> model = Net(input_dim=26, num_classes=10)
"""

__all__ = ["model", "train", "predict"]
