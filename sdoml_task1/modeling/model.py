"""Neural network model for digit classification.

This module defines the feedforward neural network architecture used
for classifying spoken digits (0-9) from MFCC audio features.

Classes
-------
Net
    A simple feedforward network with ReLU activations.

"""

from torch import nn


class Net(nn.Module):
    """Configurable multilayer perceptron for digit classification.

    Parameters
    ----------
    input_dim : int, default=26
        Size of the input feature vector (13 MFCC means + 13 stds).
    hidden_sizes : list of int, default=(64,)
        Number of neurons in each hidden layer, e.g. ``[64, 32]``.
    num_classes : int, default=10
        Number of output classes (digits 0-9).
    activation_function : str, default="relu"
        ``"relu"``, ``"sigmoid"`` or ``"tanh"``.
    regularization : str or None, default=None
        ``"l1"``, ``"weight_decay"``, ``"early_stopping"``, ``"dropout"`` or None.
        Only ``"dropout"`` changes the model itself; the others are
        applied by :func:`sdoml_task1.modeling.train.train`.
    reg_param : float, default=0.0
        Strength of the regularization (dropout probability for ``"dropout"``).

    Examples
    --------
    >>> model = Net(hidden_sizes=[64, 32], activation_function="tanh",
    ...             regularization="dropout", reg_param=0.2)
    """

    def __init__(
        self,
        input_dim=26,
        hidden_sizes=(64,),
        num_classes=10,
        activation_function="relu",
        regularization=None,
        reg_param=0.0,
    ):
        super().__init__()
        self.hidden_sizes = hidden_sizes
        self.regularization = regularization
        self.reg_param = reg_param

        self.lst = nn.ModuleList()
        self.lst.append(nn.Linear(input_dim, hidden_sizes[0]))

        for i in range(1, len(hidden_sizes)):
            self.lst.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))

        self.lst.append(nn.Linear(hidden_sizes[-1], num_classes))

        if activation_function == "relu":
            self.activation_function = nn.ReLU()
        elif activation_function == "tanh":
            self.activation_function = nn.Tanh()
        elif activation_function == "sigmoid":
            self.activation_function = nn.Sigmoid()

        if regularization == "dropout":
            self.dropouts = nn.ModuleList()
            for i in range(len(hidden_sizes)):
                self.dropouts.append(nn.Dropout(reg_param))

    def forward(self, x):
        """Run a forward pass and return logits of shape ``(batch, num_classes)``."""
        z = self.lst[0](x)
        for i in range(1, len(self.hidden_sizes) + 1):
            z = self.activation_function(z)

            if self.regularization == "dropout":
                z = self.dropouts[i - 1](z)

            z = self.lst[i](z)
        return z
