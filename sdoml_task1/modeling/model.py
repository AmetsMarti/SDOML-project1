"""Neural network model for digit classification.

This module defines the feedforward neural network architecture used
for classifying spoken digits (0-9) from MFCC audio features.

Classes
-------
Net
    A simple feedforward network with ReLU activations.

"""
import torch
import torch.nn as nn


class Net(nn.Module):
    """Simple feedforward network for digit classification.

    Architecture: Input(26) -> Linear(64) -> ReLU -> Linear(10).

    Parameters
    ----------
    input_dim : int, default=26
        Size of the input feature vector. Matches the dimensionality
        of MFCC mean+std features (13 coefficients * 2).
    num_classes : int, default=10
        Number of output classes (digits 0-9).
    """

    def __init__(self, input_dim: int = 26, num_classes: int = 10) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch_size, input_dim)``.

        Returns
        -------
        torch.Tensor
            Output logits of shape ``(batch_size, num_classes)``.
        """
        x = torch.relu(self.fc1(x))
        return self.fc3(x)