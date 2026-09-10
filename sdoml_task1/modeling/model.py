"""Neural network model for digit classification."""

import torch
import torch.nn as nn

class Net(nn.Module):
    """Simple feedforward network for digit classification.

    Parameters
    ----------
    input_dim : int, default=26
        Size of the input feature vector.
    num_classes : int, default=10
        Number of output classes.
    """
    def __init__(self, input_dim=26, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        # self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(64, num_classes)
    def forward(self, x):
        """Run a forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch_size, input_dim).
        
        Returns
        -------
        torch.Tensor
            Output logits, shape (batch_size, num_classes).
        """
        x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        return self.fc3(x)