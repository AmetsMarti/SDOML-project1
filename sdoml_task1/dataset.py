"""PyTorch dataset wrapper for precomputed audio features."""

import torch
from torch.utils.data import Dataset

class AudioMNISTFeaturesDataset(Dataset):
    """PyTorch Dataset for audio features.

    Parameters
    ----------
    X : np.ndarray
        Feature vectors.
    y : np.ndarray
        Labels.
    """
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        """Return the number of samples."""
        return len(self.X)

    def __getitem__(self, idx):
        """Return the feature vector and label at `idx`.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        torch.Tensor
            Feature vector.
        torch.Tensor
            Label.
        """
        return self.X[idx], self.y[idx]