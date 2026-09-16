
import numpy as np
import torch
from torch.utils.data import Dataset

class AudioMNISTFeaturesDataset(Dataset):
    """PyTorch Dataset for audio features.

    Wraps precomputed feature vectors and labels into a PyTorch Dataset
    suitable for use with DataLoaders during model training.

    Parameters
    ----------
    X : np.ndarray
        Feature vectors of shape ``(n_samples, n_features)``.
    y : np.ndarray
        Digit labels of shape ``(n_samples,)``.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples.
        """
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the feature vector and label at the given index.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        torch.Tensor
            Feature vector of shape ``(n_features,)``.
        torch.Tensor
            Label as a scalar tensor.
        """
        return self.X[idx], self.y[idx]