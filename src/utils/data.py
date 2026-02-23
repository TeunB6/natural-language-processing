from torch.utils.data import Dataset
from torch import Tensor


class TorchDataset(Dataset):
    """A simple PyTorch Dataset wrapper for the AG News dataset."""

    def __init__(self, X: Tensor, y: Tensor) -> None:
        """Initialize the dataset with features and labels.

        Args:
            X (Tensor): The feature tensor.
            y (Tensor): The label tensor.
        """
        self.X = X
        self.y = y

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Return a single sample from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.
        Returns:
            tuple[Tensor, Tensor]: A tuple containing the feature tensor and label tensor for the sample.
        """
        return self.X[idx], self.y[idx]
