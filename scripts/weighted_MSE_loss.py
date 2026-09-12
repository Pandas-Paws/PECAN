import torch
import torch.nn as nn
from scripts.config import cfg
import pdb

class WeightedRMSELoss(nn.Module):
    """
    Custom weighted Root Mean Squared Error (RMSE) loss function for multi-station streamflow prediction.
    """
    def __init__(self):
        """
        Initialize fixed weights (e.g., 0.8 for Station 1, 0.2 for Station 2).
        """
        super(WeightedRMSELoss, self).__init__()
        self.station_weights = torch.tensor([1, 1], dtype=torch.float32).to(cfg.DEVICE)

    def forward(self, pred, target):
        """
        Compute the weighted RMSE loss.

        Parameters:
        ----------
        pred : torch.Tensor
            Model predictions of shape (batch, seq_len, num_stations)
        target : torch.Tensor
            Ground truth values of shape (batch, seq_len, num_stations)

        Returns:
        -------
        torch.Tensor:
            Scalar loss value (Root Mean Squared Error)
        """
        mse = (pred - target) ** 2  # Compute squared error
        weighted_mse = mse * self.station_weights
        rmse = torch.sqrt(weighted_mse.mean())
        return rmse
