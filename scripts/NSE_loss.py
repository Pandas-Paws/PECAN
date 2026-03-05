# -*- coding: utf-8 -*-
import torch
import pdb

class NSELoss(torch.nn.Module):

    # Weighted Nash–Sutcliffe Efficiency (NSE) Loss with internal gauge weights.

    def __init__(self, eps: float = 1e-5):
        super(NSELoss, self).__init__()
        self.eps = eps

        # Define your weights here (adjust values and size to match num_gauges)
        self.weights = torch.tensor([0.8, 0.2])  # shape: (num_stations,)
        # If you're training on GPU, move this to CUDA later in forward()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted streamflow, shape (batch_size, num_stations)
        y_true : torch.Tensor
            Observed streamflow, shape (batch_size, num_stations)

        Returns
        -------
        torch.Tensor
            Weighted NSE loss scalar
        """
        # Move weights to same device as input tensors
        weights = self.weights.to(y_pred.device)

        mean_obs = torch.mean(y_true, dim=0, keepdim=True)

        numerator = torch.sum((y_pred - y_true) ** 2, dim=0)
        denominator = torch.sum((y_true - mean_obs) ** 2, dim=0) + self.eps

        nse = 1 - numerator / denominator
        nse_loss = 1 - nse  # loss is (1 - NSE)

        weighted_loss = nse_loss * weights
        return torch.sum(weighted_loss) / torch.sum(weights)