# -*- coding: utf-8 -*-
import os
import torch
import pdb

class NSELoss(torch.nn.Module):

    # Weighted Nash–Sutcliffe Efficiency (NSE) Loss with internal gauge weights.
    # Weights can be overridden via PECAN_LOSS_W0 (G2 outlet) and PECAN_LOSS_W1 (G1 upstream).

    def __init__(self, eps: float = 1e-5):
        super(NSELoss, self).__init__()
        self.eps = eps

        w0 = float(os.environ.get("PECAN_LOSS_W0", 0.8))
        w1 = float(os.environ.get("PECAN_LOSS_W1", 0.2))
        self.weights = torch.tensor([w0, w1])  # shape: (num_stations,); index0=outlet(06620000), index1=upstream(06614800)
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