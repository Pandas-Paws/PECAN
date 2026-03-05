from typing import Optional, Tuple

import torch
from torch import nn, Tensor

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class StandardLSTM(nn.Module):
    """
    Parameters
    ----------
    in_dim   : int       # precipitation + all other channels
    aux_dim  : int       # keep for API symmetry (often 0 for the vanilla baseline)
    hidden_dim : int
    num_layers: int
    dropout   : float    # applied between stacked LSTM layers
    batch_first: bool
    trainable_init_state: bool
    """

    def __init__(
        self,
        in_dim: int,
        aux_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = True,
        trainable_init_state: bool = False,
    ):
        super().__init__()

        self.in_dim   = in_dim
        self.aux_dim  = aux_dim                # not used for vanilla case
        self.hidden_dim  = hidden_dim
        self.num_layers  = num_layers
        self.batch_first = batch_first
        self.trainable_init_state = trainable_init_state

        # -- Layers ---------------------------------------------------
        self.lstm = nn.LSTM(
            input_size=in_dim + aux_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=batch_first,
        )
        self.readout = nn.Linear(hidden_dim, 1)   # outputs streamflow

        # -- Optional trainable initial states ------------------------
        if trainable_init_state:
            self.h0_param = nn.Parameter(torch.zeros(num_layers, 1, hidden_dim))
            self.c0_param = nn.Parameter(torch.zeros(num_layers, 1, hidden_dim))

    # -----------------------------------------------------------------
    #  Forward
    # -----------------------------------------------------------------
    def forward(
        self,
        x:  Tensor,                             # full feature tensor (B, T, C)
        xa: Optional[Tensor] = None,            # kept for API but usually None
        state: Optional[Tuple[Tensor, Tensor]] = None,
    ):
        """
        Returns
        -------
        streamflow_pred : Tensor (B, T)    same units/scale as training target
        hidden_seq      : Tensor (B, T, H)
        final_cell      : Tensor (num_layers, B, H)
        """

        if xa is not None:                     # allows compatibility elsewhere
            x = torch.cat([x, xa], dim=-1)

        # initialise hidden/cell states
        if state is None and self.trainable_init_state:
            B = x.size(0) if self.batch_first else x.size(1)
            h0 = self.h0_param.expand(self.num_layers, B, self.hidden_dim).contiguous()
            c0 = self.c0_param.expand(self.num_layers, B, self.hidden_dim).contiguous()
            state = (h0, c0)

        lstm_out, (_, c_n) = self.lstm(x, state)        # (B, T, H)
        streamflow_pred = self.readout(lstm_out).squeeze(-1)  # (B, T)
        

        return streamflow_pred, lstm_out, c_n
