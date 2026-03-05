import pdb

import numpy as np
import torch
from torch import nn, Tensor
from typing import Tuple, List
import tqdm
from pathlib import Path
import pandas as pd

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # This line checks if GPU is available


class MassConservingLSTM(nn.Module):
    """ Pytorch implementation of Mass-Conserving LSTMs. """

    def __init__(self, in_dim: int, aux_dim: int, out_dim: int, 
                 in_gate: nn.Module = None, out_gate: nn.Module = None,
                 redistribution: nn.Module = None, 
                 batch_first: bool = False, trainable_init_state=False,
                 basin_data_root: str = None, basin_id: str = None):
        """
        Parameters
        ----------
        in_dim : int
            The number of mass inputs.
        aux_dim : int
            The number of auxiliary inputs.
        out_dim : int
            The number of cells or, equivalently, outputs.
        in_gate : nn.Module, optional
            A module computing the (normalised!) input gate.
            This module must accept xm_t, xa_t and c_t as inputs
            and should produce a `in_dim` x `out_dim` matrix for every sample.
            Defaults to a time-dependent softmax input gate.
        out_gate : nn.Module, optional
            A module computing the output gate.
            This module must accept xm_t, xa_t and c_t as inputs
            and should produce a `out_dim` vector for every sample.
        redistribution : nn.Module, optional
            A module computing the redistribution matrix.
            This module must accept xm_t, xa_t and c_t as inputs
            and should produce a `out_dim` x `out_dim` matrix for every sample.
        batch_first : bool, optional
            Expects first dimension to represent samples if `True`,
            Otherwise, first dimension is expected to represent timesteps (default).
        """
        super().__init__()
        self.in_dim = in_dim
        self.aux_dim = aux_dim
        self.out_dim = out_dim
        self._seq_dim = 1 if batch_first else 0
        self.trainable_init_state = trainable_init_state
        
        # === Get watershed area ===
        self.basin_data_root = Path(basin_data_root) if basin_data_root else None
        self.basin_id = basin_id
        if self.basin_data_root and self.basin_id:
            self.area_km2 = self._get_catchment_area()
        else:
            raise ValueError("Must provide basin_data_root and basin_id to compute watershed area.")
            
        if self.trainable_init_state:
            self.init_state_param = nn.Parameter(torch.rand(1, self.out_dim) * 0.1)

        gate_inputs = self.in_dim + self.out_dim + self.aux_dim

        # initialize gates
        if out_gate is None:
            self.out_gate = _Gate(in_features=gate_inputs, out_features=out_dim)
        if in_gate is None:
            self.in_gate = _NormalizedGate(in_features=gate_inputs,
                                           out_shape=(in_dim, out_dim),
                                           normalizer="normalized_sigmoid")
        if redistribution is None:
            self.redistribution = _NormalizedGate(in_features=gate_inputs,
                                                  out_shape=(out_dim, out_dim),
                                                  normalizer="normalized_relu")
        self._reset_parameters()

    @property
    def batch_first(self) -> bool:
        return self._seq_dim != 0
            
    def _get_catchment_area(self):
        """Retrieve the catchment area from the catchment info file."""
        attributes_path = self.basin_data_root / "catchment_info.txt"
        df = pd.read_csv(attributes_path, sep=";", dtype={"gauge_id": str})
        area = df.loc[df["basin_id"].astype(str) == self.basin_id, "area_gages2"].values[0]
        return float(area)

    def reset_parameters(self, out_bias: float = -3.):
        """
        Parameters
        ----------
        out_bias : float, optional
            The initial bias value for the output gate (default to -3).
        """
        self.redistribution.reset_parameters(bias_init=nn.init.eye_)
        self.in_gate.reset_parameters(bias_init=nn.init.zeros_)
        self.out_gate.reset_parameters(
            bias_init=lambda b: nn.init.constant_(b, val=out_bias)
        )

    def _reset_parameters(self, out_bias: float = -3.):
        nn.init.constant_(self.out_gate.fc.bias, val=out_bias)

    def forward(self, xm, xa, state=None):
        xm = xm.unbind(dim=self._seq_dim)
        xa = xa.unbind(dim=self._seq_dim)
        
        batch_size = xa[0].shape[0]

        if state is None:
            state = self.init_state_param.expand(batch_size, -1) if self.trainable_init_state else torch.zeros(batch_size, self.out_dim, device=xa[0].device)

        hs, cs, os, hs_mmd = [], [], [], []
        for xm_t, xa_t in zip(xm, xa):
            # xm_t xa_t shape: [batchsize, 1] [batchsize, aux]
            hidden, state_new, o = self._step(xm_t, xa_t, state)  # h.shape=[256,16], state.shape=[256,16]
            #print("from state to state_new: ", state[0].sum().item(), state_new[0].sum().item())
            
            h_cfs = hidden * self.area_km2 * 0.40873  # Convert to cfs
            state = state_new

            hs.append(h_cfs)
            hs_mmd.append(hidden)
            cs.append(state)
            os.append(o)


        hs = torch.stack(hs, dim=self._seq_dim)  # [batch_size, seq_len, hidden_size]
        hs_mmd = torch.stack(hs_mmd, dim=self._seq_dim)  # [batch_size, seq_len, hidden_size]
        cs = torch.stack(cs, dim=self._seq_dim)  # [batch_size, seq_len, hidden_size]
        os = torch.stack(os, dim=self._seq_dim)  # [batch_size, seq_len, hidden_size]

        return hs, cs, os

    def _step(self, xt_m, xt_a, cs):
        """ Make a single time step in the MCLSTM. """
        # in this version of the MC-LSTM all available data is used to derive the gate activations. Cell states
        # are L1-normalized so that growing cell states over the sequence don't cause problems in the gates.

        # L1-normalized state
        features = torch.cat([xt_m, xt_a, cs / (cs.norm(1) + 1e-5)], dim=-1)

        # compute gate activations
        i = self.in_gate(features)
        r_m = self.redistribution(features)
        o = self.out_gate(features)  # size: [batchsize, seq_length, hidden_size], i.e., [64, seq_len, 128]

        # distribute incoming mass over the cell states
        m_in = torch.matmul(xt_m.unsqueeze(-2), i).squeeze(-2)

        # reshuffle the mass in the cell states using the redistribution matrix
        m_sys = torch.matmul(cs.unsqueeze(-2), r_m).squeeze(-2)
        #print("mclstm mass loss by redistribution: ", (m_sys.sum(dim=1)-cs.sum(dim=1)))

        # compute the new mass states
        m_new = m_in + m_sys

        # return the outgoing mass and subtract this value from the cell states.
        return o * m_new, (1 - o) * m_new, o


class _Gate(nn.Module):
    """Utility class to implement a standard sigmoid gate"""

    def __init__(self, in_features: int, out_features: int):
        super(_Gate, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.orthogonal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass through the normalised gate"""
        return torch.sigmoid(self.fc(x))


class _NormalizedGate(nn.Module):
    """Utility class to implement a gate with normalised activation function"""

    def __init__(self, in_features: int, out_shape: Tuple[int, int], normalizer: str):
        super(_NormalizedGate, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_shape[0] * out_shape[1])
        self.out_shape = out_shape

        if normalizer == "normalized_sigmoid":
            self.activation = nn.Sigmoid()
        elif normalizer == "normalized_relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(
                f"Unknown normalizer {normalizer}. Must be one of {'normalized_sigmoid', 'normalized_relu'}")
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.orthogonal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass through the normalized gate"""
        h = self.fc(x).view(-1, *self.out_shape)
        return torch.nn.functional.normalize(self.activation(h), p=1, dim=-1) # dim=-1


