import torch
import torch.nn as nn
import os
import numpy as np
import pdb
import torch.nn.functional as F
import matplotlib.pyplot as plt

RVIC_KERNEL_PATH = "/data/rdl/yihan/PECAN/watershed_data/routing_related/10180001/rvic_kernel.npy"


class pecan(nn.Module):
    """Convolutional MCR-LSTM with spatial and hidden-node mass conservation including a trash cell.

    Routing: a FIXED (non-learnable) RVIC routing kernel, replacing the
    D8+1-day-lag / learnable Muskingum-type routing used in earlier
    variants. The kernel (rvic_kernel.npy, shape (num_stations, K_lag_days,
    num_grid_cells)) is a per-source-cell impulse-response function
    produced by the real RVIC model (rvic parameters), using literature-
    grounded, DEM/flow-accumulation-derived velocity and a bounded
    diffusion value - see vic_model/build_rvic_velocity_grid.py and
    vic_model/rvic/parameters.cfg. It is loaded once as a registered
    buffer (not an nn.Parameter): no routing physics is learned here, only
    the generation-side network (_step: ConvLSTM-like gates, retention,
    MR gate) remains trainable. This sidesteps the failure modes found
    with learnable routing coefficients (loss-hacking toward degenerate
    persistence, gradient starvation over long unrolled sequences, drift
    to physically-implausible values) since there's no routing parameter
    for gradient descent to exploit.

    Each day's local runoff is pushed into a rolling (K_lag_days, N_cells)
    history buffer; routed streamflow at each gauge is a fixed linear
    combination (einsum) of that buffer against the kernel - a direct,
    single-hop path from any day's runoff to the loss, not a recurrence,
    so there's no compounding-gradient-decay issue either.
    """

    def __init__(self, in_channels: int, aux_channels: int, out_channels: int, x_dim: int, y_dim: int, routing_matrix: torch.Tensor, usgs_indices: dict, mode:str = 'mcr', mc_spatial_retention: bool = True, mc_dual_retention: bool = False, mc_groupnorm_ingate: bool = False, mc_groupnorm_groups: int = 8, mc_layernorm_ingate: bool = False, mc_groupnorm_outgate: bool = False, mc_dropout2d_ingate: bool = False, mc_groupnorm_groups_outgate: int = None, mc_dropout2d_outgate: bool = False, rvic_kernel_path: str = None):
        """
        Parameters
        ----------
        in_channels : int
            Number of input channels (e.g., precipitation, temperature).
        aux_channels : int
            Number of auxiliary input channels.
        out_channels : int
            Number of output channels (hidden state channels).
        routing_matrix : torch.Tensor
            Unused here (kept for interface compatibility with main.py /
            dataloader_seq.py - routing comes entirely from the fixed RVIC
            kernel instead).
        x_dim : int
            Spatial dimension (height).
        y_dim : int
            Spatial dimension (width).
        """
        super(pecan, self).__init__()
        self.mode = mode
        self.mc_spatial_retention = mc_spatial_retention
        self.mc_dual_retention = mc_dual_retention
        self.mc_groupnorm_ingate = mc_groupnorm_ingate
        self.mc_groupnorm_groups = mc_groupnorm_groups
        self.mc_layernorm_ingate = mc_layernorm_ingate
        self.mc_groupnorm_outgate = mc_groupnorm_outgate
        self.mc_dropout2d_ingate = mc_dropout2d_ingate
        self.mc_groupnorm_groups_outgate = mc_groupnorm_groups_outgate if mc_groupnorm_groups_outgate is not None else mc_groupnorm_groups
        self.mc_dropout2d_outgate = mc_dropout2d_outgate
        if self.mode == 'mcr':
            self.MR_gate = _ConvMRGate(out_channels, p_feat=0.1) # matches best MCR checkpoint (0.1do)
        self.in_channels = in_channels
        self.aux_channels = aux_channels
        self.out_channels = out_channels
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.usgs_indices = usgs_indices
        self.num_grid_cells = x_dim * y_dim
        self.cell_area_km2 = torch.full((self.y_dim, self.x_dim), 16.0)

        self.register_buffer("initial_state", torch.full((1, out_channels, x_dim, y_dim), 0.0)) #todo

        # -------------------------
        # Fixed RVIC routing kernel: (num_stations, K_lag_days, num_grid_cells).
        # Station order must match usgs_indices - verified against the
        # kernel-building script's own name-matched extraction.
        # -------------------------
        rvic_kernel = np.load(rvic_kernel_path if rvic_kernel_path is not None else RVIC_KERNEL_PATH)
        assert rvic_kernel.shape[0] == len(usgs_indices), \
            f"RVIC kernel has {rvic_kernel.shape[0]} stations, usgs_indices has {len(usgs_indices)}"
        assert rvic_kernel.shape[2] == self.num_grid_cells, \
            f"RVIC kernel grid size {rvic_kernel.shape[2]} != model grid {self.num_grid_cells}"
        self.register_buffer("rvic_kernel", torch.tensor(rvic_kernel, dtype=torch.float32))
        self.rvic_lag_days = rvic_kernel.shape[1]

        self.static_redistribution = True # todo

        self.static_retention_leak = True  # todo

        if not self.static_retention_leak:
            # Option 1: learn r from pooled current state (simple, effective)
            self.retention_fc = nn.Sequential(
                nn.Linear(out_channels, 1),  # output one r per sample
                nn.Sigmoid()
            )
        else:
            # fallback: static r
            if self.mc_dual_retention:
                # Separate recharge (r1, fills retention) and drainage (r2,
                # empties retention) rates instead of one shared scalar.
                # Now active in both MC and MCR modes.
                self.retention_leak_param_recharge = nn.Parameter(torch.tensor(-1.0))  # sigmoid(-1) ≈ 0.27
                self.retention_leak_param = nn.Parameter(torch.tensor(-1.0))  # sigmoid(-1) (drainage, r2)
            elif self.mc_spatial_retention:
                # Spatially-varying (per-cell) retention leak.
                # Now active in both MC and MCR modes.
                self.retention_leak_param = nn.Parameter(torch.full((1, 1, x_dim, y_dim), -1.0))  # sigmoid(-1) per cell
            else:
                self.retention_leak_param = nn.Parameter(torch.tensor(-1.0))  # sigmoid(-3)

        # Convolutional input gate. dropout=0.2 confirmed best for both modes
        # via sweep (2026-07-13): MC best is lr=0.001/dropout=0.2 (G2 NSE=0.646-
        # 0.662 depending on epoch budget; dropout 0.1 and 0.3 both regressed
        # it, bracketing 0.2 as a true local optimum). MCR best is lr=0.0005/
        # dropout=0.2 (G2 NSE=0.782/KGE=0.776, clears target).
        in_gate_dropout_p = float(os.environ.get("PECAN_DROPOUT_P", 0.2))
        print(f"[pecan model] in_gate_dropout_p = {in_gate_dropout_p} "
              f"(PECAN_DROPOUT_P env = {os.environ.get('PECAN_DROPOUT_P', '<unset, using default 0.2>')})")
        if self.mc_layernorm_ingate:
            # Full LayerNorm (channels+spatial jointly). Originally MC-only
            # (2026-07-14); un-gated from mode on 2026-07-14 so MC and MCR can
            # share identical norm/dropout config and differ only in the MR_gate
            # component, per user request.
            # GroupNorm(8) was the best batch-independent normalization found so
            # far for MC (KGE=0.748 vs BatchNorm's 0.585) but groups=4/16/32 all did
            # worse than groups=8, suggesting the win is about *not depending on
            # batch stats* rather than the specific grouping - LayerNorm removes
            # batch dependency entirely and normalizes per-sample like GroupNorm,
            # but jointly across channels+space instead of within channel groups.
            in_gate_norm = nn.LayerNorm([in_channels * out_channels, x_dim, y_dim])
        elif self.mc_groupnorm_ingate:
            # GroupNorm instead of BatchNorm2d. Originally MC-only (2026-07-14,
            # MC structural push); un-gated from mode on 2026-07-14 so MC and MCR
            # can share identical norm/dropout config and differ only in the
            # MR_gate component, per user request. MC's 60-day sequences give
            # BatchNorm noisier per-batch statistics than MCR's 180-day sequences
            # see; GroupNorm normalizes within each sample instead of across the
            # batch, so it isn't sensitive to that.
            in_gate_norm = nn.GroupNorm(num_groups=self.mc_groupnorm_groups, num_channels=in_channels * out_channels)
        else:
            in_gate_norm = nn.BatchNorm2d(in_channels * out_channels)  # Use BatchNorm instead of LayerNorm as LN performs worse here
        in_gate_dropout_layer = nn.Dropout2d(p=in_gate_dropout_p) if self.mc_dropout2d_ingate else nn.Dropout(p=in_gate_dropout_p)
        self.in_gate = nn.Sequential(
            nn.Conv2d(in_channels + aux_channels + out_channels, in_channels * out_channels, kernel_size=3, padding=1),
            in_gate_norm,
            #nn.LayerNorm([in_channels * out_channels, x_dim, y_dim]), # todo
            nn.Sigmoid(), # nn.ReLU()
            in_gate_dropout_layer
        )

        # Convolutional output gate
        if self.mc_groupnorm_outgate:
            # GroupNorm on out_gate too. Originally MC-only (2026-07-14); un-gated
            # from mode on 2026-07-14 (see in_gate above) so MC/MCR norm config can
            # be shared. Tests whether the normalization win found for in_gate
            # (GroupNorm(8) vs LayerNorm/BatchNorm) also helps out_gate, which has
            # always used LayerNorm regardless of any MC flag until now.
            out_gate_norm = nn.GroupNorm(num_groups=self.mc_groupnorm_groups_outgate, num_channels=out_channels)
        else:
            out_gate_norm = nn.LayerNorm([out_channels, x_dim, y_dim])  # LayerNorm for stability
        out_gate_layers = [
            nn.Conv2d(in_channels + aux_channels + out_channels, out_channels, kernel_size=3, padding=1),
            #nn.BatchNorm2d(out_channels), # batchnorm no good for output gate
            out_gate_norm,
            nn.Sigmoid()
        ]
        if self.mc_dropout2d_outgate:
            # DO NOT ENABLE (any mode): breaks mass conservation. See config.py
            # comment on MC_DROPOUT2D_OUTGATE for the mechanism (Dropout2d's
            # train-time inverse scaling can push the post-Sigmoid gate above 1.0,
            # and neither MC's nor MCR's mass-update clamps against o>1 the same way).
            out_gate_layers.append(nn.Dropout2d(p=in_gate_dropout_p))
        self.out_gate = nn.Sequential(*out_gate_layers)

        if self.static_redistribution:
            self.redistribution_matrix = nn.Parameter(torch.rand(out_channels, out_channels))
            self.redistribution_activation = nn.ReLU()

        if not self.static_redistribution:
            # Dynamic per-sample redistribution
            self.redistribution_fc = nn.Linear(
                out_channels,
                out_channels * out_channels
            )
            self.redistribution_activation = nn.ReLU()

        self.num_usgs_stations = len(self.usgs_indices)  # Get number of USGS stations

        # Ensure directories exist
        os.makedirs("trash_cell_outputs", exist_ok=True)
        os.makedirs("cell_state_outputs", exist_ok=True)
        os.makedirs("mr_flux_outputs", exist_ok=True)
        os.makedirs("mr_gate_outputs", exist_ok=True)

        self._reset_parameters()

    def _reset_parameters(self):
        if isinstance(self.in_gate, nn.Sequential):
            nn.init.kaiming_uniform_(self.in_gate[0].weight, nonlinearity="relu")
        else:
            nn.init.kaiming_uniform_(self.in_gate.weight)

        if isinstance(self.out_gate, nn.Sequential):
            nn.init.kaiming_uniform_(self.out_gate[0].weight, nonlinearity="relu")
        else:
            nn.init.kaiming_uniform_(self.out_gate.weight)

        if self.static_redistribution:
            with torch.no_grad():
                nn.init.xavier_uniform_(self.redistribution_matrix)

        if not self.static_redistribution:
            # Dynamic redistribution init
            nn.init.xavier_uniform_(self.redistribution_fc.weight)
            nn.init.zeros_(self.redistribution_fc.bias)

        # Initialize biases to zero
        nn.init.zeros_(self.in_gate[0].bias)
        nn.init.constant_(self.out_gate[0].bias, val=-3.0)  # More conservative output gating

    def _step(self, xm_t, xa_t, mask_t, state_t, retention_t):
        """
        Processes a single timestep t in the ConvMCR-LSTM.

        Parameters
        ----------
        xm_t : torch.Tensor
            Mass input at time t, shape (batch, in_channels, height, width).
        xa_t : torch.Tensor
            Auxiliary input at time t, shape (batch, aux_channels, height, width).
        mask_t : torch.Tensor
            Mask at time t, shape (batch, height, width).
        state_t : torch.Tensor
            Previous cell state, shape (batch, out_channels, height, width).

        Returns
        -------
        pred_usgs_streamflow : torch.Tensor
            Streamflow predictions at USGS stations, shape (batch, num_stations).
        trash_cell : torch.Tensor
            Trash cell mass loss, shape (batch, 1, height, width).
        h_next : torch.Tensor
            Updated hidden state, shape (batch, out_channels, height, width).
        c_next_main : torch.Tensor
            Updated main cell state (for fast flow), shape (batch, out_channels, height, width).
        c_next_retention : torch.Tensor
            Updated retention cell state (for slow flow), shape (batch, out_channels, height, width).
        """
        batch_size, _, height, width = xm_t.shape

        # Normalize state (L1 normalization)
        state_l1_norm = 0.9 * state_t / (state_t.norm(p=1, dim=1, keepdim=True) + 1e-8) + 0.1 * state_t

        # Check for any negative values
        if (state_l1_norm < 0).any():
            print("Warning: state_l1_norm has negative values!")
            min_val = state_l1_norm.min().item()
            print(f"Minimum value in state_l1_norm: {min_val}")

        # Concatenate inputs
        mask_spatial = mask_t[:, 0:1, :, :]  # Keep one representative channel

        xm_t = xm_t * mask_spatial  # Mask precipitation input
        xa_t = xa_t * mask_spatial  # Mask auxiliary features

        assert torch.all(xm_t >= 0), "Negative mass input detected!"

        features = torch.cat([xm_t, xa_t, state_l1_norm], dim=1)

        # Compute input and output gates
        mask_expanded = mask_spatial.unsqueeze(1).expand(-1, self.in_channels, -1, -1, -1)

        i = self.in_gate(features)  # Shape: (batch_size, in_channels * out_channels, height, width)
        i = i.view(batch_size, self.in_channels, self.out_channels, height, width)  # Shape: (batch_size, 1, in_channels * out_channels, height, width)
        i = i / (i.sum(dim=(2, 1), keepdim=True) + 1e-8)


        o = self.out_gate(features)
        o = o * mask_spatial  # **Apply Mask**

        # Distribute incoming mass (be sure to mask out unnecessary grids so mass distributes within the watershed)
        m_in = torch.einsum("bihw, biohw -> bohw", xm_t, i)
        m_in = m_in * mask_spatial  # **Ensure invalid cells get no mass**

        if self.static_redistribution:
            # Apply redistribution_fc to the reshaped state. State_t is masked.
            state_reshaped = state_t.view(batch_size, self.out_channels, -1)  # Shape: (batch, 16, 18*18)

            # Apply relu to ensure non-negative redistribution
            r_matrix = self.redistribution_activation(self.redistribution_matrix)  # (16, 16)

            # Normalize redistribution matrix (L1 normalization across rows)
            r_matrix = r_matrix / (r_matrix.sum(dim=0, keepdim=True) + 1e-8)

            # Apply redistribution matrix multiplication
            m_sys_flat = torch.matmul(r_matrix, state_reshaped)  # (batch, 16, 324)
            #print("pecan mass loss by redistribution: ", (m_sys_flat.sum(dim=[1,2])-state_reshaped.sum(dim=[1,2])))
            m_sys = m_sys_flat.view(batch_size, self.out_channels, height, width)  # Back to (batch, 16, height, width)

        if not self.static_redistribution:
            # Dynamic redistribution per sample
            state_pool = state_t.view(batch_size, self.out_channels, -1).mean(dim=2)
            h = self.redistribution_fc(state_pool)
            h = h.view(batch_size, self.out_channels, self.out_channels)
            h = self.redistribution_activation(h)
            r_matrix = F.normalize(h, p=1, dim=1, eps=1e-8)

            state_reshaped = state_t.view(batch_size, self.out_channels, -1)
            m_sys_flat = torch.bmm(r_matrix, state_reshaped)
            #print("pecan mass loss by redistribution: ", (m_sys_flat.sum(dim=[1,2])-state_reshaped.sum(dim=[1,2])))
            m_sys = m_sys_flat.view(batch_size, self.out_channels, height, width)

        # Compute new mass states
        m_new = m_in + m_sys

        # Output gate applied
        output = o * m_new

        # MCR componenet
        if self.mode == 'mcr':
            MR = self.MR_gate(state_l1_norm, 1.0 - o)
            MR_flow = MR * m_new
            o_prime = o + MR
            c_next_main = torch.clamp((1.0 - o_prime) * m_new, min = 0.0)
        else:
            MR = torch.zeros_like(m_new)                # keep shapes consistent in MC mode
            MR_flow = torch.zeros_like(m_new)
            c_next_main = (1.0 - o) * m_new

        # Update retention state and track baseflow
        if not self.static_retention_leak:
            # state_t: [B, C, H, W] -> pooled [B, C]
            state_summary = state_t.view(batch_size, self.out_channels, -1).mean(dim=2)  # (B, C)
            r = self.retention_fc(state_summary).view(batch_size, 1, 1, 1)  # broadcast to [B, 1, 1, 1]
        elif self.mc_dual_retention:
            r1 = torch.sigmoid(self.retention_leak_param_recharge)
            r2 = torch.sigmoid(self.retention_leak_param)
        else:
            r = torch.sigmoid(self.retention_leak_param)

        if self.mc_dual_retention and self.static_retention_leak:
            slow_flow = r2 * retention_t
            inflow_to_retention = r1 * c_next_main
            c_next_retention = (1 - r2) * retention_t + inflow_to_retention
        else:
            slow_flow = r * retention_t
            inflow_to_retention = r * c_next_main
            c_next_retention = (1 - r) * retention_t + inflow_to_retention


        c_next_main = c_next_main - inflow_to_retention

        # Total streamflow = fast_flow + slow_flow
        total_output = output + slow_flow

        streamflow_hidden = output.clone()
        if self.mode == 'mcr':
            #trash_cell = torch.zeros_like(output[:, 0:1, :, :]) # remove trash cell, i.e., notc, todo
            trash_cell = output[:, 0:1, :, :]
            streamflow_hidden[:, 0, :, :] = 0.0
        else:
            trash_cell = output[:, 0:1, :, :]
            streamflow_hidden[:, 0, :, :] = 0.0

        fast_flow_field = streamflow_hidden.clone()  # output with trash channel zeroed, BEFORE slow_flow added

        streamflow_hidden += slow_flow

        '''
        # trash cell also applied on slow flow (performance worse)
        trash_cell = total_output[:, 0:1, :, :]
        streamflow_hidden = total_output.clone()
        streamflow_hidden[:, 0, :, :] = 0.0
        '''

        # Hidden state update
        h_next = total_output

        # == Convert runoff to cfs before routing ===
        cell_area = self.cell_area_km2.to(xm_t.device)  # (H, W)
        runoff_mm_day = streamflow_hidden.sum(dim=1)  # (B, H, W)
        runoff_cfs = runoff_mm_day * cell_area.unsqueeze(0) * 0.40873  # (B, H, W)

        runoff_flat = runoff_cfs.view(batch_size, -1)  # Flatten for routing

        # **Mass Conservation Check**
        # Totals (mm)
        m_in_tot  = xm_t.sum()
        m_out_tot = total_output.sum()
        dS_tot = (c_next_main.sum() + c_next_retention.sum()
                     - state_t.sum() - retention_t.sum())

        err = m_in_tot - (m_out_tot + dS_tot)

        # Robust denominator: sum of magnitudes of all terms
        den = (m_in_tot.abs() + m_out_tot.abs() + dS_tot.abs()).clamp_min(1e-6)
        rel_throughput = err.abs() / den

        # Also compute "relative to input" but only when it's not a dry step
        rel_in = err.abs() / m_in_tot.abs().clamp_min(1e-6)

        # Thresholds
        dry = m_in_tot.abs() < 1e-4  # tune to your units/domain
        abs_ok = err.abs() < 1e-5    # absolute tolerance
        rel_ok = rel_throughput < 1e-2
        if self.mode == 'mcr':
            dry = m_in_tot.abs() < 1e-4  # tune to your units/domain
            abs_ok = err.abs() < 1e-5    # absolute tolerance
            rel_ok = rel_throughput < 1

        if (dry and not abs_ok) or (not dry and not rel_ok):
            tot_storage_next = (c_next_main.sum() + c_next_retention.sum())
            tot_storage_prev = (state_t.sum() + retention_t.sum())
            print(
                f"[Mass Balance] err={err.item():.3e} | rel_thr={100*rel_throughput.item():.2f}% "
                f"| rel_in={100*rel_in.item():.2f}% | in={m_in_tot.item():.3e} "
                f"| out={m_out_tot.item():.3e} | dS={dS_tot.item():.3e} "
                f"| Sprev={tot_storage_prev.item():.3e} | Snext={tot_storage_next.item():.3e}"
            )

        return runoff_flat, c_next_main, c_next_retention, trash_cell, h_next, MR, MR_flow, o, fast_flow_field


    def forward(self, xm, xa, mask, state=None, retention=None):
        """
        Forward pass for the ConvMCR-LSTM with fixed RVIC routing.
        mask dim: [batch_size, seq_length, input_num, grid_y, grid_x]
        """
        batch_size, seq_len, _, height, width = xm.shape

        # Learnable initial state
        if state is None or not isinstance(state, torch.Tensor):
            #state = self.initial_state.expand(batch_size, -1, -1, -1) * mask[0, 0, 0, :, :]
            state = self.initial_state.expand(batch_size, -1, -1, -1)

        if retention is None:
            retention = torch.zeros_like(state)

        # Store all flows over time for Conv1D processing
        all_flows = torch.zeros((batch_size, seq_len, self.num_usgs_stations), device=xm.device)

        h_states, c_states, r_states, trash_cells = [], [], [], []
        mr_gates, mr_fluxes, o_gates, fast_flows = [], [], [], []

        # Rolling local-runoff history for the fixed RVIC convolution:
        # runoff_history[:, 0, :] = today's local runoff, [:, 1, :] =
        # yesterday's, ..., [:, K-1, :] = K-1 days ago. Zero-initialized -
        # same "no history before this sequence" convention as every
        # earlier routing scheme in this codebase.
        K = self.rvic_lag_days
        runoff_history = torch.zeros((batch_size, K, self.num_grid_cells), device=xm.device)

        for t in range(seq_len):
            # 1. Call PECAN `_step()` with local forcing and current state
            (runoff_spatial_raw,
             c_next, c_next_retention,
             trash_cell, h_next,
             MR, MR_flow, o_t, fast_flow_t) = self._step(xm[:, t], xa[:, t], mask[:, t], state, retention)

            state = c_next
            retention = c_next_retention
            h_states.append(h_next)
            c_states.append(c_next+c_next_retention)
            r_states.append(retention)
            trash_cells.append(trash_cell)
            mr_gates.append(MR)
            mr_fluxes.append(MR_flow)
            o_gates.append(o_t)
            fast_flows.append(fast_flow_t)

            # 2. Push today's local runoff into the rolling history, drop
            # the oldest day
            runoff_history = torch.cat(
                [runoff_spatial_raw.unsqueeze(1), runoff_history[:, :-1, :]], dim=1
            )

            # 3. Routed streamflow at every gauge = fixed RVIC kernel dot
            # the runoff history (a direct linear combination, not a
            # recurrence - no learnable persistence to decay gradient or
            # drift to unphysical values)
            Q_t = torch.einsum("skc,bkc->bs", self.rvic_kernel, runoff_history)
            all_flows[:, t] = Q_t

            #pdb.set_trace()

        # **Get only the last time step routed flows**
        pred_streamflow = all_flows[:, -1, :]  # Shape: (batch_size, num_stations)
        pred_streamflow = F.softplus(pred_streamflow)
        #pdb.set_trace()
        return (
            pred_streamflow,  # (batch, num_stations)
            torch.stack(trash_cells, dim=1),
            torch.stack(h_states, dim=1),
            torch.stack(c_states, dim=1),
            torch.stack(r_states, dim=1),
            torch.stack(mr_gates,  dim=1),
            torch.stack(mr_fluxes, dim=1),
            torch.stack(o_gates,   dim=1),
            torch.stack(fast_flows, dim=1),
        )

    def save_trash_cell(self, trash_cell, timestamps, saving_path):
        """
        Save trash cell values for the **last time step only**.
        Input shape: (batch, seq_len, 1, H, W)
        """
        # Ensure directory exists
        os.makedirs(saving_path, exist_ok=True)

        # Get last timestep: shape (batch, seq_len, 1, H, W)
        trash = trash_cell[:, :, :, :, :].squeeze(2).cpu().numpy()

        for i in range(trash.shape[0]):
            date_str = timestamps[i].replace("-", "")  # e.g., '20121129'
            # Construct the full path for saving
            out_path = os.path.join(saving_path, f"trash_cell_{date_str}.npy")
            np.save(out_path, trash[i][-1, :, :])  # Save the last timestep's trash cell

        return trash


    def save_cell_state(self, cell_state, timestamps, saving_path):
        """
        Save cell state **sum across hidden channels**, for **last timestep only**.
        Input shape: (batch, seq_len, channels, H, W)
        Output shape per file: (H, W)
        """
        os.makedirs(saving_path, exist_ok=True)  # Ensure directory exists

        cell_sum = cell_state.sum(dim=2).cpu().numpy()  # (batch, seq_len, H, W)

        for i in range(cell_sum.shape[0]):
            date_str = timestamps[i].replace("-", "")
            out_path = os.path.join(saving_path, f"cell_state_{date_str}.npy")
            np.save(out_path, cell_sum[i][-1, :, :])

        return cell_sum

    def save_mr_flux(self, mr_flux, timestamps, saving_path, sum_channels=True):
        """
        Save MR *flux* for the last timestep of each sample.
        mr_flux: (B, T, C, H, W)
        If sum_channels=True, saves (H,W) total MR flux; otherwise saves per-channel cube.
        """
        os.makedirs(saving_path, exist_ok=True)
        x = mr_flux[:, -1]  # (B, C, H, W) last time step
        if sum_channels:
            x = x.sum(dim=1)  # (B, H, W)
            for i in range(x.shape[0]):
                date_str = timestamps[i].replace("-", "")
                np.save(os.path.join(saving_path, f"mr_flux_{date_str}.npy"),
                        x[i].detach().cpu().numpy())
        else:
            for i in range(x.shape[0]):
                date_str = timestamps[i].replace("-", "")
                np.save(os.path.join(saving_path, f"mr_flux_channels_{date_str}.npy"),
                        x[i].detach().cpu().numpy())
        return

    def save_mr_gate(self, mr_gate, timestamps, saving_path, sum_channels=True):
        """
        Save MR *gate* (dimensionless). Same behavior as flux saver.
        mr_gate: (B, T, C, H, W)
        """
        os.makedirs(saving_path, exist_ok=True)
        x = mr_gate[:, -1]
        if sum_channels:
            x = x.sum(dim=1)
            for i in range(x.shape[0]):
                date_str = timestamps[i].replace("-", "")
                np.save(os.path.join(saving_path, f"mr_gate_{date_str}.npy"),
                        x[i].detach().cpu().numpy())
        else:
            for i in range(x.shape[0]):
                date_str = timestamps[i].replace("-", "")
                np.save(os.path.join(saving_path, f"mr_gate_channels_{date_str}.npy"),
                        x[i].detach().cpu().numpy())
        return

    def save_fast_flow(self, fast_flow, timestamps, saving_path, sum_channels=True):
        """
        Save fast-pathway flux (output = o*m_new, trash channel zeroed, BEFORE
        slow_flow is added) for the last timestep of each sample.
        fast_flow: (B, T, C, H, W)
        """
        os.makedirs(saving_path, exist_ok=True)
        x = fast_flow[:, -1]  # (B, C, H, W) last time step
        if sum_channels:
            x = x.sum(dim=1)  # (B, H, W)
            for i in range(x.shape[0]):
                date_str = timestamps[i].replace("-", "")
                np.save(os.path.join(saving_path, f"fast_flow_{date_str}.npy"),
                        x[i].detach().cpu().numpy())
        else:
            for i in range(x.shape[0]):
                date_str = timestamps[i].replace("-", "")
                np.save(os.path.join(saving_path, f"fast_flow_channels_{date_str}.npy"),
                        x[i].detach().cpu().numpy())
        return

    def save_output_gate(self, o_gates, timestamps, saving_path, mean_channels=True):
        """
        Save output gate values for the last timestep of each sample.
        o_gates: (B, T, C, H, W) — sigmoid output gate, range [0, 1]
        If mean_channels=True, saves channel-mean (H, W); otherwise saves per-channel cube.
        """
        os.makedirs(saving_path, exist_ok=True)
        x = o_gates[:, -1]  # (B, C, H, W) last time step
        if mean_channels:
            x = x.mean(dim=1)  # (B, H, W)
            for i in range(x.shape[0]):
                date_str = timestamps[i].replace("-", "")
                np.save(os.path.join(saving_path, f"output_gate_{date_str}.npy"),
                        x[i].detach().cpu().numpy())
        else:
            for i in range(x.shape[0]):
                date_str = timestamps[i].replace("-", "")
                np.save(os.path.join(saving_path, f"output_gate_channels_{date_str}.npy"),
                        x[i].detach().cpu().numpy())
        return



class _ConvMRGate(nn.Module):
    """
    2-D Mass Relaxation (MR) gate with optional FEATURE DROPOUT ONLY.

    Inputs:
      c_norm, f : [B, C, H, W]
    Output:
      MR        : [B, C, H, W]

    Algebra (⊛ = conv2d):
      ov0 = (c_norm - b0) ⊛ exp(Ws or softplus(Ws_raw))
      ov1 = LN( tanh(ov0) ⊛ sigmoid(Wr) )
      ov2 = ov1 - ReLU(ov1 - f)
      MR  = ReLU(ov2 + 1 - f) + f - 1

    Dropout placement:
      - Dropout2d on tanh(ov0) BEFORE Wr conv
      - Dropout2d on ov1 BEFORE the inequality algebra
      (Keeps constraints intact.)
    """
    def __init__(
        self,
        channels: int,
        groups: int = 1,
        kernel_size: int = 3,
        use_softplus_for_Ws: bool = False,
        clamp_exp: float = 10.0,
        p_feat: float = 0.1,      # feature dropout (Dropout2d)
    ):
        super().__init__()
        # Ablation switch: disable the second LayerNorm (post-tanh, after the
        # Wr-conv) to test whether it actually helps vs. just being defensive
        # scaling for the ov1-vs-f inequality algebra. Default True (keeps
        # existing behavior for any run that doesn't set this env var).
        self.mr_gate_ln2_enabled = os.environ.get("PECAN_MRGATE_LN2", "1") == "1"
        print(f"[MR_gate] second LayerNorm (post-tanh) enabled = {self.mr_gate_ln2_enabled} "
              f"(PECAN_MRGATE_LN2 env = {os.environ.get('PECAN_MRGATE_LN2', '<unset, default 1>')})")
        # Ablation switch: disable ln0 (the FIRST LayerNorm, applied BEFORE
        # tanh) to test whether it actually helps vs. just being a
        # nice-to-have. Default True (keeps existing behavior).
        self.mr_gate_ln0_enabled = os.environ.get("PECAN_MRGATE_LN0", "1") == "1"
        print(f"[MR_gate] first LayerNorm (pre-tanh, ln0) enabled = {self.mr_gate_ln0_enabled} "
              f"(PECAN_MRGATE_LN0 env = {os.environ.get('PECAN_MRGATE_LN0', '<unset, default 1>')})")
        C = channels
        k = kernel_size
        assert k % 2 == 1, "kernel_size must be odd"
        assert C % groups == 0, "channels must be divisible by groups"
        in_per_group = C // groups
        pad = k // 2

        # Unconstrained params → transformed in forward
        self.Ws_raw = nn.Parameter(torch.empty(C, in_per_group, k, k))
        self.Wr_raw = nn.Parameter(torch.empty(C, in_per_group, k, k))
        self.b0     = nn.Parameter(torch.zeros(C, 1, 1))

        self.groups = groups
        self.pad    = pad
        self.relu   = nn.ReLU()
        # Ablation: swap ln2's normalization TYPE (LayerNorm vs GroupNorm) to test
        # whether it's LayerNorm specifically that matters, or any normalization there.
        self.mr_gate_ln2_type = os.environ.get("PECAN_MRGATE_LN2_TYPE", "layernorm")
        print(f"[MR_gate] second-norm (ln2) type = {self.mr_gate_ln2_type} "
              f"(PECAN_MRGATE_LN2_TYPE env = {os.environ.get('PECAN_MRGATE_LN2_TYPE', '<unset, default layernorm>')})")
        if self.mr_gate_ln2_type == "groupnorm":
            self.ln = nn.GroupNorm(num_groups=8, num_channels=C)
        else:
            self.ln = nn.LayerNorm(C)
        # Fix for stripe artifact: LayerNorm on ov0 before tanh. Without this,
        # ov0's magnitude tracks the raw cell state (which ranges up to ~60),
        # saturating tanh almost everywhere; once saturated, gradient to
        # Ws_raw/Wr_raw vanishes and they stay frozen near their random
        # orthogonal initialization for the entire training run, producing a
        # fixed, physically-meaningless spatial pattern from the MR gate
        # instead of a trained one. Normalizing ov0 keeps it in tanh's
        # non-saturating regime regardless of cell-state magnitude/season.
        self.ln0    = nn.LayerNorm(C)

        self.use_softplus_for_Ws = use_softplus_for_Ws
        self.clamp_exp = clamp_exp

        # Feature dropout (channel-wise)
        self.drop_feat1 = nn.Dropout2d(p_feat) if p_feat > 0 else nn.Identity()
        self.drop_feat2 = nn.Dropout2d(p_feat) if p_feat > 0 else nn.Identity()

        # Init
        nn.init.orthogonal_(self.Ws_raw.view(C, -1))
        nn.init.orthogonal_(self.Wr_raw.view(C, -1))
        nn.init.zeros_(self.b0)

    def _pos_Ws(self) -> torch.Tensor:
        if self.use_softplus_for_Ws:
            return F.softplus(self.Ws_raw)             # ≥0, smooth
        return torch.exp(self.Ws_raw.clamp(-self.clamp_exp, self.clamp_exp))

    def forward(self, c_norm: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        assert c_norm.shape == f.shape and c_norm.dim() == 4, \
            f"Expected [B,C,H,W]; got {c_norm.shape} and {f.shape}"

        # ov0 = (c_norm - b0) ⊛ exp(Ws)
        delta = c_norm - self.b0
        Ws = self._pos_Ws()
        ov0 = F.conv2d(delta, Ws, padding=self.pad, groups=self.groups)

        # per-pixel LayerNorm over channels, BEFORE tanh, so tanh never
        # saturates regardless of the raw cell-state magnitude (fixes the
        # frozen-kernel stripe artifact - see note in __init__)
        # (ablation-gated, see PECAN_MRGATE_LN0)
        if self.mr_gate_ln0_enabled:
            ov0 = self.ln0(ov0.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # feature dropout BEFORE Wr path
        ov0_act = torch.tanh(ov0)
        ov0_act = self.drop_feat1(ov0_act)

        # ov1_pre = tanh(ov0) ⊛ sigmoid(Wr)
        Wr = torch.sigmoid(self.Wr_raw)
        ov1_pre = F.conv2d(ov0_act, Wr, padding=self.pad, groups=self.groups)

        # per-pixel norm over channels (ablation-gated, see PECAN_MRGATE_LN2 /
        # PECAN_MRGATE_LN2_TYPE). GroupNorm operates directly on (B,C,H,W); LayerNorm
        # needs channels last, hence the permute only in that branch.
        if self.mr_gate_ln2_enabled:
            if self.mr_gate_ln2_type == "groupnorm":
                ov1 = self.ln(ov1_pre)
            else:
                ov1 = self.ln(ov1_pre.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        else:
            ov1 = ov1_pre

        # feature dropout BEFORE inequality algebra
        ov1 = self.drop_feat2(ov1)

        # Inequality-preserving algebra
        ov2 = ov1 - self.relu(ov1 - f)
        MR  = self.relu(ov2 + 1.0 - f) + f - 1.0
        return MR
