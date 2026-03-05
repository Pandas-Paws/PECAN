import torch
import torch.nn as nn
import os
import numpy as np
import pdb
import torch.nn.functional as F
import matplotlib.pyplot as plt


class pecan(nn.Module):
    """Convolutional MCR-LSTM with spatial and hidden-node mass conservation including a trash cell."""
    
    def __init__(self, in_channels: int, aux_channels: int, out_channels: int, x_dim: int, y_dim: int, routing_matrix: torch.Tensor, usgs_indices: dict, mode:str = 'mcr'):      
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
            Precomputed sparse routing matrix of shape (num_grid_cells, num_grid_cells).
        x_dim : int
            Spatial dimension (height).
        y_dim : int
            Spatial dimension (width).
        """
        super(pecan, self).__init__()
        self.mode = mode
        if self.mode == 'mcr':
            self.MR_gate = _ConvMRGate(out_channels) # groups=out_channels
        self.in_channels = in_channels
        self.aux_channels = aux_channels
        self.out_channels = out_channels
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.routing_matrix = routing_matrix
        self.usgs_indices = usgs_indices  
        self.num_grid_cells = routing_matrix.shape[0]
        self.cell_area_km2 = torch.full((self.y_dim, self.x_dim), 16.0)

        self.register_buffer("initial_state", torch.full((1, out_channels, x_dim, y_dim), 0.0)) #todo
                
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
            #self.retention_leak_param_recharge = nn.Parameter(torch.tensor(-1.0))  # sigmoid(-1) ≈ 0.27
            self.retention_leak_param = nn.Parameter(torch.tensor(-1.0))  # sigmoid(-3)

        # Convolutional input gate
        self.in_gate = nn.Sequential(
            nn.Conv2d(in_channels + aux_channels + out_channels, in_channels * out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels * out_channels),  # Use BatchNorm instead of LayerNorm as LN performs worse here
            #nn.LayerNorm([in_channels * out_channels, x_dim, y_dim]), # todo
            nn.Sigmoid(), # nn.ReLU()
            nn.Dropout(p=0.2)
        )

        # Convolutional output gate
        self.out_gate = nn.Sequential(
            nn.Conv2d(in_channels + aux_channels + out_channels, out_channels, kernel_size=3, padding=1),
            #nn.BatchNorm2d(out_channels), # batchnorm no good for output gate
            nn.LayerNorm([out_channels, x_dim, y_dim]),  # LayerNorm for stability
            nn.Sigmoid()
        )
        
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
        else:
            r = torch.sigmoid(self.retention_leak_param)
            #r1 = torch.sigmoid(self.retention_leak_param_recharge)
            #r2 = torch.sigmoid(self.retention_leak_param)
        '''
        slow_flow = r2 * retention_t
        inflow_to_retention = r1 * c_next_main  
        c_next_retention = (1 - r2) * retention_t + inflow_to_retention  
        
        '''
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

        return runoff_flat, c_next_main, c_next_retention, trash_cell, h_next, MR, MR_flow
        

    def forward(self, xm, xa, mask, state=None, retention=None):
        """
        Forward pass for the ConvMCR-LSTM with convolutional routing delays.
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
        all_flows = torch.zeros((batch_size, seq_len, self.num_grid_cells), device=xm.device)

        h_states, c_states, r_states, trash_cells = [], [], [], []
        mr_gates, mr_fluxes = [], []
        
        # Outlet index (the first item in usgs_indices is ALWAYS the outlet)
        outlet_row, outlet_col = list(self.usgs_indices.values())[0]
        outlet_index = outlet_row * width + outlet_col
        
        # Initialize previous routed cfs flow as zero for t=0
        routed_cfs_prev = torch.zeros((batch_size, height, width), device=xm.device)

        for t in range(seq_len):
            # 1. Call PECAN `_step()` with local forcing and current state
            (runoff_spatial_raw,
             c_next, c_next_retention,
             trash_cell, h_next,
             MR, MR_flow) = self._step(xm[:, t], xa[:, t], mask[:, t], state, retention)

            #print(f"== from state to next: {state[0].sum(dim=0).sum().item()/234:.3f}, {c_next[0].sum(dim=0).sum().item()/234:.3f}")
            #print(f"== trash cell: {trash_cell[0].sum(dim=0).sum().item()/234:.3f}")
            state = c_next
            retention = c_next_retention
            h_states.append(h_next)
            c_states.append(c_next+c_next_retention)
            r_states.append(retention)
            trash_cells.append(trash_cell)
            mr_gates.append(MR)
            mr_fluxes.append(MR_flow)
    
            # 2. Add delayed routed flow from t-1
            runoff_cfs_current = runoff_spatial_raw.view(batch_size, height, width)
            runoff_cfs_combined = runoff_cfs_current + routed_cfs_prev  # (B, H, W)
            
            runoff_cfs_flat = runoff_cfs_combined.view(batch_size, -1)
    
            # 3. Route to downstream using routing matrix
            routing_matrix_T = self.routing_matrix.transpose(0, 1)
    
            if routing_matrix_T.is_sparse:
                routed_flow_spatial = torch.sparse.mm(routing_matrix_T, runoff_cfs_flat.unsqueeze(-1)).squeeze(-1)
            else:
                routed_flow_spatial = torch.matmul(routing_matrix_T, runoff_cfs_flat.unsqueeze(-1)).squeeze(-1)
    
            all_flows[:, t] = routed_flow_spatial # todo
            #all_flows[:, t] = runoff_spatial_raw # todo, no routing
    
            # 4. Prepare routed flow (excluding outlet) for use in next timestep
            routed_internal = routed_flow_spatial.clone()
            routed_internal[:, outlet_index] = 0.0
            routed_cfs_prev = routed_internal.view(batch_size, height, width)
            
            #pdb.set_trace()
        
        # **Get only the last time step routed flows**
        last_routed_flow = all_flows[:, -1, :]  # Shape: (batch_size, num_grid_cells)
        # **Extract Streamflow at USGS Stations**
        pred_streamflow = torch.stack(
            [last_routed_flow[:, row * width + col] for _, (row, col) in self.usgs_indices.items()],
            dim=1
        )
        
        pred_streamflow = F.softplus(pred_streamflow)
        #pdb.set_trace()
        return (
            pred_streamflow,  # (batch, num_stations)
            torch.stack(trash_cells, dim=1),  
            torch.stack(h_states, dim=1),  
            torch.stack(c_states, dim=1),
            torch.stack(r_states, dim=1),
            torch.stack(mr_gates,  dim=1),
            torch.stack(mr_fluxes, dim=1)
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
        self.ln     = nn.LayerNorm(C)

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

        # feature dropout BEFORE Wr path
        ov0_act = torch.tanh(ov0)
        ov0_act = self.drop_feat1(ov0_act)

        # ov1_pre = tanh(ov0) ⊛ sigmoid(Wr)
        Wr = torch.sigmoid(self.Wr_raw)
        ov1_pre = F.conv2d(ov0_act, Wr, padding=self.pad, groups=self.groups)

        # per-pixel LayerNorm over channels
        ov1 = self.ln(ov1_pre.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # feature dropout BEFORE inequality algebra
        ov1 = self.drop_feat2(ov1)

        # Inequality-preserving algebra
        ov2 = ov1 - self.relu(ov1 - f)
        MR  = self.relu(ov2 + 1.0 - f) + f - 1.0
        return MR
