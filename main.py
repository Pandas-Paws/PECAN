import torch
import wandb
import random
import numpy as np
from scripts.config import cfg
import os
import pdb
#==============================================
#from pecan_model_routing import pecan
#from scripts.models.pecan_model_routing_vector_fastslow import pecan
from scripts.models.pecan_model_routing_fastslow import pecan
from scripts.models.mclstm import MassConservingLSTM
from scripts.models.mcrlstm import MassConservingLSTM_MR
from scripts.models.lstm import StandardLSTM
from scripts.train_seq import train_model, test_model
from scripts.train_lumped_seq import train_lumped_mcmodel, test_lumped_mcmodel, train_lumped_mcrmodel, test_lumped_mcrmodel, train_lumped_lstm, test_lumped_lstm
from scripts.dataloader_seq import load_data, load_debug_data
#==============================================

def set_random_seed(seed=3407):# 3407
    """Ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# **Set the seed before everything else**
set_random_seed()

def main():
    """Main function to run data loading, training, and testing."""
    
    
    '''
    # New vector-based routing section
    train_loader, val_loader, test_loader, vector_routing, usgs_indices = load_data(cfg.MODEL_NAME.lower())
    
    # --- move routing tensors to GPU ---
    vector_routing = {
        k: (v.to(cfg.DEVICE) if torch.is_tensor(v) else v)
        for k, v in vector_routing.items()
    }
    
    # --- convert gauges to (row,col) tuples on CPU ---
    def _to_rowcol_tuple(v, W):
        """
        Accepts:
          - (row,col) tuple/list
          - torch tensor [row,col]
          - dict with {"grid_index": [row,col]}   <-- your current case
          - dict with {"grid_index": flat_index} <-- optional
          - dict with {"row":..., "col":...}
        Returns: (row, col) as ints
        """
        # already tuple/list
        if isinstance(v, (tuple, list)) and len(v) == 2 and all(isinstance(x, (int, np.integer)) for x in v):
            return (int(v[0]), int(v[1]))
    
        # torch tensor like tensor([row,col])
        if torch.is_tensor(v):
            vv = v.detach().cpu().tolist()
            if isinstance(vv, (list, tuple)) and len(vv) == 2:
                return (int(vv[0]), int(vv[1]))
            raise ValueError(f"Tensor gauge value must be shape (2,). Got: {v.shape}")
    
        # dict formats
        if isinstance(v, dict):
            if "row" in v and "col" in v:
                return (int(v["row"]), int(v["col"]))
    
            if "grid_index" in v:
                gi = v["grid_index"]
    
                # CASE A: [row, col]
                if isinstance(gi, (list, tuple)) and len(gi) == 2:
                    return (int(gi[0]), int(gi[1]))
    
                # CASE B: flattened index
                if isinstance(gi, (int, np.integer, float)):
                    gi = int(gi)
                    return (gi // W, gi % W)
    
                raise ValueError(f"Unsupported grid_index type: {type(gi)} | value={gi}")
    
        raise ValueError(f"Cannot convert gauge value to (row,col): {v}")

    usgs_indices = {k: _to_rowcol_tuple(v, W=cfg.X_DIM) for k, v in usgs_indices.items()}
    
    if cfg.MODEL_NAME.lower() == "pecan":
        model = pecan(
            in_channels=cfg.IN_CHANNELS,
            aux_channels=cfg.AUX_CHANNELS,
            out_channels=cfg.OUT_CHANNELS,
            H=cfg.X_DIM,
            W=cfg.Y_DIM,
            vector_routing=vector_routing,
            usgs_indices=usgs_indices,
            mode=cfg.PECAN_MODE
        ).to(cfg.DEVICE)
    '''
        

    
    # This is old grid-based routing section.
    # Load train, validation, test data, and routing matrix
    train_loader, val_loader, test_loader, routing_matrix, usgs_indices = load_data(cfg.MODEL_NAME.lower())
    
    routing_matrix = routing_matrix.to(cfg.DEVICE)

    # Extract only the coordinates from the JSON before converting to tensors
    usgs_indices = {
        key: torch.tensor(value["grid_index"], dtype=torch.long, device=cfg.DEVICE) 
        for key, value in usgs_indices.items()
    }

    if cfg.MODEL_NAME.lower() == "pecan":
        model = pecan(
            in_channels=cfg.IN_CHANNELS,
            aux_channels=cfg.AUX_CHANNELS,
            out_channels=cfg.OUT_CHANNELS,
            x_dim=cfg.X_DIM,
            y_dim=cfg.Y_DIM,
            routing_matrix=routing_matrix.to(cfg.DEVICE),
            usgs_indices=usgs_indices,
            mode =  cfg.PECAN_MODE
        ).to(cfg.DEVICE)
        
    elif cfg.MODEL_NAME.lower() == "mclstm":
        model = MassConservingLSTM(
            in_dim=cfg.IN_CHANNELS,
            aux_dim = cfg.AUX_CHANNELS,
            out_dim=cfg.OUT_CHANNELS,
            batch_first = True,
            basin_data_root = cfg.BASIN_DATA_ROOT, 
            basin_id = cfg.BASIN_ID
        ).to(cfg.DEVICE)
    elif cfg.MODEL_NAME.lower() == "mcrlstm":
        model = MassConservingLSTM_MR(in_dim = cfg.IN_CHANNELS, 
                            aux_dim = cfg.AUX_CHANNELS, 
                            out_dim = cfg.OUT_CHANNELS, 
                            time_dependent=False, 
                            batch_first=True,
                            basin_data_root = cfg.BASIN_DATA_ROOT, 
                            basin_id = cfg.BASIN_ID).to(cfg.DEVICE)
    elif cfg.MODEL_NAME.lower() == "lstm":
        model = StandardLSTM(
            in_dim      = cfg.IN_CHANNELS+cfg.AUX_CHANNELS,
            aux_dim     = 0, 
            hidden_dim  = cfg.OUT_CHANNELS,
            num_layers  = 1,
            dropout     = 0.0,
            batch_first = True,
            trainable_init_state = True,
        ).to(cfg.DEVICE)
    
    # Check total trainable parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params}")

    # Train and test the model
    # For test only by reading exsiting parameters, using run_test.py
    if cfg.MODEL_NAME.lower() == "pecan":
        train_model(model, train_loader, val_loader)
        test_loss = test_model(model, test_loader)
        
    elif cfg.MODEL_NAME.lower() == "mclstm":
        train_lumped_mcmodel(model, train_loader, val_loader)
        test_loss = test_lumped_mcmodel(model, test_loader)
        
    elif cfg.MODEL_NAME.lower() == "mcrlstm":
        train_lumped_mcrmodel(model, train_loader, val_loader)
        test_loss = test_lumped_mcrmodel(model, test_loader)
        
    elif cfg.MODEL_NAME.lower() == "lstm":
        train_lumped_lstm(model, train_loader, val_loader)
        test_loss = test_lumped_lstm(model, test_loader, scaler_pickle  = "global_norm_stats_10180001.pkl")
        
    print(f"Test Loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()
