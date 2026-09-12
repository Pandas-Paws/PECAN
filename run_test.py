import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import os
from pathlib import Path
from scripts.config import cfg
from torch.utils.data import DataLoader
from scripts.config import cfg
#from scripts.models.pecan_model_routing_vector_fastslow import pecan
from scripts.models.pecan_model_routing_fastslow import pecan
from scripts.models.mclstm import MassConservingLSTM
from scripts.models.mcrlstm import MassConservingLSTM_MR
from scripts.models.lstm import StandardLSTM
from scripts.train_seq import load_trained_model, test_model
from scripts.train_lumped_seq import test_lumped_mcmodel, test_lumped_mcrmodel, test_lumped_lstm
from scripts.dataloader_seq import load_data
from scripts.config import cfg
import pdb

print(f"[DIAG] cfg.SEQ_LENGTH={cfg.SEQ_LENGTH} cfg.PECAN_MODE={cfg.PECAN_MODE} cfg.MODEL_NAME={cfg.MODEL_NAME}")

# **Load dataset**
train_loader, val_loader, test_loader, routing_matrix, usgs_indices = load_data(cfg.MODEL_NAME.lower())
print(f"[DIAG] len(test_loader.dataset)={len(test_loader.dataset)} test_loader.dataset.seq_length={test_loader.dataset.seq_length}")
routing_matrix = routing_matrix.to(cfg.DEVICE)
usgs_indices = {
    key: torch.tensor(value["grid_index"], dtype=torch.long, device=cfg.DEVICE) 
    for key, value in usgs_indices.items()
}

# **Define model before loading checkpoint**
if cfg.MODEL_NAME.lower() == "pecan":
    # Process USGS indices for PECAN
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
    checkpoint_path = "checkpoints/pecan_best_model_mcrouting_retention_cs_0.80.2_180_0.2do_epoch80_randval_cosannealing_mcrdo0.2_retrain.pkl"

    # pecan_best_model_mcrouting_retention_cs_0.80.2_180_0.2do_epoch80_randval_cosannealing.pkl
    # "checkpoints/pecan_best_model_mcrouting_retention_cs_0.80.2_180_0.2do_epoch80_randval_cosannealing_mcr_do0.2.pkl"
    # pecan_best_model_mcrouting_retention_cs_0.80.2_180_0.2do_epoch80_randval_cosannealing_mc_vbrouting.pkl

elif cfg.MODEL_NAME.lower() == "mclstm":
    model = MassConservingLSTM(
        in_dim=cfg.IN_CHANNELS,
        aux_dim = cfg.AUX_CHANNELS,
        out_dim=cfg.OUT_CHANNELS,
        basin_data_root = cfg.BASIN_DATA_ROOT, 
        basin_id = cfg.BASIN_ID,
        batch_first = True
    ).to(cfg.DEVICE)
    checkpoint_path = "checkpoints/mclstm_best_model_180_hidden64.pkl"
    
elif cfg.MODEL_NAME.lower() == "mcrlstm":
    model = MassConservingLSTM_MR(in_dim = cfg.IN_CHANNELS, 
                        aux_dim = cfg.AUX_CHANNELS, 
                        out_dim = cfg.OUT_CHANNELS, 
                        time_dependent=False, 
                        batch_first=True,
                        basin_data_root = cfg.BASIN_DATA_ROOT, 
                        basin_id = cfg.BASIN_ID).to(cfg.DEVICE)
    checkpoint_path = "checkpoints/mcrlstm_best_model_180_hidden32.pkl"
    
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
    checkpoint_path = "checkpoints/lstm_best_model_180_hidden64.pkl"
    
    
# **Load trained model**
load_trained_model(model, checkpoint_path)

# **Plot predictions on training data & save plot and CSV**
if cfg.MODEL_NAME.lower() == "pecan":
    test_loss = test_model(model, test_loader)
elif cfg.MODEL_NAME.lower() == "mclstm":
    test_loss = test_lumped_mcmodel(model, test_loader)
elif cfg.MODEL_NAME.lower() == "mcrlstm":
    test_loss = test_lumped_mcrmodel(model, test_loader)
elif cfg.MODEL_NAME.lower() == "lstm":
    test_loss = test_lumped_lstm(model, test_loader, scaler_pickle  = "global_norm_stats_10180001.pkl")
print(f"Test Loss: {test_loss:.4f}")