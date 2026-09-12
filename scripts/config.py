# scripts/config.py
import os
import torch

def _envbool(name, default):
    v = os.environ.get(name)
    return default if v is None else v == "1"

def _envfloat(name, default):
    v = os.environ.get(name)
    return default if v is None else float(v)

def _envstr(name, default):
    return os.environ.get(name, default)

def _envint(name, default):
    v = os.environ.get(name)
    return default if v is None else int(v)

class Config:
    # Basin ID
    # BASIN_ID = "06623800"  # CAMELS
    MODEL_NAME = "pecan" # pecan, lstm, mclstm, mcrlstm
    PECAN_MODE = _envstr("PECAN_MODE", "mcr") # choose from ['mcr', 'mc']. Only functional when MODEL_NAME is 'pecan'.
    MC_SPATIAL_RETENTION = _envbool("PECAN_SPATIAL_RETENTION", False) # only used when PECAN_MODE='mc'; toggles per-cell vs global-scalar retention_leak_param
    MC_DUAL_RETENTION = _envbool("PECAN_DUAL_RETENTION", False) # only used when PECAN_MODE='mc'; splits retention_leak_param into separate recharge/drainage rates
    MC_GROUPNORM_INGATE = _envbool("PECAN_GROUPNORM_INGATE", True) # only used when PECAN_MODE='mc'; swaps in_gate's BatchNorm2d for GroupNorm
    MC_GROUPNORM_GROUPS = _envint("PECAN_GROUPNORM_GROUPS", 8) # only used when MC_GROUPNORM_INGATE=True
    MC_LAYERNORM_INGATE = _envbool("PECAN_LAYERNORM_INGATE", False) # only used when PECAN_MODE='mc'; full LayerNorm instead of GroupNorm/BatchNorm2d for in_gate
    MC_GROUPNORM_OUTGATE = _envbool("PECAN_GROUPNORM_OUTGATE", False) # GroupNorm instead of LayerNorm for out_gate (applies to both MC and MCR now)
    MC_GROUPNORM_GROUPS_OUTGATE = None # only used when MC_GROUPNORM_OUTGATE=True; independent granularity for out_gate GroupNorm (defaults to MC_GROUPNORM_GROUPS if None)
    MC_DROPOUT2D_INGATE = _envbool("PECAN_DROPOUT2D_INGATE", True) # channel-wise Dropout2d instead of elementwise Dropout for in_gate (applies to both MC and MCR now)
    MC_DROPOUT2D_OUTGATE = False # only used when PECAN_MODE='mc'; adds a Dropout2d layer after out_gate's Sigmoid. DO NOT ENABLE: Dropout2d's train-time inverse scaling (1/(1-p)) can push the post-Sigmoid gate value o above 1.0, and MC mode's mass update c_next_main=(1.0-o)*m_new has no clamp (unlike MCR's clamped o_prime version), so o>1 produces negative "mass" that corrupts the recurrent state. Confirmed via state_l1_norm negative-value warnings firing thousands of times within seconds of launch.
    RVIC_KERNEL_PATH = _envstr("PECAN_RVIC_KERNEL_PATH", "watershed_data/routing_related/10180001/rvic_kernel_nhdplus_D800.npy") # None = model's default (DEM-derived, literature-grounded velocity grid, D=300). Held fixed at the uniform v=1.0 m/s / D=800 m2/s kernel (same as MC) across the sweep below, per explicit instruction.
    BASIN_ID = "10180001" # HUC 8
    
    
    # Data paths
    DATA_DIR = f"processed_data_{BASIN_ID}_PRISM/"
    
    # Define the root directory where CAMELS data is stored
    BASIN_DATA_ROOT = "watershed_data"
   
    # Climate variables to load
    VARIABLES = ["precip", "tmax", "tmin"]


    # Paths for Routing Matrix & USGS Gauge Indices
    ROUTING_MATRIX_PATH = f"watershed_data/routing_related/{BASIN_ID}/routing_matrix.npy"
    USGS_INDEX_PATH = f"watershed_data/routing_related/{BASIN_ID}/usgs_indices.json"

    # Training parameters
    BATCH_SIZE = _envint("PECAN_BATCH_SIZE", 64)
    LEARNING_RATE = _envfloat("PECAN_LR", 0.00075)
    LR_MIN = _envfloat("PECAN_LR_MIN", 1e-4)
    WEIGHT_DECAY = _envfloat("PECAN_WEIGHT_DECAY", 0.0)
    NUM_EPOCHS = _envint("PECAN_NUM_EPOCHS", 180)
    # NOTE: sweep in progress -- LR set per-launch below
    
    # PECAN_CHRONO_SPLIT=1 → contiguous water-year blocks (no leakage)
    # PECAN_CHRONO_SPLIT=0 (default) → random 85/15 split within training period
    CHRONO_SPLIT = _envbool("PECAN_CHRONO_SPLIT", False)

    # Water-year splits
    # Chrono: contiguous blocks, no leakage; Random: full train period, random 85/15
    if CHRONO_SPLIT:
        TRAIN_YEARS = list(range(1982, 2008))   # 26 WYs
        VAL_YEARS   = list(range(2008, 2013))   # 5  WYs
    else:
        TRAIN_YEARS = list(range(1982, 2013))   # 31 WYs (random split below)
        VAL_YEARS   = []                         # unused by load_data()
    TEST_YEARS      = list(range(2013, 2023))   # WY 2013 - WY 2022. 10 years
    TRAIN_VAL_SPLIT = 0.85                      # used only when CHRONO_SPLIT=False

    # Model parameters
    SEQ_LENGTH = _envint("PECAN_SEQ_LENGTH", 180)
    IN_CHANNELS = 1  # Precip as the only mass input
    AUX_CHANNELS = len(VARIABLES) - 1  # Remaining input in addition to precip
    OUT_CHANNELS = _envint("PECAN_OUT_CHANNELS", 32)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Routing
    VECTOR_ROUTING_PATH = f"watershed_data/routing_related/{BASIN_ID}/vector_routing_10180001.pt"
    CELL_TO_REACH_CSV   = f"watershed_data/routing_related/{BASIN_ID}/cell_to_catch_area.csv"
    
    # NaN Handling
    MISSING_VALUE = -9999  # Placeholder for missing values

    # Set dynamically from `dataloader.py`
    Y_DIM = None
    X_DIM = None

cfg = Config()  # Instantiate config object
