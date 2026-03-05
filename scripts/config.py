# scripts/config.py
import torch

class Config:        
    # Basin ID
    # BASIN_ID = "06623800"  # CAMELS
    MODEL_NAME = "pecan" # pecan, lstm, mclstm, mcrlstm
    PECAN_MODE = 'mcr' # choose from ['mcr', 'mc']. Only functional when MODEL_NAME is 'pecan'.
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
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 80
    
    TRAIN_YEARS = list(range(1982, 2013))  # MUST BE WATER YEARS! 31 years
    TEST_YEARS = list(range(2013, 2023))   # WY 2013 - WY 2022. 10 years
    TRAIN_VAL_SPLIT = 0.85  # train/val split
    
    # ==== Splits by whole WATER YEARS (contiguous) ====
    # Train: WY1982–WY2008, Val: WY2009–WY2012, Test: WY2013–WY2022
    #TRAIN_YEARS = list(range(1982, 2008))   # 26 WYs
    #VAL_YEARS   = list(range(2008, 2013))   # 5  WYs
    #TEST_YEARS  = list(range(2013, 2023))   # WY 2014 - WY 2022. 10 years

    # Model parameters
    SEQ_LENGTH = 60
    IN_CHANNELS = 1  # Precip as the only mass input
    AUX_CHANNELS = len(VARIABLES) - 1  # Remaining input in addition to precip
    OUT_CHANNELS = 32
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
