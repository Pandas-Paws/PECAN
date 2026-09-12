import torch
import xarray as xr
import numpy as np
import glob
import pandas as pd
import json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from scripts.datautils import load_discharge, load_meteorological_data, compute_global_normalization_stats
from scripts.config import cfg
import pdb

MISSING_VALUE = -9999  # Placeholder for missing values

class PecanDataset(Dataset):
    def __init__(self, data_dir, var_names, basin_data_root, basin_id, years, routing_matrix_path, usgs_index_path, seq_length, model_type="pecan", normalize=True, vector_routing_path=None, norm_stats=None):
        """
        Load meteorological, streamflow, routing matrix, and USGS gauge indices.
        Converts data into sequential format for training.

        Parameters
        ----------
        seq_length : int
            Number of consecutive timesteps per sample (e.g., 30 days for a month-long sequence).
        """
        super().__init__()

        self.seq_length = seq_length
        self.model_type = model_type  # "pecan", "mclstm", "mcrlstm"
        
        self.data_dir = Path(data_dir)
        self.var_names = var_names
        self.basin_data_root = Path(basin_data_root)
        self.basin_id = basin_id
        self.years = years
        self.normalize = normalize
        self._norm_stats = norm_stats  # pre-computed (var_mean, var_std, q_mean, q_std) or None

        print(f"Initializing dataset for Basin ID: {self.basin_id} | Mode: {self.model_type}")

        self.routing_matrix = self._load_routing_matrix(routing_matrix_path)
        
        # Vector routing (newly added; does not replace the old grid-based routing)
        self.vector_routing = None
        if self.model_type == "pecan":
            if vector_routing_path is None:
                raise ValueError("For model_type='pecan', vector_routing_path must be provided.")
            self.vector_routing = torch.load(vector_routing_path, map_location="cpu")
        
        
        self.usgs_indices = self._load_usgs_indices(usgs_index_path)
        self.catchment_area = self._get_catchment_area()
        print(f"Catchment area loaded: {self.catchment_area} km2")

        # Load Streamflow Data (Aligned with Water Years)
        self.qobs, self.timestamps = self._load_streamflow()
        print(f"Streamflow data shape: {self.qobs.shape}")

        # Load Meteorological Data
        self.data, self.masks = load_meteorological_data(self.data_dir, self.var_names, self.years)
        print(f"Meteorological data shape: {self.data.shape}")
        # Compute or load normalization statistics
        #self.dataset_mean, self.dataset_std = self._load_or_compute_normalization_stats()
        self.var_mean, self.var_std, self.q_mean, self.q_std = self._load_or_compute_normalization_stats()

        # Align streamflow targets with input dataset
        self.targets = self._align_targets(len(self.data))
        print(f"Final targets shape: {self.targets.shape}")
        
        # Store spatial dimensions
        self.y_dim, self.x_dim = self.data.shape[2], self.data.shape[3]

    def _load_routing_matrix(self, routing_matrix_path):
        """Load the precomputed routing matrix from a .npy file."""
        return torch.tensor(np.load(routing_matrix_path, allow_pickle=True), dtype=torch.float32)

    def _load_usgs_indices(self, usgs_index_path):
        """Load USGS station indices for streamflow extraction."""
        with open(usgs_index_path, "r") as f:
            return json.load(f)

    def _get_catchment_area(self):
        """Retrieve the catchment area from the catchment info file."""
        attributes_path = self.basin_data_root / "catchment_info.txt"
        df = pd.read_csv(attributes_path, sep=";", dtype={"gauge_id": str})
        area = df.loc[df["basin_id"].astype(str) == self.basin_id, "area_gages2"].values[0]
        return float(area)

    def _load_streamflow(self):
        """Load USGS discharge (streamflow) data for all USGS gauges in this basin and align with water years."""
        discharge_dict = load_discharge(self.basin_data_root, cfg.USGS_INDEX_PATH)
        
        # --- Select only largest drainage area gauge for lumped models ---
        if self.model_type in ["mclstm", "mcrlstm", "lstm"]:
            with open(cfg.USGS_INDEX_PATH, "r") as f:
                usgs_info = json.load(f)
    
            sorted_usgs = sorted(usgs_info.items(), key=lambda x: x[1]["drainage_area_km2"], reverse=True)
    
            for usgs_id, info in sorted_usgs:
                gauge_id_str = usgs_id.replace("USGS_", "").zfill(8)  # match discharge_dict keys
                if gauge_id_str in discharge_dict:
                    discharge_dict = {gauge_id_str: discharge_dict[gauge_id_str]}
                    print(f"[INFO] Using USGS gauge {gauge_id_str} with drainage area {info['drainage_area_km2']} km2")
                    break
            else:
                raise ValueError("No matching USGS gauge found in discharge data.")

        
        discharge_df = pd.DataFrame(discharge_dict)
        discharge_df = discharge_df.where(discharge_df > 0, np.nan).fillna(MISSING_VALUE)
        discharge_df["Time"] = pd.to_datetime(discharge_df.index)
        discharge_df["WaterYear"] = discharge_df["Time"].apply(lambda x: x.year if x.month < 10 else x.year + 1)

        # Filter data for complete water years
        discharge_df, timestamps = self._filter_to_water_years(discharge_df)

        return torch.tensor(discharge_df.drop(columns=["Time", "WaterYear"]).values, dtype=torch.float32), timestamps

    def _filter_to_water_years(self, discharge_df):
        """Ensure sequences align with full water years (October 1 - September 30)."""
        valid_indices = []
        
        for year in self.years:
            start = pd.Timestamp(f"{year-1}-10-01")  # Water Year starts on Oct 1 of the previous year
            end = pd.Timestamp(f"{year}-09-30")  # Ends on Sep 30 of the current year
            
            indices = discharge_df[(discharge_df["Time"] >= start) & (discharge_df["Time"] <= end)].index
            valid_indices.extend(indices)
    
        return discharge_df.loc[valid_indices], discharge_df["Time"].loc[valid_indices]

    def _align_targets(self, length):
        """Ensure the number of targets matches the dataset length."""
        min_length = min(length, len(self.qobs))
        if length == len(self.qobs):
            print("Same length for streamflow data and meteorological data.")
        if min_length == 0:
            raise ValueError("Streamflow data is empty or misaligned.")
        return self.qobs[:min_length]

    def _load_or_compute_normalization_stats(self):
        """Load or compute global normalization statistics."""
        if self._norm_stats is not None:
            return self._norm_stats
        return compute_global_normalization_stats()

    def __len__(self):
        """Return the number of sequences, not timesteps."""
        return max(0, self.data.shape[0] - self.seq_length + 1)

    def __getitem__(self, idx):
        start_idx = idx
        end_idx = idx + self.seq_length

        sample_seq = torch.tensor(self.data[start_idx:end_idx], dtype=torch.float32)
        mask_seq = torch.tensor(self.masks[start_idx:end_idx], dtype=torch.float32)
        target_seq = self.targets[start_idx:end_idx][-1]
        timestamp_seq = self.timestamps[start_idx:end_idx]

        # Lumped models: mask ? areal mean
        if self.model_type in ["lstm", "mclstm", "mcrlstm"]:
            sample_seq = sample_seq * mask_seq
            valid_pix  = mask_seq.sum(dim=(2, 3), keepdim=True) + 1e-6
            sample_seq = sample_seq.sum(dim=(2, 3), keepdim=True) / valid_pix
            mask_seq   = mask_seq.mean(dim=(2, 3), keepdim=True)
    
        if self.normalize:
            mean = torch.tensor(self.var_mean, dtype=torch.float32).view(1,-1,1,1)
            std  = torch.tensor(self.var_std , dtype=torch.float32).view(1,-1,1,1)
    
            if self.model_type == "lstm":
                sample_seq = (sample_seq - mean) / (std + 1e-6)       # all channels
                target_seq = (target_seq - self.q_mean) / (self.q_std + 1e-6)
            else:  # mclstm / mcrlstm
                sample_seq[:, 1:] = (sample_seq[:, 1:] - mean[:, 1:]) / (std[:, 1:] + 1e-6)
            
        return sample_seq, mask_seq, target_seq, timestamp_seq.iloc[-1].strftime("%Y-%m-%d")

def load_data(model_type):
    """Splits data into train, validation, and test sets, ensuring alignment with the water year."""
    print("Loading datasets...")

    train_water_years = list(range(min(cfg.TRAIN_YEARS), max(cfg.TRAIN_YEARS) + 1))
    test_water_years = list(range(min(cfg.TEST_YEARS), max(cfg.TEST_YEARS) + 1))

    train_dataset = PecanDataset(cfg.DATA_DIR, cfg.VARIABLES, cfg.BASIN_DATA_ROOT, cfg.BASIN_ID, years=train_water_years, 
                                  routing_matrix_path=cfg.ROUTING_MATRIX_PATH, 
                                  usgs_index_path=cfg.USGS_INDEX_PATH, 
                                  seq_length=cfg.SEQ_LENGTH, 
                                  model_type=model_type,
                                  vector_routing_path=cfg.VECTOR_ROUTING_PATH, # vector-based routing
                                  )
    
    test_dataset = PecanDataset(cfg.DATA_DIR, cfg.VARIABLES, cfg.BASIN_DATA_ROOT, cfg.BASIN_ID, years=test_water_years, 
                                  routing_matrix_path=cfg.ROUTING_MATRIX_PATH, 
                                  usgs_index_path=cfg.USGS_INDEX_PATH, 
                                  seq_length=cfg.SEQ_LENGTH, 
                                  model_type=model_type,
                                  vector_routing_path=cfg.VECTOR_ROUTING_PATH, # vector-based routing
                                  )
    
    # Ensure spatial dimensions are properly set
    if hasattr(train_dataset, "y_dim") and hasattr(train_dataset, "x_dim"):
        cfg.Y_DIM = train_dataset.y_dim
        cfg.X_DIM = train_dataset.x_dim
        print(f"Updated cfg.Y_DIM = {cfg.Y_DIM}, cfg.X_DIM = {cfg.X_DIM}")
    else:
        raise ValueError("Error: y_dim and x_dim were not properly set in PecanDataset.")

    train_size = int(cfg.TRAIN_VAL_SPLIT * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, train_dataset.routing_matrix, train_dataset.usgs_indices
    
    # Return vector routing (None for non-pecan) + usgs row/col dict, todo
    #return train_loader, val_loader, test_loader, train_dataset.vector_routing, train_dataset.usgs_indices





def load_data_chronological(model_type):
    """Use contiguous water-year blocks and train-only normalization stats."""
    print("Loading datasets...")

    # Use years directly from cfg
    train_water_years = list(cfg.TRAIN_YEARS)
    val_water_years   = list(cfg.VAL_YEARS)
    test_water_years  = list(cfg.TEST_YEARS)

    # ---- Compute train-only normalization stats ONCE ----
    var_mean, var_std, q_mean, q_std = compute_global_normalization_stats()
    norm_stats = (var_mean, var_std, q_mean, q_std)
    print("[Norm] Using train-only stats "
          f"(vars C={len(var_mean)}, gauges G={len(np.atleast_1d(q_mean))})")

    # ---- Build datasets (all reuse the same stats) ----
    train_dataset = PecanDataset(
        cfg.DATA_DIR, cfg.VARIABLES, cfg.BASIN_DATA_ROOT, cfg.BASIN_ID,
        years=train_water_years,
        routing_matrix_path=cfg.ROUTING_MATRIX_PATH,
        usgs_index_path=cfg.USGS_INDEX_PATH,
        seq_length=cfg.SEQ_LENGTH,
        model_type=model_type,
        normalize=True,
        norm_stats=norm_stats,
        vector_routing_path=cfg.VECTOR_ROUTING_PATH,
    )

    val_dataset = PecanDataset(
        cfg.DATA_DIR, cfg.VARIABLES, cfg.BASIN_DATA_ROOT, cfg.BASIN_ID,
        years=val_water_years,
        routing_matrix_path=cfg.ROUTING_MATRIX_PATH,
        usgs_index_path=cfg.USGS_INDEX_PATH,
        seq_length=cfg.SEQ_LENGTH,
        model_type=model_type,
        normalize=True,
        norm_stats=norm_stats,
        vector_routing_path=cfg.VECTOR_ROUTING_PATH,
    )

    test_dataset = PecanDataset(
        cfg.DATA_DIR, cfg.VARIABLES, cfg.BASIN_DATA_ROOT, cfg.BASIN_ID,
        years=test_water_years,
        routing_matrix_path=cfg.ROUTING_MATRIX_PATH,
        usgs_index_path=cfg.USGS_INDEX_PATH,
        seq_length=cfg.SEQ_LENGTH,
        model_type=model_type,
        normalize=True,
        norm_stats=norm_stats,
        vector_routing_path=cfg.VECTOR_ROUTING_PATH,
    )

    # Ensure spatial dims are set
    if hasattr(train_dataset, "y_dim") and hasattr(train_dataset, "x_dim"):
        cfg.Y_DIM = train_dataset.y_dim
        cfg.X_DIM = train_dataset.x_dim
        print(f"Updated cfg.Y_DIM = {cfg.Y_DIM}, cfg.X_DIM = {cfg.X_DIM}")
    else:
        raise ValueError("Error: y_dim and x_dim were not properly set in PecanDataset.")

    # DataLoaders (shuffle only train)
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_dataset,  batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, train_dataset.routing_matrix, train_dataset.usgs_indices



      
def load_debug_data():
    """
    Loads a very small dataset for debugging by forcing overfitting.
    Returns small train_loader, val_loader, and test_loader.
    """
    print("Loading small dataset for debugging...")

    seq_length = cfg.SEQ_LENGTH

    # **Load Routing Matrix**
    routing_matrix = torch.tensor(np.load(cfg.ROUTING_MATRIX_PATH, allow_pickle=True), dtype=torch.float32)

    # **Load USGS Indices**
    with open(cfg.USGS_INDEX_PATH, "r") as f:
        usgs_indices = json.load(f)

    # **Initialize Tiny Train Dataset (Only 100 Samples)**
    small_train_dataset = PecanDataset(
        cfg.DATA_DIR, cfg.VARIABLES, cfg.BASIN_DATA_ROOT, cfg.BASIN_ID,
        years=cfg.TRAIN_YEARS, routing_matrix_path=cfg.ROUTING_MATRIX_PATH,
        usgs_index_path=cfg.USGS_INDEX_PATH,
        seq_length=seq_length
    )

    # **Limit to Only 8 Samples**
    small_train_dataset = torch.utils.data.Subset(small_train_dataset, range(20))  

    # **Create Small DataLoaders**
    small_train_loader = DataLoader(small_train_dataset, batch_size=2, shuffle=True, num_workers=0)
    small_val_loader = DataLoader(small_train_dataset, batch_size=2, shuffle=False, num_workers=0)  # Use the same small dataset for validation

    print(f"Small dataset size: {len(small_train_dataset)} samples")

    return small_train_loader, small_val_loader, routing_matrix, usgs_indices



