# -*- coding: utf-8 -*-
import sqlite3
from pathlib import Path, PosixPath
from typing import List, Tuple
import numpy as np
import pandas as pd
from numba import njit
import pdb
import json
import glob
import xarray as xr
import pickle
from scripts.config import cfg


MISSING_VALUE = -9999

def load_discharge(basin_root: PosixPath, usgs_json_path: str) -> dict:
    '''
    Load daily streamflow discharge data for all USGS gauges listed in the JSON file.

    Parameters
    ----------
    basin_root : PosixPath
        Path to the main directory of the basin dataset.
    usgs_json_path : str
        Path to the JSON file containing USGS gauge mappings.

    Returns
    -------
    dict
        A dictionary where keys are USGS gauge IDs (padded to 8 digits) and values are pandas Series of discharge values.

    Raises
    ------
    RuntimeError
        If no discharge file is found for any USGS gauge.
    '''
    # Load USGS gauge ID mappings from JSON
    with open(usgs_json_path, "r") as f:
        usgs_gauge_mapping = json.load(f)  # JSON keys are USGS gauge IDs

    # Path to streamflow data
    discharge_path = basin_root / "usgs_streamflow"
    discharge_data = {}

    for gauge_id, info in usgs_gauge_mapping.items():
        gauge_id_str = gauge_id.replace("USGS_", "").zfill(8)

        # Ensure drainage area is in the JSON
        if "drainage_area_km2" not in info:
            print(f"Warning: Missing drainage area for USGS {gauge_id_str}. Skipping...")
            continue

        drainage_area_km2 = info["drainage_area_km2"]

        # Search for the corresponding discharge file
        files = list(discharge_path.glob(f"**/{gauge_id_str}_streamflow_qc.txt"))

        if not files:
            print(f"Warning: No file found for USGS Gauge {gauge_id_str}")
            continue  # Skip to next gauge

        file_path = files[0]  # Take the first matching file

        # Load discharge data
        col_names = ["gauge_id", "Year", "Mnth", "Day", "QObs", "flag"]
        df = pd.read_csv(file_path, sep="\s+", header=None, names=col_names)

        # Create datetime index
        df.index = pd.to_datetime(df.Year.astype(str) + "/" + df.Mnth.astype(str) + "/" + df.Day.astype(str), format="%Y/%m/%d")

        # Normalize discharge from cubic feet per second to mm per day (using individual drainage area)
        if cfg.MODEL_NAME == 'lstm':
            df.QObs = df.QObs * 2.44657554555 / drainage_area_km2

        # Store in dictionary
        discharge_data[gauge_id_str] = df.QObs

    if not discharge_data:
        raise RuntimeError(f"No discharge data found for any USGS gauge listed in {usgs_json_path}")

    return discharge_data
    
    
def load_meteorological_data(data_dir: str, var_names: list, years: list) -> tuple:
    """
    Load and combine meteorological variables from multiple NetCDF files.

    Parameters
    ----------
    data_dir : str
        Path to directory containing NetCDF files.
    var_names : list of str
        List of variable names to extract (e.g., ['tmax', 'tmin', 'precip']).
    years : list of int
        List of years to include (e.g., train or test years).

    Returns
    -------
    tuple
        (data, masks) where:
        - data (numpy array): Meteorological data of shape (time, channels, y, x).
        - masks (numpy array): Boolean mask indicating valid data points.
    """

    all_data = []
    all_masks = []

    for var in var_names:
        var_files = sorted(glob.glob(f"{data_dir}/{var}_*.nc"))  # Load all NetCDF files for this variable

        if not var_files:
            raise ValueError(f"No NetCDF files found for variable '{var}' in {data_dir}")

        var_data = []
        var_mask = []

        for file in var_files:
            ds = xr.open_dataset(file)
            time_index = pd.to_datetime(ds.time.values)

            # Convert to water year (Oct 1 - Sep 30)
            water_years = np.where(time_index.month >= 10, time_index.year + 1, time_index.year)
            ds = ds.assign_coords(water_year=("time", water_years))
            

            # **Filter only selected years**
            ds = ds.sel(time=ds["water_year"].isin(years))
            if len(ds.time) == 0:
                continue  # Skip if no valid data in selected years
                
            data_array = ds[var].values  # Shape: (time, y, x)

            # **Flip data to match spatial alignment**
            data_array = np.flip(data_array, axis=1)  

            # Create mask for valid data points (1 = valid, 0 = missing)
            mask = ~np.isnan(data_array)
            mask = mask.astype(np.float32)

            # Replace NaNs with MISSING_VALUE
            data_array = np.where(np.isnan(data_array), MISSING_VALUE, data_array)

            var_data.append(data_array)
            var_mask.append(mask)

            ds.close()

        if var_data:
            var_data = np.concatenate(var_data, axis=0)  # Shape: (time, y, x)
            var_mask = np.concatenate(var_mask, axis=0)  # Shape: (time, y, x)
        else:
            var_data = np.zeros((1, 1, 1))  # Placeholder with minimal shape
            var_mask = np.zeros((1, 1, 1))


        all_data.append(var_data)
        all_masks.append(var_mask)

    # Stack variables along the channel dimension (time, channels, y, x)
    data = np.stack(all_data, axis=1)
    masks = np.stack(all_masks, axis=1)

    return data, masks

    
def compute_global_normalization_stats():
    pkl = Path(f"global_norm_stats_{cfg.BASIN_ID}.pkl")
    
    if pkl.exists():
        with pkl.open("rb") as f:
            stats = pickle.load(f)
        print("read norm pkl.")
        return stats["var_mean"], stats["var_std"], stats["q_mean"], stats["q_std"]

    
    # -- 1.  Forcings ---------------------------------------------
    data, masks = load_meteorological_data(Path(cfg.DATA_DIR),
                                           cfg.VARIABLES, cfg.TRAIN_YEARS)
    # masks ==1 valid, 0 invalid
    data   = np.where(masks > 0, data, np.nan)       # put NaN where missing
    var_mean = np.nanmean(data, axis=(0, 2, 3))      # C-vector
    var_std  = np.nanstd (data, axis=(0, 2, 3))

    # ------------------------------------------------------------------
    # 2. Streamflow — SINGLE GAUGE ONLY
    # ------------------------------------------------------------------
    discharge = load_discharge(Path(cfg.BASIN_DATA_ROOT), cfg.USGS_INDEX_PATH)

    # Load gauge metadata
    with open(cfg.USGS_INDEX_PATH, "r") as f:
        usgs_info = json.load(f)

    # Sort gauges by drainage area (descending)
    sorted_usgs = sorted(
        usgs_info.items(),
        key=lambda x: x[1]["drainage_area_km2"],
        reverse=True
    )

    # Pick the largest gauge that actually exists in discharge dict
    q = None
    chosen_gauge = None
    for usgs_id, info in sorted_usgs:
        gid = usgs_id.replace("USGS_", "").zfill(8)
        if gid in discharge:
            q = discharge[gid]
            chosen_gauge = gid
            break

    if q is None:
        raise RuntimeError("No matching USGS gauge found for normalization.")

    print(f"[Norm] Using single gauge {chosen_gauge} "
          f"(area={usgs_info['USGS_' + chosen_gauge]['drainage_area_km2']} km2)")

    # Restrict to TRAIN years only (same logic as dataset)
    q = q[(q.index.year >= min(cfg.TRAIN_YEARS) - 1) &
          (q.index.year <= max(cfg.TRAIN_YEARS))]

    q = q.replace(-9999, np.nan).dropna()

    q_mean = float(q.mean())
    q_std  = float(q.std())

    # ------------------------------------------------------------------
    # 3. Cache stats
    # ------------------------------------------------------------------
    with pkl.open("wb") as f:
        pickle.dump(
            {
                "var_mean": var_mean,
                "var_std":  var_std,
                "q_mean":   q_mean,
                "q_std":    q_std,
                "gauge":    chosen_gauge
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL
        )

    return var_mean, var_std, q_mean, q_std



def compute_global_normalization_stats_chronological():
    """
    Train-only normalization stats for the current basin and TRAIN_YEARS.
    - Forcings: per-channel mean/std over [time, y, x], masking invalid pixels.
    - Streamflow: per-gauge mean/std, filtered by WATER YEARS in TRAIN_YEARS.
    Caches to disk with a key that depends on basin, variables, and year span.
    """
    # Cache key includes basin, vars, and train-year span
    key = f"{cfg.BASIN_ID}_{min(cfg.TRAIN_YEARS)}-{max(cfg.TRAIN_YEARS)}_{tuple(cfg.VARIABLES)}"
    pkl = Path(f"global_norm_stats_{key}.pkl")

    if pkl.exists():
        with pkl.open("rb") as f:
            stats = pickle.load(f)
        return stats["var_mean"], stats["var_std"], stats["q_mean"], stats["q_std"]

    # ---------- 1) Forcings (train years only) ----------
    data, masks = load_meteorological_data(Path(cfg.DATA_DIR), cfg.VARIABLES, cfg.TRAIN_YEARS)
    # masks: 1 valid, 0 invalid
    data = np.where(masks > 0, data, np.nan)                          # [T, C, Y, X]
    var_mean = np.nanmean(data, axis=(0, 2, 3))                       # [C]
    var_std  = np.nanstd (data, axis=(0, 2, 3))
    var_std  = np.maximum(var_std, 1e-6)

    # If you keep precip (channel 0) unnormalized in PECAN, lock it to 0/1:
    # (PECAN normalizes aux channels only; this keeps arrays aligned)
    # var_mean[0] = 0.0
    # var_std[0]  = 1.0

    # ---------- 2) Streamflow (per gauge, TRAIN water years only) ----------
    discharge = load_discharge(Path(cfg.BASIN_DATA_ROOT), cfg.USGS_INDEX_PATH)
    # Make a DataFrame with gauges as columns
    q_df = pd.DataFrame(discharge)                         # index = datetime, columns = gauges
    q_df = q_df.replace(MISSING_VALUE, np.nan)

    # Water year filter: WY = year + (month >= 10)
    wy = q_df.index.year + (q_df.index.month >= 10)
    mask_wy = (wy >= min(cfg.TRAIN_YEARS)) & (wy <= max(cfg.TRAIN_YEARS))
    q_df = q_df.loc[mask_wy]

    q_mean = q_df.mean(axis=0).to_numpy()                  # [num_gauges]
    q_std  = q_df.std (axis=0).to_numpy()
    q_std  = np.maximum(q_std, 1e-6)

    with pkl.open("wb") as f:
        pickle.dump({"var_mean": var_mean, "var_std": var_std,
                     "q_mean": q_mean,   "q_std": q_std}, f)
    return var_mean, var_std, q_mean, q_std

