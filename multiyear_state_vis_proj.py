#!/usr/bin/env python3
# Updated: save ALL figures to /scratch/yihan/PECAN/

import os, warnings, re
from datetime import datetime
from collections import defaultdict

import numpy as np
import xarray as xr
import rioxarray
import geopandas as gpd
import matplotlib.pyplot as plt
from affine import Affine
from pyproj import CRS, Transformer

from met_brewer import met_brew
from matplotlib.colors import LinearSegmentedColormap

from scripts.datautils import load_meteorological_data
import pdb

# =========================
# USER SETTINGS
# =========================
BASIN_ID = "10180001"
DATA_DIR = "/home/eecs/erichson/yihan/PECAN/"
PRISM_DATA_DIR = "/home/eecs/erichson/yihan/PECAN/processed_data_10180001_PRISM/"
SHAPEFILE_PATH = "/home/eecs/erichson/yihan/PECAN/HUC8_NorthPlatteHeadwater/shapefile/NorthPlatteHeadwater.shp"
PRISM_NC_PATH = "/home/eecs/erichson/yihan/PECAN/processed_data_10180001_PRISM/tmin_2022_regridded_clipped_utm12.nc"

MODEL_MCR = "mcr_retrain"
MODEL_MC  = "mc_sq60_lr0.0015"

# ---- NEW: where to save all figures ----
FIG_DIR = "/scratch/yihan/PECAN/"

# Flip tiles on load? (only if .npy were written south-up)
FLIP_INPUT_NPY_UPDOWN = False

# Treat explicit zeros as "no data" for *states*, not for flux/gate
TREAT_ZERO_AS_NODATA_STATES = True
ZERO_NODATA_EPS = 1e-10

DEFAULT_FALLBACK_EPSG_IF_UNKNOWN = "EPSG:32612"

# Choose a MetBrewer palette (optional)
colors = met_brew("Hokusai2", brew_type="continuous")
cmap_cont = LinearSegmentedColormap.from_list("cmap_cont", colors)

# MR gate plotting range (typical gates are [0,1])
MRGATE_VMIN = 0.0
MRGATE_VMAX = 1.0
MRGATE_CMAP = "turbo"

# =========================
# HELPERS
# =========================
def _log(s): print(f"[info] {s}")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def outfig(name: str) -> str:
    """Return full output path under FIG_DIR."""
    return os.path.join(FIG_DIR, name)

def safe_stack_mean(list_of_arrays):
    if not list_of_arrays:
        return None
    arrs = [np.asarray(a) for a in list_of_arrays
            if a is not None and np.asarray(a).size and np.isfinite(a).any()]
    if not arrs:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(np.stack(arrs), axis=0)

def load_matrices_by_date(folder):
    files = sorted([f for f in os.listdir(folder) if f.endswith(".npy")]) if os.path.isdir(folder) else []
    matrices, dates = [], []
    for f in files:
        m = re.search(r"(\d{8})", f)  # YYYYMMDD anywhere
        if not m:
            continue
        try:
            dt = datetime.strptime(m.group(1), "%Y%m%d")
        except Exception:
            continue
        try:
            A = np.load(os.path.join(folder, f))
            if FLIP_INPUT_NPY_UPDOWN:
                A = np.flipud(A)
            matrices.append(A)
            dates.append(dt)
        except Exception:
            continue
    return matrices, dates

def guess_crs_for_crsless_shapefile(gdf: gpd.GeoDataFrame) -> CRS:
    xmin, ymin, xmax, ymax = gdf.total_bounds
    if (xmin >= -180 and xmax <= 180 and ymin >= -90 and ymax <= 90):
        _log("Shapefile has no CRS; bounds look like lon/lat. Assuming EPSG:4326.")
        return CRS.from_epsg(4326)
    _log(f"Shapefile has no CRS; assuming {DEFAULT_FALLBACK_EPSG_IF_UNKNOWN}.")
    return CRS.from_string(DEFAULT_FALLBACK_EPSG_IF_UNKNOWN)

def get_netcdf_grid_and_crs(nc_path: str):
    if not os.path.exists(nc_path):
        raise FileNotFoundError(f"PRISM NetCDF not found: {nc_path}")
    ds = xr.open_dataset(nc_path)
    if len(ds.data_vars) == 0:
        raise RuntimeError("NetCDF has no data variables.")
    varname = list(ds.data_vars)[0]
    da = ds[varname]
    if not da.rio.crs:
        raise RuntimeError("NetCDF variable has no CRS. Write the CRS via da.rio.write_crs(...).")
    tf: Affine = da.rio.transform()
    H = da.sizes[da.rio.y_dim]
    W = da.sizes[da.rio.x_dim]
    return tf, W, H, da.rio.crs

def build_edges_from_northup_affine(tf: Affine, W: int, H: int):
    x_edges = tf.c + tf.a * np.arange(W + 1)   # a > 0
    y_edges = tf.f + tf.e * np.arange(H + 1)   # e < 0 for north-up
    XE, YE = np.meshgrid(x_edges, y_edges)
    return XE, YE

def normalize_edges_and_data(XE, YE, A):
    if A is None:
        return XE, YE, A

    # 1) If X decreases to the right, flip horizontally
    if XE.shape[1] > 1 and XE[0, 1] < XE[0, 0]:
        XE = np.fliplr(XE)
        YE = np.fliplr(YE)
        A  = np.fliplr(A)

    # 2) If Y decreases downward (north-up), flip edges to south-up,
    #    and flip A accordingly so the plotted raster aligns.
    if YE.shape[0] > 1 and YE[1, 0] < YE[0, 0]:
        XE = np.flipud(XE)
        YE = np.flipud(YE)
    A = np.flipud(A)

    return XE, YE, A

def compute_global_vmax(dicts, default=1.0):
    vals = []
    for d in dicts:
        for arr in d.values():
            if arr is not None and np.isfinite(arr).any():
                vals.append(np.nanmax(arr))
    return float(np.nanpercentile(vals, 98)) if vals else float(default)

def compute_global_vabsmax(dicts, default=1.0):
    vals = []
    for d in dicts:
        for arr in d.values():
            if arr is not None and np.isfinite(arr).any():
                vals.append(np.nanmax(np.abs(arr)))
    return float(np.nanpercentile(vals, 98)) if vals else float(default)

def _apply_nodata_states(arr):
    if not TREAT_ZERO_AS_NODATA_STATES:
        return arr
    out = arr.copy()
    out[np.isfinite(out) & (np.abs(out) <= ZERO_NODATA_EPS)] = np.nan
    return out

def add_dict_arrays(d1, d2):
    keys = sorted(set(d1.keys()) | set(d2.keys()))
    out = {}
    for k in keys:
        a = d1.get(k, None)
        b = d2.get(k, None)
        if a is None or b is None:
            out[k] = None
            continue
        out[k] = np.asarray(a) + np.asarray(b)
    return out

def subtract_dict_arrays(d1, d2):
    keys = sorted(set(d1.keys()) | set(d2.keys()))
    out = {}
    for k in keys:
        a = d1.get(k, None)
        b = d2.get(k, None)
        if a is None or b is None:
            out[k] = None
            continue
        out[k] = np.asarray(a) - np.asarray(b)
    return out

def monthly_mean_from_daily_datetime_dict(daily_dt_dict):
    by_month = defaultdict(list)
    for dt, A in daily_dt_dict.items():
        if A is None or not np.isfinite(A).any():
            continue
        by_month[dt.month].append(A)
    return {m: safe_stack_mean(v) for m, v in by_month.items()}

def preprocess_field_dict(field_dict, mask, is_flux: bool):
    out = {}
    for dt, A in field_dict.items():
        if A is None:
            out[dt] = None
            continue
        A = np.asarray(A).squeeze()
        A = np.where(mask == 1, A, np.nan)
        if not is_flux:
            A = _apply_nodata_states(A)
        out[dt] = A
    return out
    
def scale_dict_arrays(d, scale):
    """Return new dict with each array divided by scale."""
    out = {}
    for k, A in d.items():
        if A is None:
            out[k] = None
        else:
            out[k] = np.asarray(A) / float(scale)
    return out
def load_model_outputs(model_name: str, want_cell_retention: bool, want_mrflux: bool, want_mrgate: bool):
    dir_trash = os.path.join(DATA_DIR, f"trash_cell_outputs_{model_name}")
    trash_mats, trash_dates = load_matrices_by_date(dir_trash)
    trash = dict(zip(trash_dates, trash_mats))

    cell = retention = {}
    if want_cell_retention:
        dir_cell = os.path.join(DATA_DIR, f"cell_state_outputs_{model_name}")
        dir_ret  = os.path.join(DATA_DIR, f"retention_state_outputs_{model_name}")
        cell_mats, cell_dates = load_matrices_by_date(dir_cell)
        ret_mats,  ret_dates  = load_matrices_by_date(dir_ret)
        cell = dict(zip(cell_dates, cell_mats))
        retention = dict(zip(ret_dates, ret_mats))

    mrflux = {}
    if want_mrflux:
        dir_mr = os.path.join(DATA_DIR, f"mr_flux_outputs_{model_name}")
        mr_mats, mr_dates = load_matrices_by_date(dir_mr)
        mrflux = dict(zip(mr_dates, mr_mats))

    mrgate = {}
    if want_mrgate:
        dir_g = os.path.join(DATA_DIR, f"mr_gate_outputs_{model_name}")
        g_mats, g_dates = load_matrices_by_date(dir_g)
        mrgate = dict(zip(g_dates, g_mats))

    return trash, cell, retention, mrflux, mrgate

# =========================
# MAKE OUTPUT DIR
# =========================
ensure_dir(FIG_DIR)
_log(f"All figures will be saved to: {FIG_DIR}")

# =========================
# LOAD MASK
# =========================
try:
    _, masks = load_meteorological_data(PRISM_DATA_DIR, ["precip"], [2013])
    mask = masks[0, 0, :, :]
    H_mask, W_mask = mask.shape
    _log(f"Mask loaded from datautils: shape (H,W)=({H_mask},{W_mask})")
except Exception as e:
    _log(f"Could not load mask via load_meteorological_data: {e}")
    mask = None
    H_mask = W_mask = None

# =========================
# GEO
# =========================
tf, W_nc, H_nc, nc_crs = get_netcdf_grid_and_crs(PRISM_NC_PATH)
_log(f"NetCDF CRS: {nc_crs}")
_log(f"NetCDF size: W={W_nc}, H={H_nc}")

if (H_mask is not None) and (H_mask != H_nc or W_mask != W_nc):
    _log(f"WARNING: mask shape ({H_mask},{W_mask}) != NetCDF shape ({H_nc},{W_nc}). Using NetCDF shape for edges.")

if not os.path.exists(SHAPEFILE_PATH):
    raise FileNotFoundError(f"Shapefile not found: {SHAPEFILE_PATH}")

gdf = gpd.read_file(SHAPEFILE_PATH)
if gdf.crs is None:
    shp_crs = guess_crs_for_crsless_shapefile(gdf)
    gdf = gdf.set_crs(shp_crs)
else:
    shp_crs = gdf.crs
_log(f"Shapefile CRS: {shp_crs}")

XE_nc, YE_nc = build_edges_from_northup_affine(tf, W_nc, H_nc)
to_shp = Transformer.from_crs(nc_crs, shp_crs, always_xy=True)
XE, YE = to_shp.transform(XE_nc, YE_nc)

gdf_shp = gdf.to_crs(shp_crs)
xmin, ymin, xmax, ymax = gdf_shp.total_bounds
xlim = (xmin, xmax)
ylim = (ymin, ymax)

# choose mask: prefer loaded mask else all-ones of NetCDF shape
if mask is None or mask.shape != (H_nc, W_nc):
    if mask is not None and mask.shape != (H_nc, W_nc):
        _log("WARNING: mask shape != NetCDF grid; ignoring mask.")
    mask = np.ones((H_nc, W_nc), dtype=np.uint8)

# =========================
# PLOTTING
# =========================
def overlay_basin(ax):
    gdf_shp.boundary.plot(ax=ax, color="black", linewidth=2)

def plot_monthly(field_name, data_dict, vmax, title, fname, cmap="turbo", vmin=0.0):
    fig, axes = plt.subplots(2, 6, figsize=(16, 5))
    fig.subplots_adjust(left=0, right=0.90, top=1, bottom=0, wspace=0, hspace=0)

    for ax in axes.flatten():
        for spine in ax.spines.values():
            spine.set_visible(False)

    last_im = None

    for i, month in enumerate(range(1, 13)):
        r, c = divmod(i, 6)
        ax = axes[r, c]
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_anchor("NW")
        ax.margins(0)

        arr = data_dict.get(month)
        if arr is not None and np.isfinite(arr).any():
            XEp, YEp, arrp = normalize_edges_and_data(XE, YE, arr)
            last_im = ax.pcolormesh(
                XEp, YEp, arrp,
                cmap=cmap,
                vmin=vmin, vmax=vmax,
                shading="auto"
            )

        overlay_basin(ax)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0, pos.width, pos.height])

    if last_im is not None:
        cbar_ax = fig.add_axes([0.92, 0.12, 0.02, 0.65])
        fig.colorbar(last_im, cax=cbar_ax, label=field_name)

    outpath = outfig(fname)
    fig.savefig(outpath, dpi=600, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    _log(f"Saved: {outpath}")

def plot_monthly_diverging(field_name, data_dict, vabs, title, fname, cmap="bwr"):
    plot_monthly(field_name, data_dict, vmax=+vabs, title=title, fname=fname, cmap=cmap, vmin=-vabs)

# =========================
# LOAD DAILY MATRICES (MCR + MC)
# =========================
trash_mcr, cell_mcr, retention_mcr, mrflux_mcr, mrgate_mcr = load_model_outputs(
    MODEL_MCR, want_cell_retention=True, want_mrflux=True, want_mrgate=True
)
trash_mc, _, _, _, _ = load_model_outputs(
    MODEL_MC, want_cell_retention=False, want_mrflux=False, want_mrgate=False
)

# preprocess (mask + nodata for states)
trash_mcr        = preprocess_field_dict(trash_mcr,        mask, is_flux=False)
cell_mcr         = preprocess_field_dict(cell_mcr,         mask, is_flux=False)
retention_mcr    = preprocess_field_dict(retention_mcr,    mask, is_flux=False)
mrflux_mcr       = preprocess_field_dict(mrflux_mcr,       mask, is_flux=True)
mrgate_mcr       = preprocess_field_dict(mrgate_mcr,       mask, is_flux=False)
trash_mc         = preprocess_field_dict(trash_mc,         mask, is_flux=False)

_log(f"MCR: trash={len(trash_mcr)}, cell={len(cell_mcr)}, retention={len(retention_mcr)}, mrflux={len(mrflux_mcr)}, mrgate={len(mrgate_mcr)}")
_log(f"MC : trash={len(trash_mc)}")

# =========================
# BUILD MONTHLY MEANS (MCR)
# =========================
common_dt_mcr_all = sorted(
    set(trash_mcr.keys())
    & set(cell_mcr.keys())
    & set(retention_mcr.keys())
    & set(mrflux_mcr.keys())
    & set(mrgate_mcr.keys())
)
_log(f"MCR common dates (trash+cell+retention+mrflux+mrgate) = {len(common_dt_mcr_all)}")

trash_mcr_aligned      = {dt: trash_mcr[dt] for dt in common_dt_mcr_all}
cell_mcr_aligned       = {dt: cell_mcr[dt] for dt in common_dt_mcr_all}
retention_mcr_aligned  = {dt: retention_mcr[dt] for dt in common_dt_mcr_all}
mrflux_mcr_aligned     = {dt: mrflux_mcr[dt] for dt in common_dt_mcr_all}
mrgate_mcr_aligned     = {dt: mrgate_mcr[dt] for dt in common_dt_mcr_all}

monthly_trash_mcr      = monthly_mean_from_daily_datetime_dict(trash_mcr_aligned)
monthly_cell_mcr       = monthly_mean_from_daily_datetime_dict(cell_mcr_aligned)
monthly_retention_mcr  = monthly_mean_from_daily_datetime_dict(retention_mcr_aligned)
monthly_mrflux_mcr     = monthly_mean_from_daily_datetime_dict(mrflux_mcr_aligned)
monthly_mrgate_mcr     = monthly_mean_from_daily_datetime_dict(mrgate_mcr_aligned)

monthly_fast_mcr = {}
for m in range(1, 13):
    a = monthly_cell_mcr.get(m, None)
    b = monthly_retention_mcr.get(m, None)
    monthly_fast_mcr[m] = None if (a is None or b is None) else (a - b)

g_trash_vmax_mcr      = compute_global_vmax([monthly_trash_mcr], default=1.0)
g_retention_vmax_mcr  = compute_global_vmax([monthly_retention_mcr], default=1.0)
g_fast_vmax_mcr       = compute_global_vmax([monthly_fast_mcr], default=1.0)
g_mrflux_vabs_mcr     = compute_global_vabsmax([monthly_mrflux_mcr], default=1.0)

plot_monthly("Trash Cell Mass", monthly_trash_mcr, g_trash_vmax_mcr,
             "Monthly Mean: Trash Cell Mass (MCR)",
             f"climatology_trash_monthly_proj_{MODEL_MCR}.png", cmap="turbo", vmin=0.0)

plot_monthly("Total Cell State Mass", monthly_cell_mcr, 60,
             "Monthly Mean: Total Cell State Mass (MCR)",
             f"climatology_cell_monthly_proj_{MODEL_MCR}.png", cmap="turbo", vmin=0.0)

plot_monthly("Retention State Mass", monthly_retention_mcr, g_retention_vmax_mcr,
             "Monthly Mean: Retention State Mass (MCR)",
             f"climatology_retention_monthly_proj_{MODEL_MCR}.png", cmap="turbo", vmin=0.0)

plot_monthly("Fast Flow State Mass", monthly_fast_mcr, g_fast_vmax_mcr,
             "Monthly Mean: Fast Flow State Mass (MCR)",
             f"climatology_fast_monthly_proj_{MODEL_MCR}.png", cmap="turbo", vmin=0.0)

plot_monthly_diverging("MR Flux", monthly_mrflux_mcr, g_mrflux_vabs_mcr,
                       "Monthly Mean: MR Flux (MCR, diverging)",
                       f"climatology_mrflux_monthly_proj_{MODEL_MCR}.png", cmap="bwr")
# ---- MR GATE ----
# ---- MR GATE (mean across channels) ----
N_CH = 32  # <-- set to your out_channels that were summed when saving
monthly_mrgate_mean_mcr = scale_dict_arrays(monthly_mrgate_mcr, N_CH)

plot_monthly("MR Gate (mean over channels)", monthly_mrgate_mean_mcr, vmax=0.4,
             title="Monthly Mean: MR Gate (MCR, mean over channels)",
             fname=f"climatology_mrgate_monthly_proj_{MODEL_MCR}.png",
             cmap=MRGATE_CMAP, vmin=MRGATE_VMIN)


# (trash + mrflux) for MCR
monthly_trash_plus_mrflux_mcr = add_dict_arrays(monthly_trash_mcr, monthly_mrflux_mcr)
vabs_trash_plus_mrflux_mcr = compute_global_vabsmax([monthly_trash_plus_mrflux_mcr], default=1.0)

plot_monthly_diverging("Trash + MR Flux (MCR)", monthly_trash_plus_mrflux_mcr, vabs_trash_plus_mrflux_mcr,
                       "Monthly Mean: Trash + MR Flux (MCR)",
                       f"climatology_trash_plus_mrflux_monthly_proj_{MODEL_MCR}.png",
                       cmap="bwr")

# =========================
# DELTA: (trash_mcr + mrflux_mcr) - trash_mc
# =========================
common_dt_delta = sorted(set(trash_mcr_aligned.keys()) & set(mrflux_mcr_aligned.keys()) & set(trash_mc.keys()))
_log(f"Common dates for delta (trash_mcr + mrflux_mcr - trash_mc) = {len(common_dt_delta)}")

trash_mcr_delta  = {dt: trash_mcr[dt] for dt in common_dt_delta}
mrflux_mcr_delta = {dt: mrflux_mcr[dt] for dt in common_dt_delta}
trash_mc_delta   = {dt: trash_mc[dt] for dt in common_dt_delta}

trash_plus_mrflux_mcr_daily_dt = add_dict_arrays(trash_mcr_delta, mrflux_mcr_delta)
monthly_trash_plus_mrflux_mcr_for_delta = monthly_mean_from_daily_datetime_dict(trash_plus_mrflux_mcr_daily_dt)
monthly_trash_mc_for_delta              = monthly_mean_from_daily_datetime_dict(trash_mc_delta)
monthly_delta = subtract_dict_arrays(monthly_trash_plus_mrflux_mcr_for_delta, monthly_trash_mc_for_delta)

vabs_delta = compute_global_vabsmax([monthly_delta], default=1.0)

plot_monthly_diverging("(Trash+MRFlux)_MCR - Trash_MC", monthly_delta, vabs_delta,
                       "Monthly Mean: (Trash + MR Flux)_MCR − Trash_MC",
                       "climatology_delta_trashplusmrflux_mcr_minus_trash_mc_monthly_proj.png",
                       cmap="bwr")

_log("All climatology figures saved.")