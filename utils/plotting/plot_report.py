"""
Validation Plots - DeepESD Style

Generate comprehensive validation plots matching DeepESD report scales:
- RMSE (spatial map + distribution)
- Mean Bias (spatial map + distribution)
- Wasserstein Distance  [pr only]
- Bias (SDII)           [pr only]
- Bias (RX1day)         [pr only]
- Bias (p98)            [tasmax only]
- Bias (TXx)            [tasmax only]
- Power Spectral Density (overall + specific dates)
- Daily Comparisons (GT vs Prediction vs Bias)
- PDF Comparison

Usage:
    python validation_plots_complete.py
"""


import os

import argparse
import numpy as np
import sys
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import wasserstein_distance
import warnings
from matplotlib.gridspec import GridSpec
from typing import Any, Callable
from numpy import fft
warnings.filterwarnings('ignore')


import pickle


import matplotlib
sys.path.append("/leonardo_work/ICT26_ESP/sdigioia/CORDEX-ML/GNN4CD-CORDEXML")

from utils.plots import extremes_cmap
from utils.tools import date_to_idxs_new
from utils.tools import set_seed_everything, date_to_idxs_new3, return_train_test_idxs

import matplotlib.ticker as ticker
import matplotlib.colors as colors
import matplotlib.patches as patches
# import colorcet as cc

from matplotlib.ticker import ScalarFormatter, NullFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

os.environ["CARTOPY_DATA_DIR"] = "/leonardo_work/ICT26_ESP/sdigioia/CORDEX-ML/cartopy/"

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False



import xarray as xr

#DOMAIN="NZ"



#with open(output_path+output_file, 'rb') as f1:
#        graph = pickle.load(f1)


#pr_pred_normalized= graph.pr_normalized


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#-- paths
parser.add_argument('--input_path', type=str, help='path to input directory')
parser.add_argument('--output_path', type=str, help='path to output directory')
parser.add_argument('--plot_path', type=str, help='path to plot directory')
parser.add_argument('--val_file', type=str, help='validation graph pred')
parser.add_argument('--val_year', type=int, help='validation year')
parser.add_argument('--season_file', type=str)
parser.add_argument('--var', type=str, default='pr')
parser.add_argument('--domain', type=str, default='ALPS')
parser.add_argument('--experiment', type=str, default="ESD_pseudo_reality")

# ==================== Configuration ====================

# Input file (validation results)
#ESD_pseudo_reality
#pr

#Emulator_hist_future
#pr


MODEL_NAME = 'GNN4CD'

WET_THRESHOLD = 1.0   # mm/day

# ---- Fixed color scales matching DeepESD report ----
# pr
PR_RMSE_VMAX   = 12.0  # RMSE        (mm/day)
PR_BIAS_VMAX   = 2.0   # Mean Bias   (mm/day)
PR_SDII_VMAX   = 3.0   # Bias SDII   (mm/day)
PR_RX1DAY_VMAX = 40.0  # Bias RX1day (mm/day)
PR_WD_VMAX     = 5.0   # Wasserstein Distance

# tasmax
TX_RMSE_VMAX   = 4.0   # RMSE        (K)
TX_BIAS_VMAX   = 1.0   # Mean Bias   (K)
TX_P98_VMAX    = 3.0   # Bias p98    (K)
TX_TXX_VMAX    = 3.0   # Bias TXx    (K)

# daily comparison bias
PR_DAILY_BIAS_VMAX = None   # None = auto (actual min/max)
TX_DAILY_BIAS_VMAX = 5.5    # fixed (K)

# ---- Date labels per experiment ----
# ESD validates on 1975; Emulator validates on historical 1975 but future comparison = 2085
DATE_LABELS = {
    'ESD_pseudo_reality':   {1975: ['1975-01-15', '1975-07-15'], 1980: ['1980-01-15', '1980-07-15']},
    'Emulator_hist_future': {2098: ['2098-01-15', '2098-07-15']},
}

# ==================== Utility Functions ====================

def date_to_idxs_from_timeindex(
    year_start, month_start, day_start, time_index,
    year_end=None, month_end=None, day_end=None    
):
    """
    Compute start/end indices using the actual time_index array.
    The time index must be sorted. The period [day_start, day_end]
    can be obtain with [start_idx, end_idx)
    Args:
        time_index: numpy array datetime64
    Returns:
        start_idx: int
        end_idx: int
    """

    # Build datetime64 timestamps
    start_ts = np.datetime64(f"{year_start:04d}-{month_start:02d}-{day_start:02d}T00:00:00")
    if year_end is not None:
        end_ts   = np.datetime64(f"{year_end:04d}-{month_end:02d}-{day_end:02d}T23:59:59")

    # Find indices using binary search
    # Note - np.searchsorted returns the insertion position that would keep the array sorted,
    # even if the exact timestamp is not present.
    start_idx = np.searchsorted(time_index, start_ts, side="left")
    start_idx = int(start_idx)
    if year_end is not None:
        end_idx   = np.searchsorted(time_index, end_ts,   side="right")
        end_idx = int(end_idx) 
        return start_idx, end_idx
    else:
        return start_idx


def make_spatial_map(ax, lon, lat, values, vmin, vmax, cmap, title, plot_fn="pcolormesh"):
    if HAS_CARTOPY and hasattr(ax, 'projection'):
        if plot_fn == "scatter":
            im = ax.scatter(lon, lat, c=values, s=15, vmin=vmin, vmax=vmax,
                        cmap=cmap, marker='s', transform=ccrs.PlateCarree())
        elif plot_fn == "pcolormesh":
            im = ax.pcolormesh(lon, lat, values, vmin=vmin, vmax=vmax,
                        cmap=cmap, transform=ccrs.PlateCarree(), edgecolor = 'face', linewidth = 0.01)
        ax.coastlines(resolution='10m', linewidth=0.8, color='black')
        #ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')
        ax.set_xlim([lon.min()-0.5, lon.max()+0.5])
        ax.set_ylim([lat.min()-0.5, lat.max()+0.5])
    else:
        if plot_fn == "scatter":
            im = ax.scatter(lon, lat, c=values, vmin=vmin, vmax=vmax,
                            cmap=cmap)
        elif plot_fn == "pcolormesh":
            im = ax.pcolormesh(lon, lat, values, s=15, vmin=vmin, vmax=vmax,
                cmap=cmap, marker='s', edgecolor = 'face', linewidth = 0.01)
        ax.set_xlim([lon.min()-0.5, lon.max()+0.5])
        ax.set_ylim([lat.min()-0.5, lat.max()+0.5])
        ax.set_aspect('equal')
    ax.set_title(title, fontsize=14, fontweight='bold')
    return im

def draw_maps_page(pdf, lon, lat, pred, target):

    vmin_f = 0 if VAR == 'pr' else 275
    vmax_f = np.nanpercentile(np.concatenate([pred, target]), 98) + 4 if VAR == 'pr' else 295

    if HAS_CARTOPY:
        fig = plt.figure(figsize=(16, 6))
        ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
        ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    cmap_field = 'magma' if VAR == 'tasmax' else 'turbo'
    fig.suptitle(f'Average | {DOMAIN} | {VAR}',
                    fontsize=14, fontweight='bold', y=0.98)

    im1 = make_spatial_map(ax1, lon, lat, target, vmin_f, vmax_f, cmap_field,
                            f'Ground Truth ({val_year})')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label=f'{VAR} value')

    im2 = make_spatial_map(ax2, lon, lat, pred, vmin_f, vmax_f, cmap_field,
                            f'Prediction ({MODEL_NAME})')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label=f'{VAR} value')

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def draw_metric_page(pdf, lon, lat, metric, vmin, vmax, cmap, label, title, text=""):
    """Standard 2-panel page: spatial map + boxplot with fixed scale."""
    if HAS_CARTOPY:
        fig = plt.figure(figsize=(16, 6))
        ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
        ax2 = fig.add_subplot(1, 2, 2)
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    im = make_spatial_map(ax1, lon, lat, metric, vmin, vmax, cmap, 'Spatial Map')
    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label(label, fontsize=12)

    valid = metric[~np.isnan(metric)]
    ax2.boxplot([valid], patch_artist=True,
                boxprops=dict(facecolor='lightblue'),
                medianprops=dict(color='orange', linewidth=2))
    ax2.set_ylabel(label, fontsize=12)
    ax2.set_title('Distribution', fontsize=14, fontweight='bold')
    ax2.set_xticklabels([''])
    ax2.set_ylim(vmin, vmax)
    if vmin < 0:
        ax2.axhline(0, color='red', linestyle='--', linewidth=1)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.text(0.95, 0.95, f'Mean: {np.nanmean(metric):.2f}' + text,
             transform=ax2.transAxes, ha='right', va='top', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def nodes_to_grid(values, y_dim, x_dim):
    grid = np.flip(values.reshape(y_dim, x_dim, -1), axis=0).squeeze()
    return grid

# def nodes_to_grid(values, lon, lat):
#     """
#     Map scattered nodes onto a regular 2-D grid sorted by lat/lon.
#     This is essential for correct PSD computation — random node ordering
#     produces an artificially rising spectrum (noise-dominated).
#     """
#     unique_lons = np.sort(np.unique(np.round(lon, 4)))
#     unique_lats = np.sort(np.unique(np.round(lat, 4)))[::-1]  # north-first

#     lon_idx = {v: i for i, v in enumerate(unique_lons)}
#     lat_idx = {v: i for i, v in enumerate(unique_lats)}

#     grid = np.full((len(unique_lats), len(unique_lons)), np.nan)
#     for i in range(len(values)):
#         c = lon_idx.get(round(lon[i], 4))
#         r = lat_idx.get(round(lat[i], 4))
#         if r is not None and c is not None:
#             grid[r, c] = values[i]

#     # Fill NaN holes with spatial mean before FFT
#     grid = np.where(np.isnan(grid), np.nanmean(grid), grid)
#     return grid

def _filter_by_season(x: xr.Dataset, season: str | None) -> xr.Dataset:
    """
    Filter the dataset by season label.

    Parameters
    ----------
    x : xr.Dataset
        Dataset to filter.
    season : str | None
        Season name (winter, summer, spring, autumn) or None to skip filtering.

    Returns
    -------
    xr.Dataset
        Filtered dataset.
    """
    if season is None:
        return x
    if season == "winter":
        return x.where(x["time.season"] == "DJF", drop=True)
    if season == "summer":
        return x.where(x["time.season"] == "JJA", drop=True)
    if season == "spring":
        return x.where(x["time.season"] == "MAM", drop=True)
    if season == "autumn":
        return x.where(x["time.season"] == "SON", drop=True)
    return x

def old_psd(
    x0: xr.Dataset,
    x1: xr.Dataset,
    var: str,
    season: str | None = None,
):
    """
    Compute the power spectral density for x0 and x1 datasets.

    Parameters
    ----------
    x0 : xr.Dataset
        Ground truth dataset.
    x1 : xr.Dataset
        Predicted dataset.
    var : str
        Variable name to analyse.
    season : str | None, optional
        Season to filter before computing the PSD.

    Returns
    -------
    tuple[xr.Dataset, xr.Dataset]
        Power spectral densities for x0 and x1.
    """
    x0 = _filter_by_season(x0, season)
    x1 = _filter_by_season(x1, season)

    x0_da = x0[var]
    x1_da = x1[var]

    x0_np = np.nan_to_num(x0_da.values, nan=0.0)
    x1_np = np.nan_to_num(x1_da.values, nan=0.0)

    fft_x0 = fft.fftshift(fft.fft2(x0_np, axes=(-2, -1)), axes=(-2, -1))
    fft_x1 = fft.fftshift(fft.fft2(x1_np, axes=(-2, -1)), axes=(-2, -1))

    power_x0 = np.abs(fft_x0) ** 2
    power_x1 = np.abs(fft_x1) ** 2

    psd_x0_list = [_radial_average(p) for p in power_x0]
    psd_x1_list = [_radial_average(p) for p in power_x1]

    avg_psd_x0 = np.mean(psd_x0_list, axis=0)
    avg_psd_x1 = np.mean(psd_x1_list, axis=0)

    psd_x0_da = xr.DataArray(avg_psd_x0, dims=["wavenumber"], name="PSD_x0")
    psd_x1_da = xr.DataArray(avg_psd_x1, dims=["wavenumber"], name="PSD_x1")

    return psd_x0_da, psd_x1_da

def _radial_average(array_2d: np.ndarray) -> np.ndarray:
    """
    Compute the radial average of a two-dimensional field.

    Parameters
    ----------
    array_2d : np.ndarray
        Two-dimensional array to average.

    Returns
    -------
    np.ndarray
        Radially averaged profile.

    """
    y, x = np.indices(array_2d.shape)
    center = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])
    r = np.hypot(x - center[0], y - center[1]).astype(np.int32)
    tbin = np.bincount(r.ravel(), array_2d.ravel())
    nr = np.bincount(r.ravel())
    return tbin / np.maximum(nr, 1)

def compute_psd_new(field_0, field_1, lon, lat):
    """
    Compute radially-averaged 2-D PSD on the proper regular grid.

    Parameters
    ----------
    x0 : numpy.ndarray
        Ground truth dataset.
    x1 : numpy.ndarray
        Predicted dataset.

    Returns
    -------
    tuple[xr.Dataset, xr.Dataset]
        Power spectral densities for x0 and x1.
    """
    x0_grid          = nodes_to_grid(field_0, lon, lat)
    x1_grid          = nodes_to_grid(field_1, lon, lat)
    nrows, ncols  = x0_grid.shape


    x0_np = np.nan_to_num(x0_grid, nan=0.0)
    x1_np = np.nan_to_num(x1_grid, nan=0.0)

    fft_x0 = fft.fftshift(fft.fft2(x0_np, axes=(-2, -1)), axes=(-2, -1))
    fft_x1 = fft.fftshift(fft.fft2(x1_np, axes=(-2, -1)), axes=(-2, -1))

    power_x0 = np.abs(fft_x0) ** 2
    power_x1 = np.abs(fft_x1) ** 2

    psd_x0_list = [_radial_average(p) for p in power_x0]
    psd_x1_list = [_radial_average(p) for p in power_x1]

    avg_psd_x0 = np.mean(psd_x0_list, axis=0)
    avg_psd_x1 = np.mean(psd_x1_list, axis=0)

    psd_x0_da = xr.DataArray(avg_psd_x0, dims=["wavenumber"], name="PSD_x0")
    psd_x1_da = xr.DataArray(avg_psd_x1, dims=["wavenumber"], name="PSD_x1")

    return psd_x0_da, psd_x1_da

def compute_psd_2d(field_values, lon, lat):
    """
    Compute radially-averaged 2D PSD matching DeepESD eval_diagnostics.psd exactly:
      - No mean subtraction
      - No windowing
      - Full-diagonal radial average via bincount
    """
    grid = nodes_to_grid(field_values, lon, lat)

    fft_grid = np.fft.fftshift(np.fft.fft2(grid))
    power    = np.abs(fft_grid) ** 2

    y, x     = np.indices(power.shape)
    center_x = (x.max() - x.min()) / 2.0
    center_y = (y.max() - y.min()) / 2.0
    r        = np.hypot(x - center_x, y - center_y).astype(np.int32)
    tbin     = np.bincount(r.ravel(), power.ravel())
    nr       = np.bincount(r.ravel())
    radial_psd  = tbin / np.maximum(nr, 1)
    wavenumbers = np.arange(len(radial_psd))
    return wavenumbers, radial_psd

# def compute_psd_2d(field_values, lon, lat):
#     """
#     Compute radially-averaged 2-D PSD on the proper regular grid.
#     Returns (wavenumbers, psd_power).
#     """
#     grid          = nodes_to_grid(field_values, lon, lat)
#     grid = np.nan_to_num(grid, nan=0.0)
#     nrows, ncols  = grid.shape[0], grid.shape[1]

#     # Remove mean (detrend)
#     grid = grid - grid.mean()

#     # 2-D Hann window to reduce spectral leakage
#     window = np.outer(np.hanning(nrows), np.hanning(ncols))
#     grid   = grid * window

#     fft = np.fft.fft2(grid)
#     psd = (np.abs(fft) ** 2) / (nrows * ncols)   # normalise

#     # Shift zero-frequency to centre for radial averaging
#     psd    = np.fft.fftshift(psd)
#     cy, cx = nrows // 2, ncols // 2
#     y, x   = np.indices(psd.shape)
#     r      = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
#     max_r  = min(cy, cx)

#     radial_psd  = np.array([psd[r == ri].mean() for ri in range(max_r)])
#     wavenumbers = np.arange(max_r)
#     return wavenumbers, radial_psd


# ==================== Metric computations ====================

def compute_rmse(pred, target):
    axis = 1 if pred.ndim == 2 else 2
    return np.sqrt(np.nanmean((pred - target) ** 2, axis=axis))

def compute_bias(pred, target):
    axis = 1 if pred.ndim == 2 else 2
    return np.nanmean(pred - target, axis=axis)

def compute_wasserstein_per_node(pred, target):
    n  = pred.shape[0]
    wd = np.zeros(n)
    for i in range(n):
        p, t  = pred[i], target[i]
        valid = ~(np.isnan(p) | np.isnan(t))
        p, t  = p[valid], t[valid]
        wet   = (p > WET_THRESHOLD) | (t > WET_THRESHOLD)
        p, t  = p[wet], t[wet]
        wd[i] = wasserstein_distance(p, t) if (len(p) > 0 and len(t) > 0) else np.nan
    return wd

def compute_sdii(precip):
    axis = 1 if precip.ndim == 2 else 2
    wet_mask = precip >= WET_THRESHOLD
    wet = np.where(wet_mask, precip, np.nan)
    return np.nanmean(wet, axis=axis)

def compute_rx1day(precip):
    axis = 1 if precip.ndim == 2 else 2
    return np.nanmax(precip, axis=axis)

def compute_p98(field):
    axis = 1 if field.ndim == 2 else 2
    return np.nanpercentile(field, q=98, axis=axis)

def compute_txx(field):
    axis = 1 if field.ndim == 2 else 2
    return np.nanmax(field, axis=axis)


# ==================== Main ====================

if __name__ == '__main__':

    args = parser.parse_args()

    DOMAIN=args.domain
    EXPERIMENT=args.experiment
    #EXPERIMENT="ESD_pseudo_reality"
    VAR        = args.var                  # 'pr' or 'tasmax'

    
    #output_path=f"/leonardo_work/ICT25_ESP/sdigioia/CORDEX-ML/Emulator_output/{DOMAIN}_domain/{EXPERIMENT}/predictions/qloss-precipitation-woGRU_5/"
    #plot_path=f"/leonardo_work/ICT25_ESP/sdigioia/CORDEX-ML/Emulator_output/{DOMAIN}_domain/{EXPERIMENT}/predictions/qloss-precipitation-woGRU_5/plots/"
    input_path  = args.input_path
    output_path = args.output_path
    plot_path = args.plot_path
    val_file= args.val_file
    val_year = args.val_year

    val_file_help="/leonardo_work/ICT26_ESP/sdigioia/CORDEX-ML/CORDEX-domains/ALPS_domain/train/ESD_pseudo_reality/target/pr_tasmax_CNRM-CM5_1961-1980.nc"
    season_file=args.season_file

    VAL_FILE = input_path + val_file
    OUTPUT_PDF = plot_path + f"GNN4CD_{EXPERIMENT}_{DOMAIN}_{val_year}_{VAR}.pdf"

    print("Loading validation data...")
    with open(VAL_FILE, 'rb') as f:
        data = pickle.load(f)

    date_labels_experiment = DATE_LABELS.get(EXPERIMENT)
    date_labels = date_labels_experiment.get(val_year)
    
    # Grid dimensions: inferred from data for non-ALPS domains
    DOMAIN_GRID = {
        'ALPS': None,   # read from netCDF below
        'SA':   (128, 128),
    }

    if args.domain == 'ALPS':
        data2 = xr.open_dataset(val_file_help)
        y = data2.y.to_numpy()
        x = data2.x.to_numpy()
        y_dim = y.shape[0]
        x_dim = x.shape[0]
    elif args.domain in DOMAIN_GRID and DOMAIN_GRID[args.domain] is not None:
        y_dim, x_dim = DOMAIN_GRID[args.domain]
    else:
        raise ValueError(f"Unknown domain '{args.domain}'. Add it to DOMAIN_GRID or use --domain ALPS.")
    
    # Corresponding time indices within the validation year array
    # Day 14 = Jan 15, Day 195 = Jul 15
    time_index = data.times
    year_1, month_1, day_1 = map(int, date_labels[0].split("-"))
    year_2, month_2, day_2 = map(int, date_labels[1].split("-"))
    day_idx_1 = date_to_idxs_from_timeindex(year_1, month_1, day_1, time_index)
    day_idx_2 = date_to_idxs_from_timeindex(year_2, month_2, day_2, time_index)
    SAMPLE_TIME_INDICES = [day_idx_1, day_idx_2]
        

    # --- Detect format: HeteroData vs plain dict ---
    IS_HETERODATA = hasattr(data, '_node_store_dict')

    if IS_HETERODATA:
        pred   = data.pr_gnn4cd   if VAR == 'pr' else data.tasmax_gnn4cd
        target = data.target if VAR == 'pr' else data.target
        lon    = data['high'].lon
        lat    = data['high'].lat           
        times  = data.times if hasattr(data, 'times') else np.arange(pred.shape[1])
    else:
        pred   = data['pr_gnn4cd']   if VAR == 'pr' else data['tasmax_gnn4cd']
        target = data['pr_target']   if VAR == 'pr' else data['tasmax_target']
        lon    = data['lon']
        lat    = data['lat']
        times  = data['times'] if 'times' in data else np.arange(pred.shape[1])
        
    print(target.shape)
    
    #sys.exit()

    def to_np(x):
        if isinstance(x, np.ndarray): return x
        return x.cpu().numpy() if hasattr(x, 'cpu') else np.array(x)
    pred, target, lon, lat = to_np(pred), to_np(target), to_np(lon), to_np(lat)

    pred_grid = nodes_to_grid(pred, y_dim, x_dim)
    target_grid = nodes_to_grid(target, y_dim, x_dim)
    lon_grid = nodes_to_grid(lon, y_dim, x_dim)
    lat_grid = nodes_to_grid(lat, y_dim, x_dim)


    # # Align time dimension (Emulator_hist_future may have mismatch due to leap years)
    # if pred.shape[1] != target.shape[1]:
    #     min_len = min(pred.shape[1], target.shape[1])
    #     print(f"WARNING: time mismatch pred={pred.shape[1]}, target={target.shape[1]}, truncating to {min_len}")
    #     pred   = pred[:,   :min_len]
    #     target = target[:, :min_len]
    # ====================
    print(f"pred shape  : {pred.shape}")
    print(f"target shape: {target.shape}")
    print(f"nodes       : {len(lon)}")

    print(f"pred_grid shape  : {pred_grid.shape}")
    print(f"target_grid shape  : {target_grid.shape}")
    print(f"lon_grid/lat_grid shape  : {lon_grid.shape}")

    pdf = PdfPages(OUTPUT_PDF)
    print(f"\nGenerating: {OUTPUT_PDF}")

    # ================= AVERAGE ===================
    print("  Average...")
    pred_avg = np.nanmean(pred_grid, axis=2)
    target_avg = np.nanmean(target_grid, axis=2)
    draw_maps_page(pdf, lon_grid, lat_grid, pred_avg, target_avg)

    # ==================== RMSE ====================
    print("  RMSE...")
    rmse = compute_rmse(pred_grid, target_grid)
    rmse_vmax = TX_RMSE_VMAX if VAR == 'tasmax' else PR_RMSE_VMAX
    draw_metric_page(pdf, lon_grid, lat_grid, rmse, 0, rmse_vmax, 'viridis',
                     'RMSE', f'RMSE | {MODEL_NAME} | {VAR}')

    # ==================== Mean Bias ====================
    print("  Mean Bias...")
    bias      = compute_bias(pred_grid, target_grid)
    bias_vmax = TX_BIAS_VMAX if VAR == 'tasmax' else PR_BIAS_VMAX
    cmap_bias = 'RdBu_r' if VAR == 'tasmax' else 'BrBG'
    draw_metric_page(pdf, lon_grid, lat_grid, bias, -bias_vmax, bias_vmax, cmap_bias,
                     'Mean Bias', f'Mean Bias | {MODEL_NAME} | {VAR}')

    # ==================== Variable-specific metrics ====================
    if VAR == 'pr':
        print("  Wasserstein Distance...")
        wd = compute_wasserstein_per_node(pred, target)
        wd = nodes_to_grid(wd, y_dim, x_dim)
        draw_metric_page(pdf, lon_grid, lat_grid, wd, 0, PR_WD_VMAX, 'viridis',
                         'Wasserstein Distance',
                         f'Wasserstein Distance | {MODEL_NAME} | {VAR}',
                         text=f"\nWet threshold: {WET_THRESHOLD} mm/day")

        print("  Bias (SDII)...")
        sdii_bias = compute_sdii(pred_grid) - compute_sdii(target_grid)
        draw_metric_page(pdf, lon_grid, lat_grid, sdii_bias, -PR_SDII_VMAX, PR_SDII_VMAX, 'BrBG',
                         'Bias (SDII)', f'Bias (SDII) | {MODEL_NAME} | {VAR}',
                         text=f"\nWet threshold: {WET_THRESHOLD} mm/day")

        print("  Bias (RX1day)...")
        rx1_bias = compute_rx1day(pred_grid) - compute_rx1day(target_grid)
        draw_metric_page(pdf, lon_grid, lat_grid, rx1_bias, -PR_RX1DAY_VMAX, PR_RX1DAY_VMAX, 'BrBG',
                         'Bias (RX1day)', f'Bias (RX1day) | {MODEL_NAME} | {VAR}')

    else:  # tasmax
        print("  Bias (p98)...")
        p98_bias = compute_p98(pred_grid) - compute_p98(target_grid)
        draw_metric_page(pdf, lon_grid, lat_grid, p98_bias, -TX_P98_VMAX, TX_P98_VMAX, 'RdBu_r',
                         'Bias (p98)', f'Bias (p98) | {MODEL_NAME} | {VAR}')

        print("  Bias (TXx)...")
        txx_bias = compute_txx(pred_grid) - compute_txx(target_grid)
        draw_metric_page(pdf, lon_grid, lat_grid, txx_bias, -TX_TXX_VMAX, TX_TXX_VMAX, 'RdBu_r',
                         'Bias (TXx)', f'Bias (TXx) | {MODEL_NAME} | {VAR}')

    # ==================== Overall PSD ====================
    print("  Overall PSD (per time step, may take ~1 min)...")
    T = pred.shape[1]
    psd_p_sum = psd_t_sum = None
    for _t in range(T):
        wn, _p = compute_psd_2d(pred[:,   _t], y_dim, x_dim)
        _,  _q = compute_psd_2d(target[:, _t], y_dim, x_dim)
        if psd_p_sum is None:
            psd_p_sum = np.zeros_like(_p)
            psd_t_sum = np.zeros_like(_q)
        psd_p_sum += _p;  psd_t_sum += _q
    psd_p = psd_p_sum / T
    psd_t = psd_t_sum / T

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle(f'Overall Power Spectral Density | {DOMAIN} | {VAR}',
                 fontsize=14, fontweight='bold')
    ax.loglog(wn[1:], psd_t[1:], label='Ground Truth',               color='black', linewidth=2)
    ax.loglog(wn[1:], psd_p[1:], label=f'Prediction ({MODEL_NAME})', color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Wavenumber', fontsize=12);  ax.set_ylabel('Power', fontsize=12)
    ax.legend(fontsize=11);  ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout();  pdf.savefig(fig, bbox_inches='tight');  plt.close()

    # ==================== PSD for specific dates ====================
    for time_idx, date_label in zip(SAMPLE_TIME_INDICES, date_labels):
        print(f"  PSD {date_label}...")
        wn, psd_p = compute_psd_2d(pred[:,   time_idx], y_dim, x_dim)
        _,  psd_t = compute_psd_2d(target[:, time_idx], y_dim, x_dim)
        fig, ax = plt.subplots(figsize=(10, 7))
        fig.suptitle(f'Power Spectral Density | {DOMAIN} | {VAR} | {date_label}',
                     fontsize=14, fontweight='bold')
        ax.loglog(wn[1:], psd_t[1:], label='Ground Truth',               color='black', linewidth=2)
        ax.loglog(wn[1:], psd_p[1:], label=f'Prediction ({MODEL_NAME})', color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Wavenumber', fontsize=12);  ax.set_ylabel('Power', fontsize=12)
        ax.legend(fontsize=11);  ax.grid(True, alpha=0.3, which='both')
        plt.tight_layout();  pdf.savefig(fig, bbox_inches='tight');  plt.close()

    # # ==================== Overall PSD ====================
    # print("  Overall PSD...")
    # mean_pred   = np.nanmean(pred,   axis=1)
    # mean_target = np.nanmean(target, axis=1)
    # if args.domain=='ALPS':
    #     wn,  psd_p = compute_psd_2d(mean_pred,   y_dim, x_dim) #x, y)
    #     _,   psd_t = compute_psd_2d(mean_target, y_dim, x_dim) #x, y)
    # else:
    #     wn,  psd_p = compute_psd_2d(mean_pred,   lon, lat)
    #     _,   psd_t = compute_psd_2d(mean_target, lon, lat)
    # fig, ax = plt.subplots(figsize=(10, 7))
    # fig.suptitle(f'Overall Power Spectral Density | {DOMAIN} | {VAR}',
    #              fontsize=14, fontweight='bold')
    # ax.loglog(wn[1:], psd_t[1:], label='Ground Truth',                linewidth=2)
    # ax.loglog(wn[1:], psd_p[1:], label=f'Prediction ({MODEL_NAME})',  linewidth=2, linestyle='--')
    # ax.set_xlabel('Wavenumber', fontsize=12)
    # ax.set_ylabel('Power',      fontsize=12)
    # ax.legend(fontsize=11)
    # ax.grid(True, alpha=0.3, which='both')
    # plt.tight_layout()
    # pdf.savefig(fig, bbox_inches='tight')
    # plt.close()

    # # ==================== PSD for specific dates ====================
    # for time_idx, date_label in zip(SAMPLE_TIME_INDICES, date_labels):
    #     print(f"  PSD {date_label}...")
    #     if args.domain=='ALPS':
    #         wn,  psd_p = compute_psd_2d(pred[:,   time_idx],   y_dim, x_dim) #x, y)
    #         _,   psd_t = compute_psd_2d(target[:,   time_idx], y_dim, x_dim) #x, y)
    #     else:
    #         wn, psd_p = compute_psd_2d(pred[:,   time_idx], lon, lat)
    #         _,  psd_t = compute_psd_2d(target[:, time_idx], lon, lat)

    #     fig, ax = plt.subplots(figsize=(10, 7))
    #     fig.suptitle(f'Power Spectral Density | {DOMAIN} | {VAR} | {date_label}',
    #                  fontsize=14, fontweight='bold')
    #     ax.loglog(wn[1:], psd_t[1:], label='Ground Truth',                linewidth=2)
    #     ax.loglog(wn[1:], psd_p[1:], label=f'Prediction ({MODEL_NAME})',  linewidth=2, linestyle='--')
    #     ax.set_xlabel('Wavenumber', fontsize=12)
    #     ax.set_ylabel('Power',      fontsize=12)
    #     ax.legend(fontsize=11)
    #     ax.grid(True, alpha=0.3, which='both')
    #     plt.tight_layout()
    #     pdf.savefig(fig, bbox_inches='tight')
    #     plt.close()

    # ==================== Daily Comparisons ====================

    # # Fixed scales per day (original)
    # if VAR == "pr":
    #     vmin_f_list = [0, 0]
    #     vmax_f_list = [37, 75]
    #     bias_vmax_d_list = [22, 61]
    # else:
    #     vmin_f_list = [252, 272]
    #     vmax_f_list = [294, 306]
    #     bias_vmax_d_list = [5, 6.1]

    i = 0
    for time_idx, date_label in zip(SAMPLE_TIME_INDICES, date_labels):
        print(f"  Daily comparison {date_label}...")

        fp = pred_grid[:, :, time_idx]
        ft = target_grid[:, :, time_idx]
        fb = fp - ft

        # Scale from ground truth
        vmin_f = float(np.nanmin(ft))
        vmax_f = float(np.nanmax(ft))
        bias_vmax_d = float(np.nanmax(np.abs(fb)))

        if HAS_CARTOPY:
            fig = plt.figure(figsize=(18, 6))
            ax1 = fig.add_subplot(1, 3, 1, projection=ccrs.PlateCarree())
            ax2 = fig.add_subplot(1, 3, 2, projection=ccrs.PlateCarree())
            ax3 = fig.add_subplot(1, 3, 3, projection=ccrs.PlateCarree())
        else:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        cmap_field = 'magma' if VAR == 'tasmax' else 'turbo'
        fig.suptitle(f'Daily Comparison | {DOMAIN} | {VAR} | {date_label}',
                     fontsize=14, fontweight='bold', y=0.98)

        im1 = make_spatial_map(ax1, lon_grid, lat_grid, ft, vmin_f, vmax_f, cmap_field,
                               f'Ground Truth ({date_label})')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label=f'{VAR} value')

        im2 = make_spatial_map(ax2, lon_grid, lat_grid, fp, vmin_f, vmax_f, cmap_field,
                               f'Prediction ({MODEL_NAME})')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label=f'{VAR} value')

        im3 = make_spatial_map(ax3, lon_grid, lat_grid, fb, -bias_vmax_d, bias_vmax_d, cmap_bias,
                               'Bias (Pred - GT)')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label='Bias')

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        i += 1

    # ==================== PDF Comparison ====================
    print("  PDF Comparison...")
    pf = pred.flatten();   pf = pf[~np.isnan(pf)]
    tf = target.flatten(); tf = tf[~np.isnan(tf)]
    # if VAR == 'pr':
    #     pf = pf[pf > WET_THRESHOLD]
    #     tf = tf[tf > WET_THRESHOLD]

    wd_all = wasserstein_distance(pf, tf)

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.suptitle(f'Distribution Comparison | {DOMAIN} | {VAR} | {MODEL_NAME}',
                 fontsize=14, fontweight='bold')

    if VAR == "tasmax" or VAR == "t2m":
        min_val_tasmax = 275
        max_val_tasmax= 325
        bins = np.arange(min_val_tasmax,max_val_tasmax,1).astype(np.float32)
    elif VAR == "pr":
        min_val_pr = 0
        max_val_pr = 1000
        bins = np.arange(min_val_pr,max_val_pr,1).astype(np.float32)
    
    if VAR == "pr":
        hist_vals, bins = np.histogram(pf, bins=bins, density=False)
        bins_mid = (bins[:-1] + bins[1:]) / 2
        Ntot = np.nansum(hist_vals)
        hist_vals_target, bins_target = np.histogram(tf, bins=bins, density=False)
        bins_target_mid = (bins_target[:-1] + bins_target[1:]) / 2
        Ntot_target = np.nansum(hist_vals_target)
        ax.scatter(bins_target_mid, hist_vals_target/Ntot_target, color='black', s=80, label='Ground Truth', alpha=0.4, zorder=2)
        ax.scatter(bins_mid, hist_vals/Ntot, color='darkorange', s=80, label=f'Prediction ({MODEL_NAME})', alpha=0.4, zorder=2)
    else:
        ax.hist(tf, bins=bins, density=True, alpha=0.6, label='Ground Truth',              color='steelblue')
        ax.hist(pf, bins=bins, density=True, alpha=0.6, label=f'Prediction ({MODEL_NAME})', color='darkorange')
    ax.set_xlabel(f'{VAR} value', fontsize=12)
    ax.set_ylabel('Density',      fontsize=12)
    ax.set_title('Probability Density Function (PDF)', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    stats = (f"Ground Truth:\n Mean: {np.mean(tf):.3f}\n Std: {np.std(tf):.3f}\n\n"
             f"Prediction:\n Mean: {np.mean(pf):.3f}\n Std: {np.std(pf):.3f}\n\n"
             f"Wasserstein Distance:\n {wd_all:.4f}")
    ax.text(0.98, 0.98, stats, transform=ax.transAxes, va='top', ha='right',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    if VAR == "pr":
        ax.set_yscale('log')
        ax.set_xscale('log')
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    pdf.close()
    n_pages = 11 if VAR == 'pr' else 9
    print(f"\n✓ Done: {OUTPUT_PDF}  ({n_pages} pages)")
