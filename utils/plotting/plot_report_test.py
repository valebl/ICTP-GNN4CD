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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import wasserstein_distance
import warnings
from numpy import fft
warnings.filterwarnings('ignore')
import pickle
import xarray as xr
import json
from utils.plotting.setup_cartopy import setup_cartopy

# Set-up cartopy before imports
setup_cartopy()
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#-- paths
parser.add_argument('--input_path', type=str, help='path to input directory')
parser.add_argument('--output_path', type=str, help='path to output directory')
parser.add_argument('--plot_path', type=str, help='path to plot directory')
parser.add_argument('--val_file', type=str, help='validation graph pred')
parser.add_argument('--var', type=str, default='pr')
parser.add_argument('--domain', type=str, default='ALPS')
parser.add_argument('--experiment', type=str, default="ESD_pseudo_reality")
parser.add_argument('--comparison_data', type=str, default="")
parser.add_argument('--period', type=str, default="")
parser.add_argument('--val_mode', type=str, default="")
parser.add_argument('--report_name', type=str, default="")
parser.add_argument('--test_id', type=str, default="")
parser.add_argument('--test_year', type=str, default="")
parser.add_argument('--config_file', type=str)

# ==================== Utility Functions ====================

def add_test_id(fig, run_id):
    fig.text(
        0.99, 0.995, run_id,
        ha='right', va='top',
        fontsize=9, color='gray',
        transform=fig.transFigure,
        zorder=1000
    )

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

    avg_cfg = CONFIG["average_field_ranges"][VAR]

    # vmin is always fixed
    vmin_f = avg_cfg["vmin"]

    # vmax logic differs by variable
    if VAR == "pr":
        # dynamic percentile-based vmax
        vmax_f = np.nanpercentile(np.concatenate([pred, target]), 98) + avg_cfg["vmax_offset"]
    else:
        # tasmax: experiment-dependent fixed vmax
        if EXPERIMENT == "Emulator_hist_future":
            vmax_f = avg_cfg["vmax_emulator"]
        else:
            vmax_f = avg_cfg["vmax_esd"]

    if HAS_CARTOPY:
        fig = plt.figure(figsize=(16, 6))
        ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
        ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    if VAR == "tasmax":
        cmap_field = CONFIG["colormaps"]["field_tasmax"]
    else:
        cmap_field = CONFIG["colormaps"]["field_pr"]

    fig.suptitle(f'Average | {DOMAIN} | {VAR}',
                    fontsize=14, fontweight='bold', y=0.98)

    im1 = make_spatial_map(ax1, lon, lat, target, vmin_f, vmax_f, cmap_field,
                            comparison_label)
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label=f'{VAR} value')

    im2 = make_spatial_map(ax2, lon, lat, pred, vmin_f, vmax_f, cmap_field,
                            prediction_label)
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label=f'{VAR} value')

    plt.tight_layout()
    add_test_id(fig, test_id)
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
    add_test_id(fig, test_id)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def nodes_to_grid(values, y_dim, x_dim):
    grid = np.flip(values.reshape(y_dim, x_dim, -1), axis=0).squeeze()
    return grid


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

def compute_psd_2d(field_values, lon, lat, t, label):
    """
    Compute radially-averaged 2D PSD matching DeepESD eval_diagnostics.psd exactly:
      - No mean subtraction
      - No windowing
      - Full-diagonal radial average via bincount
    """
    grid = nodes_to_grid(field_values, lon, lat)
    if np.isnan(grid).any():
        print(f"Time {t}: found {np.isnan(grid).sum()} nan values in {label} - filling with zeros.")
        grid[np.isnan(grid)] = 0.0

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


# ==================== Metric computations ====================

def compute_rmse(pred, target):
    axis = 1 if pred.ndim == 2 else 2
    return np.sqrt(np.nanmean((pred - target) ** 2, axis=axis))

def compute_bias(pred, target):
    axis = 1 if pred.ndim == 2 else 2
    return np.nanmean(pred, axis=axis) - np.nanmean(target, axis=axis)

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

    DOMAIN = args.domain
    EXPERIMENT = args.experiment
    VAR = args.var                  # 'pr' or 'tasmax'
    CONFIG_FILE = args.config_file

    input_path  = args.input_path
    plot_path = args.plot_path
    val_file= args.val_file
    period = args.period
    val_mode = args.val_mode
    report_name = args.report_name
    test_id = args.test_id

    log_file = plot_path+f"logs/{report_name.lower()}.txt"
    if os.path.isfile(log_file):
        os.remove(log_file)
    log = open(log_file, "a")
    sys.stdout = log

    # Load plot configuration
    with open(CONFIG_FILE, "r") as f:
        CONFIG = json.load(f)

    MODEL_NAME = CONFIG["model"]["name"]
    WET_THRESHOLD = CONFIG["thresholds"]["wet_threshold_mm_day"]
    PR_RMSE_VMAX   = CONFIG["color_scales"]["pr"]["rmse_vmax"]
    PR_BIAS_VMAX   = CONFIG["color_scales"]["pr"]["bias_vmax"]
    PR_SDII_VMAX   = CONFIG["color_scales"]["pr"]["sdii_vmax"]
    PR_RX1DAY_VMAX = CONFIG["color_scales"]["pr"]["rx1day_vmax"]
    PR_WD_VMAX     = CONFIG["color_scales"]["pr"]["wasserstein_vmax"]

    TX_RMSE_VMAX   = CONFIG["color_scales"]["tasmax"]["rmse_vmax"]
    TX_BIAS_VMAX   = CONFIG["color_scales"]["tasmax"]["bias_vmax"]
    TX_P98_VMAX    = CONFIG["color_scales"]["tasmax"]["p98_vmax"]
    TX_TXX_VMAX    = CONFIG["color_scales"]["tasmax"]["txx_vmax"]

    PR_DAILY_BIAS_VMAX = CONFIG["color_scales"]["daily_bias"]["pr_vmax"]
    TX_DAILY_BIAS_VMAX = CONFIG["color_scales"]["daily_bias"]["tasmax_vmax"]

    if period == 'historical':
        years_test = list(range(1981, 2001))
        prediction_label = "GNN4CD (1981-2000)"
    elif period == 'mid_century':
        years_test = list(range(2041, 2061))
        prediction_label = "GNN4CD (2041-2060)"
    else:
        years_test = list(range(2080, 2100))
        prediction_label = "GNN4CD (2080-2099)"

    if val_mode == "val_1":
        comparison_label = "val_1 (1980)"
    elif val_mode == "val_2":
        comparison_label = "val_2 (2098-2099)"
    elif val_mode == "era5":
        if VAR == "pr":
            comparison_label = "era5 (1981-2000)"
        else:
            comparison_label = "era5 (1981-1990)"

    if DOMAIN == "ALPS":
        gcm_model = "CNRM-CM5"
        if VAR == "pr":
            remap_op = "remapnn"
        else:
            remap_op = "remapnn"
    elif DOMAIN == "NZ" or DOMAIN == "SA":
        gcm_model = "ACCESS-CM2"
        if VAR == "pr":
            remap_op = "remapcon"
        else:
            remap_op = "remapnn"

    val_file_help=f"/leonardo_work/ICT26_ESP/sdigioia/CORDEX-ML/CORDEX-domains/{DOMAIN}_domain/train/ESD_pseudo_reality/target/pr_tasmax_{gcm_model}_1961-1980.nc"

    VAL_FILE = input_path + val_file
    OUTPUT_PDF = plot_path + f"{report_name}.pdf"

    print(f"Creating plot report for predictions at path: {VAL_FILE}")
    print("\nLoading validation data...")
    with open(VAL_FILE, 'rb') as f:
        data = pickle.load(f)
    
    data2 = xr.open_dataset(val_file_help)
    
    if DOMAIN == 'ALPS':
        y = data2.y.to_numpy()
        x = data2.x.to_numpy()
        y_dim = y.shape[0]
        x_dim = x.shape[0]
    
    time_index = data.times
        
    # --- Detect format: HeteroData vs plain dict ---
    IS_HETERODATA = hasattr(data, '_node_store_dict')

    if IS_HETERODATA:
        pred   = data.pr_gnn4cd   if VAR == 'pr' else data.tasmax_gnn4cd
        lon    = data['high'].lon
        lat    = data['high'].lat           
        times  = data.times if hasattr(data, 'times') else np.arange(pred.shape[1])
    else:
        pred   = data['pr_gnn4cd']   if VAR == 'pr' else data['tasmax_gnn4cd']
        lon    = data['lon']
        lat    = data['lat']
        times  = data['times'] if 'times' in data else np.arange(pred.shape[1])

    if val_mode == "val_1":
        comparison_data = xr.open_dataset(f"/leonardo_work/ICT26_ESP/sdigioia/CORDEX-ML/CORDEX-domains/{DOMAIN}_domain/train/ESD_pseudo_reality/target/pr_tasmax_{gcm_model}_1961-1980.nc")
        target = comparison_data[VAR].to_numpy()
        y_dim = target.shape[1]
        x_dim = target.shape[2]
        target = target.transpose(1, 2, 0)
        target = target.reshape(target.shape[0]*target.shape[1], -1)
        time_index_comparison = comparison_data["time"].to_numpy()
        idx_start, idx_end = date_to_idxs_from_timeindex(1980,1,1,time_index_comparison,1980,12,31)
        idx_end = min(idx_end, target.shape[1]-1)
        target = target[:,idx_start:idx_end]
        print(f"\nval_1 - comparison data: shape = {target.shape}, time_bounds: [{time_index_comparison[idx_start]}, {time_index_comparison[idx_end]}]")
    elif val_mode == "val_2":
        comparison_data = xr.open_dataset(f"/leonardo_work/ICT26_ESP/sdigioia/CORDEX-ML/CORDEX-domains/{DOMAIN}_domain/train/Emulator_hist_future/target/pr_tasmax_{gcm_model}_1961-1980_2080-2099.nc")
        target = comparison_data[VAR].to_numpy()
        y_dim = target.shape[1]
        x_dim = target.shape[2]
        target = target.transpose(1, 2, 0)
        target = target.reshape(target.shape[0]*target.shape[1], -1)
        time_index_comparison = comparison_data["time"].to_numpy()
        idx_start, idx_end = date_to_idxs_from_timeindex(2098,1,1,time_index_comparison,2099,12,31)
        idx_end = min(idx_end, target.shape[1]-1)
        target = target[:,idx_start:idx_end]
        print(f"\nval_2 - comparison data: shape = {target.shape}, time_bounds: [{time_index_comparison[idx_start]}, {time_index_comparison[idx_end]}]")
    elif val_mode == "era5":
        if VAR == "pr":
            ERA5_VAR = "tp"
            ERA5_YEARS = "1981-2000"
            ERA5_TIME_VAR = "time"
        else:
            ERA5_VAR = "tasmax"
            ERA5_YEARS = "1981-2000"
            ERA5_TIME_VAR = "valid_time"

        comparison_data = xr.open_dataset(f"/leonardo_work/ICT26_ESP/SHARED/ERA5_daily/{DOMAIN}_domain/{ERA5_VAR}_daily_{remap_op}-cordexml_{ERA5_YEARS}.nc")
        target = comparison_data[ERA5_VAR].to_numpy()
        y_dim = target.shape[1]
        x_dim = target.shape[2]
        target = target.transpose(1, 2, 0)
        target = target.reshape(target.shape[0]*target.shape[1], -1)
        time_index_comparison = comparison_data[ERA5_TIME_VAR].to_numpy()
        print(f"\nera5 - comparison data: shape = {target.shape}, time_bounds: [{time_index_comparison[0]}, {time_index_comparison[-1]}]")
    else:
        raise Exception("val_mode should be either 'val_1', 'val_2' or 'era5'")
        
    print(target.shape)
    
    ### This is just for debugging purposes, when we want to see the reults for a single year!
    if args.test_year != "":
        test_year = int(args.test_year)
        idx_start, idx_end = date_to_idxs_from_timeindex(test_year,1,1,time_index,test_year,12,31)
        pred = pred[:,idx_start:idx_end]
        prediction_label = "GNN4CD (test_year)"
        OUTPUT_PDF = plot_path + f"{report_name}_{test_year}.pdf"

    def to_np(x):
        if isinstance(x, np.ndarray): return x
        return x.cpu().numpy() if hasattr(x, 'cpu') else np.array(x)
    if args.domain=='ALPS':
        pred, target, lon, lat, x, y= to_np(pred), to_np(target), to_np(lon), to_np(lat), to_np(x), to_np(y)
        print(x.size)
        print(y.size)
    else:
        pred, target, lon, lat = to_np(pred), to_np(target), to_np(lon), to_np(lat)

    pred, target, lon, lat = to_np(pred), to_np(target), to_np(lon), to_np(lat)

    pred_grid = nodes_to_grid(pred, y_dim, x_dim)
    target_grid = nodes_to_grid(target, y_dim, x_dim)
    lon_grid = nodes_to_grid(lon, y_dim, x_dim)
    lat_grid = nodes_to_grid(lat, y_dim, x_dim)

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

    # ==================== Mean Bias ====================
    print("  Mean Bias...")
    bias      = compute_bias(pred_grid, target_grid)
    bias_vmax = TX_BIAS_VMAX if VAR == 'tasmax' else PR_BIAS_VMAX
    if VAR == "tasmax":
        cmap_bias = CONFIG["colormaps"]["bias_tasmax"]
    else:
        cmap_bias = CONFIG["colormaps"]["bias_pr"]
    draw_metric_page(pdf, lon_grid, lat_grid, bias, -bias_vmax, bias_vmax, cmap_bias,
                     'Mean Bias', f'Mean Bias | {MODEL_NAME} | {VAR}')

    # ==================== Variable-specific metrics ====================
    if VAR == 'pr':
        print("  Bias (SDII)...")
        sdii_bias = compute_sdii(pred_grid) - compute_sdii(target_grid)
        draw_metric_page(pdf, lon_grid, lat_grid, sdii_bias, -PR_SDII_VMAX, PR_SDII_VMAX, 'BrBG',
                         'Bias (SDII)', f'Bias (SDII) | {MODEL_NAME} | {VAR}',
                         text=f"\nWet threshold: {WET_THRESHOLD} mm/day")
    else:  # tasmax
        print("  Bias (p98)...")
        p98_bias = compute_p98(pred_grid) - compute_p98(target_grid)
        draw_metric_page(pdf, lon_grid, lat_grid, p98_bias, -TX_P98_VMAX, TX_P98_VMAX, 'RdBu_r',
                         'Bias (p98)', f'Bias (p98) | {MODEL_NAME} | {VAR}')

    # ==================== Overall PSD ====================
    print("  Overall PSD (per time step, may take ~1 min)...")
    # predictions
    T = pred.shape[1]
    psd_p_sum = None
    for _t in range(T):
        wn, _p = compute_psd_2d(pred[:,   _t], y_dim, x_dim, _t, "predictions")
        if psd_p_sum is None:
            psd_p_sum = np.zeros_like(_p)
        psd_p_sum += _p
    psd_p = psd_p_sum / T
    # Target
    T = target.shape[1]
    psd_t_sum = None
    for _t in range(T):
        _,  _q = compute_psd_2d(target[:, _t], y_dim, x_dim, _t, "target")
        if psd_t_sum is None:
            psd_t_sum = np.zeros_like(_q)
        psd_t_sum += _q
    psd_t = psd_t_sum / T

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle(f'Overall Power Spectral Density | {DOMAIN} | {VAR}',
                 fontsize=14, fontweight='bold')
    ax.loglog(wn[1:], psd_t[1:], label=comparison_label,               color='black', linewidth=2)
    ax.loglog(wn[1:], psd_p[1:], label=prediction_label, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Wavenumber', fontsize=12);  ax.set_ylabel('Power', fontsize=12)
    ax.legend(fontsize=11);  ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    add_test_id(fig, test_id)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # ==================== PDF Comparison ====================
    print("  PDF Comparison...")
    pf = pred.flatten();   pf = pf[~np.isnan(pf)]
    tf = target.flatten(); tf = tf[~np.isnan(tf)]

    fig, ax = plt.subplots(figsize=(7, 7.5))
    fig.suptitle(f'Distribution Comparison | {DOMAIN} | {VAR} | {MODEL_NAME}',
                 fontsize=14, fontweight='bold', y=0.95)

    # Histogram bin configuration from JSON
    hist_cfg = CONFIG["histogram_bins"][VAR]

    min_val = hist_cfg["min"]
    max_val = hist_cfg["max"]
    step    = hist_cfg["step"]

    bins = np.arange(min_val, max_val, step).astype(np.float32)
    
    if VAR == "pr":
        hist_vals, bins = np.histogram(pf, bins=bins, density=False)
        bins_mid = (bins[:-1] + bins[1:]) / 2
        Ntot = np.nansum(hist_vals)
        hist_vals_target, bins_target = np.histogram(tf, bins=bins, density=False)
        bins_target_mid = (bins_target[:-1] + bins_target[1:]) / 2
        Ntot_target = np.nansum(hist_vals_target)
        ax.scatter(bins_target_mid, hist_vals_target/Ntot_target, color='black', s=80, label=comparison_label, alpha=0.4, zorder=2)
        ax.scatter(bins_mid, hist_vals/Ntot, color='darkorange', s=80, label=prediction_label, alpha=0.4, zorder=2)
    else:
        ax.hist(tf, bins=bins, density=True, alpha=0.6, label=comparison_label,              color='steelblue')
        ax.hist(pf, bins=bins, density=True, alpha=0.6, label=prediction_label, color='darkorange')
    ax.set_xlabel(f'{VAR} value', fontsize=12)
    ax.set_ylabel('Density',      fontsize=12)
    ax.set_title('Probability Density Function (PDF)', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    stats = (f"Comparison:\n Mean: {np.mean(tf):.3f}\n Std: {np.std(tf):.3f}\n\n"
             f"Prediction:\n Mean: {np.mean(pf):.3f}\n Std: {np.std(pf):.3f}\n\n"
    )
    ax.text(0.98, 0.98, stats, transform=ax.transAxes, va='top', ha='right',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    if VAR == "pr":
        ax.set_yscale('log')
        ax.set_xscale('log')

    fig.subplots_adjust(top=0.85)
    add_test_id(fig, test_id)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    pdf.close()
    n_pages = 11 if VAR == 'pr' else 9
    print(f"\n✓ Done: {OUTPUT_PDF}  ({n_pages} pages)")
