import argparse
import gc
import os
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap

# Compatibility for old-GNN pickle files saved with NumPy 2.x and read in the
# RLenv NumPy 1.x runtime. NumPy 2 pickles may reference numpy._core modules.
try:
    import sys
    import numpy.core as _np_core
    import numpy.core._multiarray_umath as _np_umath
    import numpy.core.fromnumeric as _np_fromnumeric
    import numpy.core.multiarray as _np_multiarray
    import numpy.core.numeric as _np_numeric

    sys.modules.setdefault("numpy._core", _np_core)
    sys.modules.setdefault("numpy._core._multiarray_umath", _np_umath)
    sys.modules.setdefault("numpy._core.fromnumeric", _np_fromnumeric)
    sys.modules.setdefault("numpy._core.multiarray", _np_multiarray)
    sys.modules.setdefault("numpy._core.numeric", _np_numeric)
except Exception:
    pass

os.environ.setdefault("CARTOPY_DATA_DIR", "/leonardo_work/ICT26_ESP_0/wtang/cartopy_data")
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False


def as_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def maybe_inverse_log1p_precip(field, label):
    field = field.astype(np.float32, copy=False)
    finite = field[np.isfinite(field)]
    if finite.size == 0:
        return field

    # Test predictions should be in mm/day. Some intermediate files can still
    # carry log1p precipitation; in that case a 20-year Rx1day plot becomes
    # almost blank on a 0-225 mm/day color scale. SA precipitation in mm/day
    # should normally have values well above this range over ACCESS-CM2 tests.
    q999 = float(np.nanpercentile(finite, 99.9))
    vmax = float(np.nanmax(finite))
    if vmax <= 12.0 and q999 <= 8.0:
        print(f"[plot_test_extreme_report] {label}: detected log1p-like precipitation scale; applying expm1")
        field = np.expm1(field).clip(min=0.0).astype(np.float32, copy=False)
        finite = field[np.isfinite(field)]
        if finite.size == 0:
            return field
        q999 = float(np.nanpercentile(finite, 99.9))
        vmax = float(np.nanmax(finite))

    # Some test pipelines can leave precipitation in kg m-2 s-1. The requested
    # figures use mm/day; if the whole multi-year file is still tiny, convert it.
    # This is deliberately conservative so already-correct mm/day files are not
    # rescaled.
    if vmax <= 1.0 and q999 <= 0.1:
        print(f"[plot_test_extreme_report] {label}: detected flux-like precipitation scale; multiplying by 86400")
        return (field * 86400.0).clip(min=0.0).astype(np.float32, copy=False)
    return field


def _select_prediction_field(data, field_name=None):
    if field_name:
        field = getattr(data, field_name)
    elif hasattr(data, "pr_gnn4cd"):
        field = data.pr_gnn4cd
    elif hasattr(data, "tasmax_gnn4cd"):
        field = data.tasmax_gnn4cd
    else:
        raise AttributeError("Could not find pr_gnn4cd/tasmax_gnn4cd in prediction file")

    return field


def orient_field(field, n_times, n_nodes, label):
    """
    Return prediction as (node, time).

    Test files from the deterministic Attention path usually store
    (node, time). DDPM files can store either a 2-D primary field or a
    sample stack. Use explicit time/node sizes instead of shape heuristics
    so we do not accidentally average over the time axis.
    """
    field = np.squeeze(as_numpy(field))
    if field.ndim == 1:
        if n_times == 1 and field.shape[0] == n_nodes:
            return field.reshape(n_nodes, 1)
        raise ValueError(f"{label}: cannot orient 1-D field with shape {field.shape}")

    if field.ndim == 2:
        if field.shape == (n_nodes, n_times):
            return field
        if field.shape == (n_times, n_nodes):
            return field.T
        raise ValueError(
            f"{label}: expected (nodes,time)=({n_nodes},{n_times}) "
            f"or (time,nodes)=({n_times},{n_nodes}), got {field.shape}"
        )

    if field.ndim == 3:
        shape = field.shape
        node_axes = [i for i, size in enumerate(shape) if size == n_nodes]
        time_axes = [i for i, size in enumerate(shape) if size == n_times]
        if not node_axes or not time_axes:
            raise ValueError(
                f"{label}: cannot identify node/time axes in 3-D field {shape}; "
                f"n_nodes={n_nodes}, n_times={n_times}"
            )

        node_axis = node_axes[0]
        time_axis = time_axes[0] if time_axes[0] != node_axis else time_axes[-1]
        sample_axes = [i for i in range(3) if i not in (node_axis, time_axis)]
        if len(sample_axes) != 1:
            raise ValueError(f"{label}: cannot identify sample axis in 3-D field {shape}")

        field = np.moveaxis(field, (sample_axes[0], node_axis, time_axis), (0, 1, 2))
        return np.nanmean(field, axis=0)

    raise ValueError(f"{label}: unsupported field ndim={field.ndim}, shape={field.shape}")


def get_field(data, field_name=None, n_times=None):
    if n_times is None:
        raise ValueError("n_times must be provided to orient prediction fields safely")

    high = data["high"]
    n_nodes = as_numpy(get_attr(high, "lat")).size
    label = field_name or "prediction"

    # DDPM test files may keep the full sample stack in pr_gnn4cd when no
    # ensemble method was selected. For a climatological report, prefer the
    # already-saved mean field if it exists, otherwise orient and average the
    # stack explicitly.
    if field_name in (None, "", "pr_gnn4cd") and hasattr(data, "pr_residual_ddpm_mean"):
        field = data.pr_residual_ddpm_mean
        label = "pr_residual_ddpm_mean"
    else:
        field = _select_prediction_field(data, field_name)

    field = orient_field(field, n_times=n_times, n_nodes=n_nodes, label=label)
    return maybe_inverse_log1p_precip(field, label)


def get_times(data):
    times = []
    for t in np.asarray(data.times, dtype=object):
        if isinstance(t, np.datetime64):
            times.append(t.astype("datetime64[D]"))
        elif isinstance(t, (int, np.integer)):
            # Some PyG pickles store datetime64 values as integer nanoseconds.
            # Convert those back explicitly; otherwise str(t)[:10] becomes a
            # meaningless digit prefix and date matching fails.
            times.append(np.datetime64(int(t), "ns").astype("datetime64[D]"))
        elif hasattr(t, "year"):
            times.append(np.datetime64(f"{int(t.year):04d}-{int(t.month):02d}-{int(t.day):02d}"))
        else:
            text = str(t)
            if text.isdigit() and len(text) > 12:
                times.append(np.datetime64(int(text), "ns").astype("datetime64[D]"))
            else:
                times.append(np.datetime64(text[:10]))
    return np.asarray(times)


def find_daily_index(times, target_day, source_path):
    matches = np.where(times == target_day)[0]
    if len(matches) > 0:
        return int(matches[0])

    # Keep the report usable for no-leap / shifted test files. This should be a
    # rare fallback, and the printed message makes the substitution explicit.
    day_diffs = np.abs((times - target_day).astype("timedelta64[D]").astype(int))
    nearest = int(np.nanargmin(day_diffs))
    if day_diffs[nearest] <= 3:
        print(
            "[plot_test_extreme_report] requested daily date "
            f"{str(target_day)} not found in {os.path.basename(source_path)}; "
            f"using nearest available date {str(times[nearest])}"
        )
        return nearest

    raise ValueError(
        f"Daily date {target_day} not found in {source_path}. "
        f"Available range is {times[0]} to {times[-1]} with {len(times)} entries."
    )


def get_attr(obj, name):
    if hasattr(obj, name):
        return getattr(obj, name)
    return obj[name]


def infer_grid(data, n_nodes, grid_h=None, grid_w=None):
    if grid_h is None or grid_w is None:
        side = int(round(np.sqrt(n_nodes)))
        if side * side != n_nodes:
            raise ValueError(
                "Please pass --grid-h/--grid-w; node count is not a square: "
                f"{n_nodes}"
            )
        grid_h = side
        grid_w = side

    high = data["high"]
    lat = as_numpy(get_attr(high, "lat"))
    lon = as_numpy(get_attr(high, "lon"))
    if lat.size != n_nodes or lon.size != n_nodes:
        raise ValueError(
            f"High-grid lon/lat size mismatch: lat={lat.size}, lon={lon.size}, "
            f"field nodes={n_nodes}"
        )
    return int(grid_h), int(grid_w), lat.reshape(-1), lon.reshape(-1)


def to_grid(field_node_time, grid_h, grid_w):
    return field_node_time.reshape(grid_h, grid_w, -1)


def annual_rx1day(field_node_time, times):
    years = np.asarray([int(str(t)[:4]) for t in times])
    out = []
    for year in sorted(np.unique(years)):
        idx = years == year
        if np.any(idx):
            out.append(np.nanmax(field_node_time[:, idx], axis=1))
    return np.stack(out, axis=1)


def rx1day_climatology(field_node_time, times):
    return np.nanmean(annual_rx1day(field_node_time, times), axis=1)


def make_precip_cmap():
    return LinearSegmentedColormap.from_list(
        "precip_sa",
        ["#f7fbff", "#c7ecee", "#62c6d7", "#2b8cbe", "#4146ac", "#8c1d82", "#d7191c", "#fee08b"],
        N=256,
    )


def make_rx1day_cmap():
    return LinearSegmentedColormap.from_list(
        "rx1day_sa",
        ["#6b3f00", "#b87919", "#ead7a0", "#f7f7f7", "#a9ddd7", "#35978f", "#004b3a"],
        N=256,
    )


def make_delta_cmap():
    return LinearSegmentedColormap.from_list(
        "delta_rx1day",
        ["#6b3f00", "#b87919", "#ead7a0", "#f7f7f7", "#a9ddd7", "#35978f", "#004b3a"],
        N=256,
    )


def plot_map(ax, lon, lat, values, title, cmap, vmin, vmax, cbar_label):
    values = np.asarray(values)
    lon = np.asarray(lon)
    lat = np.asarray(lat)

    transform = ccrs.PlateCarree() if HAS_CARTOPY else None
    use_mesh = lon.ndim == 2 and lat.ndim == 2 and lon.shape == values.shape and lat.shape == values.shape
    if use_mesh:
        kwargs = {"transform": transform} if HAS_CARTOPY else {}
        mesh = ax.pcolormesh(
            lon,
            lat,
            values,
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            edgecolor="face",
            linewidth=0.01,
            **kwargs,
        )
    else:
        kwargs = {"transform": transform} if HAS_CARTOPY else {}
        mesh = ax.scatter(
            lon.reshape(-1),
            lat.reshape(-1),
            c=values.reshape(-1),
            s=14,
            marker="s",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            **kwargs,
        )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    if HAS_CARTOPY and hasattr(ax, "set_extent"):
        ax.coastlines(resolution="10m", linewidth=0.8, color="black")
        ax.add_feature(cfeature.BORDERS, linewidth=0.6, edgecolor="black")
        ax.set_extent(
            [
                float(np.nanmin(lon)) - 0.5,
                float(np.nanmax(lon)) + 0.5,
                float(np.nanmin(lat)) - 0.5,
                float(np.nanmax(lat)) + 0.5,
            ],
            crs=ccrs.PlateCarree(),
        )
    else:
        ax.set_xlim([float(np.nanmin(lon)) - 0.5, float(np.nanmax(lon)) + 0.5])
        ax.set_ylim([float(np.nanmin(lat)) - 0.5, float(np.nanmax(lat)) + 0.5])
        ax.set_aspect("equal")
    cbar = plt.colorbar(mesh, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label(cbar_label)
    return mesh


def load_prediction(path, field_name, grid_h, grid_w):
    with open(path, "rb") as f:
        data = pickle.load(f)
    times = get_times(data)
    field = get_field(data, field_name, n_times=len(times))
    grid_h, grid_w, lat, lon = infer_grid(data, field.shape[0], grid_h, grid_w)
    print(
        f"[plot_test_extreme_report] loaded {os.path.basename(path)}: "
        f"field={field.shape}, times={len(times)}, "
        f"min/max={np.nanmin(field):.3f}/{np.nanmax(field):.3f}"
    )
    del data
    gc.collect()
    return field, times, grid_h, grid_w, lat, lon


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-file", required=True)
    parser.add_argument("--mid-century-file", required=True)
    parser.add_argument("--end-century-file", required=True)
    parser.add_argument("--delta-historical-file", default="")
    parser.add_argument("--delta-mid-century-file", default="")
    parser.add_argument("--delta-end-century-file", default="")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--report-name", default="ACCESS_CM2_pr_test_extreme_report")
    parser.add_argument("--field-name", default="")
    parser.add_argument("--daily-date", default="1998-11-13")
    parser.add_argument("--grid-h", type=int, default=128)
    parser.add_argument("--grid-w", type=int, default=128)
    parser.add_argument("--daily-vmax", type=float, default=225.0)
    parser.add_argument("--rx1day-vmax", type=float, default=225.0)
    parser.add_argument("--delta-vmax", type=float, default=45.0)
    parser.add_argument("--delta-mid-vmax", type=float, default=None)
    parser.add_argument("--delta-end-vmax", type=float, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    field_name = args.field_name or None

    hist, hist_times, grid_h, grid_w, lat, lon = load_prediction(
        args.historical_file, field_name, args.grid_h, args.grid_w
    )
    mid, mid_times, _, _, _, _ = load_prediction(
        args.mid_century_file, field_name, args.grid_h, args.grid_w
    )
    end, end_times, _, _, _, _ = load_prediction(
        args.end_century_file, field_name, args.grid_h, args.grid_w
    )

    hist_rx = rx1day_climatology(hist, hist_times)

    # Optional split: use the main historical file for Emulator historical
    # diagnostics, but use ESD pseudo-reality files for climate-change signal.
    # If no delta files are provided, the report falls back to the same files.
    delta_hist = hist
    delta_hist_times = hist_times
    delta_mid = mid
    delta_mid_times = mid_times
    delta_end = end
    delta_end_times = end_times
    if args.delta_historical_file:
        delta_hist, delta_hist_times, _, _, _, _ = load_prediction(
            args.delta_historical_file, field_name, args.grid_h, args.grid_w
        )
    if args.delta_mid_century_file:
        delta_mid, delta_mid_times, _, _, _, _ = load_prediction(
            args.delta_mid_century_file, field_name, args.grid_h, args.grid_w
        )
    if args.delta_end_century_file:
        delta_end, delta_end_times, _, _, _, _ = load_prediction(
            args.delta_end_century_file, field_name, args.grid_h, args.grid_w
        )

    delta_hist_rx = rx1day_climatology(delta_hist, delta_hist_times)
    mid_rx = rx1day_climatology(delta_mid, delta_mid_times)
    end_rx = rx1day_climatology(delta_end, delta_end_times)

    eps = 1e-6
    mid_delta = 100.0 * (mid_rx - delta_hist_rx) / np.maximum(delta_hist_rx, eps)
    end_delta = 100.0 * (end_rx - delta_hist_rx) / np.maximum(delta_hist_rx, eps)

    target_day = np.datetime64(args.daily_date)
    daily_idx = find_daily_index(hist_times, target_day, args.historical_file)
    daily = hist[:, daily_idx]
    daily_label = str(hist_times[daily_idx])

    print(
        "[plot_test_extreme_report] diagnostics:",
        f"hist_rx min/max={np.nanmin(hist_rx):.3f}/{np.nanmax(hist_rx):.3f}",
        f"daily min/max={np.nanmin(daily):.3f}/{np.nanmax(daily):.3f}",
        f"daily_date={daily_label}",
        f"mid_delta min/max={np.nanmin(mid_delta):.3f}/{np.nanmax(mid_delta):.3f}",
        f"end_delta min/max={np.nanmin(end_delta):.3f}/{np.nanmax(end_delta):.3f}",
    )

    precip_cmap = make_precip_cmap()
    rx1day_cmap = make_rx1day_cmap()
    delta_cmap = make_delta_cmap()
    delta_mid_vmax = args.delta_mid_vmax if args.delta_mid_vmax is not None else args.delta_vmax
    delta_end_vmax = args.delta_end_vmax if args.delta_end_vmax is not None else args.delta_vmax

    figures = [
        (
            hist_rx,
            "Rx1day climatology, ACCESS-CM2 historical (1981-2000)",
            rx1day_cmap,
            0,
            args.rx1day_vmax,
            "Rx1day [mm day$^{-1}$]",
            "rx1day_climatology_1981_2000",
        ),
        (
            daily,
            f"Daily precipitation, ACCESS-CM2 {daily_label}",
            precip_cmap,
            0,
            args.daily_vmax,
            "Precipitation [mm day$^{-1}$]",
            f"daily_precip_{daily_label}",
        ),
        (
            mid_delta,
            "Delta Rx1day, ACCESS-CM2 2041-2060 vs 1981-2000",
            delta_cmap,
            -delta_mid_vmax,
            delta_mid_vmax,
            "Delta Rx1day [%]",
            "delta_rx1day_2041_2060_vs_1981_2000",
        ),
        (
            end_delta,
            "Delta Rx1day, ACCESS-CM2 2080-2099 vs 1981-2000",
            delta_cmap,
            -delta_end_vmax,
            delta_end_vmax,
            "Delta Rx1day [%]",
            "delta_rx1day_2080_2099_vs_1981_2000",
        ),
    ]

    pdf_path = os.path.join(args.output_path, f"{args.report_name}.pdf")
    with PdfPages(pdf_path) as pdf:
        for values, title, cmap, vmin, vmax, label, stem in figures:
            if HAS_CARTOPY:
                fig, ax = plt.subplots(figsize=(8, 7), subplot_kw={"projection": ccrs.PlateCarree()})
            else:
                fig, ax = plt.subplots(figsize=(8, 7))
            plot_map(ax, lon, lat, values, title, cmap, vmin, vmax, label)
            fig.tight_layout()
            fig.savefig(os.path.join(args.output_path, f"{args.report_name}_{stem}.png"), dpi=220)
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    main()
