import argparse
import os
import pickle
import re

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from torch_geometric.data import HeteroData

from diffusion import ResidualDDPM
from losses import (
    apply_coarse_additive_correction,
    apply_coarse_conservation,
    reconstruct_precip_mm,
    reconstruct_temperature,
    spatial_confidence_gate_from_precip,
)
from models import ResidualDenoiserLR


def write_log(msg, path, mode="a"):
    with open(path, mode) as f:
        f.write(msg)


def infer_year(time_value):
    if hasattr(time_value, "year"):
        return int(time_value.year)
    if isinstance(time_value, np.datetime64):
        return int(str(time_value.astype("datetime64[Y]"))[:4])

    text = str(time_value)
    if re.fullmatch(r"-?\d+", text):
        value = int(text)
        abs_value = abs(value)
        try:
            if abs_value >= 10**17:
                return int(str(np.datetime64(value, "ns").astype("datetime64[Y]"))[:4])
            if abs_value >= 10**14:
                return int(str(np.datetime64(value, "us").astype("datetime64[Y]"))[:4])
            if abs_value >= 10**11:
                return int(str(np.datetime64(value, "ms").astype("datetime64[Y]"))[:4])
            if abs_value >= 10**8:
                return int(str(np.datetime64(value, "s").astype("datetime64[Y]"))[:4])
        except (OverflowError, ValueError):
            pass

    match = re.search(r"(19|20)\d{2}", text)
    if match is None:
        return None
    return int(match.group(0))


def normalize_times_for_output(times):
    out = []
    for time_value in np.array(times, dtype=object):
        if isinstance(time_value, np.datetime64):
            out.append(time_value.astype("datetime64[ns]"))
            continue
        if hasattr(time_value, "year"):
            try:
                out.append(np.datetime64(time_value, "ns"))
            except (TypeError, ValueError):
                out.append(
                    np.datetime64(
                        f"{int(time_value.year):04d}-"
                        f"{int(time_value.month):02d}-"
                        f"{int(time_value.day):02d}",
                        "ns",
                    )
                )
            continue

        text = str(time_value)
        if re.fullmatch(r"-?\d+", text):
            value = int(text)
            abs_value = abs(value)
            try:
                if abs_value >= 10**17:
                    out.append(np.datetime64(value, "ns"))
                    continue
                if abs_value >= 10**14:
                    out.append(np.datetime64(value, "us").astype("datetime64[ns]"))
                    continue
                if abs_value >= 10**11:
                    out.append(np.datetime64(value, "ms").astype("datetime64[ns]"))
                    continue
                if abs_value >= 10**8:
                    out.append(np.datetime64(value, "s").astype("datetime64[ns]"))
                    continue
            except (OverflowError, ValueError):
                pass
        out.append(time_value)
    return np.array(out)


def time_key(time_value):
    """Stable key for aligning numpy datetime/cftime/int-encoded times."""
    normalized = normalize_times_for_output([time_value])[0]
    if isinstance(normalized, np.datetime64):
        return str(normalized.astype("datetime64[ns]"))
    return str(normalized)


def load_test_low_input(
    file_path,
    file_name,
    graph,
    input_graph_path=None,
    params=("q", "t", "u", "v", "z"),
    levels=("500", "700", "850"),
):
    """
    Load low-resolution predictors from a CORDEX-ML test netCDF file.

    This mirrors the Attention test loader but is kept local so the DDPM
    predictor can run on test periods that are not present in train/low_input.npy.
    The low grid is then reduced to the graph low nodes using unique_src.npy when
    available, or the low-to-high graph edge sources otherwise.
    """
    # The low-resolution branch must use exactly the same channel order used
    # during DDPM training. In the SA preprocessing this order is stored in
    # low_input_metadata.json (currently levels 500, 700, 850). Falling back to
    # a hard-coded order here previously permuted the pressure-level channels
    # during test inference and produced grid-like spatial artifacts.
    metadata_file = None
    if input_graph_path:
        metadata_file = os.path.join(input_graph_path, "low_input_metadata.json")
    if metadata_file and os.path.exists(metadata_file):
        import json

        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        params = tuple(metadata.get("variables", params))
        levels = tuple(str(level) for level in metadata.get("levels", levels))
    else:
        params = tuple(params)
        levels = tuple(str(level) for level in levels)

    ds = xr.open_dataset(os.path.join(file_path, file_name), engine="netcdf4")
    n_params = len(params)
    n_levels = len(levels)
    time_dim = len(ds["time"])
    lat_low = ds["lat"].values
    lon_low = ds["lon"].values
    lat_dim = len(lat_low)
    lon_dim = len(lon_low)

    x_low = np.zeros(
        (time_dim, n_params, n_levels, lat_dim, lon_dim),
        dtype=np.float32,
    )
    for var_idx, var in enumerate(params):
        for lev_idx, lev in enumerate(levels):
            x_low[:, var_idx, lev_idx] = ds[f"{var}_{lev}"].values

    if lat_low[0] > lat_low[-1]:
        x_low = np.flip(x_low, axis=3)
    if lon_low[0] > lon_low[-1]:
        x_low = np.flip(x_low, axis=4)

    # time, var, lev, lat, lon -> low_node, time, var, lev
    x_low = np.transpose(x_low, (3, 4, 0, 1, 2))
    x_low = x_low.reshape(-1, *x_low.shape[2:])

    src = graph["low", "to", "high"].edge_index[0].cpu().numpy()
    unique_src = np.unique(src)
    unique_src_file = None
    if input_graph_path:
        unique_src_file = os.path.join(input_graph_path, "unique_src.npy")
    if unique_src_file and os.path.exists(unique_src_file):
        unique_src = np.load(unique_src_file)

    if unique_src.shape[0] != x_low.shape[0]:
        x_low = x_low[unique_src]

    time_index = ds["time"].values
    ds.close()
    return x_low, time_index, params, levels


def output_file_for_step(output_file, step):
    root, ext = os.path.splitext(output_file)
    return f"{root}_step{step:04d}{ext}"


parser = argparse.ArgumentParser()
parser.add_argument("--gnn_pred_file", type=str, required=True)
parser.add_argument("--graph_file", type=str, required=True)
parser.add_argument("--static_file", type=str, default=None)
parser.add_argument("--low_input_file", type=str, default="")
parser.add_argument("--time_index_file", type=str, default="")
parser.add_argument("--test_input_path_p", type=str, default="")
parser.add_argument("--predictors_filename", type=str, default="")
parser.add_argument("--input_graph_path", type=str, default="")
parser.add_argument("--lr_norm_file", type=str, required=True)
parser.add_argument("--residual_norm_file", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)
parser.add_argument("--output_file", type=str, default="G_SA_pr_attention_residual_ddpm.pkl")
parser.add_argument("--log_file", type=str, default="log.txt")
parser.add_argument("--target_type", type=str, default="precipitation")
parser.add_argument("--ddpm_timesteps", type=int, default=1000)
parser.add_argument("--beta_start", type=float, default=1e-4)
parser.add_argument("--beta_end", type=float, default=0.02)
parser.add_argument("--unet_base", type=int, default=32)
parser.add_argument("--lr_hidden", type=int, default=32)
parser.add_argument("--time_emb_dim", type=int, default=64)
parser.add_argument("--n_samples", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--grid_h", type=int, default=128)
parser.add_argument("--grid_w", type=int, default=128)
parser.add_argument("--use_spatial_confidence_gate", action="store_true")
parser.add_argument("--no-use_spatial_confidence_gate", dest="use_spatial_confidence_gate", action="store_false")
parser.set_defaults(use_spatial_confidence_gate=False)
parser.add_argument("--gate_smooth_kernel", type=int, default=9)
parser.add_argument("--gate_threshold", type=float, default=1.0)
parser.add_argument("--gate_tau", type=float, default=0.5)
parser.add_argument("--gate_min", type=float, default=0.0)
parser.add_argument(
    "--ensemble_method",
    type=str,
    default="raw",
    choices=("raw", "mean", "smooth_gnn_best_member"),
)
parser.add_argument("--ensemble_smooth_kernel", type=int, default=9)
parser.add_argument("--coarse_block_size", type=int, default=8)
parser.add_argument("--coarse_scale_min", type=float, default=0.0)
parser.add_argument("--coarse_scale_max", type=float, default=4.0)
parser.add_argument("--apply_coarse_conservation", action="store_true")
parser.add_argument("--no-apply_coarse_conservation", dest="apply_coarse_conservation", action="store_false")
parser.set_defaults(apply_coarse_conservation=True)
parser.add_argument("--noise_scale", type=float, default=1.0)
parser.add_argument("--reverse_noise_scale", type=float, default=1.0)
parser.add_argument("--residual_scale", type=float, default=1.0)
parser.add_argument("--snapshot_interval", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--year_filter", type=int, default=0)


if __name__ == "__main__":
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    log_path = os.path.join(args.output_path, args.log_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    write_log(
        "Attention-conditioned residual DDPM prediction\n"
        f"Device: {device}\n"
        f"DDPM timesteps: {args.ddpm_timesteps}\n",
        log_path,
        "w",
    )

    with open(args.gnn_pred_file, "rb") as f:
        gnn_data = pickle.load(f)

    time_mask = None
    out_times = gnn_data.times
    if args.year_filter > 0:
        times_arr = np.array(gnn_data.times, dtype=object)
        time_mask = np.array([infer_year(t) == args.year_filter for t in times_arr])
        if not np.any(time_mask):
            preview = [str(t) for t in times_arr[:5]]
            raise ValueError(
                f"No times matching year_filter={args.year_filter}. "
                f"First times are: {preview}"
            )
        out_times = times_arr[time_mask]

    if args.target_type == "precipitation":
        x_gnn_raw = np.clip(gnn_data.pr_gnn4cd, 0, None)
        x_gnn_model = np.log1p(x_gnn_raw)
    elif args.target_type in ("temperature", "tasmax"):
        x_gnn_raw = gnn_data.tasmax_gnn4cd
        x_gnn_model = x_gnn_raw
    else:
        raise ValueError(f"Unsupported target_type: {args.target_type}")

    if time_mask is not None:
        x_gnn_raw = x_gnn_raw[:, time_mask]
        x_gnn_model = x_gnn_model[:, time_mask]

    x_gnn_t = torch.from_numpy(x_gnn_model.T.astype(np.float32)).to(device)
    x_gnn_raw_t = torch.from_numpy(x_gnn_raw.T.astype(np.float32)).to(device)
    n_times, n_nodes = x_gnn_t.shape
    write_log(f"Attention predictions: {n_times} timesteps, {n_nodes} nodes\n", log_path)

    with open(args.graph_file, "rb") as f:
        graph = pickle.load(f)
    low2high_edge_index = graph["low", "to", "high"].edge_index.to(device)

    n_static = 0
    static_feats = torch.zeros(n_nodes, 0, device=device)
    if args.static_file and os.path.exists(args.static_file):
        orog = np.load(args.static_file).reshape(-1, 1).astype(np.float32)
        orog = (orog - orog.mean()) / (orog.std() + 1e-8)
        static_feats = torch.from_numpy(orog).to(device)
        n_static = 1

    if args.test_input_path_p and args.predictors_filename:
        low_input, time_index, predictor_params, predictor_levels = load_test_low_input(
            file_path=args.test_input_path_p,
            file_name=args.predictors_filename,
            graph=graph,
            input_graph_path=args.input_graph_path,
        )
        write_log(
            "Loaded test low-resolution predictors from "
            f"{os.path.join(args.test_input_path_p, args.predictors_filename)}\n",
            log_path,
        )
        write_log(
            "Test predictor channel order: "
            f"params={list(predictor_params)}, levels={list(predictor_levels)}\n",
            log_path,
        )
    else:
        if not args.low_input_file or not args.time_index_file:
            raise ValueError(
                "Provide either test_input_path_p/predictors_filename for test "
                "prediction or low_input_file/time_index_file for preprocessed "
                "validation prediction."
            )
        low_input = np.load(args.low_input_file, allow_pickle=True)
        time_index = np.load(args.time_index_file, allow_pickle=True)
    n_low, _, n_vars, n_lev = low_input.shape
    n_lr = n_vars * n_lev

    gnn_times = np.array(gnn_data.times, dtype=object)
    if time_mask is not None:
        gnn_times = gnn_times[time_mask]
    time_index_obj = np.array(time_index, dtype=object)
    time_to_idx = {time_key(t): i for i, t in enumerate(time_index_obj)}
    missing_times = [time_key(t) for t in gnn_times if time_key(t) not in time_to_idx]
    if missing_times:
        raise ValueError(
            "Could not align DDPM low-resolution predictors with GNN prediction "
            f"times. First missing times: {missing_times[:5]}"
        )
    aligned_idxs = np.array([time_to_idx[time_key(t)] for t in gnn_times])

    lr_aligned = low_input[:, aligned_idxs, :, :]
    lr_aligned = lr_aligned.reshape(n_low, n_times, n_lr).transpose(1, 0, 2)
    norm = np.load(args.lr_norm_file)
    lr_aligned = (lr_aligned - norm["mean"]) / norm["std"]
    lr_t = torch.from_numpy(lr_aligned.astype(np.float32)).to(device)
    write_log(f"LR features: {tuple(lr_t.shape)}\n", log_path)

    residual_norm = np.load(args.residual_norm_file)
    residual_mean = float(residual_norm["mean"])
    residual_std = float(residual_norm["std"])
    normalize_residual = bool(residual_norm["enabled"])
    residual_mean_t = torch.tensor(residual_mean, device=device, dtype=torch.float32)
    residual_std_t = torch.tensor(residual_std, device=device, dtype=torch.float32)
    write_log(
        f"Residual normalization: enabled={normalize_residual}, "
        f"mean={residual_mean:.6f}, std={residual_std:.6f}\n",
        log_path,
    )

    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    model = ResidualDenoiserLR(
        n_static=n_static,
        n_lr=n_lr,
        lr_hidden=args.lr_hidden,
        time_emb_dim=args.time_emb_dim,
        unet_base=args.unet_base,
        grid_h=args.grid_h,
        grid_w=args.grid_w,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    unexpected = [key for key in unexpected if key != "degree"]
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch. Missing keys: {missing}; unexpected keys: {unexpected}"
        )
    model = model.to(device).eval()

    ddpm = ResidualDDPM(
        timesteps=args.ddpm_timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        device=device,
    )

    target_out = None
    if hasattr(gnn_data, "target"):
        target_out = gnn_data.target
        if time_mask is not None and hasattr(target_out, "shape") and target_out.shape[-1] == len(time_mask):
            target_out = target_out[:, time_mask]

    def refine_from_residual(batch_gnn, batch_gnn_raw, residual):
        # NEW: optional spatial confidence gate for prediction. The original
        # branch reconstructed with the full DDPM residual:
        #     pred, _ = reconstruct_precip_mm(batch_gnn, residual)
        # When enabled, the residual is suppressed where the smoothed
        # deterministic Attention baseline has low wet confidence.
        residual_for_refine = residual
        if (
            args.use_spatial_confidence_gate
            and args.target_type == "precipitation"
        ):
            gate = spatial_confidence_gate_from_precip(
                batch_gnn_raw,
                grid_h=args.grid_h,
                grid_w=args.grid_w,
                smooth_kernel=args.gate_smooth_kernel,
                threshold=args.gate_threshold,
                tau=args.gate_tau,
                gate_min=args.gate_min,
            )
            residual_for_refine = residual * gate

        # NEW: optional residual amplitude scaling. This is a prediction-only
        # post-processing control for testing whether smaller stochastic
        # corrections reduce RMSE without retraining. The original behavior is:
        #     residual_for_refine = residual_for_refine
        if args.residual_scale != 1.0:
            residual_for_refine = residual_for_refine * args.residual_scale

        if args.target_type == "precipitation":
            pred, _ = reconstruct_precip_mm(batch_gnn, residual_for_refine)
        else:
            pred, _ = reconstruct_temperature(batch_gnn, residual_for_refine)

        if args.apply_coarse_conservation:
            if args.target_type == "precipitation":
                pred = apply_coarse_conservation(
                    pred,
                    batch_gnn_raw,
                    grid_h=args.grid_h,
                    grid_w=args.grid_w,
                    block_size=args.coarse_block_size,
                    scale_min=args.coarse_scale_min,
                    scale_max=args.coarse_scale_max,
                )
            else:
                pred = apply_coarse_additive_correction(
                    pred,
                    batch_gnn_raw,
                    grid_h=args.grid_h,
                    grid_w=args.grid_w,
                    block_size=args.coarse_block_size,
                )
        return pred

    def smooth_node_time_np(field_node_time, kernel_size):
        if kernel_size <= 1:
            return field_node_time.astype(np.float32, copy=False)
        field = torch.from_numpy(field_node_time.astype(np.float32, copy=False))
        n_nodes_field, n_time_field = field.shape
        field = field.T.reshape(-1, 1, args.grid_h, args.grid_w)
        pad = kernel_size // 2
        field = F.pad(field, (pad, pad, pad, pad), mode="replicate")
        smooth = F.avg_pool2d(field, kernel_size=kernel_size, stride=1)
        return smooth.reshape(n_time_field, n_nodes_field).T.numpy()

    def select_smooth_gnn_best_member(stacked):
        """
        NEW: post-hoc multi-sample selection.

        For each day, choose the sample whose smoothed precipitation field is
        closest to the smoothed deterministic Attention baseline. This keeps
        stochastic details but discourages large-scale spatial displacement,
        which is the main source of RMSE inflation.
        """
        baseline_smooth = smooth_node_time_np(
            x_gnn_raw,
            kernel_size=args.ensemble_smooth_kernel,
        )
        selected = np.empty_like(stacked[0])
        chosen = []
        for t in range(stacked.shape[2]):
            base_t = baseline_smooth[:, t]
            scores = []
            for k in range(stacked.shape[0]):
                sample_smooth_t = smooth_node_time_np(
                    stacked[k, :, t:t + 1],
                    kernel_size=args.ensemble_smooth_kernel,
                )[:, 0]
                scores.append(float(np.sqrt(np.mean((sample_smooth_t - base_t) ** 2))))
            best_idx = int(np.argmin(scores))
            selected[:, t] = stacked[best_idx, :, t]
            chosen.append(best_idx)
        counts = np.bincount(chosen, minlength=stacked.shape[0])
        write_log(
            "Smooth-GNN best-member counts: "
            + ", ".join(f"s{k}={int(v)}" for k, v in enumerate(counts))
            + "\n",
            log_path,
        )
        return selected

    def save_output(samples, output_file):
        out = HeteroData()
        out.times = normalize_times_for_output(out_times)
        out["low"].lat = gnn_data["low"].lat
        out["low"].lon = gnn_data["low"].lon
        out["high"].lat = gnn_data["high"].lat
        out["high"].lon = gnn_data["high"].lon
        if target_out is not None:
            out.target = target_out

        stacked = np.stack(samples)
        mean_field = np.mean(stacked, axis=0)
        if args.ensemble_method == "mean":
            primary_field = mean_field
        elif args.ensemble_method == "smooth_gnn_best_member" and len(samples) > 1:
            primary_field = select_smooth_gnn_best_member(stacked)
        else:
            # Original behavior: n_samples=1 saves the single sample; n_samples>1
            # saved the full stack as pr_gnn4cd. For plotting/reporting, the new
            # ensemble modes above provide a 2-D primary field instead.
            primary_field = samples[0] if args.n_samples == 1 else stacked

        if args.target_type == "precipitation":
            out.pr_gnn4cd = primary_field
            out.pr_attention_baseline = x_gnn_raw
            out.pr_residual_ddpm_samples = stacked
            out.pr_residual_ddpm_mean = mean_field
        else:
            out.tasmax_gnn4cd = primary_field
            out.tasmax_attention_baseline = x_gnn_raw
            out.tasmax_residual_ddpm_samples = stacked
            out.tasmax_residual_ddpm_mean = mean_field

        out_path = os.path.join(args.output_path, output_file)
        with open(out_path, "wb") as f:
            pickle.dump(out, f)
        write_log(f"Saved to {out_path}\n", log_path)

    all_samples = []
    snapshot_samples = {}
    for sample_idx in range(args.n_samples):
        refined_batches = []
        snapshot_batches = {}
        for i in range(0, n_times, args.batch_size):
            batch_gnn = x_gnn_t[i:i + args.batch_size]
            batch_gnn_raw = x_gnn_raw_t[i:i + args.batch_size]
            batch_lr = lr_t[i:i + args.batch_size]

            sample_out = ddpm.sample(
                model,
                batch_gnn,
                static_feats,
                low2high_edge_index,
                batch_lr,
                noise_scale=args.noise_scale,
                reverse_noise_scale=args.reverse_noise_scale,
                snapshot_interval=args.snapshot_interval,
            )
            if args.snapshot_interval > 0:
                residual_model, residual_snapshots_model = sample_out
            else:
                residual_model = sample_out
                residual_snapshots_model = {}

            residual = residual_model * residual_std_t + residual_mean_t
            pred = refine_from_residual(batch_gnn, batch_gnn_raw, residual)
            refined_batches.append(pred.cpu().numpy())

            for step, residual_step_model in residual_snapshots_model.items():
                residual_step = residual_step_model * residual_std_t + residual_mean_t
                pred_step = refine_from_residual(batch_gnn, batch_gnn_raw, residual_step)
                snapshot_batches.setdefault(step, []).append(pred_step.cpu().numpy())

        sample = np.concatenate(refined_batches, axis=0).T
        all_samples.append(sample)
        for step, batches in snapshot_batches.items():
            snapshot_sample = np.concatenate(batches, axis=0).T
            snapshot_samples.setdefault(step, []).append(snapshot_sample)
        write_log(f"Sample {sample_idx + 1}/{args.n_samples} done\n", log_path)

    save_output(all_samples, args.output_file)
    for step in sorted(snapshot_samples):
        save_output(snapshot_samples[step], output_file_for_step(args.output_file, step))
