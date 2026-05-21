import argparse
import os
import pickle
import re

import numpy as np
import torch
from torch_geometric.data import HeteroData

from diffusion.residual_flow_matching import ResidualFlowMatching
from losses import (
    apply_coarse_additive_correction,
    apply_coarse_conservation,
    reconstruct_precip_mm,
    reconstruct_temperature,
)
from models.residual_denoiser_lr import ResidualDenoiserLR


def write_log(msg, path, mode="a"):
    with open(path, mode) as f:
        f.write(msg)


def infer_year(time_value):
    """Return a year from datetime/cftime/numpy/string-like time values."""
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

    match = re.search(r"(19|20)\d{2}", str(time_value))
    if match is None:
        return None
    return int(match.group(0))


def normalize_times_for_output(times):
    """
    plot_report.py expects a datetime64-like time index. Some Valentina tasmax
    outputs store times as Unix nanosecond integers, so convert those back.
    """
    out = []
    for time_value in np.array(times, dtype=object):
        if isinstance(time_value, np.datetime64):
            out.append(time_value.astype("datetime64[ns]"))
            continue
        if hasattr(time_value, "year"):
            out.append(np.datetime64(time_value, "ns"))
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


parser = argparse.ArgumentParser()
parser.add_argument("--gnn_pred_file", type=str, required=True)
parser.add_argument("--graph_file", type=str, required=True)
parser.add_argument("--static_file", type=str, default=None)
parser.add_argument("--low_input_file", type=str, required=True)
parser.add_argument("--time_index_file", type=str, required=True)
parser.add_argument("--lr_norm_file", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)
parser.add_argument("--output_file", type=str, default="G_SA_pr_residual_fm.pkl")
parser.add_argument("--log_file", type=str, default="log.txt")
parser.add_argument("--target_type", type=str, default="precipitation")
parser.add_argument("--n_inference_steps", type=int, default=50)
parser.add_argument("--unet_base", type=int, default=32)
parser.add_argument("--lr_hidden", type=int, default=16)
parser.add_argument("--time_emb_dim", type=int, default=64)
parser.add_argument("--n_samples", type=int, default=10)
parser.add_argument("--sigma_min", type=float, default=1e-4)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--grid_h", type=int, default=128)
parser.add_argument("--grid_w", type=int, default=128)
parser.add_argument("--coarse_block_size", type=int, default=8)
parser.add_argument("--apply_coarse_conservation", action="store_true")
parser.add_argument("--no-apply_coarse_conservation", dest="apply_coarse_conservation",
                    action="store_false")
parser.set_defaults(apply_coarse_conservation=True)
parser.add_argument("--noise_scale", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--year_filter", type=int, default=0,
                    help="If >0, keep only prediction times whose string starts with this year.")


if __name__ == "__main__":
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    log_path = os.path.join(args.output_path, args.log_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    write_log(f"Residual high-frequency FM prediction\nDevice: {device}\n", log_path, "w")

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
    write_log(f"GNN predictions: {n_times} timesteps, {n_nodes} nodes\n", log_path)

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

    low_input = np.load(args.low_input_file, allow_pickle=True)
    time_index = np.load(args.time_index_file, allow_pickle=True)
    n_low, _, n_vars, n_lev = low_input.shape
    n_lr = n_vars * n_lev

    gnn_times = np.array(gnn_data.times, dtype=object)
    if time_mask is not None:
        gnn_times = gnn_times[time_mask]
    time_index_obj = np.array(time_index, dtype=object)
    time_to_idx = {str(t): i for i, t in enumerate(time_index_obj)}
    aligned_idxs = np.array([time_to_idx[str(t)] for t in gnn_times])

    lr_aligned = low_input[:, aligned_idxs, :, :]
    lr_aligned = lr_aligned.reshape(n_low, n_times, n_lr).transpose(1, 0, 2)
    norm = np.load(args.lr_norm_file)
    lr_aligned = (lr_aligned - norm["mean"]) / norm["std"]
    lr_t = torch.from_numpy(lr_aligned.astype(np.float32)).to(device)
    write_log(f"LR features: {tuple(lr_t.shape)}\n", log_path)

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
    cfm = ResidualFlowMatching(sigma_min=args.sigma_min)

    all_samples = []
    for sample_idx in range(args.n_samples):
        refined_batches = []
        for i in range(0, n_times, args.batch_size):
            batch_gnn = x_gnn_t[i:i + args.batch_size]
            batch_gnn_raw = x_gnn_raw_t[i:i + args.batch_size]
            batch_lr = lr_t[i:i + args.batch_size]
            residual = cfm.sample(
                model,
                batch_gnn,
                static_feats,
                low2high_edge_index,
                batch_lr,
                n_steps=args.n_inference_steps,
                noise_scale=args.noise_scale,
            )
            if args.target_type == "precipitation":
                pred, _ = reconstruct_precip_mm(batch_gnn, residual)
            else:
                pred, _ = reconstruct_temperature(batch_gnn, residual)

            if args.apply_coarse_conservation:
                if args.target_type == "precipitation":
                    pred = apply_coarse_conservation(
                        pred,
                        batch_gnn_raw,
                        grid_h=args.grid_h,
                        grid_w=args.grid_w,
                        block_size=args.coarse_block_size,
                    )
                else:
                    pred = apply_coarse_additive_correction(
                        pred,
                        batch_gnn_raw,
                        grid_h=args.grid_h,
                        grid_w=args.grid_w,
                        block_size=args.coarse_block_size,
                    )
            refined_batches.append(pred.cpu().numpy())

        sample = np.concatenate(refined_batches, axis=0).T
        all_samples.append(sample)
        write_log(f"Sample {sample_idx + 1}/{args.n_samples} done\n", log_path)

    out = HeteroData()
    out.times = normalize_times_for_output(out_times)
    out["low"].lat = gnn_data["low"].lat
    out["low"].lon = gnn_data["low"].lon
    out["high"].lat = gnn_data["high"].lat
    out["high"].lon = gnn_data["high"].lon
    if hasattr(gnn_data, "target"):
        target = gnn_data.target
        if time_mask is not None and hasattr(target, "shape") and target.shape[-1] == len(time_mask):
            target = target[:, time_mask]
        out.target = target

    if args.target_type == "precipitation":
        out.pr_gnn4cd = all_samples[0] if args.n_samples == 1 else np.stack(all_samples)
        out.pr_gnn_baseline = gnn_data.pr_gnn4cd
        out.pr_residual_fm_mean = np.mean(np.stack(all_samples), axis=0)
    else:
        out.tasmax_gnn4cd = all_samples[0] if args.n_samples == 1 else np.stack(all_samples)
        out.tasmax_gnn_baseline = gnn_data.tasmax_gnn4cd
        out.tasmax_residual_fm_mean = np.mean(np.stack(all_samples), axis=0)

    out_path = os.path.join(args.output_path, args.output_file)
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    write_log(f"Saved to {out_path}\n", log_path)
