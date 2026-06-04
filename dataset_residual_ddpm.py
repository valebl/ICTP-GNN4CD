import pickle
import re

import numpy as np
import torch
from torch.utils.data import Dataset


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


def parse_years(years):
    if not years:
        return None
    if isinstance(years, str):
        items = re.split(r"[_ ,]+", years.strip())
        return {int(item) for item in items if item}
    return {int(item) for item in years}


class ResidualDDPMDatasetLR(Dataset):
    """
    Dataset for DDPM-style full residual learning after deterministic GNN output.

    Precipitation residual is learned in log1p space:
        residual = log1p(target_pr) - log1p(gnn_pr)

    Temperature residual is learned in raw-value space:
        residual = target_tasmax - gnn_tasmax

    Unlike Experiments_GNN_diffusion2/3, this dataset does not remove the
    smoothed large-scale component. The diffusion model learns the full residual,
    matching the residual setup used in the reference DDPM implementation.
    """

    def __init__(
        self,
        gnn_pred_file,
        target_file,
        time_index_file,
        low_input_file,
        target_type="precipitation",
        train_years="",
        lr_mean=None,
        lr_std=None,
    ):
        with open(gnn_pred_file, "rb") as f:
            data = pickle.load(f)

        if target_type == "precipitation":
            x_gnn_raw = data.pr_gnn4cd
        elif target_type in ("temperature", "tasmax"):
            x_gnn_raw = data.tasmax_gnn4cd
        else:
            raise ValueError(f"Unsupported target_type: {target_type}")

        target = np.load(target_file, allow_pickle=True)
        time_index = np.load(time_index_file, allow_pickle=True)
        low_input = np.load(low_input_file, allow_pickle=True)

        if target.shape[0] < target.shape[1]:
            target = target.T

        gnn_times = np.array(data.times, dtype=object)
        selected_years = parse_years(train_years)
        if selected_years is not None:
            time_mask = np.array([infer_year(t) in selected_years for t in gnn_times])
            if not np.any(time_mask):
                preview = [str(t) for t in gnn_times[:5]]
                raise ValueError(
                    f"No training times matched train_years={sorted(selected_years)}. "
                    f"First prediction times are: {preview}"
                )
            gnn_times = gnn_times[time_mask]
            x_gnn_raw = x_gnn_raw[:, time_mask]

        time_index_obj = np.array(time_index, dtype=object)
        time_to_idx = {str(t): i for i, t in enumerate(time_index_obj)}
        aligned_idxs = np.array([time_to_idx[str(t)] for t in gnn_times])

        target_aligned = target[:, aligned_idxs]
        if target_type == "precipitation":
            x_gnn = np.log1p(np.clip(x_gnn_raw, 0, None))
            x_hr = np.log1p(np.clip(target_aligned, 0, None))
        else:
            x_gnn = x_gnn_raw
            x_hr = target_aligned

        x_gnn_t = torch.from_numpy(x_gnn.T.astype(np.float32))
        x_hr_t = torch.from_numpy(x_hr.T.astype(np.float32))
        residual = x_hr_t - x_gnn_t

        n_low, _, n_vars, n_lev = low_input.shape
        n_lr = n_vars * n_lev
        lr_aligned = low_input[:, aligned_idxs, :, :]
        lr_aligned = lr_aligned.reshape(n_low, -1, n_lr).transpose(1, 0, 2)
        if lr_mean is None or lr_std is None:
            lr_mean = lr_aligned.mean(axis=(0, 1), keepdims=True)
            lr_std = lr_aligned.std(axis=(0, 1), keepdims=True) + 1e-8
        lr_aligned = (lr_aligned - lr_mean) / lr_std

        print(
            "[ResidualDDPMDatasetLR] "
            f"{x_gnn_t.shape[0]} timesteps, {x_gnn_t.shape[1]} nodes, "
            f"{n_low} LR nodes, {n_lr} LR features | "
            f"years={train_years or 'all'} | "
            f"full residual mean={float(residual.mean()):.4f}, "
            f"std={float(residual.std()):.4f}"
        )

        self.x_gnn = x_gnn_t
        self.x_hr = x_hr_t
        self.residual = residual
        self.lr_feats = torch.from_numpy(lr_aligned.astype(np.float32))
        self.lr_mean = lr_mean
        self.lr_std = lr_std

    def __len__(self):
        return self.residual.shape[0]

    def __getitem__(self, idx):
        return self.x_gnn[idx], self.x_hr[idx], self.residual[idx], self.lr_feats[idx]
