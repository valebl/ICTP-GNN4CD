import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _box_blur_fields(fields, grid_h=128, grid_w=128, kernel_size=9, chunk_size=256):
    """
    fields: torch.Tensor (T, N)
    Returns a spatially smoothed tensor with the same shape.
    """
    if kernel_size <= 1:
        return fields
    if fields.shape[1] != grid_h * grid_w:
        raise ValueError(
            f"Expected {grid_h * grid_w} nodes, got {fields.shape[1]}. "
            "Pass matching --grid_h/--grid_w if the grid changed."
        )

    pad = kernel_size // 2
    blurred = []
    for i in range(0, fields.shape[0], chunk_size):
        x = fields[i:i + chunk_size].reshape(-1, 1, grid_h, grid_w)
        x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        x = F.avg_pool2d(x, kernel_size=kernel_size, stride=1)
        blurred.append(x.reshape(-1, grid_h * grid_w))
    return torch.cat(blurred, dim=0)


class ResidualFMDatasetLR(Dataset):
    """
    Dataset for stochastic high-frequency residual flow matching.

    Training target for precipitation:
        residual_raw = log1p(target_pr) - log1p(gnn_pr)

    Training target for temperature:
        residual_raw = target_tasmax - gnn_tasmax

    In both cases:
        residual_hf  = residual_raw - smooth(residual_raw)

    The model only learns residual_hf. This keeps the GNN large-scale field as
    the anchor and asks the generative model to add convective-scale structure.
    """

    def __init__(self, gnn_pred_file, target_file, time_index_file, low_input_file,
                 target_type="precipitation", grid_h=128, grid_w=128,
                 highpass_kernel=9):
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
        residual_raw = x_hr_t - x_gnn_t
        residual_smooth = _box_blur_fields(
            residual_raw, grid_h=grid_h, grid_w=grid_w, kernel_size=highpass_kernel
        )
        residual_hf = residual_raw - residual_smooth

        n_low, _, n_vars, n_lev = low_input.shape
        n_lr = n_vars * n_lev
        lr_aligned = low_input[:, aligned_idxs, :, :]
        lr_aligned = lr_aligned.reshape(n_low, -1, n_lr).transpose(1, 0, 2)
        lr_mean = lr_aligned.mean(axis=(0, 1), keepdims=True)
        lr_std = lr_aligned.std(axis=(0, 1), keepdims=True) + 1e-8
        lr_aligned = (lr_aligned - lr_mean) / lr_std

        print(
            "[ResidualFMDatasetLR] "
            f"{x_gnn_t.shape[0]} timesteps, {x_gnn_t.shape[1]} nodes, "
            f"{n_low} LR nodes, {n_lr} LR features | "
            f"raw residual std={float(residual_raw.std()):.4f}, "
            f"highpass residual std={float(residual_hf.std()):.4f}"
        )

        self.x_gnn = x_gnn_t
        self.x_hr = x_hr_t
        self.residual = residual_hf
        self.lr_feats = torch.from_numpy(lr_aligned.astype(np.float32))
        self.lr_mean = lr_mean
        self.lr_std = lr_std

    def __len__(self):
        return self.residual.shape[0]

    def __getitem__(self, idx):
        return self.x_gnn[idx], self.x_hr[idx], self.residual[idx], self.lr_feats[idx]
