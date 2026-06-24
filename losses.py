import torch
import torch.nn.functional as F


def smooth_node_field(field, grid_h=128, grid_w=128, kernel_size=9):
    """Box-smooth flattened node fields laid out on a regular high-res grid."""
    if kernel_size <= 1:
        return field
    pad = kernel_size // 2
    grid = field.reshape(-1, 1, grid_h, grid_w)
    grid = F.pad(grid, (pad, pad, pad, pad), mode="replicate")
    smooth = F.avg_pool2d(grid, kernel_size=kernel_size, stride=1)
    return smooth.reshape_as(field)


def spatial_confidence_gate_from_precip(
    gnn_pr_mm,
    grid_h=128,
    grid_w=128,
    smooth_kernel=9,
    threshold=1.0,
    tau=0.5,
    gate_min=0.0,
):
    """
    Spatial confidence gate for residual corrections.

    NEW: this gate makes the diffusion residual active mostly where the
    deterministic Attention/GNN baseline already has a smooth wet signal. It
    is meant to reduce false displaced precipitation structures that increase
    RMSE while retaining stochastic texture in plausible wet regions.
    """
    smooth_ref = smooth_node_field(
        gnn_pr_mm,
        grid_h=grid_h,
        grid_w=grid_w,
        kernel_size=smooth_kernel,
    )
    gate = torch.sigmoid((smooth_ref - threshold) / max(float(tau), 1e-6))
    if gate_min > 0:
        gate = gate_min + (1.0 - gate_min) * gate
    return gate.clamp(min=0.0, max=1.0)


def wet_day_loss(pred_mm, target_mm, threshold=1.0, sharpness=4.0):
    """
    Wet/dry auxiliary loss for precipitation.

    The DDPM predicts a residual, not a wet probability directly. We first
    reconstruct precipitation in mm/day, then classify each pixel as wet when
    precipitation is above threshold. BCE-with-logits is used for numerical
    stability.
    """
    logits = sharpness * (pred_mm - threshold)
    logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
    logits = logits.clamp(min=-50.0, max=50.0)
    target_prob = torch.nan_to_num((target_mm > threshold).float(), nan=0.0)
    return F.binary_cross_entropy_with_logits(logits, target_prob)


def reconstruct_precip_mm(x_gnn_log, residual_log, log_clip=None):
    pred_log = x_gnn_log + residual_log
    if log_clip is not None and log_clip > 0:
        pred_log = torch.nan_to_num(
            pred_log,
            nan=0.0,
            posinf=float(log_clip),
            neginf=-float(log_clip),
        )
        pred_log = pred_log.clamp(min=0.0, max=float(log_clip))
    return torch.expm1(pred_log).clamp(min=0.0), pred_log


def reconstruct_temperature(x_gnn, residual):
    pred = x_gnn + residual
    return pred, pred


def apply_coarse_conservation(
    pred_mm,
    ref_mm,
    grid_h=128,
    grid_w=128,
    block_size=8,
    eps=1e-6,
    scale_min=0.0,
    scale_max=4.0,
):
    """
    Multiplicative coarse correction for precipitation.

    The reference field is the deterministic Attention prediction. This keeps
    the DDPM correction from drifting too far in coarse-block mean while still
    allowing high-frequency stochastic structure.
    """
    pred = pred_mm.reshape(-1, 1, grid_h, grid_w)
    ref = ref_mm.reshape(-1, 1, grid_h, grid_w)
    pred_c = F.avg_pool2d(pred, kernel_size=block_size, stride=block_size)
    ref_c = F.avg_pool2d(ref, kernel_size=block_size, stride=block_size)
    scale = ref_c / (pred_c + eps)
    scale = torch.clamp(scale, scale_min, scale_max)
    scale = F.interpolate(
        scale,
        size=(grid_h, grid_w),
        mode="bilinear",
        align_corners=False,
    )
    return (pred * scale).reshape_as(pred_mm).clamp(min=0.0)


def apply_coarse_additive_correction(pred, ref, grid_h=128, grid_w=128, block_size=8):
    """Additive coarse correction for temperature-like variables."""
    pred_grid = pred.reshape(-1, 1, grid_h, grid_w)
    ref_grid = ref.reshape(-1, 1, grid_h, grid_w)
    pred_c = F.avg_pool2d(pred_grid, kernel_size=block_size, stride=block_size)
    ref_c = F.avg_pool2d(ref_grid, kernel_size=block_size, stride=block_size)
    delta = ref_c - pred_c
    delta = F.interpolate(
        delta,
        size=(grid_h, grid_w),
        mode="bilinear",
        align_corners=False,
    )
    return (pred_grid + delta).reshape_as(pred)
