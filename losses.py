import torch
import torch.nn.functional as F


def endpoint_from_velocity(x_t, v_pred, t):
    """Approximate x1 for a linear FM path using x_t + (1-t) * v."""
    return x_t + (1.0 - t.view(-1, 1)) * v_pred


def psd_loss(pred_log, target_log, grid_h=128, grid_w=128):
    pred = pred_log.reshape(-1, grid_h, grid_w)
    target = target_log.reshape(-1, grid_h, grid_w)
    pred_fft = torch.fft.rfft2(pred, norm="ortho")
    target_fft = torch.fft.rfft2(target, norm="ortho")
    pred_amp = torch.log1p(torch.abs(pred_fft))
    target_amp = torch.log1p(torch.abs(target_fft))
    return F.mse_loss(pred_amp, target_amp)


def quantile_loss(pred_mm, target_mm, quantiles=(0.5, 0.9, 0.95, 0.99)):
    losses = []
    pred_flat = pred_mm.reshape(pred_mm.shape[0], -1)
    target_flat = target_mm.reshape(target_mm.shape[0], -1)
    for q in quantiles:
        pred_q = torch.quantile(pred_flat, q, dim=1)
        target_q = torch.quantile(target_flat, q, dim=1)
        losses.append(F.l1_loss(pred_q, target_q))
    return torch.stack(losses).mean()


def coarse_conservation_loss(pred_mm, target_mm, grid_h=128, grid_w=128, block_size=8):
    pred = pred_mm.reshape(-1, 1, grid_h, grid_w)
    target = target_mm.reshape(-1, 1, grid_h, grid_w)
    pred_c = F.avg_pool2d(pred, kernel_size=block_size, stride=block_size)
    target_c = F.avg_pool2d(target, kernel_size=block_size, stride=block_size)
    return F.l1_loss(pred_c, target_c)


def wet_day_loss(pred_mm, target_mm, threshold=1.0, sharpness=4.0):
    pred_prob = torch.sigmoid(sharpness * (pred_mm - threshold))
    target_prob = (target_mm > threshold).float()
    return F.binary_cross_entropy(pred_prob, target_prob)


def reconstruct_precip_mm(x_gnn_log, residual_log):
    pred_log = x_gnn_log + residual_log
    return torch.expm1(pred_log).clamp(min=0.0), pred_log


def reconstruct_temperature(x_gnn, residual):
    pred = x_gnn + residual
    return pred, pred


def apply_coarse_conservation(pred_mm, ref_mm, grid_h=128, grid_w=128,
                              block_size=8, eps=1e-6):
    """
    Rescale the field toward the coarse-block mean of ref_mm.

    The coarse scale factor is bilinearly upsampled instead of nearest-neighbor
    upsampled; nearest preserves block means more exactly, but leaves visible
    8x8 mosaic artifacts in daily precipitation maps.
    """
    pred = pred_mm.reshape(-1, 1, grid_h, grid_w)
    ref = ref_mm.reshape(-1, 1, grid_h, grid_w)
    pred_c = F.avg_pool2d(pred, kernel_size=block_size, stride=block_size)
    ref_c = F.avg_pool2d(ref, kernel_size=block_size, stride=block_size)
    scale = ref_c / (pred_c + eps)
    scale = torch.clamp(scale, 0.25, 4.0)
    scale = F.interpolate(
        scale,
        size=(grid_h, grid_w),
        mode="bilinear",
        align_corners=False,
    )
    return (pred * scale).reshape_as(pred_mm).clamp(min=0.0)


def apply_coarse_additive_correction(pred, ref, grid_h=128, grid_w=128,
                                     block_size=8):
    """
    Add a smooth coarse correction so block means approach ref.

    This is appropriate for temperature-like variables. Precipitation uses the
    multiplicative version above to preserve non-negativity.
    """
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
