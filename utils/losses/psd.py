import torch
import torch.nn as nn
from .registry import register_loss


@register_loss("PSD_Loss")
class PSD_Loss(nn.Module):
    """
    Pure radial power-spectral-density loss for single-channel 2D fields.
    Returns ONLY the spectral term -- combine with your other losses
    externally
 
    pred, target: flattened (B*H*W,) node tensors, OR (B, H, W).
    """
 
    def __init__(
            self,
            eps=1e-6,
            wavenumber_weight=0.0,
            y_dim=128,
            x_dim=128,
            apply_expm1=False,
            wet_threshold=0.1,
            min_wet_weight=0.1,
        ):
        super().__init__()
        self.eps = eps
        self.wavenumber_weight = wavenumber_weight
        self.y_dim = y_dim
        self.x_dim = x_dim
        self.apply_expm1 = apply_expm1
        self.wet_threshold = wet_threshold
        self.min_wet_weight = min_wet_weight  # floor so dry days aren't zeroed out entirely
 
    def forward(self, pred, target):
        if pred.dim() == 1:
            n = pred.shape[0]
            grid_size = self.y_dim * self.x_dim
            assert n % grid_size == 0, (
                f"PSD_Loss: flattened input length {n} is not a multiple of "
                f"y_dim*x_dim={grid_size}."
            )
            B = n // grid_size
            pred = pred.view(B, self.y_dim, self.x_dim)
            target = target.view(B, self.y_dim, self.x_dim)
        else:
            B = pred.shape[0]
 
        if self.apply_expm1:
            pred = torch.expm1(pred)
            target = torch.expm1(target)
 
        pred = torch.nan_to_num(pred, nan=0.0)
        target = torch.nan_to_num(target, nan=0.0)
 
        per_sample_losses = []
        per_sample_weights = []
        for b in range(B):
            psd_pred, _ = self._compute_psd_radial(pred[b])
            psd_true, _ = self._compute_psd_radial(target[b])
 
            loss_b = (torch.log(psd_pred + self.eps) -
                      torch.log(psd_true + self.eps)) ** 2
 
            if self.wavenumber_weight != 0:
                k = torch.arange(loss_b.shape[-1], device=loss_b.device)
                loss_b = loss_b * (1 + self.wavenumber_weight * k)
 
            per_sample_losses.append(loss_b.mean())
 
            # wet-pixel fraction weighting: wet-pixel fraction, floored so dry
            # days still contribute a little rather than being zeroed out
            wet_frac = (target[b] > self.wet_threshold).float().mean()
            per_sample_weights.append(wet_frac.clamp(min=self.min_wet_weight))
 
        losses = torch.stack(per_sample_losses)
        weights = torch.stack(per_sample_weights)
        return (losses * weights).sum() / weights.sum()
 
    def _compute_psd_radial(self, data):
        data = torch.nan_to_num(data.float(), nan=0.0)
 
        F = torch.fft.fft2(data, norm="ortho")
        F = torch.fft.fftshift(F)
        power = (F.real ** 2 + F.imag ** 2)
 
        H, W = power.shape
        y = torch.arange(H, device=data.device)
        x = torch.arange(W, device=data.device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
 
        center_y = (H - 1) / 2.0
        center_x = (W - 1) / 2.0
        r = torch.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2).to(torch.long)
 
        r_flat = r.view(-1)
        p_flat = power.view(-1)
        K = r_flat.max().item() + 1
 
        radial_sum = torch.zeros(K, device=data.device)
        radial_sum.scatter_add_(0, r_flat, p_flat)
        counts = torch.zeros(K, device=data.device)
        counts.scatter_add_(0, r_flat, torch.ones_like(p_flat))
        radial_psd = radial_sum / counts.clamp_min(1)
 
        wavenumbers = torch.arange(K, device=data.device)
        return radial_psd, wavenumbers
