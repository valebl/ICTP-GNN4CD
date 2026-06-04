import math

import torch
import torch.nn as nn


def _num_groups(channels, max_groups=8):
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal flow-time embedding, similar to diffusion timestep embeddings."""

    def __init__(self, dim, scale=1.0):
        super().__init__()
        self.dim = dim
        self.scale = scale

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=device, dtype=t.dtype)
            / max(half - 1, 1)
        )
        emb = (self.scale * t[:, None].float()) * freqs[None, :].float()
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class AdaptiveGroupNormResidualBlock(nn.Module):
    """
    Residual block with timestep-conditioned adaptive group normalization.

    This mirrors the useful part of the reference residual diffusion model:
    the scalar diffusion/flow time modulates feature maps through a learned
    scale and shift, so the same U-Net can behave differently near noise and
    near the final residual field.
    """

    def __init__(self, in_channels, out_channels, time_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm1 = nn.GroupNorm(_num_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.norm2 = nn.GroupNorm(_num_groups(out_channels), out_channels)
        self.act = nn.GELU()
        self.time_proj = nn.Linear(time_dim, 2 * out_channels)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x, time_emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        scale, shift = self.time_proj(time_emb).chunk(2, dim=1)
        h = h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        return h + self.skip(x)


class PlainResidualBlock(nn.Module):
    """Residual conv block matching the reference model's non-FiLM blocks."""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )
        self.act = nn.LeakyReLU(0.01)

    def forward(self, x):
        h = self.act(self.conv1(x))
        h = self.conv2(h)
        return self.act(h + self.skip(x))


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.block = PlainResidualBlock(in_channels, out_channels, kernel_size)

    def forward(self, x):
        skip = self.block(x)
        return nn.functional.avg_pool2d(skip, kernel_size=2, stride=2), skip


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, kernel_size=3):
        super().__init__()
        self.block = PlainResidualBlock(
            in_channels + skip_channels, out_channels, kernel_size
        )

    def forward(self, x, skip):
        x = nn.functional.interpolate(
            x, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.block(torch.cat([x, skip], dim=1))


class LowResolutionPredictorBranch(nn.Module):
    """
    Bottleneck encoder for coarse predictors after low-to-high graph scatter.

    In the reference model, coarse predictors follow a separate pathway and are
    concatenated near the bottleneck. Here low-resolution climate predictors are
    projected on low nodes, scattered to high nodes using the graph, then
    downsampled to the high-resolution branch bottleneck scale.
    """

    def __init__(self, in_channels, base_channels):
        super().__init__()
        b = base_channels
        self.block0 = PlainResidualBlock(in_channels, b, kernel_size=3)
        self.block1 = PlainResidualBlock(b, b * 2, kernel_size=3)
        self.block2 = PlainResidualBlock(b * 2, b * 4, kernel_size=3)
        self.block3 = PlainResidualBlock(b * 4, b * 4, kernel_size=3)

    def forward(self, x):
        x = self.block0(x)
        x = nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        x = self.block1(x)
        x = nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        x = self.block2(x)
        x = nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return self.block3(x)


class ResidualConditionedUNet(nn.Module):
    """
    Paper-inspired conditional residual U-Net used as a DDPM noise predictor.

    High-resolution branch:
      noisy residual x_t, GNN conditional mean, and static terrain features.

    Low-resolution branch:
      atmospheric predictors scattered from low nodes to high nodes, encoded
      separately, and fused near the bottleneck, following the reference code.
    """

    def __init__(self, high_in_channels, lr_channels, base=32, time_dim=256):
        super().__init__()
        b = base
        self.lr_branch = LowResolutionPredictorBranch(lr_channels, b)

        self.film0 = AdaptiveGroupNormResidualBlock(high_in_channels, b, time_dim)
        self.down0 = DownBlock(b, b, kernel_size=3)
        self.film1 = AdaptiveGroupNormResidualBlock(b, b * 2, time_dim)
        self.down1 = DownBlock(b * 2, b * 2, kernel_size=3)
        self.film2 = AdaptiveGroupNormResidualBlock(b * 2, b * 4, time_dim)
        self.down2 = DownBlock(b * 4, b * 4, kernel_size=3)

        self.film3 = AdaptiveGroupNormResidualBlock(b * 8, b * 8, time_dim)
        self.mid0 = PlainResidualBlock(b * 8, b * 8, kernel_size=3)
        self.mid1 = PlainResidualBlock(b * 8, b * 8, kernel_size=5)
        self.film4 = AdaptiveGroupNormResidualBlock(b * 8, b * 8, time_dim)
        self.mid2 = PlainResidualBlock(b * 8, b * 16, kernel_size=3)

        self.up2 = UpBlock(b * 16, b * 4, b * 8, kernel_size=3)
        self.film5 = AdaptiveGroupNormResidualBlock(b * 8, b * 8, time_dim)
        self.up1 = UpBlock(b * 8, b * 2, b * 4, kernel_size=5)
        self.film6 = AdaptiveGroupNormResidualBlock(b * 4, b * 4, time_dim)
        self.up0 = UpBlock(b * 4, b, b * 2, kernel_size=3)
        self.film7 = AdaptiveGroupNormResidualBlock(b * 2, b * 2, time_dim)

        self.out0 = PlainResidualBlock(b * 2, b * 2, kernel_size=3)
        self.out1 = PlainResidualBlock(b * 2, b * 2, kernel_size=3)
        self.out2 = PlainResidualBlock(b * 2, b, kernel_size=5)
        mid_channels = max(b // 2, 16)
        self.out = nn.Sequential(
            nn.Conv2d(b, b, 3, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(b, mid_channels, 3, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(mid_channels, 1, 3, padding=1),
        )

    def forward(self, high_grid, lr_grid, time_emb):
        x = self.film0(high_grid, time_emb)
        x, skip0 = self.down0(x)
        x = self.film1(x, time_emb)
        x, skip1 = self.down1(x)
        x = self.film2(x, time_emb)
        x, skip2 = self.down2(x)

        lr_bottleneck = self.lr_branch(lr_grid)
        if lr_bottleneck.shape[-2:] != x.shape[-2:]:
            lr_bottleneck = nn.functional.interpolate(
                lr_bottleneck,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        x = self.film3(torch.cat([x, lr_bottleneck], dim=1), time_emb)
        x = self.mid0(x)
        x = self.mid1(x)
        x = self.film4(x, time_emb)
        x = self.mid2(x)

        x = self.up2(x, skip2)
        x = self.film5(x, time_emb)
        x = self.up1(x, skip1)
        x = self.film6(x, time_emb)
        x = self.up0(x, skip0)
        x = self.film7(x, time_emb)

        x = self.out0(x)
        x = self.out1(x)
        x = self.out2(x)
        return self.out(x)


class ResidualDenoiserLR(nn.Module):
    """
    Strong conditional U-Net noise predictor for residual DDPM.

    This model is designed for the same interface as Experiments_GNN_diffusion4:
    it starts from an already trained deterministic GNN prediction and learns
    the DDPM noise in a noisy full residual. The backbone follows the reference
    residual diffusion model more closely than diffusion3: high-resolution
    noisy residual/static/conditional-mean branch, separate low-resolution
    predictor branch, bottleneck fusion, and timestep-adaptive residual blocks.
    """

    def __init__(
        self,
        n_static=1,
        n_lr=15,
        lr_hidden=32,
        time_emb_dim=64,
        unet_base=32,
        grid_h=128,
        grid_w=128,
    ):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w

        time_dim = time_emb_dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        self.lr_proj = nn.Sequential(
            nn.Linear(n_lr, lr_hidden),
            nn.GELU(),
            nn.Linear(lr_hidden, lr_hidden),
            nn.GELU(),
        )
        self.register_buffer("degree", torch.empty(0), persistent=False)

        high_in_channels = 2 + n_static
        self.unet = ResidualConditionedUNet(
            high_in_channels=high_in_channels,
            lr_channels=lr_hidden,
            base=unet_base,
            time_dim=time_dim,
        )

    def _scatter_lr(self, lr_feats, low2high_edge_index, n_high):
        bsz, _, channels = lr_feats.shape
        src_idx = low2high_edge_index[0]
        dst_idx = low2high_edge_index[1]
        edge_count = low2high_edge_index.shape[1]

        src_feats = lr_feats[:, src_idx, :]
        high_lr = torch.zeros(
            bsz, n_high, channels, device=lr_feats.device, dtype=lr_feats.dtype
        )
        dst_exp = dst_idx.view(1, -1, 1).expand(bsz, -1, channels)
        high_lr.scatter_add_(1, dst_exp, src_feats)

        if self.degree.numel() != n_high:
            deg = torch.zeros(n_high, device=lr_feats.device)
            deg.scatter_add_(0, dst_idx, torch.ones(edge_count, device=lr_feats.device))
            self.degree = deg.clamp(min=1.0).view(1, -1, 1)

        return high_lr / self.degree

    def forward(self, x_t, cond_gnn, static_feats, low2high_edge_index, lr_feats, t):
        bsz, n_nodes = x_t.shape
        h, w = self.grid_h, self.grid_w

        lr_proj = self.lr_proj(lr_feats.reshape(-1, lr_feats.shape[-1]))
        lr_proj = lr_proj.reshape(bsz, -1, lr_proj.shape[-1])
        lr_high = self._scatter_lr(lr_proj, low2high_edge_index, n_nodes)
        lr_grid = lr_high.permute(0, 2, 1).reshape(bsz, -1, h, w)

        channels = [
            x_t.reshape(bsz, 1, h, w),
            cond_gnn.reshape(bsz, 1, h, w),
        ]
        if static_feats.shape[-1] > 0:
            static_grid = static_feats.T.reshape(static_feats.shape[-1], h, w)
            channels.append(static_grid.unsqueeze(0).expand(bsz, -1, -1, -1))
        high_grid = torch.cat(channels, dim=1)

        time_emb = self.time_mlp(t)
        v_grid = self.unet(high_grid, lr_grid, time_emb)
        return v_grid.reshape(bsz, n_nodes)
