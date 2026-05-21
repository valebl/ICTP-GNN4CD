import math
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / max(half - 1, 1)
        )
        emb = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_c), out_c),
            nn.GELU(),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_c), out_c),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class VelocityUNet(nn.Module):
    def __init__(self, in_channels, base=32):
        super().__init__()
        b = base
        self.enc0 = DoubleConv(in_channels, b)
        self.down1 = nn.Conv2d(b, b * 2, 2, stride=2)
        self.enc1 = DoubleConv(b * 2, b * 2)
        self.down2 = nn.Conv2d(b * 2, b * 4, 2, stride=2)
        self.enc2 = DoubleConv(b * 4, b * 4)
        self.down3 = nn.Conv2d(b * 4, b * 8, 2, stride=2)
        self.bottleneck = DoubleConv(b * 8, b * 8)
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, 2, 2)
        self.dec2 = DoubleConv(b * 8, b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, 2)
        self.dec1 = DoubleConv(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, 2, 2)
        self.dec0 = DoubleConv(b * 2, b)
        self.out = nn.Conv2d(b, 1, 1)

    def forward(self, x):
        e0 = self.enc0(x)
        e1 = self.enc1(self.down1(e0))
        e2 = self.enc2(self.down2(e1))
        bn = self.bottleneck(self.down3(e2))
        d2 = self.dec2(torch.cat([self.up3(bn), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up2(d2), e1], dim=1))
        d0 = self.dec0(torch.cat([self.up1(d1), e0], dim=1))
        return self.out(d0)


class ResidualDenoiserLR(nn.Module):
    """
    U-Net velocity model for high-frequency residual flow matching.

    Inputs are rasterized to a 128x128 grid:
      - noisy residual state x_t
      - log1p GNN precipitation condition
      - optional static fields, currently orography
      - low-resolution atmospheric features scattered to high-resolution nodes
      - scalar flow time
    """

    def __init__(self, n_static=1, n_lr=15, lr_hidden=16,
                 time_emb_dim=64, unet_base=32, grid_h=128, grid_w=128):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w

        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_proj = nn.Linear(time_emb_dim, 1)
        self.lr_proj = nn.Sequential(nn.Linear(n_lr, lr_hidden), nn.GELU())
        self.register_buffer("degree", None)

        in_channels = 2 + n_static + lr_hidden + 1
        self.unet = VelocityUNet(in_channels=in_channels, base=unet_base)

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

        if self.degree is None or self.degree.shape[1] != n_high:
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

        t_emb = self.time_proj(self.time_emb(t))
        t_grid = t_emb.view(bsz, 1, 1, 1).expand(bsz, 1, h, w)

        channels = [
            x_t.reshape(bsz, 1, h, w),
            cond_gnn.reshape(bsz, 1, h, w),
        ]
        if static_feats.shape[-1] > 0:
            static_grid = static_feats.T.reshape(static_feats.shape[-1], h, w)
            channels.append(static_grid.unsqueeze(0).expand(bsz, -1, -1, -1))
        channels.extend([lr_grid, t_grid])

        v_grid = self.unet(torch.cat(channels, dim=1))
        return v_grid.reshape(bsz, n_nodes)
