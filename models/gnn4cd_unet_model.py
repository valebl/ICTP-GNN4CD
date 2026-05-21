import torch.nn as nn
import torch_geometric.nn as geometric_nn
from torch_geometric.nn import GATv2Conv, GraphConv
import numpy as np
import torch

from typing import Optional
from torch import Tensor

from torch_geometric.nn.inits import ones, zeros
from torch_geometric.typing import OptTensor
from torch_geometric.utils import scatter

from .registry import register_model


class GATBlock(nn.Module):
    def __init__(self, in_dim, out_dim, heads=2, dropout=0.2):
        super().__init__()
        self.gat = GATv2Conv(in_dim, out_dim, heads=heads,
                              dropout=dropout, aggr='add',
                              add_self_loops=True, bias=True)
        self.norm = nn.LayerNorm(out_dim * heads)
        self.act = nn.ReLU()
        # we need same shape to apply skip connection as x + h
        self.rescale = None
        if in_dim != out_dim * heads:
            self.rescale = nn.Linear(in_dim, out_dim * heads)

    def forward(self, x, edge_index):
        h = self.gat(x, edge_index)
        if self.rescale is not None:
            x = self.rescale(x)
        x = x + h          # residual
        x = self.norm(x)
        x = self.act(x)
        return x

class Processor(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.block1 = GATBlock(hidden, 32, heads=2, dropout=0.2)
        self.block2 = GATBlock(64, 32, heads=2, dropout=0.2)
        self.block3 = GATBlock(64, 32, heads=2, dropout=0.2)

    def forward(self, x, edge_index):
        x = self.block1(x, edge_index)
        x = self.block2(x, edge_index)
        x = self.block3(x, edge_index)
        return x


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
    def __init__(self, in_channels, out_channels=1, base=16):
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
        self.out = nn.Conv2d(b, out_channels, 1)

    def forward(self, x):
        e0 = self.enc0(x)
        e1 = self.enc1(self.down1(e0))
        e2 = self.enc2(self.down2(e1))
        bn = self.bottleneck(self.down3(e2))
        d2 = self.dec2(torch.cat([self.up3(bn), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up2(d2), e1], dim=1))
        d0 = self.dec0(torch.cat([self.up1(d1), e0], dim=1))
        return self.out(d0)


def nodes_to_grid(feat, grid_row, grid_col, batch_size, height, width):
    n_nodes = grid_row.shape[0]
    channels = feat.shape[-1]
    feat_b = feat.reshape(batch_size, n_nodes, channels)
    grid = torch.zeros(
        batch_size, channels, height, width,
        device=feat.device, dtype=feat.dtype
    )
    grid[:, :, grid_row, grid_col] = feat_b.permute(0, 2, 1)
    return grid


def grid_to_nodes(grid, grid_row, grid_col):
    feat_b = grid[:, :, grid_row, grid_col]
    return feat_b.permute(0, 2, 1).reshape(-1, grid.shape[1])


class CFMHead(nn.Module):
    def __init__(self, node_dim=64, cond_dim=64, time_emb_dim=32, hidden=128,
                 output_dim=1, unet_base=16):
        """
        Predicts the flow-matching velocity with a U-Net:
            v_theta(x_t, t, node_emb)  ->  (N, output_dim)

        cond_dim, time_emb_dim, and hidden are kept for config compatibility
        with the original per-node MLP CFM head.
        """
        super().__init__()
        self.output_dim = output_dim
        self.unet = VelocityUNet(
            in_channels=node_dim + output_dim + 1,
            out_channels=output_dim,
            base=unet_base,
        )
        self._grid_built = False
        self.grid_row = None
        self.grid_col = None
        self.grid_h = None
        self.grid_w = None

    def _build_grid_mapping(self, data):
        batch_vec = getattr(data['high'], 'batch', None)
        if batch_vec is None:
            mask0 = torch.ones(
                data['high'].x.shape[0], dtype=torch.bool, device=data['high'].x.device
            )
        else:
            mask0 = batch_vec == 0

        lon_np = data['high'].lon[mask0].detach().cpu().numpy().astype(np.float64)
        lat_np = data['high'].lat[mask0].detach().cpu().numpy().astype(np.float64)

        unique_lats = np.sort(np.unique(np.round(lat_np, 4)))[::-1]
        unique_lons = np.sort(np.unique(np.round(lon_np, 4)))

        lat2row = {round(float(v), 4): i for i, v in enumerate(unique_lats)}
        lon2col = {round(float(v), 4): i for i, v in enumerate(unique_lons)}

        self.grid_row = torch.tensor(
            [lat2row[round(float(v), 4)] for v in lat_np],
            dtype=torch.long, device=data['high'].x.device
        )
        self.grid_col = torch.tensor(
            [lon2col[round(float(v), 4)] for v in lon_np],
            dtype=torch.long, device=data['high'].x.device
        )
        self.grid_h = len(unique_lats)
        self.grid_w = len(unique_lons)
        self._grid_built = True

    def forward(self, node_emb, x_t, t, data):
        if not self._grid_built:
            self._build_grid_mapping(data)
        elif self.grid_row.device != node_emb.device:
            self.grid_row = self.grid_row.to(node_emb.device)
            self.grid_col = self.grid_col.to(node_emb.device)

        batch_vec = getattr(data['high'], 'batch', None)
        batch_size = 1 if batch_vec is None else int(batch_vec.max().item()) + 1

        cond_grid = nodes_to_grid(
            node_emb, self.grid_row, self.grid_col,
            batch_size, self.grid_h, self.grid_w
        )
        xt_grid = nodes_to_grid(
            x_t, self.grid_row, self.grid_col,
            batch_size, self.grid_h, self.grid_w
        )

        if t.numel() == 1:
            t_sample = t.reshape(1).expand(batch_size)
        else:
            t_sample = t.reshape(batch_size, -1)[:, 0]
        t_grid = t_sample.view(batch_size, 1, 1, 1).expand(
            batch_size, 1, self.grid_h, self.grid_w
        )

        unet_input = torch.cat([cond_grid, xt_grid, t_grid], dim=1)
        velocity_grid = self.unet(unet_input)
        return grid_to_nodes(velocity_grid, self.grid_row, self.grid_col)


class SinusoidalTimeEmbedding(nn.Module):
    """Maps scalar t in [0,1] to a (1, dim) vector. dim must be even.
    """
    def __init__(self, dim = 32):
        super().__init__()
        assert dim % 2 == 0
        half = dim // 2
        freqs = torch.exp(
            -torch.arange(half, dtype=torch.float32) * (np.log(10000) / (half - 1))
        )
        # register_buffer saves freqs in the model's state_dict so it moves correctly with
        # .to(device) and gets saved/loaded with the model, but is never updated by the optimizer
        self.register_buffer("freqs", freqs)
 
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: scalar or (1,)  ->  (1, dim)
        t = t.view(1, 1).float()
        args = t * self.freqs.unsqueeze(0) * 2 * np.pi    # (1, half)
        return torch.cat([args.sin(), args.cos()], dim=-1) # (1, dim)


@register_model("GNN4CD_UNet_Model")
class GNN4CD_UNet_Model(nn.Module):
    """
    GNN4CD with a Conditional Flow Matching generative head.
    """
    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument("--rnn_n_layers", type=int, default=2)
        parser.add_argument("--x_low_encoding_dim", type=int, default=128)
        parser.add_argument("--x_low2high_dim", type=float, default=64)
        parser.add_argument("--cfm_cond_dim", type=int, default=64)
        parser.add_argument("--cfm_time_emb_dim", type=int, default=32)
        parser.add_argument("--cfm_hidden", type=int, default=128)
        parser.add_argument("--cfm_unet_base", type=int, default=16)
        parser.add_argument("--n_steps", type=int, default=5)
        parser.add_argument("--n_samples", type=int, default=10)
        return parser
    
    def __init__(
        self,
        x_low_var_dim,
        x_low_lev_dim,
        x_high_dim,
        output_dim,
        history_length,
        rnn_n_layers,
        x_low_encoding_dim,
        x_low2high_dim,
        cfm_cond_dim,
        cfm_time_emb_dim,
        cfm_hidden,
        cfm_unet_base,
        n_steps,
        n_samples,
        target_type=None
        ):

        super().__init__()

        seq_length = history_length + 1
        rnn_input_dim = x_low_var_dim * x_low_lev_dim
        rnn_hidden_dim = x_low_var_dim * x_low_lev_dim

        self.n_steps = n_steps
        self.n_samples = n_samples
        self.target_type = target_type

        # input shape (N,L,Hin)
        self.rnn = nn.Sequential(
            nn.GRU(rnn_input_dim, rnn_hidden_dim, rnn_n_layers, batch_first=True),
        )

        self.dense = nn.Sequential(
            nn.Linear(rnn_hidden_dim*seq_length, x_low_encoding_dim),
            nn.ReLU()
        )

        self.downscaler = geometric_nn.Sequential('x, edge_index', [
            (GraphConv((x_low_encoding_dim, x_high_dim), out_channels=x_low2high_dim, aggr='mean'), 'x, edge_index -> x')
            ])
        
        self.processor = Processor(64)

        # New generative head (replaces self.predictor)
        self.cfm_head = CFMHead(
            node_dim=64,
            cond_dim=cfm_cond_dim,
            time_emb_dim=cfm_time_emb_dim,
            hidden=cfm_hidden,
            output_dim=output_dim,
            unet_base=cfm_unet_base,
        )

    def _sample(self, encod_high, data):
        # inference mode
        N = encod_high.shape[0]
        output_dim = self.cfm_head.output_dim
    
        # fixed-step Euler integration
        dt = 1.0 / self.n_steps
        t_grid = torch.linspace(0.0, 1.0 - dt, self.n_steps, device=encod_high.device)
        samples = []
        for _ in range(self.n_samples):
            x = torch.randn(N, output_dim, device=encod_high.device)
            for t_val in t_grid:
                v = self.cfm_head(encod_high, x, t_val.unsqueeze(0), data)
                x = x + v * dt
            if self.target_type == "precipitation":
                x = torch.clamp(x, min=0.0)
            samples.append(x)
            
        return torch.stack(samples, dim=-1).squeeze(1)
    
    def forward(self, data):
        encod_rnn, _ = self.rnn(data.x_dict['low']) # out, h
        encod_rnn = encod_rnn.flatten(start_dim=1)
        encod_rnn = self.dense(encod_rnn)
        encod_low2high  = self.downscaler((encod_rnn, data.x_dict['high']), data['low', 'to', 'high'].edge_index)
        encod_high = self.processor(encod_low2high , data.edge_index_dict[('high','within','high')])
        
        # training/validation mode: target present -> run CFM path
        y = getattr(data['high'], 'y', None)
        if y is not None:
            x_1 = y.unsqueeze(-1)                   # (N, 1) - ground truth
            x_1 = torch.nan_to_num(x_1, nan=0.0)
            x_0 = torch.randn_like(x_1)             # (N, 1) - noise
            batch_vec = getattr(data['high'], 'batch', None)
            if batch_vec is None:
                t = torch.rand(1, device=x_1.device).expand_as(x_1)
            else:
                batch_size = int(batch_vec.max().item()) + 1
                t_per_sample = torch.rand(batch_size, device=x_1.device)
                t = t_per_sample[batch_vec].unsqueeze(-1)
            x_t = (1 - t) * x_0 + t * x_1           # linear interpolation

            v_pred = self.cfm_head(encod_high, x_t, t, data)
            v_target = x_1 - x_0                    # the "true" velocity

            out = torch.cat([v_pred, v_target], dim=1)

        # validation/prediction mode: -> sample
        if not self.training:
            samples = self._sample(encod_high, data)
            if "out" in locals():
                out = [out, samples]
            else:
                out = samples

        return out
