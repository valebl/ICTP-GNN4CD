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


class CFMHead(nn.Module):
    def __init__(self, node_dim=64, cond_dim=64, time_emb_dim=32, hidden=128, output_dim=1):
        """
        Predicts the flow-matching velocity:
            v_theta(x_t, t, node_emb)  ->  (N, output_dim)
    
        Args:
            node_dim    : output dim of the Processor (64 = 32 * 2 heads)
            cond_dim    : internal size of the conditioning projection
            time_emb_dim: sinusoidal embedding size for t (must be even)
            hidden      : hidden size of the velocity MLP
            output_dim  : number of predicted variables (1 for precipitation)
        """
        super().__init__()
        self.cond_mlp = nn.Sequential(
            nn.Linear(node_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)
 
        # input: (x_t (output_dim), t_emb (time_emb_dim), c (cond_dim))
        self.net = nn.Sequential(
            nn.Linear(output_dim + time_emb_dim + cond_dim, hidden),
            nn.SiLU(), # SiLU frequenctly used in flow matching, smooth and continuous
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, output_dim),
        )
 
    def forward(self, node_emb, x_t, t):
        N = node_emb.shape[0]
        c = self.cond_mlp(node_emb)                 # (N, cond_dim)
        t_emb = self.time_emb(t).expand(N, -1)      # (N, time_emb_dim)
        inp = torch.cat([x_t, t_emb, c], dim=-1)
        return self.net(inp)                        # (N, output_dim)


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
        self.register_buffer("freqs", freqs)
 
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: scalar or (1,)  →  (1, dim)
        t = t.view(1, 1).float()
        args = t * self.freqs.unsqueeze(0) * 2 * np.pi    # (1, half)
        return torch.cat([args.sin(), args.cos()], dim=-1) # (1, dim)


@register_model("GNN4CD_CFM_Model")
class GNN4CD_CFM_Model(nn.Module):
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
        n_steps,
        n_samples
        ):

        super().__init__()

        seq_length = history_length + 1
        rnn_input_dim = x_low_var_dim * x_low_lev_dim
        rnn_hidden_dim = x_low_var_dim * x_low_lev_dim

        self.n_steps = n_steps
        self.n_samples = n_samples

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
        )

    def _sample(self, encod_high):
        # inference mode
        N = encod_high.shape[0]
        output_dim = self.cfm_head.net[-1].out_features
    
        # fixed-step Euler integration
        dt = 1.0 / self.n_steps
        t_grid = torch.linspace(0.0, 1.0 - dt, self.n_steps, device=encod_high.device)
        samples = []
        for _ in range(self.n_samples):
            x = torch.randn(N, output_dim, device=encod_high.device)
            for t_val in t_grid:
                v = self.cfm_head(encod_high, x, t_val.unsqueeze(0))
                x = x + v * dt
            samples.append(x)
            
        samples_mean = torch.stack(samples, dim=-1).squeeze(1).mean(dim=-1, keepdim=True)
        
        return samples_mean
    
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
            x_0 = torch.randn_like(x_1)             # (N, 1) - noise
            t = torch.rand(1, device=x_1.device)    # scalar in [0,1]
            x_t = (1 - t) * x_0 + t * x_1           # linear interpolation

            v_pred = self.cfm_head(encod_high, x_t, t)
            v_target = x_1 - x_0                    # the "true" velocity

            out = torch.cat([v_pred, v_target], dim=1)

        # validation/prediction mode: -> sample
        if not self.training:
            samples_mean = self._sample(encod_high)
            if "out" in locals():
                out = torch.cat([out, samples_mean], dim=1)
            else:
                out = samples_mean

        return out