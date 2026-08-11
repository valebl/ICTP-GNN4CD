import torch.nn as nn
import torch_geometric.nn as geometric_nn
from torch_geometric.nn import GATv2Conv, GraphConv
import numpy as np
import torch

from typing import Optional
from torch import Tensor
from torch_scatter import scatter_softmax

from torch_geometric.nn.inits import ones, zeros
from torch_geometric.typing import OptTensor
from torch_geometric.utils import scatter

from .registry import register_model

### Modified the Encoder and Downscaler:
# - the encoder is now two TransformerEncoderLayer modules, on time and then on layers
# - the dowmscaler now adopts bipartite attention with edge features

### Modified the Processor by:
# - using 'add' aggregation instead of 'mean'
# - using LayerNorm instead of BatchNorm
# - using 3 layers instead of 5


class TimeLevelAxialAttention(nn.Module):
    def __init__(self, var_dim, lev_dim, seq_length, d_model=128, n_heads=4, n_layers=2, out_dim=128):
        super().__init__()
        self.var_dim = var_dim
        self.lev_dim = lev_dim
        self.d_model = d_model

        # project variables → model dimension
        self.proj = nn.Linear(var_dim, d_model)

        # embeddings
        self.time_emb = nn.Embedding(seq_length, d_model)
        self.level_emb = nn.Embedding(lev_dim, d_model)

        # axial transformer layers
        self.temporal_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=4*d_model,
                batch_first=True,
                dropout=0.1,
                activation="gelu"
            )
            for _ in range(n_layers)
        ])

        self.vertical_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=4*d_model,
                batch_first=True,
                dropout=0.1,
                activation="gelu"
            )
            for _ in range(n_layers)
        ])

        self.norm_out = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, out_dim)

    def forward(self, x):
        # x: (N, T, L, F)
        N, T, L, F = x.shape

        # project features
        h = self.proj(x)  # (N, T, L, d)

        # positional embeddings
        t_idx = torch.arange(T, device=x.device).view(1, T, 1)
        l_idx = torch.arange(L, device=x.device).view(1, 1, L)
        pos = self.time_emb(t_idx) + self.level_emb(l_idx)  # (1,T,1,d) + (1,1,L,d)

        h = h + pos  # (N,T,L,d)

        # ---- Axial Attention ----
        # 1) Temporal attention: reshape to (N*L, T, d)
        ht = h.permute(0, 2, 1, 3).reshape(N*L, T, self.d_model)
        for layer in self.temporal_layers:
            ht = layer(ht)
        ht = ht.reshape(N, L, T, self.d_model).permute(0, 2, 1, 3)

        # 2) Vertical attention: reshape to (N*T, L, d)
        hv = ht.reshape(N*T, L, self.d_model)
        for layer in self.vertical_layers:
            hv = layer(hv)
        hv = hv.reshape(N, T, L, self.d_model)

        seq = hv.reshape(N, T*L, self.d_model)
        h = seq.mean(dim=1)          # (N, d_model)
        return self.out(h)


class CrossAttEdgeFeatDownscaler(nn.Module):
    """
    Bipartite attention-based downscaler with edge features.
    Transfers low-res atmospheric encoding onto high-res nodes,
    conditioned on static high-res features and geometric edge attributes.
    """
    def __init__(self, low_dim, high_dim, edge_dim, hidden_dim, out_dim):
        super().__init__()
        # project low-res encoding + edge attr into messages
        self.message_mlp = nn.Sequential(
            nn.Linear(low_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # attention scoring: how relevant is each low-res neighbor?
        self.attn_mlp = nn.Sequential(
            nn.Linear(hidden_dim + high_dim, 1)
        )
        # fusion: combine aggregated message with local high-res context
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim + high_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x_low, x_high, edge_index, edge_attr_low2high):
        # edge_index: [2, E], edge_index[0]=low src, edge_index[1]=high dst
        src, dst = edge_index
        # build messages from each low-res neighbor
        m = self.message_mlp(torch.cat([x_low[src], edge_attr_low2high], dim=-1))
        # attention weight conditioned on destination high-res features
        a = self.attn_mlp(torch.cat([m, x_high[dst]], dim=-1))
        a = scatter_softmax(a, dst, dim=0)
        # weighted aggregation
        agg = scatter(a * m, dst, dim=0, dim_size=x_high.size(0), reduce='sum')
        # fuse with high-res static context
        out = self.fusion_mlp(torch.cat([agg, x_high], dim=-1))
        return self.norm(out)


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


@register_model("GNN4CD_AxialAttention_CrossEdgeFeat_Multivar_Model")
class GNN4CD_AxialAttention_CrossEdgeFeat_Multivar_Model(nn.Module):

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument("--rnn_n_layers", type=int, default=2)
        parser.add_argument("--x_low_encoding_dim", type=int, default=128)
        parser.add_argument("--x_low2high_dim", type=float, default=64)
        parser.add_argument("--edge_attr_dim", type=int, default=4)
        parser.add_argument("--d_encoder", type=int, default=128)
        return parser
    
    def __init__(
        self,
        x_low_var_dim,
        x_low_lev_dim,
        x_high_dim,
        output_dim,
        history_length,
        x_low_encoding_dim,
        x_low2high_dim,
        edge_attr_dim,
        d_encoder
        ):

        super().__init__()

        seq_length = history_length + 1
        self.x_low_var_dim = x_low_var_dim
        self.x_low_lev_dim = x_low_lev_dim

        self.encoder = TimeLevelAxialAttention(
            var_dim=x_low_var_dim,
            lev_dim=x_low_lev_dim,
            seq_length=seq_length,
            d_model=d_encoder,
            n_heads=4,
            n_layers=2,
            out_dim=x_low_encoding_dim
        )

        self.downscaler = CrossAttEdgeFeatDownscaler(
            low_dim=x_low_encoding_dim,
            high_dim=x_high_dim,
            edge_dim=edge_attr_dim,
            hidden_dim=x_low2high_dim,
            out_dim=x_low2high_dim
        )
        
        self.processor = Processor(64)

        self.predictor = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, data):
        # reshape low-res input
        x_low = data.x_dict['low']  # (N, T, var_dim * lev_dim)
        N, T, flat_dim = x_low.shape
        x_low = x_low.view(N, T, self.x_low_var_dim, self.x_low_lev_dim).swapaxes(-2,-1) # (N, T, lev_dim, var_dim)
        encod_low = self.encoder(x_low)  # (N, x_low_encoding_dim)
        encod_low2high  = self.downscaler(encod_low, data.x_dict['high'], data['low', 'to', 'high'].edge_index, data['low', 'to', 'high'].edge_attr)
        encod_high = self.processor(encod_low2high , data.edge_index_dict[('high','within','high')])
        out = self.predictor(encod_high)
        return out