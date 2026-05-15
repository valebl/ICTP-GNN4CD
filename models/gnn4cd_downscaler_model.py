import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as geometric_nn
from torch_geometric.nn import GATv2Conv, GraphConv

from .registry import register_model


class GATBlock(nn.Module):
    def __init__(self, in_dim, out_dim, heads=2, dropout=0.2):
        super().__init__()
        self.gat = GATv2Conv(
            in_dim,
            out_dim,
            heads=heads,
            dropout=dropout,
            aggr='add',
            add_self_loops=True,
            bias=True,
        )
        self.norm = nn.LayerNorm(out_dim * heads)
        self.act = nn.ReLU()
        self.rescale = None
        if in_dim != out_dim * heads:
            self.rescale = nn.Linear(in_dim, out_dim * heads)

    def forward(self, x, edge_index):
        h = self.gat(x, edge_index)
        if self.rescale is not None:
            x = self.rescale(x)
        x = x + h
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


class LowResSpatialEncoder(nn.Module):
    """
    Spatial GNN on the low-resolution graph (lon, lat).
    Operates on each time step separately.
    """

    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.gnn1 = GATv2Conv(in_dim, hidden_dim, heads=2, aggr='add')
        self.gnn2 = GATv2Conv(2 * hidden_dim, hidden_dim, heads=2, aggr='add')
        self.norm = nn.LayerNorm(2 * hidden_dim)

    def forward(self, x_low, edge_index_low):
        # x_low: (N_low, in_dim)
        h = self.gnn1(x_low, edge_index_low)   # (N_low, 2*hidden_dim)
        h = F.relu(h)
        h = self.gnn2(h, edge_index_low)       # (N_low, 2*hidden_dim)
        h = self.norm(h)
        return h  # (N_low, 2*hidden_dim)


class TemporalTransformer(nn.Module):
    """
    Temporal encoder over the sequence of low-res spatial embeddings.
    """

    def __init__(self, hidden_dim, n_heads=4, n_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, H_seq):
        # H_seq: (N_low, T, hidden_dim)
        return self.transformer(H_seq)  # (N_low, T, hidden_dim)


class CrossResDownscaler(nn.Module):
    """
    Cross-resolution GNN: low-res embeddings -> high-res nodes.
    """

    def __init__(self, low_dim, high_in_dim, out_dim):
        super().__init__()
        self.conv = GraphConv((low_dim, high_in_dim), out_dim, aggr='mean')

    def forward(self, h_low, x_high, edge_index_low2high):
        # h_low:  (N_low, low_dim)
        # x_high: (N_high, high_in_dim)
        return self.conv((h_low, x_high), edge_index_low2high)  # (N_high, out_dim)


class SpatioTemporalDownscaler(nn.Module):
    """
    Full spatio-temporal low-res encoder + cross-resolution downscaler.
    Replaces the old RNN + flatten + dense + GraphConv.
    """

    def __init__(
        self,
        low_in_dim,
        low_hidden_dim,
        high_in_dim,
        out_dim,
        seq_length,
        n_heads=4,
        n_layers=2,
    ):
        super().__init__()

        # spatial encoder outputs low_hidden_dim
        self.spatial = LowResSpatialEncoder(low_in_dim, low_hidden_dim // 2)
        # temporal transformer works on low_hidden_dim
        self.temporal = TemporalTransformer(low_hidden_dim, n_heads=n_heads, n_layers=n_layers)
        # cross-resolution GNN
        self.cross = CrossResDownscaler(low_hidden_dim, high_in_dim, out_dim)

        self.T = seq_length
        self.low_hidden_dim = low_hidden_dim

    def forward(self, x_low_seq, x_high, edge_low, edge_low2high):
        """
        x_low_seq: (N_low, T, low_in_dim)
        x_high:    (N_high, high_in_dim)
        edge_low:  edge_index for low-res graph ('low','within','low')
        edge_low2high: edge_index for ('low','to','high')
        """

        N_low, T, _ = x_low_seq.shape
        assert T == self.T, f"Expected T={self.T}, got {T}"

        # 1. spatial GNN at each time step
        H = []
        for t in range(T):
            h_t = self.spatial(x_low_seq[:, t, :], edge_low)  # (N_low, low_hidden_dim)
            H.append(h_t)
        H = torch.stack(H, dim=1)  # (N_low, T, low_hidden_dim)

        # 2. temporal transformer
        H = self.temporal(H)       # (N_low, T, low_hidden_dim)

        # 3. use last time step embedding
        h_low_final = H[:, -1, :]  # (N_low, low_hidden_dim)

        # 4. cross-resolution GNN
        h_high = self.cross(h_low_final, x_high, edge_low2high)  # (N_high, out_dim)

        return h_high


@register_model("GNN4CDDownscaler_Model")
class GNN4CDDownscaler_Model(nn.Module):

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument("--rnn_n_layers", type=int, default=2)  # kept for compatibility, unused now
        parser.add_argument("--x_low_encoding_dim", type=int, default=128)
        parser.add_argument("--x_low2high_dim", type=float, default=64)
        parser.add_argument("--st_n_heads", type=int, default=4)
        parser.add_argument("--st_n_layers", type=int, default=2)
        return parser
    
    def __init__(
        self,
        x_low_var_dim,
        x_low_lev_dim,
        x_high_dim,
        output_dim,
        history_length,
        rnn_n_layers,          # kept for signature compatibility
        x_low_encoding_dim,    # this will be the low_hidden_dim
        x_low2high_dim,
        st_n_heads=4,
        st_n_layers=2,
        ):

        super().__init__()

        seq_length = history_length + 1
        low_in_dim = x_low_var_dim * x_low_lev_dim   # per-time-step feature dim

        # new spatio-temporal downscaler
        self.downscaler = SpatioTemporalDownscaler(
            low_in_dim=low_in_dim,
            low_hidden_dim=x_low_encoding_dim,
            high_in_dim=x_high_dim,
            out_dim=x_low2high_dim,
            seq_length=seq_length,
            n_heads=st_n_heads,
            n_layers=st_n_layers,
        )

        self.processor = Processor(64)
    
        self.predictor = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, data):
        # x_low_seq: (N_low, T, low_in_dim)
        x_low_seq = data.x_dict['low']  # we already have this shape for the RNN
        x_high = data.x_dict['high']

        edge_low = data.edge_index_dict[('low', 'within', 'low')]
        edge_low2high = data['low', 'to', 'high'].edge_index
        edge_high = data.edge_index_dict[('high', 'within', 'high')]

        # spatio-temporal low-res encoder + cross-res downscaler
        encod_low2high = self.downscaler(
            x_low_seq=x_low_seq,
            x_high=x_high,
            edge_low=edge_low,
            edge_low2high=edge_low2high,
        )

        # high-res processor + predictor (unchanged)
        encod_high = self.processor(encod_low2high, edge_high)
        out = self.predictor(encod_high)
        return out
