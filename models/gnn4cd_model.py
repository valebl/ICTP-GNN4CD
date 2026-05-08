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

### Modified the Processor by:
# - using 'add' aggregation instead of 'mean'
# - using LayerNorm instead of BatchNorm
# - using 3 layers instead of 5

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
        # self.block4 = GATBlock(64, 32, heads=2, dropout=0.2)
        # self.block5 = GATBlock(64, 32, heads=2, dropout=0.2)

    def forward(self, x, edge_index):
        x = self.block1(x, edge_index)
        x = self.block2(x, edge_index)
        x = self.block3(x, edge_index)
        # x = self.block4(x, edge_index)
        # x = self.block5(x, edge_index)
        return x


@register_model("GNN4CD_Model")
class GNN4CD_Model(nn.Module):

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument("--rnn_n_layers", type=int, default=2)
        parser.add_argument("--x_low_encoding_dim", type=int, default=128)
        parser.add_argument("--x_low2high_dim", type=float, default=64)
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
        x_low2high_dim
        ):

        super().__init__()

        seq_length = history_length + 1
        rnn_input_dim = x_low_var_dim * x_low_lev_dim
        rnn_hidden_dim = x_low_var_dim * x_low_lev_dim

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
    
        self.predictor = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
            )

    def forward(self, data):
        encod_rnn, _ = self.rnn(data.x_dict['low']) # out, h
        encod_rnn = encod_rnn.flatten(start_dim=1)
        encod_rnn = self.dense(encod_rnn)
        # print(f"encod_rnn: {encod_rnn.shape}, data.x_dict['high']: {data.x_dict['high'].shape}, data['low', 'to', 'high'].edge_index: {data['low', 'to', 'high'].edge_index.shape}")
        encod_low2high  = self.downscaler((encod_rnn, data.x_dict['high']), data['low', 'to', 'high'].edge_index)
        encod_high = self.processor(encod_low2high , data.edge_index_dict[('high','within','high')])
        out = self.predictor(encod_high)
        return out