"""
Noisy variant of GNN4CD_model_mod1 for probabilistic downscaling with CRPS.

Gaussian noise is injected at three stages of the forward pass, mirroring
deep4downscaling/deep/models/gnn/noisy.py (Noisy_P_GNN4CD):
  - Stage 1: between the GRU/dense temporal encoding and the downscaler.
  - Stage 2: between the downscaler and the processor.
  - Stage 3: between the processor and the final predictor MLP.

During training (model.train() or torch.is_grad_enabled()), the model performs
`members_for_training` independent noisy forward passes and returns a list of
tensors (one per member). This list is directly compatible with CRPSLoss.

During inference (model.eval() + torch.no_grad()), a single noisy forward pass
is returned as a plain tensor. To build an ensemble, the test loader can be run
ensemble_size times.
"""

from math import ceil

import torch
import torch.nn as nn
import torch_geometric.nn as geometric_nn
from torch_geometric.nn import GATv2Conv, GraphConv


class GATBlock(nn.Module):
    def __init__(self, in_dim, out_dim, heads=2, dropout=0.2):
        super().__init__()
        self.gat = GATv2Conv(in_dim, out_dim, heads=heads,
                             dropout=dropout, aggr='add',
                             add_self_loops=True, bias=True)
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


class GNN4CD_model_mod1_Noisy(nn.Module):
    def __init__(self, encoding_dim=128, seq_length=25, h_in=5*5, h_hid=5*5,
                 n_layers=2, high_in=6+1, low2high_out=64, high_out=64,
                 num_channels_noise=5, members_for_training=5):
        super().__init__()

        if num_channels_noise <= 0:
            raise ValueError("num_channels_noise must be greater than 0.")

        self.num_channels_noise = num_channels_noise
        self.members_for_training = members_for_training

        # Progressive noise reduction across stages (matching Noisy_P_GNN4CD)
        self.num_noise_2 = num_channels_noise - ceil(0.3 * num_channels_noise)
        self.num_noise_3 = num_channels_noise - ceil(0.6 * num_channels_noise)

        self.rnn = nn.Sequential(
            nn.GRU(h_in, h_hid, n_layers, batch_first=True),
        )

        # Dense projection: input is h_in*seq_length + noise_1
        self.dense = nn.Sequential(
            nn.Linear(h_in * seq_length + num_channels_noise, encoding_dim),
            nn.ReLU()
        )

        self.downscaler = geometric_nn.Sequential('x, edge_index', [
            (GraphConv((encoding_dim, high_in), out_channels=low2high_out, aggr='mean'),
             'x, edge_index -> x')
        ])

        # The processor's first block expects low2high_out + num_noise_2 features
        self.processor = Processor(low2high_out + self.num_noise_2)

        # Predictor: input is high_out + noise_3
        self.predictor = nn.Sequential(
            nn.Linear(high_out + self.num_noise_3, high_out),
            nn.ReLU(),
            nn.Linear(high_out, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def _forward_once(self, data) -> torch.Tensor:
        encod_rnn, _ = self.rnn(data.x_dict['low'])
        encod_rnn = encod_rnn.flatten(start_dim=1)

        # Stage 1 noise: after GRU, before dense projection
        noise_1 = torch.randn(encod_rnn.shape[0], self.num_channels_noise,
                              device=encod_rnn.device)
        encod_rnn = torch.cat([encod_rnn, noise_1], dim=-1)
        encod_rnn = self.dense(encod_rnn)

        encod_low2high = self.downscaler(
            (encod_rnn, data.x_dict['high']),
            data['low', 'to', 'high'].edge_index
        )

        # Stage 2 noise: after downscaler, before processor
        noise_2 = torch.randn(encod_low2high.shape[0], self.num_noise_2,
                              device=encod_low2high.device)
        encod_low2high = torch.cat([encod_low2high, noise_2], dim=-1)

        encod_high = self.processor(
            encod_low2high,
            data.edge_index_dict[('high', 'within', 'high')]
        )

        # Stage 3 noise: after processor, before predictor
        noise_3 = torch.randn(encod_high.shape[0], self.num_noise_3,
                              device=encod_high.device)
        encod_high = torch.cat([encod_high, noise_3], dim=-1)

        x_high = self.predictor(encod_high)
        return x_high

    def forward(self, data):
        is_ensemble_mode = self.training or torch.is_grad_enabled()
        members_to_iterate = self.members_for_training if is_ensemble_mode else 1
        out_members = [self._forward_once(data) for _ in range(members_to_iterate)]
        if is_ensemble_mode:
            return out_members
        else:
            return out_members[0]
