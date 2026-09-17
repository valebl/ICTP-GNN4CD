import torch.nn as nn
from torch_geometric.nn import GATv2Conv
import numpy as np
import torch
from torch_scatter import scatter_softmax
from torch_geometric.utils import scatter

from .registry import register_model

### Modified the Encoder and Downscaler:
# - the encoder is now two TransformerEncoderLayer modules, on time and then on layers
# - the dowmscaler now adopts bipartite attention with edge features

### Modified the Processor by:
# - using 'add' aggregation instead of 'mean'
# - using LayerNorm instead of BatchNorm
# - using 3 layers instead of 5
# - adding noise for CRPS loss in the GATBlock (see below)

### Added noise injection for CRPS training (per-node, following Bris CRPS-FFT (Nordhagen et al. 2025):
# - ConditionalLayerNorm: per-node noise-conditioned LayerNorm (normalize,
#   then apply a noise-conditioned per-node scale/shift), following
#   AIFS-CRPS / Bris CRPS-FFT (Lang et al. 2024b; Nordhagen et al. 2025)
# - z_node has shape (N_high, node_noise_dim): independent noise per node,
#   not one global vector broadcast to every node, so the injected
#   stochasticity is itself spatially varying
# - injected at every GATBlock inside the processor (all 3 message-passing
#   steps), not just at the encoder/predictor bottlenecks, matching the
#   design in Nordhagen et al. 2025
# - z_node can be passed explicitly (z= kwarg) to draw multiple ensemble
#   members from the same input for CRPS estimation, or left None to be
#   sampled internally

### Ensemble generation (M_train members) for CRPS training:
# - encoder and downscaler are deterministic (no noise) and run once per
#   member/input, only the processor depends on z_node
# - downscaler/processor/predictor run in a loop, once per member, reusing
#   the original single-graph edges each time
# - use_checkpointing=True additionally wraps each member's downscaler/
#   processor/predictor call in gradient checkpointing, to have lower peak memory,
#   at the expense of some recompute needed


class TimeLevelAxialAttention(nn.Module):
    def __init__(self, var_dim, lev_dim, seq_length, d_model=128, n_heads=4, n_layers=2, out_dim=128):
        super().__init__()
        self.var_dim = var_dim
        self.lev_dim = lev_dim
        self.d_model = d_model

        # project variables to model dimension
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


class ConditionalLayerNorm(nn.Module):
    """
    Per-node noise-conditioned LayerNorm, following the AIFS-CRPS /
    Bris-CRPS-FFT design (Lang et al. 2024b; Nordhagen et al. 2025):
    normalize, then apply a per-node (scale, shift) produced by an MLP
    from a per-node noise vector z_node.

    h_out = LayerNorm(h) * (1 + gamma(z_node)) + beta(z_node)

    z_node has shape (N, noise_dim) and is an independent noise per node,
    so the injected stochasticity is itself spatially varying.
    Zero-init the last linear so at initialization this is a plain
    (unconditioned) LayerNorm, and the noise pathway is learned in
    gradually.
    """
    def __init__(self, dim, noise_dim, hidden_dim=64):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * dim)
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, h, z_node):
        h_n = self.norm(h)
        gamma_beta = self.mlp(z_node)  # (N, 2*dim)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return h_n * (1 + gamma) + beta


class GATBlock(nn.Module):
    """
    Noise is injected at every message-passing step via each block's
    ConditionalLayerNorm. The same z_node is reused across all 3 blocks within
    one ensemble member.
    """
    def __init__(self, in_dim, out_dim, node_noise_dim, heads=2, dropout=0.2):
        super().__init__()
        self.gat = GATv2Conv(in_dim, out_dim, heads=heads,
                              dropout=dropout, aggr='add',
                              add_self_loops=True, bias=True)
        self.cond_norm = ConditionalLayerNorm(out_dim * heads, node_noise_dim)
        self.act = nn.ReLU()
        # we need same shape to apply skip connection as x + h
        self.rescale = None
        if in_dim != out_dim * heads:
            self.rescale = nn.Linear(in_dim, out_dim * heads)

    def forward(self, x, edge_index, z_node):
        h = self.gat(x, edge_index)
        if self.rescale is not None:
            x = self.rescale(x)
        x = x + h          # residual
        x = self.cond_norm(x, z_node)
        x = self.act(x)
        return x


class Processor(nn.Module):
    def __init__(self, hidden, node_noise_dim):
        super().__init__()
        self.block1 = GATBlock(hidden, 32, node_noise_dim, heads=2, dropout=0.2)
        self.block2 = GATBlock(64, 32, node_noise_dim, heads=2, dropout=0.2)
        self.block3 = GATBlock(64, 32, node_noise_dim, heads=2, dropout=0.2)

    def forward(self, x, edge_index, z_node):
        x = self.block1(x, edge_index, z_node)
        x = self.block2(x, edge_index, z_node)
        x = self.block3(x, edge_index, z_node)
        return x


@register_model("GNN4CD_AxialAttention_CrossEdgeFeat_CRPS_Model")
class GNN4CD_AxialAttention_CrossEdgeFeat_CRPS_Model(nn.Module):

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument("--rnn_n_layers", type=int, default=2)
        parser.add_argument("--x_low_encoding_dim", type=int, default=128)
        parser.add_argument("--x_low2high_dim", type=float, default=64)
        parser.add_argument("--edge_attr_dim", type=int, default=4)
        parser.add_argument("--d_encoder", type=int, default=128)
        parser.add_argument("--node_noise_dim", type=int, default=4)
        parser.add_argument("--M_train", type=int, default=2)
        parser.add_argument("--use_checkpointing", action="store_true", default=False)
        parser.add_argument("--n_members", type=int, default=10)
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
        d_encoder,
        node_noise_dim=4,
        M_train=2,
        use_checkpointing=False,
        n_members=10
        ):

        super().__init__()

        seq_length = history_length + 1
        self.x_low_var_dim = x_low_var_dim
        self.x_low_lev_dim = x_low_lev_dim
        self.node_noise_dim = node_noise_dim
        self.M_train = M_train
        self.use_checkpointing = use_checkpointing
        self.n_members = n_members

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

        self.processor = Processor(64, node_noise_dim)

        self.predictor = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def sample_noise(self, M, N_high, device):
        """Per-node noise: independent draw for each of the N_high nodes,
        for each of the M ensemble members. Shape (M, N_high, node_noise_dim)."""
        return torch.randn(M, N_high, self.node_noise_dim, device=device)

    def _downscale_process_predict(self, encod_low, x_high, edge_index_l2h,
                                    edge_attr_l2h, edge_index_hh, z_node):
        """
        This method contains the memory-heavy part of the forward pass for a single ensemble
        (everything from the downscaler through to the predictor head).
        It should be defined as a separate method so it can be wrapped in
        gradient checkpointing (use_checkpointing=True) to trade compute
        for lower peak memory per member.
        """
        encod_low2high = self.downscaler(
            encod_low, x_high, edge_index_l2h, edge_attr_l2h
        )  # (N_high, x_low2high_dim)

        encod_high = self.processor(encod_low2high, edge_index_hh, z_node)  # (N_high, 64)

        out = self.predictor(encod_high)  # (N_high, output_dim)
        return out

    def _run_ensemble(self, data, z):
        """
        Forward pass for M ensemble members (M = z.shape[0]).
        Loops over members one at a time, reusing the original single-graph
        edges for every member. The encoder and downscaler are deterministic (no
        noise), noise only enters inside the processor via each member's
        z_node, matching Bris CRPS-FFT where noise is injected into the
        processor's latent space rather than at the encoder.
        Returns shape (M, N_high, output_dim).
        z: (M, N_high, node_noise_dim), per-node noise for each member.
        """
        M = z.shape[0]

        x_low = data.x_dict['low']  # (N_low, T, var_dim * lev_dim)
        N_low, T, flat_dim = x_low.shape
        x_low = x_low.view(N_low, T, self.x_low_var_dim, self.x_low_lev_dim).swapaxes(-2, -1)

        x_high = data.x_dict['high']
        N_high = x_high.shape[0]

        edge_index_l2h = data['low', 'to', 'high'].edge_index
        edge_attr_l2h = data['low', 'to', 'high'].edge_attr
        edge_index_hh = data.edge_index_dict[('high', 'within', 'high')]

        # encoder does not depend on noise so it run once, shared across members
        encod_low = self.encoder(x_low)  # (N_low, x_low_encoding_dim)

        outs = []
        for m in range(M):
            z_m = z[m]  # (N_high, node_noise_dim)

            if self.use_checkpointing and self.training:
                out_m = torch.utils.checkpoint.checkpoint(
                    self._downscale_process_predict,
                    encod_low, x_high, edge_index_l2h,
                    edge_attr_l2h, edge_index_hh, z_m,
                    use_reentrant=False,
                )
            else:
                out_m = self._downscale_process_predict(
                    encod_low, x_high, edge_index_l2h,
                    edge_attr_l2h, edge_index_hh, z_m,
                )
            outs.append(out_m)

        return torch.stack(outs, dim=0)  # (M, N_high, output_dim)

    def forward(self, data, z=None):
        """
        1. Training mode: returns an ensemble of shape (M_train, N_high,
        output_dim), use directly with CRPSLoss. Computed as a loop over
        members.
        2. Eval mode: returns a single stochastic sample of shape
        (N_high, output_dim). Use generate_ensemble() for a full ensemble
        at eval time.
        A specific z (shape (N_high, node_noise_dim) or (M, N_high,
        node_noise_dim)) can be passed to control the noise draw(s)
        explicitly, e.g. for reproducibility.
        """
        device = data.x_dict['low'].device
        N_high = data.x_dict['high'].shape[0]

        if z is None:
            M = self.M_train if self.training else 1
            z = self.sample_noise(M, N_high, device)
        elif z.dim() == 2:
            z = z.unsqueeze(0)

        out = self._run_ensemble(data, z)  # (M, N_high, output_dim)

        if not self.training and out.shape[0] == 1:
            return out.squeeze(0)
        return out

    @torch.no_grad()
    def generate_ensemble(self, data, n_members=10):
        """
        Draw n_members stochastic samples for the same input, for CRPS
        estimation.
        """
        was_training = self.training
        self.eval()
        N_high = data.x_dict['high'].shape[0]
        z = self.sample_noise(n_members, N_high, data.x_dict['low'].device)
        out = self._run_ensemble(data, z)  # (n_members, N_high, output_dim)
        if was_training:
            self.train()
        return out