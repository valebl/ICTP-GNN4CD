import torch.nn as nn
import torch_geometric.nn as geometric_nn
from torch_geometric.nn import GATv2Conv, GraphConv
import torch
###########################################################
### This GNN4CD_model architecture is used to downscale the precipitation in the CORDEX-ML experiment  ###
### The CORDEX-ML experiment is described at:  ###
###########################################################

#Strengthed version of P_GNN4CD_model with enhanced terrain encoding, fusion layer, residual connections, and an enhanced predictor.
class P_GNN4CD_model(nn.Module):
    def __init__(self, encoding_dim=64, seq_l=2, h_in=5*3, h_hid=128, n_layers=2, 
                 high_in=3, low2high_out=64, high_out=64):
        super(P_GNN4CD_model, self).__init__()
        
        self.n_layers = n_layers
        self.h_hid = h_hid
        
        # GRU temporal encoding
        self.rnn = nn.GRU(h_in, h_hid, n_layers, batch_first=True, 
                         dropout=0.1 if n_layers > 1 else 0)
        
        self.dense = nn.Sequential(
            nn.Linear(h_hid, encoding_dim),
            nn.BatchNorm1d(encoding_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Downscaler
        self.downscaler = geometric_nn.Sequential('x, edge_index', [
            (GraphConv((encoding_dim, high_in), out_channels=low2high_out, aggr='mean'), 
             'x, edge_index -> x')
        ])
        
        #Strengthed topography encoder
        self.terrain_encoder = nn.Sequential(
            nn.Linear(high_in, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # Combine the downscaled atmospheric features with the terrain features
        self.fusion = nn.Sequential(
            nn.Linear(low2high_out + 64, high_out*2),  # 大气+地形
            nn.BatchNorm1d(high_out*2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        #Add residual connections in the processor
        # First GATv2 layer
        self.gat1 = GATv2Conv(high_out*2, high_out, heads=2, dropout=0.2, 
                             aggr='add', add_self_loops=True, bias=True)
        self.bn1 = geometric_nn.BatchNorm(high_out*2)
        
        # Second GATv2 layer
        self.gat2 = GATv2Conv(high_out*2, high_out, heads=2, dropout=0.2,
                             aggr='add', add_self_loops=True, bias=True)
        self.bn2 = geometric_nn.BatchNorm(high_out*2)
        
        # Third GATv2 layer
        self.gat3 = GATv2Conv(high_out*2, high_out, heads=2, dropout=0.2,
                             aggr='add', add_self_loops=True, bias=True)
        self.bn3 = geometric_nn.BatchNorm(high_out*2)
        
        # Fourth GATv2 layer
        self.gat4 = GATv2Conv(high_out*2, high_out, heads=2, dropout=0.2,
                             aggr='add', add_self_loops=True, bias=True)
        self.bn4 = geometric_nn.BatchNorm(high_out*2)
        
        # last GATv2 layer 
        self.gat5 = GATv2Conv(high_out*2, high_out, heads=1, dropout=0.0,
                             aggr='add', add_self_loops=True, bias=True)
        
        # Strengthed predictor with more layers and non-linearities, and also takes the terrain features as input for better predictions.
        self.predictor = nn.Sequential(
            nn.Linear(high_out + 64, high_out),  # terrain features are also included
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(high_out, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, data):
        output, h_n = self.rnn(data.x_dict['low'])
        encod_rnn = h_n[-1]
        encod_rnn = self.dense(encod_rnn)

        encod_low2high = self.downscaler(
            (encod_rnn, data.x_dict['high']), 
            data["low", "to", "high"].edge_index
        )

        terrain_features = self.terrain_encoder(data.x_dict['high'])

        fused = torch.cat([encod_low2high, terrain_features], dim=-1)
        x = self.fusion(fused)

        edge_index = data.edge_index_dict[('high', 'within', 'high')]

        identity = x
        x = self.gat1(x, edge_index)
        x = self.bn1(x)
        x = torch.relu(x)
        x = x + identity

        identity = x
        x = self.gat2(x, edge_index)
        x = self.bn2(x)
        x = torch.relu(x)
        x = x + identity

        identity = x
        x = self.gat3(x, edge_index)
        x = self.bn3(x)
        x = torch.relu(x)
        x = x + identity

        identity = x
        x = self.gat4(x, edge_index)
        x = self.bn4(x)
        x = torch.relu(x)
        x = x + identity

        x = self.gat5(x, edge_index)
        x = torch.relu(x)

        # splice terrain
        x = torch.cat([x, terrain_features], dim=-1)  # 64+64=128
        x_high = self.predictor(x)     # 128 input

        return x_high
