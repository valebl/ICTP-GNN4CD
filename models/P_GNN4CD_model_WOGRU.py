import torch.nn as nn
import torch_geometric.nn as geometric_nn
from torch_geometric.nn import GATv2Conv, GraphConv
import torch
import torch.nn as nn
###########################################################
### This GNN4CD_model architecture is used to downscale the precipitation in the CORDEX-ML experiment  ###
### The CORDEX-ML experiment is described at:  ###
###########################################################


class P_GNN4CD_model(nn.Module):
    """
    GNN4CD model to emulate high-resol precipitation maps

    Here we assume that the high-resol predictors are lat, lon and orography
    If you want to use only orography please move to high_in=1
    """

    def __init__(self, encoding_dim=64, seq_l=2, h_in=5*3, h_hid=128, n_layers=2, high_in=3, low2high_out=64, high_out=64):
        super(P_GNN4CD_model, self).__init__()

        # input shape (N,L,Hin)
        #self.rnn = nn.Sequential(
        #    nn.GRU(h_in, h_hid, n_layers, batch_first=True),
        #)

        self.dense = nn.Sequential(
            nn.Linear(h_in*seq_l, encoding_dim),
            nn.ReLU()
        )

        self.downscaler = geometric_nn.Sequential('x, edge_index', [
            (GraphConv((encoding_dim, high_in), out_channels=low2high_out, aggr='mean'), 'x, edge_index -> x')
            ])
        
        self.processor = geometric_nn.Sequential('x, edge_index', [
            (geometric_nn.BatchNorm(low2high_out), 'x -> x'),
            (GATv2Conv(in_channels=low2high_out, out_channels=high_out, heads=2, dropout=0.2, aggr='mean', add_self_loops=True, bias=True), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_out*2), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=high_out*2, out_channels=high_out, heads=2, dropout=0.2, aggr='mean', add_self_loops=True, bias=True),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_out*2), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=high_out*2, out_channels=high_out, heads=2, dropout=0.2, aggr='mean', add_self_loops=True, bias=True),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_out*2), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=high_out*2, out_channels=high_out, heads=2, dropout=0.2, aggr='mean', add_self_loops=True, bias=True),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(high_out*2), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=high_out*2, out_channels=high_out, heads=1, dropout=0.0, aggr='mean', add_self_loops=True, bias=True), 'x, edge_index -> x'),
            nn.ReLU(),
            ])

        self.predictor = nn.Sequential(
            nn.Linear(high_out, high_out),
            nn.ReLU(),
            nn.Linear(high_out, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
            )


    def forward(self, data):
        #encod_rnn, _ = self.rnn(data.x_dict['low']) # out, h
        encod_x = data.x_dict['low'].flatten(start_dim=1)
        encod_2 = self.dense(encod_x)
        encod_low2high  = self.downscaler((encod_2, data.x_dict['high']), data["low", "to", "high"].edge_index)
        encod_high = self.processor(encod_low2high , data.edge_index_dict[('high','within','high')])
        x_high = self.predictor(encod_high)

        return x_high


#Add terrain_weight


# class P_GNN4CD_model(nn.Module):
#     """
#     去掉GRU + 加入地形残差融合
#     """
#     def __init__(self, encoding_dim=64, seq_l=2, h_in=5*3, h_hid=128, n_layers=2, 
#                  high_in=3, low2high_out=64, high_out=64):
#         super(P_GNN4CD_model, self).__init__()
        
#         self.h_hid = h_hid
#         self.n_layers = n_layers
        
#         # ===== 去掉GRU，直接用MLP处理输入特征 =====
#         # 输入维度：h_in * seq_l（所有时间步的特征展平）
#         self.dense = nn.Sequential(
#             nn.Linear(h_in * seq_l, encoding_dim),
#             nn.BatchNorm1d(encoding_dim),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(encoding_dim, encoding_dim),
#             nn.BatchNorm1d(encoding_dim),
#             nn.ReLU(),
#         )
        
#         # ===== Downscaler =====
#         self.downscaler = geometric_nn.Sequential('x, edge_index', [
#             (GraphConv((encoding_dim, high_in), out_channels=low2high_out, aggr='mean'), 
#              'x, edge_index -> x')
#         ])
        
#         # ===== 地形编码器（轻量级）=====
#         self.terrain_encoder = nn.Sequential(
#             nn.Linear(high_in, low2high_out),
#             nn.BatchNorm1d(low2high_out),
#             nn.ReLU(),
#             nn.Linear(low2high_out, low2high_out),
#             nn.Tanh()
#         )
        
#         # 可学习的地形权重（初始化为小值，保护大气信号）
#         self.terrain_weight = nn.Parameter(torch.tensor(0.1))
        
#         # ===== GNN Processor =====
#         self.processor = geometric_nn.Sequential('x, edge_index', [
#             (geometric_nn.BatchNorm(low2high_out), 'x -> x'),
#             (GATv2Conv(in_channels=low2high_out, out_channels=high_out, heads=2, 
#                       dropout=0.2, aggr='mean', add_self_loops=True, bias=True), 
#              'x, edge_index -> x'),
#             (geometric_nn.BatchNorm(high_out*2), 'x -> x'),
#             nn.ReLU(),
#             (GATv2Conv(in_channels=high_out*2, out_channels=high_out, heads=2, 
#                       dropout=0.2, aggr='mean', add_self_loops=True, bias=True),
#              'x, edge_index -> x'),
#             (geometric_nn.BatchNorm(high_out*2), 'x -> x'),
#             nn.ReLU(),
#             (GATv2Conv(in_channels=high_out*2, out_channels=high_out, heads=2, 
#                       dropout=0.2, aggr='mean', add_self_loops=True, bias=True),
#              'x, edge_index -> x'),
#             (geometric_nn.BatchNorm(high_out*2), 'x -> x'),
#             nn.ReLU(),
#             (GATv2Conv(in_channels=high_out*2, out_channels=high_out, heads=2, 
#                       dropout=0.2, aggr='mean', add_self_loops=True, bias=True),
#              'x, edge_index -> x'),
#             (geometric_nn.BatchNorm(high_out*2), 'x -> x'),
#             nn.ReLU(),
#             (GATv2Conv(in_channels=high_out*2, out_channels=high_out, heads=1, 
#                       dropout=0.0, aggr='mean', add_self_loops=True, bias=True), 
#              'x, edge_index -> x'),
#             nn.ReLU(),
#         ])
        
#         # ===== Predictor =====
#         self.predictor = nn.Sequential(
#             nn.Linear(high_out, high_out),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(high_out, 32),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(32, 1)
#         )

#     def forward(self, data):
#         # 1. 直接展平低分辨率特征（去掉GRU）
#         x_low = data.x_dict['low']           # (N, seq_l, h_in)
#         x_low_flat = x_low.flatten(start_dim=1)  # (N, seq_l * h_in)
#         encod_rnn = self.dense(x_low_flat)        # (N, encoding_dim)
        
#         # 2. Downscaler
#         encod_low2high = self.downscaler(
#             (encod_rnn, data.x_dict['high']), 
#             data["low", "to", "high"].edge_index
#         )                                         # (N_high, low2high_out)
        
#         # 3. 地形残差融合
#         terrain_features = self.terrain_encoder(data.x_dict['high'])
#         encod_with_terrain = encod_low2high + self.terrain_weight * terrain_features
        
#         # 4. GNN Processor
#         encod_high = self.processor(
#             encod_with_terrain, 
#             data.edge_index_dict[('high', 'within', 'high')]
#         )
        
#         # 5. 预测
#         x_high = self.predictor(encod_high)
        
#         return x_high