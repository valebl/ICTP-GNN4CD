import torch.nn as nn
import torch_geometric.nn as geometric_nn
from torch_geometric.nn import GATv2Conv, GraphConv
import torch
###########################################################
### This GNN4CD_model architecture is used to downscale the precipitation in the CORDEX-ML experiment  ###
### The CORDEX-ML experiment is described at:  ###
###########################################################


# class P_GNN4CD_model(nn.Module):
#     """
#     GNN4CD model to emulate high-resol precipitation/temperature maps
#     优化版本 - 增强时序建模能力
#     """
#     def __init__(self, encoding_dim=64, seq_l=2, h_in=5*3, h_hid=128, n_layers=2, 
#                  high_in=3, low2high_out=64, high_out=64):
#         super(P_GNN4CD_model, self).__init__()
        
#         # ===== 修改1: 增加隐藏维度，不用Sequential =====
#         self.n_layers = n_layers
#         self.h_hid = h_hid
        
#         # GRU: 增加隐藏维度到128，添加dropout
#         self.rnn = nn.GRU(
#             h_in, h_hid, n_layers, 
#             batch_first=True, 
#             dropout=0.1 if n_layers > 1 else 0
#         )
        
#         # ===== 修改2: Dense层输入改为h_hid，添加正则化 =====
#         self.dense = nn.Sequential(
#             nn.Linear(h_hid, encoding_dim),
#             nn.BatchNorm1d(encoding_dim),
#             nn.ReLU(),
#             nn.Dropout(0.1)
#         )
        
#         # Downscaler: 从低分辨率到高分辨率的映射
#         self.downscaler = geometric_nn.Sequential('x, edge_index', [
#             (GraphConv((encoding_dim, high_in), out_channels=low2high_out, aggr='mean'), 
#              'x, edge_index -> x')
#         ])
        
#         # Processor: 高分辨率空间处理，添加BatchNorm
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
        
#         # Predictor: 最终预测层，添加Dropout
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
#         # ===== 修改3: 使用GRU的最后隐藏状态 =====
#         # output shape: (batch, seq_len, h_hid)
#         # h_n shape: (n_layers, batch, h_hid)
#         output, h_n = self.rnn(data.x_dict['low'])
        
#         # 使用最后一层的隐藏状态（推荐）
#         encod_rnn = h_n[-1]  # shape: (batch, h_hid=128)
        
#         # 或者使用最后一个时间步的输出（备选）
#         # encod_rnn = output[:, -1, :]  # shape: (batch, h_hid=128)
        
#         # Dense层处理
#         encod_rnn = self.dense(encod_rnn)
        
#         # Downscaler: 映射到高分辨率
#         encod_low2high = self.downscaler(
#             (encod_rnn, data.x_dict['high']), 
#             data["low", "to", "high"].edge_index
#         )
        
#         # Processor: 空间细化
#         encod_high = self.processor(
#             encod_low2high, 
#             data.edge_index_dict[('high','within','high')]
#         )
        
#         # Predictor: 最终预测
#         x_high = self.predictor(encod_high)
        
#         return x_high
    
#增强地形特征编码、融合层、残差连接和增强预测器的改进版本
class P_GNN4CD_model(nn.Module):
    def __init__(self, encoding_dim=64, seq_l=2, h_in=5*3, h_hid=128, n_layers=2, 
                 high_in=3, low2high_out=64, high_out=64):
        super(P_GNN4CD_model, self).__init__()
        
        self.n_layers = n_layers
        self.h_hid = h_hid
        
        # GRU时序编码
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
        
        # ===== 改进1: 增强地形特征编码 =====
        # 专门处理高分辨率地形信息
        self.terrain_encoder = nn.Sequential(
            nn.Linear(high_in, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # ===== 改进2: 融合层 - 结合大气信息和地形信息 =====
        self.fusion = nn.Sequential(
            nn.Linear(low2high_out + 64, high_out*2),  # 大气+地形
            nn.BatchNorm1d(high_out*2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # ===== 改进3: Processor中添加残差连接 =====
        # 第一个GATv2层
        self.gat1 = GATv2Conv(high_out*2, high_out, heads=2, dropout=0.2, 
                             aggr='add', add_self_loops=True, bias=True)
        self.bn1 = geometric_nn.BatchNorm(high_out*2)
        
        # 第二个GATv2层
        self.gat2 = GATv2Conv(high_out*2, high_out, heads=2, dropout=0.2,
                             aggr='add', add_self_loops=True, bias=True)
        self.bn2 = geometric_nn.BatchNorm(high_out*2)
        
        # 第三个GATv2层
        self.gat3 = GATv2Conv(high_out*2, high_out, heads=2, dropout=0.2,
                             aggr='add', add_self_loops=True, bias=True)
        self.bn3 = geometric_nn.BatchNorm(high_out*2)
        
        # 第四个GATv2层
        self.gat4 = GATv2Conv(high_out*2, high_out, heads=2, dropout=0.2,
                             aggr='add', add_self_loops=True, bias=True)
        self.bn4 = geometric_nn.BatchNorm(high_out*2)
        
        # 最后一层
        self.gat5 = GATv2Conv(high_out*2, high_out, heads=1, dropout=0.0,
                             aggr='add', add_self_loops=True, bias=True)
        
        # ===== 改进4: 增强的预测器，保留更多细节 =====
        self.predictor = nn.Sequential(
            nn.Linear(high_out + 64, high_out),  # 拼接地形特征
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

        x = self.gat5(x, edge_index)   # 输出64
        x = torch.relu(x)

        # 再次拼接地形特征
        x = torch.cat([x, terrain_features], dim=-1)  # 64+64=128
        x_high = self.predictor(x)     # 输入128 ✅

        return x_high

#仅加地形权重和残差链接


# class P_GNN4CD_model(nn.Module):
#     def __init__(self, encoding_dim=64, seq_l=2, h_in=5*3, h_hid=128, n_layers=2, 
#                  high_in=3, low2high_out=64, high_out=64):
#         super(P_GNN4CD_model, self).__init__()
        
#         self.n_layers = n_layers
#         self.h_hid = h_hid
        
#         # ===== 时序编码（不变）=====
#         self.rnn = nn.GRU(
#             h_in, h_hid, n_layers, 
#             batch_first=True, 
#             dropout=0.1 if n_layers > 1 else 0
#         )
        
#         self.dense = nn.Sequential(
#             nn.Linear(h_hid, encoding_dim),
#             nn.BatchNorm1d(encoding_dim),
#             nn.ReLU(),
#             nn.Dropout(0.1)
#         )
        
#         # ===== Downscaler（不变）=====
#         self.downscaler = geometric_nn.Sequential('x, edge_index', [
#             (GraphConv((encoding_dim, high_in), out_channels=low2high_out, aggr='mean'), 
#              'x, edge_index -> x')
#         ])
        
#         # ===== 改进：轻量级地形编码器 =====
#         # 输出维度和low2high_out一样，用加法融合，不破坏大气信号
#         self.terrain_encoder = nn.Sequential(
#             nn.Linear(high_in, low2high_out),
#             nn.BatchNorm1d(low2high_out),
#             nn.ReLU(),
#             nn.Linear(low2high_out, low2high_out),
#             nn.Tanh()  # 用Tanh限制范围，防止地形信号过大
#         )
        
#         # 地形融合权重（可学习的标量，初始化为小值）
#         # 让模型自己学习地形的重要程度
#         self.terrain_weight = nn.Parameter(torch.tensor(0.1))
        
#         # ===== GNN Processor（保持原有结构，添加残差）=====
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
        
#         # ===== Predictor（不变）=====
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
#         # 1. 时序编码
#         output, h_n = self.rnn(data.x_dict['low'])
#         encod_rnn = h_n[-1]
#         encod_rnn = self.dense(encod_rnn)
        
#         # 2. Downscaler：低分辨率 → 高分辨率
#         encod_low2high = self.downscaler(
#             (encod_rnn, data.x_dict['high']), 
#             data["low", "to", "high"].edge_index
#         )
        
#         # 3. 地形融合（加法残差，保护大气信号）
#         terrain_features = self.terrain_encoder(data.x_dict['high'])
        
#         # 用可学习权重控制地形的贡献
#         # 初始值0.1确保训练初期大气信号主导
#         encod_with_terrain = encod_low2high + self.terrain_weight * terrain_features
        
#         # 4. GNN Processor
#         encod_high = self.processor(
#             encod_with_terrain, 
#             data.edge_index_dict[('high','within','high')]
#         )
        
#         # 5. 预测
#         x_high = self.predictor(encod_high)
        
#         return x_high