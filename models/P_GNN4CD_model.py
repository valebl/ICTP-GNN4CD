import torch.nn as nn
import torch_geometric.nn as geometric_nn
from torch_geometric.nn import GATv2Conv, GraphConv

###########################################################
### This GNN4CD_model architecture is used to downscale the precipitation in the CORDEX-ML experiment  ###
### The CORDEX-ML experiment is described at:  ###
###########################################################


class P_GNN4CD_model(nn.Module):
    """
    GNN4CD model to emulate high-resol precipitation maps
    """

    def __init__(self, encoding_dim=64, seq_l=2, h_in=5*3, h_hid=5*3, n_layers=2, high_in=3, low2high_out=64, high_out=64):
        super(P_GNN4CD_model, self).__init__()

        # input shape (N,L,Hin)
        self.rnn = nn.Sequential(
            nn.GRU(h_in, h_hid, n_layers, batch_first=True),
        )

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
        encod_rnn, _ = self.rnn(data.x_dict['low']) # out, h
        encod_rnn = encod_rnn.flatten(start_dim=1)
        encod_rnn = self.dense(encod_rnn)
        encod_low2high  = self.downscaler((encod_rnn, data.x_dict['high']), data["low", "to", "high"].edge_index)
        encod_high = self.processor(encod_low2high , data.edge_index_dict[('high','within','high')])
        x_high = self.predictor(encod_high)

        return x_high



# class P_GNN4CD_model(nn.Module):
#     """
#     GNN4CD model to emulate high-resol temperature maps
#     """
#     def __init__(self, encoding_dim=64, seq_l=2, h_in=5*3, h_hid=128, n_layers=2, 
#                  high_in=3, low2high_out=64, high_out=64):
#         super(P_GNN4CD_model, self).__init__()
        
#         # ===== 修改1: 增加隐藏维度，添加dropout，不用Sequential =====
#         self.n_layers = n_layers
#         self.h_hid = h_hid
        
#         # GRU不用Sequential包装，这样可以获取hidden state
#         self.rnn = nn.GRU(h_in, h_hid, n_layers, batch_first=True, 
#                          dropout=0.1 if n_layers > 1 else 0)
        
#         # ===== 修改2: 调整dense层，输入是h_hid而不是h_in*seq_l =====
#         self.dense = nn.Sequential(
#             nn.Linear(h_hid, encoding_dim),
#             nn.BatchNorm1d(encoding_dim),
#             nn.ReLU(),
#             nn.Dropout(0.1)
#         )
        
#         self.downscaler = geometric_nn.Sequential('x, edge_index', [
#             (GraphConv((encoding_dim, high_in), out_channels=low2high_out, aggr='mean'), 
#              'x, edge_index -> x')
#         ])
        
#         # ===== 修改3: 在processor中添加残差连接的可能性 =====
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
#         # ===== 关键修改：使用GRU的最后隐藏状态 =====
#         # output shape: (batch, seq_len, h_hid)
#         # h_n shape: (n_layers, batch, h_hid)
#         output, h_n = self.rnn(data.x_dict['low'])
        
#         # 使用最后一层的隐藏状态
#         encod_rnn = h_n[-1]  # shape: (batch, h_hid)
        
#         # 或者使用最后一个时间步的输出（两种方法选一个）
#         # encod_rnn = output[:, -1, :]  # shape: (batch, h_hid)
        
#         encod_rnn = self.dense(encod_rnn)
#         encod_low2high = self.downscaler((encod_rnn, data.x_dict['high']), 
#                                         data["low", "to", "high"].edge_index)
#         encod_high = self.processor(encod_low2high, 
#                                     data.edge_index_dict[('high','within','high')])
#         x_high = self.predictor(encod_high)
        
#         return x_high