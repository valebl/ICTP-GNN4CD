import pickle
import sys
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_convert

from torch_geometric.data import Data, HeteroData

import torch
import numpy as np
from typing import Sequence, Union
from torch_geometric.data import Data, Batch
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.utils import degree

import torch_geometric.transforms as T
transform = T.AddLaplacianEigenvectorPE(k=2)

Graph = Union[HeteroData, None]
Targets = Sequence[Union[np.ndarray, None]]
DoubleCl = Union[bool, None]
Additional_Features = Sequence[torch.tensor]
Model_Name = Union[str, None]


class Dataset_Graph(Dataset):

    def __init__(
        self,
        graph: Graph,
        targets: Targets,
        model_name: Model_Name,
        **kwargs: Additional_Features
    ):
        self.graph = graph
        self.targets = targets
        self.model_name = model_name
        self.additional_feature_keys = []
        for key, value in kwargs.items():
            setattr(self, key, value)
            self.additional_feature_keys.append(key)
        self._check_temporal_consistency()
        #self._add_node_degree()
        if "1h" in self.model_name:
            self._get_features = self.__get_features_1h
        elif "3h" in self.model_name:
            self._get_features = self.__get_features_3h
        elif "6h" in self.model_name:
            self._get_features = self.__get_features_6h
        else:
            self._get_features = self.__get_features_24h

    def __len__(self):
        #return len(self.features)
        return self.graph['low'].x.shape[1]
        
    def _check_temporal_consistency(self):
        if self.targets is not None:
            assert (self.graph['low'].x.shape[1]) == self.targets.shape[1], "Temporal dimension inconsistency."

    def _set_snapshot_count(self):
        self.snapshot_count = len(self)
    
    def _get_high_nodes_degree(self, snapshot):
        node_degree = (degree(snapshot['high','within','high'].edge_index[0], snapshot['high'].num_nodes) / 8).unsqueeze(-1)
        return node_degree

    def __get_features_3h(self, time_index: int):
        x_low = self.graph['low'].x[:,time_index-24:time_index+1:3,:]   # model HiResPrecipNet
        x_low = x_low.flatten(start_dim=2, end_dim=-1)                  # model HiResPrecipNet
        return x_low

    def __get_features_6h(self, time_index: int):
        x_low = self.graph['low'].x[:,time_index-24:time_index+1:6,:]   # model HiResPrecipNet
        x_low = x_low.flatten(start_dim=2, end_dim=-1)                  # model HiResPrecipNet
        return x_low

    def __get_features_24h(self, time_index: int):
        x_low = self.graph['low'].x[:,time_index-24:time_index+1,:]     # model TEST
        x_low = x_low.flatten(start_dim=2, end_dim=-1)                  # model TEST
        return x_low

    def __get_features_1h(self, time_index: int):
        x_low = self.graph['low'].x[:,time_index,:]                     # model HiResPrecipNet
        x_low = x_low.flatten(start_dim=1, end_dim=-1)                  # model HiResPrecipNet
        return x_low
    
    def _get_target(self, time_index: int):
        return self.targets[:,time_index]

    def _get_train_mask(self, target: torch.tensor):
        return ~torch.isnan(target)
    
    def _get_additional_feature(self, time_index: int, feature_key: str):
        feature = getattr(self, feature_key)[:,time_index]
        return feature
    
    def _get_additional_features(self, time_index: int):
        additional_features = {
            key: self._get_additional_feature(time_index, key)
            for key in self.additional_feature_keys
        }
        return additional_features
    
    def __getitem__(self, time_index: int):
        x_low = self._get_features(time_index)
        y = self._get_target(time_index) if self.targets is not None else None
        train_mask = self._get_train_mask(y) if y is not None else None

        additional_features = self._get_additional_features(time_index)

        snapshot = HeteroData()

        for key, value in additional_features.items():
            if value.shape[0] == self.graph['high'].x.shape[0]:
                snapshot['high'][key] = value
            elif value.shape[0] == self.graph['low'].x.shape[0]:
                snapshot['high'][key] = value
       
        snapshot['high'].y = y
        snapshot['high'].train_mask = train_mask
        snapshot.num_nodes = self.graph.num_nodes
        snapshot['high'].num_nodes = self.graph['high'].num_nodes
        snapshot['low'].num_nodes = self.graph['low'].num_nodes
        snapshot.t = time_index
        
        #snapshot['low', 'within', 'low'].edge_index = self.graph['low', 'within', 'low'].edge_index
        snapshot['high', 'within', 'high'].edge_index = self.graph['high', 'within', 'high'].edge_index
        snapshot['low', 'to', 'high'].edge_index = self.graph['low', 'to', 'high'].edge_index
        # snapshot['low', 'to', 'high'].edge_weight = self.graph['low', 'to', 'high'].edge_weight

        snapshot['low'].x = x_low 
        #snapshot['high'].x_empty = self.graph['high'].x
        snapshot['high'].x = self.graph['high'].z_std # torch.zeros((snapshot['high'].num_nodes,1))
        snapshot['high'].z_std = self.graph['high'].z_std
        snapshot['high'].land_std = self.graph['high'].land_std

        snapshot['high'].lon = self.graph['high'].lon
        snapshot['high'].lat = self.graph['high'].lat
        snapshot['low'].lon = self.graph['low'].lon
        snapshot['low'].lat = self.graph['low'].lat

        #snapshot['high'].laplacian_eigenvector_pe = self.graph['high'].laplacian_eigenvector_pe     
        #snapshot['high'].deg = self._get_high_nodes_degree(snapshot)

        return snapshot

        node_idx = torch.randint(high=snapshot['high'].num_nodes, size=(1,)).item()
        num_hops = torch.randint(low=2, high=5, size=(1,)).item()

        subset_low, _, _, _ = k_hop_subgraph(node_idx=node_idx, num_hops=1,
                edge_index=snapshot['low', 'to', 'high'].edge_index,
                relabel_nodes=False)
        
        subset_high, _, _, _ = k_hop_subgraph(node_idx=node_idx, num_hops=2,
                edge_index=snapshot['high', 'within', 'high'].edge_index,
                relabel_nodes=False)

        print(subset_low)
        print(subset_high)
        
        
        subset_dict = {
            'low': subset_low,
            'high': subset_high
        }

        s = snapshot.subgraph(subset_dict=subset_dict)
        s['node_idx'] = node_idx
        
        return s
        

class Iterable_Graph(object):

    def __init__(self, dataset_graph, shuffle, idxs_vector=None):
        self.dataset_graph = dataset_graph
        self.shuffle = shuffle
        self.idxs_vector = idxs_vector

    def __len__(self):
        return len(self.idxs_vector)

    def __next__(self):
        if self.prog_idx < self.idxs_vector.shape[0]:
            self.idx = self.sampling_vector[self.prog_idx].item()
            self.prog_idx = self.prog_idx + 1
            return self.idx
        else:
            self.prog_idx = 0
            self.idx = 0
            raise StopIteration

    def __iter__(self):
        self.prog_idx = 0
        self.idx = 0
        if self.idxs_vector is not None:
            if self.shuffle:
                rnd_idxs = torch.randperm(self.idxs_vector.shape[0])
                self.sampling_vector = self.idxs_vector[rnd_idxs].view(self.idxs_vector.size())
            else:
                self.sampling_vector = self.idxs_vector
        else:
            if self.shuffle:
                self.sampling_vector = torch.randperm(len(self)-24) + 24 # from 24 to len
            else:
                self.sampling_vector = torch.arange(24, len(self))
        return self


def custom_collate_fn_graph(batch_list):
    return Batch.from_data_list(batch_list)


