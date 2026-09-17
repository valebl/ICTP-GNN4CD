import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData, Batch
from torch.utils.data import Sampler
from typing import Sequence, Union
from torch_geometric.utils import degree
import copy
import torch_geometric.transforms as T
transform = T.AddLaplacianEigenvectorPE(k=2)

class Graph_Dataset(Dataset):
    """
    Multivariable target convention: `target` can be either
      - (num_nodes, time)              for a single target variable, or
      - (num_nodes, time, n_vars)      for n_vars target variables,
        stacked along a trailing axis in the same order as
        args.target_variables (e.g. "tas,tasmax,...").
    `_get_target` below is shape-agnostic: it just slices the time axis
    (axis=1), so `snapshot['high'].y` ends up (num_nodes,) or
    (num_nodes, n_vars) accordingly. This mirrors the model's output
    convention of (N, n_target_variables, output_dim), so single- and
    multi-variable runs share the same code path throughout.
    """

    def __init__(
        self,
        graph: Union[HeteroData, None],
        low_input: Sequence[Union[torch.tensor, None]],
        high_input: Sequence[Union[torch.tensor, None]],
        target: Sequence[Union[torch.tensor, None]],
        history_length: int,
        **kwargs: Sequence[torch.tensor]
    ):
        self.graph = graph
        self.low_input = low_input
        self.high_input = high_input
        self.target = target
        self.history_length = history_length
        # 1 for a single-variable target, n_vars for a stacked (N,T,n_vars) target
        self.n_target_variables = target.shape[-1] if (target is not None and target.dim() == 3) else 1
        self.additional_feature_keys = []
        for key, value in kwargs.items():
            setattr(self, key, value)
            self.additional_feature_keys.append(key)
        self._set_graph_static_high_x(high_input)

    def __len__(self):
        if self.target is not None:
            return len(self.target)
        
    def _get_high_nodes_degree(self, snapshot):
        node_degree = (degree(snapshot['high','within','high'].edge_index[0], snapshot['high'].num_nodes) / 8).unsqueeze(-1)
        return node_degree

    def _get_features(self, idx: int):
        x_low = self.low_input[:,idx-self.history_length:idx+1,:]
        # x_low = x_low.flatten(start_dim=2, end_dim=-1)            
        return x_low

    def _get_target(self, idx: int):
        # (num_nodes,) for a single target variable, or
        # (num_nodes, n_vars) when self.target is (num_nodes, time, n_vars)
        return self.target[:,idx]

    def _get_train_mask(self, target: torch.tensor):
        return ~torch.isnan(target)
    
    def _get_additional_feature(self, idx: int, feature_key: str):
        feature = getattr(self, feature_key)[:,idx]
        return feature
    
    def _get_additional_features(self, idx: int):
        additional_features = {
            key: self._get_additional_feature(idx, key)
            for key in self.additional_feature_keys
        }
        return additional_features
    
    def _set_graph_static_high_x(self, x):
        self.graph["high"].x = x

    def set_additional_features(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
            self.additional_feature_keys.append(key)
    
    def __getitem__(self, idx: int):

        snapshot = copy.deepcopy(self.graph)

        # Get the dynamic initial node features (low input and target)
        # The static initial node features (elevation, land use) were 
        # already in the self.graph HeteroData object
        snapshot['low'].x = self._get_features(idx)
        y = self._get_target(idx) if self.target is not None else None
        train_mask = self._get_train_mask(y) if y is not None else None
        snapshot['high'].y = y
        snapshot['high'].train_mask = train_mask
        
        snapshot.num_nodes = self.graph.num_nodes
        snapshot.idxs = idx

        additional_features = self._get_additional_features(idx)
        for key, value in additional_features.items():
            if value.shape[0] == self.graph['high'].x.shape[0]:
                snapshot['high'][key] = value
            elif value.shape[0] == self.graph['low'].x.shape[0]:
                snapshot['high'][key] = value

        return snapshot

def custom_collate_fn_graph(batch_list):
    return Batch.from_data_list(batch_list)