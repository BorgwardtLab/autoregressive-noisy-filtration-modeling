"""A graph transformer for binary graph classification"""
import torch
from torch import nn
from torch_geometric.nn import GINConv, GPSConv, global_add_pool


class GraphDiscriminator(nn.Module):
    def __init__(self, num_layers, hidden_dim, feature_dims):
        super().__init__()
        self.featurizers = nn.ModuleDict()
        for feature_name, dim in feature_dims.items():
            mlp = nn.Sequential(
                nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
            )
            norm = nn.BatchNorm1d(dim)
            ln = nn.LayerNorm(hidden_dim)
            self.featurizers[feature_name] = nn.Sequential(norm, mlp, ln)

        layers = []
        for idx in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            conv = GPSConv(hidden_dim, GINConv(mlp), heads=4, attn_type="multihead")
            layers.append(conv)
        self.layers = nn.ModuleList(layers)
        self.classification_layer = nn.Linear(hidden_dim, 1)

    def forward(self, batch):
        edge_index, batch_enumeration = batch.edge_index, batch.batch
        x = 0
        for feature_name, featurizer in self.featurizers.items():
            feature = getattr(batch, feature_name)
            assert feature.ndim == 2
            if feature_name == "eig_vecs":
                random_flips = (
                    2
                    * torch.randint(
                        0,
                        2,
                        (batch.num_graphs, feature.size(-1)),
                        device=feature.device,
                    )
                    - 1
                )
                random_flip_per_node = random_flips[batch_enumeration]
                assert random_flip_per_node.shape == (batch.num_nodes, feature.size(-1))
                feature = feature * random_flip_per_node
            x = x + featurizer(feature)
        for layer in self.layers:
            x = layer(x, edge_index, batch=batch_enumeration)
        return self.classification_layer(global_add_pool(x, batch_enumeration))
