import time
from copy import deepcopy
from typing import List, Optional

import torch
import torch_geometric
from einops import rearrange
from hydra.utils import instantiate
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.utils import cumsum, to_dense_adj, to_dense_batch

from anfm.data.base.features import KNodeCycles, global_laplacian
from anfm.models.edge_predictor import EdgePredictor, sample_edges
from anfm.models.robust_rwpe import RobustAddRandomWalkPE
from anfm.utils import dense_to_sparse_adj, dense_to_sparse_batch


def to_sparse_batch(x, mask):
    """Inverse of utils.to_dense_batch"""
    x = rearrange(x, "b n d -> (b n) d")
    flat_mask = mask.flatten()
    return x[flat_mask]


class NodeIndividualizer(nn.Module):
    KNOWN_INDIVIDUALIZATIONS = (
        "ordering",
        "random",
    )

    def __init__(self, individualizations, max_nodes, embed_dim):
        super().__init__()
        self.individualizations = set(individualizations)
        self.embed_dim = embed_dim

        if not all(
            indiv in self.KNOWN_INDIVIDUALIZATIONS for indiv in individualizations
        ):
            raise ValueError("Unknown individualization")

        if "ordering" in self.individualizations:
            self.pos_embed = nn.Embedding(max_nodes, embed_dim)

        if "random" in self.individualizations:
            # the random noise changes over the filtration (i.e. time)
            self.rand_mean = nn.Parameter(torch.zeros(embed_dim))
            self.rand_log_std = nn.Parameter(torch.zeros(embed_dim))

        if "random-shared" in self.individualizations:
            # the random noise is constant over time
            self.rand_shared_mean = nn.Parameter(torch.zeros(embed_dim))
            self.rand_shared_log_std = nn.Parameter(torch.zeros(embed_dim))

        if "random-shared-fixed" in self.individualizations:
            # as random-shared, but don't learn the standard deviations
            self.rand_shared_mean_fixed = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, graphs):
        x = torch.zeros(
            (graphs.num_nodes, self.embed_dim), device=graphs.edge_index.device
        )
        individualizations_applied = 0
        if "ordering" in self.individualizations:
            x = x + self.pos_embed(graphs.ordering)
            individualizations_applied += 1

        if "random" in self.individualizations:
            x = x + self.rand_mean + self.rand_log_std.exp() * torch.randn_like(x)
            individualizations_applied += 1

        assert individualizations_applied == len(self.individualizations)

        return x


class DataNormalization(nn.Module):
    def __init__(self, dim, eps=1e-5, momentum=0.025):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        running_mean = torch.zeros((dim,))
        running_var = torch.ones((dim,))
        self._freeze_ema = False
        self.register_buffer("running_mean", running_mean)
        self.register_buffer("running_var", running_var)

    def freeze_ema(self):
        self._freeze_ema = True

    def forward(self, x):
        assert x.shape[-1] == self.dim
        assert not torch.isnan(x).any()
        update_stats = (
            not self._freeze_ema
            and self.training
            and not torch.is_inference_mode_enabled()
        )
        if update_stats:
            var, mean = torch.var_mean(
                x, dim=[i for i in range(x.ndim - 1)], keepdim=True
            )
            self.running_mean = (
                self.running_mean * (1 - self.momentum)
                + self.momentum * mean.squeeze().detach()
            )
            self.running_var = (
                self.running_var * (1 - self.momentum)
                + self.momentum * var.squeeze().detach()
            )
        x = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        assert not torch.isnan(x).any()
        return x


class FeatureTransform(nn.Module):
    """Transforms a list of features into a single feature vector."""

    def __init__(self, feature_dims, emb_dim):
        super().__init__()
        self.feature_dims = feature_dims
        self.emb_dim = emb_dim
        self.separate_transforms = nn.ModuleList(
            [
                nn.Sequential(
                    DataNormalization(dim),
                    nn.Linear(dim, emb_dim),
                    nn.ReLU(),
                    nn.Linear(emb_dim, emb_dim),
                    nn.LayerNorm(emb_dim),
                )
                for dim in feature_dims
            ]
        )
        self.shared_transform = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.ReLU())

    def forward(self, features):
        assert len(features) == len(self.feature_dims) and all(
            feature.shape[-1] == dim
            for feature, dim in zip(features, self.feature_dims)
        )
        transformed = [
            transform(feature)
            for transform, feature in zip(self.separate_transforms, features)
        ]
        aggregated = torch.mean(torch.stack(transformed, dim=0), dim=0)
        return self.shared_transform(aggregated)


def initial_graph(num_nodes, num_laplacian_eigenvecs, distribution="empty"):
    if distribution == "empty":
        disconnected_graph = Data(
            edge_index=torch.zeros(
                (2, 0),
                dtype=torch.long,
            ),
            x=torch.zeros((num_nodes, 0)),
            ordering=torch.arange(num_nodes).long(),
            edge_attr=torch.ones((0,)),
            eigenvectors=torch.zeros(
                (num_nodes, num_laplacian_eigenvecs),
            ),
            eigenvalues=torch.zeros((num_nodes, num_laplacian_eigenvecs)),
            num_nodes=num_nodes,
        )
        feature_extractor = KNodeCycles()
        kcyclesx, kcyclesy = feature_extractor.k_cycles(
            to_dense_adj(disconnected_graph.edge_index, max_num_nodes=num_nodes)
        )
        disconnected_graph.cycle_features = kcyclesx[0]
        disconnected_graph.global_cycle_features = kcyclesy

        return disconnected_graph
    else:
        raise NotImplementedError()


class MMLinkPredictor(nn.Module):
    FORWARD_TIME = 0
    EXTRA_FEATURE_TIME = 0
    FILTRATION_CONV_TIME = 0

    def __init__(
        self,
        filtration_size,
        num_layers,
        embed_dim,
        num_edge_types,
        num_components,
        max_nodes,
        time_kwargs,
        graph_kwargs,
        node_individualization=("ordering",),
        node_extra_feature_dims=(4, 4, 2, 20),
        graph_extra_feature_dims=(4,),
        node_extra_features=("eigenvectors", "eigenvalues", "cycle_features", "rwpe"),
        graph_extra_features=("global_cycle_features",),
        use_prev_adj=False,
        skip_connections=False,
        dot_product_dim=None,
        initial_distribution="empty",
        transform=None,
        num_hidden_edge_predictor=1,
    ):
        super().__init__()
        self.mixer_model = MixerModel(
            filtration_size,
            num_layers,
            embed_dim,
            num_edge_types,
            max_nodes,
            time_kwargs,
            graph_kwargs,
            node_individualization=node_individualization,
            node_extra_feature_dims=node_extra_feature_dims,
            graph_extra_feature_dims=graph_extra_feature_dims,
            node_extra_features=node_extra_features,
            graph_extra_features=graph_extra_features,
            use_prev_adj=use_prev_adj,
            skip_connections=skip_connections,
            initial_distribution=initial_distribution,
        )
        try:
            idx = node_extra_features.index("rwpe")
            self.rwpe_dim = node_extra_feature_dims[idx]
        except ValueError:
            self.rwpe_dim = 0
        try:
            idx = node_extra_features.index("eigenvectors")
            self.num_laplacian_eigenvecs = node_extra_feature_dims[idx]
        except ValueError:
            self.num_laplacian_eigenvecs = 0
        self.cycle_features = (
            "cycle_features" in node_extra_features
            or "global_cycle_features" in graph_extra_features
        )
        self.filtration_size = filtration_size
        self.initial_distribution = initial_distribution
        self.transform = transform
        dot_product_dim = dot_product_dim if dot_product_dim is not None else embed_dim
        self.edge_predictor = EdgePredictor(
            num_edge_types,
            num_components,
            embed_dim,
            dot_product_dim,
            num_hidden=num_hidden_edge_predictor,
        )

    def forward(
        self,
        filtrations,
        edge_attr=None,
        global_x=None,
        generation_step=None,
        max_num_nodes=None,
    ):
        assert not self.mixer_model.use_prev_adj
        x = self.mixer_model(
            filtrations,
            edge_attr,
            global_x,
            generation_step=generation_step,
        )
        dense_x, mask = to_dense_batch(
            x, batch=filtrations.batch, max_num_nodes=max_num_nodes
        )
        if generation_step is None:
            dense_x = rearrange(dense_x, "(t b) n d -> t b n d", t=self.filtration_size)
            mask = rearrange(mask, "(t b) n -> t b n", t=self.filtration_size)
        return *self.edge_predictor(dense_x, mask), mask

    @torch.inference_mode
    def sample(self, num_nodes: List[int], device="cpu", return_filtrations=True):
        """Generate a batch of graphs.

        :param num_nodes: List of the number of nodes
        :return:
        """
        if self.training:
            raise RuntimeError("Can only sample in evaluation mode")
        max_num_nodes = max(num_nodes)

        if self.rwpe_dim > 0:
            rwpe_transform = RobustAddRandomWalkPE(self.rwpe_dim, "rwpe")
        else:
            rwpe_transform = None
        current_graphs = [
            initial_graph(
                n,
                num_laplacian_eigenvecs=self.num_laplacian_eigenvecs,
                distribution=self.initial_distribution,
            ).to(device)
            for n in num_nodes
        ]
        current_batch = Batch.from_data_list(current_graphs)
        current_batch = current_batch.to(device)
        if rwpe_transform is not None:
            current_batch = rwpe_transform(current_batch)
        if self.transform is not None:
            current_batch = self.transform(current_batch)

        all_batches = [current_batch]

        for graph_idx in range(self.filtration_size):
            assert not self.mixer_model.use_prev_adj
            t0 = time.time()
            mixture_logits, logits, mask = self.forward(
                current_batch,
                generation_step=graph_idx,
                max_num_nodes=max_num_nodes,
            )
            sampled_edges = sample_edges(logits, mixture_logits, mask)
            t1 = time.time()
            self.FORWARD_TIME += t1 - t0
            # sampled edges is of shape (batch size, num nodes, num nodes, num edge types), one-hot
            assert sampled_edges.ndim == 4
            adj = torch.argmax(sampled_edges, dim=-1)

            t0 = time.time()
            if self.cycle_features:
                feature_extractor = KNodeCycles()
                kcyclesx, kcyclesy = feature_extractor.k_cycles(adj.float())
                kcyclesx = dense_to_sparse_batch(kcyclesx, mask=mask)
            else:
                kcyclesx, kcyclesy = None, None

            if self.num_laplacian_eigenvecs > 0:
                all_eigenvector_encodings, all_eigenvalue_encodings = global_laplacian(
                    current_batch.edge_index,
                    max_num_nodes,
                    self.num_laplacian_eigenvecs,
                    batch=current_batch.batch,
                    fast=len(num_nodes) % 16 == 0,
                    return_dense=False,
                )
                all_eigenvector_encodings = all_eigenvector_encodings.to(device)
                all_eigenvalue_encodings = all_eigenvalue_encodings.to(device)
            else:
                all_eigenvector_encodings, all_eigenvalue_encodings = None, None
            t1 = time.time()
            self.EXTRA_FEATURE_TIME += t1 - t0

            new_edge_idx, new_edge_attr, new_batch_enumeration = dense_to_sparse_adj(
                adj, mask
            )
            new_batch = Batch(
                edge_index=new_edge_idx,
                edge_attr=new_edge_attr,
                batch=new_batch_enumeration,
                x=current_batch.x,
                ordering=current_batch.ordering,
                eigenvectors=all_eigenvector_encodings,
                eigenvalues=all_eigenvalue_encodings,
                cycle_features=kcyclesx,
                global_cycle_features=kcyclesy,
            )
            t0 = time.time()
            if rwpe_transform is not None:
                new_batch = rwpe_transform(new_batch)
            t1 = time.time()
            self.EXTRA_FEATURE_TIME += t1 - t0

            all_batches.append(new_batch)
            current_batch = new_batch

        t0 = time.time()
        num_graphs = len(num_nodes)
        offsets = cumsum(torch.tensor(num_nodes, device=device))
        filtrations = [[] for _ in range(num_graphs)]
        sampled_graphs = []
        # assert batch.global_cycle_features.shape[0] == num_graphs
        if return_filtrations:
            for filtration_idx, batch in enumerate(all_batches):
                for graph_idx, n in enumerate(num_nodes):
                    start = offsets[graph_idx]
                    end = offsets[graph_idx + 1]
                    assert end - start == n
                    edge_mask = (start <= batch.edge_index[0]) & (
                        batch.edge_index[0] < end
                    )
                    edge_index = batch.edge_index[:, edge_mask]
                    assert (edge_index >= start).all() and (edge_index < end).all()
                    graph = Data(
                        edge_index=edge_index - start,
                        x=batch.x[start:end],
                        ordering=batch.ordering[start:end],
                        eigenvectors=batch.eigenvectors[start:end],
                        eigenvalues=batch.eigenvalues[start:end],
                        cycle_features=batch.cycle_features[start:end]
                        if self.cycle_features
                        else None,
                        global_cycle_features=batch.global_cycle_features[
                            graph_idx
                        ].unsqueeze(0)
                        if self.cycle_features
                        else None,
                        rwpe=batch.rwpe[start:end]
                        if rwpe_transform is not None
                        else None,
                    )
                    if filtration_idx != len(all_batches) - 1:
                        filtrations[graph_idx].append(graph)
                    else:
                        sampled_graphs.append(graph)
            assert all(n == g.num_nodes for n, g in zip(num_nodes, sampled_graphs))
        else:
            filtrations = None
            batch = all_batches[-1]
            for graph_idx, n in enumerate(num_nodes):
                start = offsets[graph_idx]
                end = offsets[graph_idx + 1]
                assert end - start == n
                edge_mask = (start <= batch.edge_index[0]) & (batch.edge_index[0] < end)
                edge_index = batch.edge_index[:, edge_mask]
                assert (edge_index >= start).all() and (edge_index < end).all()
                graph = Data(
                    edge_index=edge_index - start,
                    x=batch.x[start:end],
                    ordering=batch.ordering[start:end],
                    eigenvectors=batch.eigenvectors[start:end],
                    eigenvalues=batch.eigenvalues[start:end],
                    cycle_features=batch.cycle_features[start:end]
                    if self.cycle_features
                    else None,
                    global_cycle_features=batch.global_cycle_features[
                        graph_idx
                    ].unsqueeze(0)
                    if self.cycle_features
                    else None,
                    rwpe=batch.rwpe[start:end] if rwpe_transform is not None else None,
                )
                sampled_graphs.append(graph)
        t1 = time.time()
        self.FILTRATION_CONV_TIME += t1 - t0
        return filtrations, sampled_graphs


class MMRegressor(nn.Module):
    def __init__(self, output_dim, *args, **kwargs):
        super().__init__()
        self.model = MixerModel(*args, **kwargs)
        self.dense = nn.Linear(kwargs["embed_dim"], output_dim)

    def forward(
        self,
        filtrations,
        edge_attr=None,
        global_x=None,
        generation_step=None,
    ):
        x = self.model(
            filtrations,
            edge_attr=edge_attr,
            global_x=global_x,
            generation_step=generation_step,
        )
        return self.dense(
            torch_geometric.nn.global_mean_pool(x, batch=filtrations.batch)
        )


class MixerModel(nn.Module):
    def __init__(
        self,
        filtration_size,
        num_layers,
        embed_dim,
        num_edge_types,
        max_nodes,
        time_kwargs,
        graph_kwargs,
        node_individualization=("ordering",),
        node_extra_feature_dims=(4, 4, 2, 20),
        graph_extra_feature_dims=(4,),
        node_extra_features=("eigenvectors", "eigenvalues", "cycle_features", "rwpe"),
        graph_extra_features=("global_cycle_features",),
        use_prev_adj=False,
        skip_connections=False,
        initial_distribution="empty",
    ):
        super().__init__()
        self.initial_distribution = initial_distribution
        self.filtration_size = filtration_size
        self.use_prev_adj = use_prev_adj
        self.num_edge_types = num_edge_types
        self.node_extra_features = node_extra_features
        self.graph_extra_features = graph_extra_features
        self.node_extra_feature_dims = node_extra_feature_dims
        self.graph_extra_feature_dims = graph_extra_feature_dims

        if node_extra_feature_dims is not None:
            assert len(node_extra_feature_dims) == len(node_extra_features)
        if graph_extra_feature_dims is not None:
            assert len(graph_extra_feature_dims) == len(graph_extra_features)

        first_graph_kwargs = deepcopy(graph_kwargs)
        first_graph_kwargs["global_embed_dim"] = (
            embed_dim if graph_extra_feature_dims is not None else None
        )
        self._first_layer = MixerBlock(
            embed_dim,
            time_kwargs,
            first_graph_kwargs,
            max_nodes,
            node_individualization=node_individualization,
            node_extra_feature_dims=node_extra_feature_dims,
            graph_extra_feature_dims=graph_extra_feature_dims,
            skip_connections=skip_connections,
        )
        self._other_layers = nn.ModuleList(
            [
                MixerBlock(
                    embed_dim,
                    time_kwargs,
                    graph_kwargs,
                    max_nodes,
                    node_individualization=None,
                    skip_connections=skip_connections,
                )
                for _ in range(num_layers - 1)
            ]
        )

    def forward(
        self,
        filtrations,
        edge_attr=None,
        global_x=None,
        generation_step=None,
    ):
        bs = filtrations.batch[-1].item() + 1
        if generation_step is None:
            assert bs % self.filtration_size == 0
            t = torch.repeat_interleave(
                torch.arange(
                    self.filtration_size, device=filtrations.edge_index.device
                ),
                bs // self.filtration_size,
            ).long()
        else:
            t = (
                generation_step
                * torch.ones(bs, device=filtrations.edge_index.device).long()
            )

        if self.node_extra_features is not None:
            node_extra_features = [
                getattr(filtrations, feature) for feature in self.node_extra_features
            ]
            assert all(
                feat.size(-1) == dim
                for feat, dim in zip(node_extra_features, self.node_extra_feature_dims)
            )
        else:
            node_extra_features = None

        if self.graph_extra_features is not None:
            graph_extra_features = [
                getattr(filtrations, feature) for feature in self.graph_extra_features
            ]
            assert all(
                feat.size(-1) == dim
                for feat, dim in zip(
                    graph_extra_features, self.graph_extra_feature_dims
                )
            )
        else:
            graph_extra_features = None

        x = self._first_layer(
            filtrations,
            self.filtration_size,
            node_extra_features=node_extra_features,
            graph_extra_features=graph_extra_features,
            edge_attr=edge_attr,
            t=t,
            global_x=global_x,
            generation_step=generation_step,
        )
        for layer in self._other_layers:
            x = layer(
                filtrations,
                self.filtration_size,
                x=x,
                edge_attr=edge_attr,
                t=t,
                global_x=global_x,
                generation_step=generation_step,
            )
        return x


class MixerBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        time_kwargs,
        graph_kwargs,
        max_nodes,
        node_individualization=None,
        node_extra_feature_dims=None,
        graph_extra_feature_dims=None,
        skip_connections=False,
    ):
        super().__init__()
        self.skip_connections = skip_connections

        if node_individualization is not None:
            self.individualizer = NodeIndividualizer(
                node_individualization,
                max_nodes,
                embed_dim,
            )
        else:
            self.individualizer = None

        if node_extra_feature_dims is not None:
            self.node_feature_transform = FeatureTransform(
                node_extra_feature_dims, embed_dim
            )

        else:
            self.node_feature_transform = None

        if graph_extra_feature_dims is not None:
            self.graph_feature_transform = FeatureTransform(
                graph_extra_feature_dims, embed_dim
            )
        else:
            self.graph_feature_transform = None

        self.graph_model = instantiate(graph_kwargs)
        self.time_model = instantiate(time_kwargs)

    def forward(
        self,
        filtrations,
        filtration_size,
        x=None,
        edge_attr=None,
        t=None,
        global_x=None,
        node_extra_features: Optional[List[torch.Tensor]] = None,
        graph_extra_features: Optional[List[torch.Tensor]] = None,
        generation_step=None,
    ):
        if node_extra_features is not None:
            feats = self.node_feature_transform(node_extra_features)
            # assert not torch.isnan(feats).any(), node_extra_features
            if x is None:
                x = feats
            else:
                x = x + feats

        if graph_extra_features is not None:
            feats = self.graph_feature_transform(graph_extra_features)
            if global_x is None:
                global_x = feats
            else:
                global_x = global_x + feats
        else:
            assert self.graph_feature_transform is None

        if self.individualizer is not None:
            indiv = self.individualizer(filtrations)
            assert x.shape == indiv.shape, (x.shape, indiv.shape)
            x = x + indiv

        original_x = x
        result = self.graph_model(
            filtrations.edge_index,
            batch=filtrations.batch,
            x=x,
            edge_attr=edge_attr,
            t=t,
            global_x=global_x,
        )
        x, mask = result["node_embeddings"], result["mask"]
        sparse_x = to_sparse_batch(x, mask)
        if self.skip_connections and original_x is not None:
            sparse_x = sparse_x + original_x
        original_x = sparse_x
        sparse_x = self.time_model(sparse_x, filtration_size, generation_step)
        if self.skip_connections:
            sparse_x = sparse_x + original_x
        return sparse_x
