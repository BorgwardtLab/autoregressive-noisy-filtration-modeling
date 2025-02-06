import glob
import json
import os
import pickle
from itertools import chain
from typing import Callable, Dict, List, Optional, Tuple, Union

import h5py
import joblib
import networkx as nx
import numpy as np
import torch
from einops import rearrange
from filelock import FileLock
from loguru import logger
from torch_geometric.data import Batch, Data
from torch_geometric.transforms import AddRandomWalkPE
from torch_geometric.utils import (
    dense_to_sparse,
    from_networkx,
    to_dense_adj,
    to_networkx,
)
from tqdm import tqdm

import anfm
import anfm.filtration.scheduling
from anfm import DATA_DIR
from anfm.data.base.eval import GraphSet
from anfm.data.base.features import KNodeCycles, global_laplacian


def _edge_list_to_torch(edge_list, num_nodes, num_edge_types):
    assert num_edge_types == 2
    if len(edge_list) == 0:
        edge_index = torch.zeros((0, 2)).long()
    else:
        edge_index = torch.Tensor([[edge[0], edge[1]] for edge in edge_list]).long()
        edge_index = torch.cat([edge_index, torch.flip(edge_index, dims=(1,))], dim=0)
    edge_index = edge_index.t().contiguous()
    adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).long()[0]
    onehot_adj = torch.nn.functional.one_hot(adj, num_edge_types)
    assert onehot_adj.ndim == 3
    edge_index = edge_index.t().contiguous()
    return adj, onehot_adj, edge_index


def graph_to_filtration(
    graph: Data,
    filtration_fn: Union[str, Callable],
    schedule_fn: Union[str, Callable],
    filtration_size: int,
    num_edge_types: int = 2,
    num_laplacian_eigvecs: int = 0,
    random_walk_dim: int = 0,
    noise_schedule: Optional[List[float]] = None,
) -> Tuple[List[Data], torch.Tensor]:
    """Given a graph, create a filtration of subgraphs and the target adjacency matrices."""
    if isinstance(schedule_fn, str):
        schedule_fn = getattr(anfm.filtration.scheduling, schedule_fn)
    if isinstance(filtration_fn, str):
        filtration_fn = getattr(anfm.filtration, filtration_fn)

    graph_data = []

    nx_graph = to_networkx(
        graph,
        to_undirected=True,
    )
    curvature, node_values = filtration_fn(nx_graph)

    filtration = schedule_fn(curvature, filtration_size)
    assert len(filtration) == filtration_size + 1
    edge_list = list(nx_graph.edges)

    all_adjacencies = []
    noisy_adjacencies = []

    assert (
        noise_schedule is None
        or (isinstance(noise_schedule, str) and noise_schedule.startswith("dense"))
        or len(filtration) == len(noise_schedule)
    ), (len(filtration), len(noise_schedule))

    if isinstance(noise_schedule, str) and noise_schedule.startswith("dense"):
        _, *edge_probs = noise_schedule.split("-")
        edge_probs = torch.tensor([float(prob) for prob in edge_probs])
        _, final_onehot_adj, _ = _edge_list_to_torch(
            edge_list, graph.num_nodes, num_edge_types
        )
        edge_type_count = torch.sum(
            torch.triu(torch.permute(final_onehot_adj, (2, 0, 1)), diagonal=1),
            dim=(1, 2),
        )
        assert edge_type_count.numel() == num_edge_types, edge_type_count.shape
        final_density = torch.sum(edge_type_count[1:])
        edge_type_count = None

    if random_walk_dim > 0:
        rwpe_transformer = AddRandomWalkPE(random_walk_dim, attr_name="rwpe")
    else:
        rwpe_transformer = None

    for filtr_idx, edge_subset in enumerate(filtration):
        current_edges = [edge_list[i] for i in edge_subset]
        adj, onehot_adj, edge_index = _edge_list_to_torch(
            current_edges, graph.num_nodes, num_edge_types
        )
        if noise_schedule is not None:
            if isinstance(noise_schedule, str) and noise_schedule.startswith("dense"):
                current_edge_count = torch.sum(
                    torch.triu(torch.permute(onehot_adj, (2, 0, 1)), diagonal=1),
                    dim=(1, 2),
                )
                assert current_edge_count.ndim == 1
                current_density = torch.sum(current_edge_count[1:])
                sampling_probs = (1 - current_density / final_density) * edge_probs + (
                    current_density / final_density
                ) * final_onehot_adj
                sampling_probs = torch.where(
                    onehot_adj[..., 0].unsqueeze(-1).bool(), sampling_probs, onehot_adj
                )
            else:
                edge_type_count = torch.sum(
                    torch.triu(torch.permute(onehot_adj, (2, 0, 1)), diagonal=1),
                    dim=(1, 2),
                )
                assert edge_type_count.numel() == num_edge_types, edge_type_count.shape
                noisiness = noise_schedule[filtr_idx]
                edge_probs = edge_type_count / torch.sum(edge_type_count)
                transition_matrix = noisiness * (edge_probs).unsqueeze(0) + (
                    1 - noisiness
                ) * torch.eye(num_edge_types)
                sampling_probs = transition_matrix[adj, :]
            assert torch.allclose(
                sampling_probs.sum(dim=-1),
                torch.ones((graph.num_nodes, graph.num_nodes)),
            )
            noisy_adjacencies.append(sampling_probs)
            dist = torch.distributions.Categorical(probs=sampling_probs)
            adj = dist.sample()
            assert adj.ndim == 2
            adj = torch.triu(adj, diagonal=1)
            adj = adj + adj.t()
            onehot_adj = torch.nn.functional.one_hot(adj, num_edge_types)
            edge_index, _ = dense_to_sparse(adj)
            assert edge_index.ndim == 2

        all_adjacencies.append(onehot_adj)

        attributes = {
            "edge_index": edge_index,
            "num_nodes": graph.num_nodes,
            "node_ordering_values": torch.from_numpy(node_values),
        }
        if num_laplacian_eigvecs > 0:
            eigenvector_encoding, eigenvalue_encoding = global_laplacian(
                edge_index,
                graph.num_nodes,
                k=num_laplacian_eigvecs,
                return_dense=False,
            )
            assert len(eigenvector_encoding) == graph.num_nodes, (
                len(eigenvector_encoding),
                graph.num_nodes,
            )
            attributes["eigenvectors"] = eigenvector_encoding
            attributes["eigenvalues"] = eigenvalue_encoding
        current_graph = Data(**attributes)
        if rwpe_transformer is not None:
            current_graph = rwpe_transformer(current_graph)
        graph_data.append(current_graph)

    if noise_schedule is None:
        target = torch.stack(all_adjacencies[1:], dim=0)
    else:
        target = torch.stack(noisy_adjacencies[1:], dim=0).to(torch.float16)

    # drop the last entry of graph_data, which is the full graph we want to predict
    graph_data = graph_data[:-1]

    return graph_data, target


def add_cycle_features(filtrations, max_num_nodes):
    feature_extractor = KNodeCycles()
    batch = Batch.from_data_list(list(chain(*filtrations)))
    adjacencies = to_dense_adj(
        batch.edge_index, batch.batch, max_num_nodes=max_num_nodes
    )
    kcyclesx, kcyclesy = feature_extractor.k_cycles(adjacencies)
    kcyclesx = rearrange(kcyclesx, "(b t) n d -> b t n d", b=len(filtrations))
    kcyclesy = rearrange(kcyclesy, "(b t) d -> b t d", b=len(filtrations))
    bs, filtration_size, num_nodes, _ = kcyclesx.shape
    for i in range(bs):
        for t in range(filtration_size):
            filtrations[i][t].cycle_features = kcyclesx[
                i, t, : filtrations[i][t].num_nodes
            ]
            filtrations[i][t].global_cycle_features = kcyclesy[None, i, t]


def add_cycle_features_batched(filtrations, max_num_nodes, batch_size=512):
    assert isinstance(filtrations, list)
    for i in range(0, len(filtrations), batch_size):
        add_cycle_features(filtrations[i : i + batch_size], max_num_nodes)


class OnDiskFiltrationSequences:
    """A class for storing and appending filtration sequences to an HDF5 file."""

    def __init__(self, filename: str):
        self.filename = filename
        self.hdf5_file = h5py.File(filename, "r")
        self.edge_index = self.hdf5_file["edge_index"]
        self.edge_ptr = self.hdf5_file["edge_ptr"]
        self.num_edges = self.hdf5_file["num_edges"]
        self.node_ptr = self.hdf5_file["node_ptr"]
        self.num_nodes = self.hdf5_file["num_nodes"]
        self.node_attributes = self.hdf5_file["node_attributes"]
        self.graph_attributes = self.hdf5_file["graph_attributes"]

    def __len__(self):
        return self.hdf5_file.attrs["n_filtrations"]

    def __getitem__(self, idx: int) -> List[Data]:
        assert 0 <= idx < len(self), idx
        num_nodes = self.num_nodes[idx]
        filtration_size = self.hdf5_file.attrs["filtration_size"]
        total_num_edges = self.num_edges[idx].sum()
        total_num_nodes = num_nodes * filtration_size

        whole_edge_index = torch.from_numpy(
            self.edge_index[
                :, self.edge_ptr[idx] : self.edge_ptr[idx] + total_num_edges
            ]
        )
        edge_indices = torch.split(
            whole_edge_index, self.num_edges[idx].tolist(), dim=1
        )

        whole_node_attributes = {
            attr: torch.from_numpy(
                self.node_attributes[attr][
                    self.node_ptr[idx] : self.node_ptr[idx] + total_num_nodes
                ]
            )
            for attr in self.node_attributes
        }
        node_attributes = {
            attr: torch.split(whole_node_attributes[attr], int(num_nodes), dim=0)
            for attr in whole_node_attributes
        }

        graph_attributes = {
            attr: torch.from_numpy(
                self.graph_attributes[attr][
                    idx * filtration_size : (idx + 1) * filtration_size
                ]
            )
            for attr in self.graph_attributes
        }
        data_list = [
            Data(
                edge_index=edge_indices[i],
                **{attr: node_attributes[attr][i] for attr in node_attributes},
                **{key: value[i] for key, value in graph_attributes.items()},
                num_nodes=num_nodes,
            )
            for i in range(filtration_size)
        ]
        return data_list

    @staticmethod
    def from_data_list(
        data_lists: List[List[Data]], filename: str
    ) -> "OnDiskFiltrationSequences":
        if os.path.exists(filename):
            raise ValueError(f"File {filename} already exists")

        filtration_size = len(data_lists[0])
        with h5py.File(filename, "w") as hdf5_file:
            hdf5_file.attrs["n_filtrations"] = len(data_lists)
            hdf5_file.attrs["filtration_size"] = filtration_size
            hdf5_file.create_dataset(
                "edge_index",
                data=np.concatenate(
                    [data.edge_index.numpy() for data in chain(*data_lists)], axis=1
                ),
                dtype=np.int64,
            )

            num_edges = np.array(
                [
                    [data.edge_index.shape[1] for data in data_list]
                    for data_list in data_lists
                ]
            )
            num_nodes = np.array([data_list[0].num_nodes for data_list in data_lists])
            edge_ptr = np.zeros(len(data_lists), dtype=np.int64)
            edge_ptr[1:] = np.cumsum(num_edges.sum(axis=1))[:-1]
            node_ptr = np.zeros(len(data_lists), dtype=np.int64)
            node_ptr[1:] = np.cumsum(filtration_size * num_nodes)[:-1]

            hdf5_file.create_dataset("num_edges", data=num_edges, dtype=np.int64)
            hdf5_file.create_dataset("num_nodes", data=num_nodes, dtype=np.int64)
            hdf5_file.create_dataset("edge_ptr", data=edge_ptr, dtype=np.int64)
            hdf5_file.create_dataset("node_ptr", data=node_ptr, dtype=np.int64)

            node_attributes = hdf5_file.create_group("node_attributes")
            for attr in data_lists[0][0].node_attrs():
                node_attributes.create_dataset(
                    attr,
                    data=np.concatenate(
                        [data[attr].numpy() for data in chain(*data_lists)]
                    ),
                    dtype=np.float32,
                )

            graph_attributes = hdf5_file.create_group("graph_attributes")
            for attr in ["global_cycle_features"]:
                graph_attributes.create_dataset(
                    attr,
                    data=np.stack([data[attr].numpy() for data in chain(*data_lists)]),
                    dtype=np.float32,
                )

        return OnDiskFiltrationSequences(filename)

    def close(self):
        self.hdf5_file.close()


class ShardedFiltrationSequences:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(path, exist_ok=True)
        num_shards = len(list(glob.glob(os.path.join(path, "*.hdf5"))))
        self.shards = [
            OnDiskFiltrationSequences(os.path.join(path, f"shard_{i}.hdf5"))
            for i in range(num_shards)
        ]
        self.num_filtrations = [len(shard) for shard in self.shards]
        self.cumsum_num_filtrations = np.cumsum(self.num_filtrations)
        self.closed = False

    def __len__(self):
        return sum(len(shard) for shard in self.shards)

    def __getitem__(self, idx: int) -> List[Data]:
        if self.closed:
            raise ValueError("Filtration sequences have been closed")
        shard_idx = np.searchsorted(self.cumsum_num_filtrations, idx, side="right")
        assert 0 <= shard_idx < len(self.shards), (
            idx,
            shard_idx,
            self.cumsum_num_filtrations,
        )
        preceding_elements = (
            self.cumsum_num_filtrations[shard_idx - 1] if shard_idx > 0 else 0
        )
        return self.shards[shard_idx][idx - preceding_elements]

    def add_shard(self, data_list: List[List[Data]]):
        fpath = os.path.join(self.path, f"shard_{len(self.shards)}.hdf5")
        if os.path.exists(fpath):
            raise ValueError(f"File {fpath} already exists")
        shard = OnDiskFiltrationSequences.from_data_list(data_list, fpath)
        self.shards.append(shard)
        self.num_filtrations.append(len(shard))
        self.cumsum_num_filtrations = np.cumsum(self.num_filtrations)

    def close(self):
        while self.shards:
            self.shards.pop().close()
        self.closed = True

    def __del__(self):
        self.close()


class ShardedTargets:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(path, exist_ok=True)
        num_shards = len(list(glob.glob(os.path.join(path, "*.hdf5"))))
        self.shards = [
            h5py.File(os.path.join(path, f"shard_{i}.hdf5"), "r")
            for i in range(num_shards)
        ]
        self.num_targets = [len(shard["targets"]) for shard in self.shards]
        self.cumsum_num_targets = np.cumsum(self.num_targets)

    def __len__(self):
        return sum(self.num_targets)

    def __getitem__(self, idx: int) -> torch.Tensor:
        shard_idx = np.searchsorted(self.cumsum_num_targets, idx, side="right")
        assert 0 <= idx < len(self), idx
        preceding_elements = (
            self.cumsum_num_targets[shard_idx - 1] if shard_idx > 0 else 0
        )
        assert 0 <= idx - preceding_elements < len(self.shards[shard_idx]["targets"])
        return torch.from_numpy(
            self.shards[shard_idx]["targets"][idx - preceding_elements]
        )

    def add_shard(self, targets: List[torch.Tensor]):
        fpath = os.path.join(self.path, f"shard_{len(self.shards)}.hdf5")
        if os.path.exists(fpath):
            raise ValueError(f"File {fpath} already exists")
        targets = np.stack([target.numpy() for target in targets], axis=0)
        with h5py.File(fpath, "w") as hdf5_file:
            hdf5_file.create_dataset("targets", data=targets, dtype=np.float16)

        self.shards.append(h5py.File(fpath, "r"))
        self.num_targets.append(targets.shape[0])
        self.cumsum_num_targets = np.cumsum(self.num_targets)

    def close(self):
        while self.shards:
            self.shards.pop().close()
        self.closed = True

    def __del__(self):
        self.close()


def sample_with_holdout(
    num_samples: int,
    sampler: Callable[[], nx.Graph],
    holdout: Optional[GraphSet] = None,
) -> List[nx.Graph]:
    samples = []
    while len(samples) < num_samples:
        sample = sampler()
        if holdout is None or sample not in holdout:
            samples.append(sample)
    return samples


def pad_targets(target_data: torch.Tensor, max_nodes: int) -> torch.Tensor:
    assert target_data.ndim == 4 and target_data.size(1) == target_data.size(
        2
    ), target_data.shape
    filtration_length, n, _, n_edge_types = target_data.shape
    padded = torch.zeros(
        (filtration_length, max_nodes, max_nodes, n_edge_types),
        dtype=target_data.dtype,
        device=target_data.device,
    )
    padded[:, :n, :n, :] = target_data
    return padded


def nx_graphs_to_filtrations(
    nx_graphs: List[nx.Graph],
    num_repetitions: int,
    filtration_kwargs: Dict,
    shard_size: int = 1000,
    parent_dir: Optional[str] = None,
) -> Tuple[ShardedFiltrationSequences, ShardedTargets]:
    max_num_nodes = max(g.number_of_nodes() for g in nx_graphs)
    parent_dir = DATA_DIR if parent_dir is None else os.path.join(DATA_DIR, parent_dir)
    identifier = joblib.hash(
        (nx_graphs, num_repetitions, filtration_kwargs, shard_size)
    )
    os.makedirs(parent_dir, exist_ok=True)
    path = os.path.join(parent_dir, identifier)
    filtration_shard_directory = os.path.join(path, "filtrations")
    target_shard_directory = os.path.join(path, "targets")

    # During training, we may use multiple GPUs, hence multiple processes.
    # We use a file lock to avoid race conditions.
    with FileLock(os.path.join(parent_dir, f"{identifier}.lock")):
        if os.path.exists(path):
            logger.info(f"Loading existing data from {path}...")
            sharded_filtrations = ShardedFiltrationSequences(filtration_shard_directory)
            sharded_targets = ShardedTargets(target_shard_directory)
            with open(os.path.join(path, "graph_set.pkl"), "rb") as f:
                graph_set = pickle.load(f)
            if (
                len(sharded_filtrations) != len(nx_graphs) * num_repetitions
                or len(sharded_targets) != len(nx_graphs) * num_repetitions
            ):
                sharded_filtrations.close()
                sharded_targets.close()
                raise ValueError(
                    f"Filtration sequences and targets have inconsistent lengths, {len(sharded_filtrations)} != {len(nx_graphs) * num_repetitions} or {len(sharded_targets)} != {len(nx_graphs) * num_repetitions}"
                )
            return sharded_filtrations, sharded_targets, graph_set

        os.makedirs(path, exist_ok=False)
        logger.info(f"Generating data for {path}...")

        graph_set = GraphSet(nx_graphs)
        with open(os.path.join(path, "graph_set.pkl"), "wb") as f:
            pickle.dump(graph_set, f)

        with open(os.path.join(path, "nx_graphs.pkl"), "wb") as f:
            pickle.dump(nx_graphs, f)

        with open(os.path.join(path, "kwargs.json"), "w") as f:
            json.dump(
                {
                    "filtration_kwargs": filtration_kwargs,
                    "num_repetitions": num_repetitions,
                    "shard_size": shard_size,
                },
                f,
            )

        sharded_filtrations = ShardedFiltrationSequences(filtration_shard_directory)
        sharded_targets = ShardedTargets(target_shard_directory)
        filtration_buffer = []
        target_buffer = []
        for g in tqdm(nx_graphs):
            data = from_networkx(g)
            for _ in range(num_repetitions):
                filtration, target = graph_to_filtration(data, **filtration_kwargs)
                target = pad_targets(target, max_num_nodes)
                filtration_buffer.append(filtration)
                target_buffer.append(target)
                if len(filtration_buffer) == shard_size:
                    add_cycle_features_batched(filtration_buffer, max_num_nodes)
                    sharded_filtrations.add_shard(filtration_buffer)
                    sharded_targets.add_shard(target_buffer)
                    filtration_buffer = []
                    target_buffer = []

        if len(filtration_buffer) > 0:
            add_cycle_features_batched(filtration_buffer, max_num_nodes)
            sharded_filtrations.add_shard(filtration_buffer)
            sharded_targets.add_shard(target_buffer)

    return sharded_filtrations, sharded_targets, graph_set
