from anfm.evaluation.spectre_utils import (  # isort: skip
    clustering_stats,  # isort: skip
    degree_stats,  # isort: skip
    spectral_stats,  # isort: skip
    orbit_stats_all,  # isort: skip
)

import numpy as np
import torch
from torch.utils.data import Dataset

from anfm.data.base.data_generation import nx_graphs_to_filtrations
from anfm.data.base.eval import ratio_novel, ratio_unique


class AbstractFiltrationDataset(Dataset):
    def __init__(
        self,
        nx_graphs,
        num_repetitions,
        filtration_kwargs,
        shard_size=1000,
        parent_dir=None,
        seed=0,
    ):
        (
            self.sharded_filtrations,
            self.sharded_targets,
            self.graph_set,
        ) = nx_graphs_to_filtrations(
            nx_graphs, num_repetitions, filtration_kwargs, shard_size, parent_dir
        )
        self.max_num_nodes = max(g.number_of_nodes() for g in self.nx_graphs)
        self.rng = np.random.default_rng(seed)

    @property
    def nx_graphs(self):
        return self.graph_set.nx_graphs

    def __getitem__(self, item):
        filtration, target = self.sharded_filtrations[item], self.sharded_targets[item]

        random_flips = (
            2
            * self.rng.integers(
                0,
                2,
                size=(
                    len(filtration),
                    filtration[0].eigenvectors.shape[-1],
                ),
            )
            - 1
        )

        for flip, filtr in zip(random_flips, filtration):
            filtr.eigenvectors = filtr.eigenvectors * torch.from_numpy(flip).to(
                device=filtr.eigenvectors.device
            )

        return filtration, target

    def __contains__(self, g):
        return g in self.graph_set

    def __len__(self):
        return len(self.sharded_filtrations)

    def sample_num_nodes(self):
        idx = int(self.rng.integers(0, len(self.nx_graphs)))
        return self.nx_graphs[idx].number_of_nodes()

    @staticmethod
    def evaluate_graphs(graphs, val_graphs, train_graph_set, comprehensive=False):
        result = {}
        result["unique_accuracy"] = ratio_unique(graphs)
        result["novel_accuracy"] = ratio_novel(graphs, train_graph_set)
        result["clustering_stats"] = clustering_stats(val_graphs, graphs)
        result["degree_stats"] = degree_stats(val_graphs, graphs)
        result["spectral_stats"] = spectral_stats(
            val_graphs,
            graphs,
            n_eigvals=-1,
        )
        if comprehensive:
            result["orbit"] = orbit_stats_all(val_graphs, graphs)
        return result

    @property
    def max_nodes(self):
        return self.max_num_nodes

    def close(self):
        self.sharded_filtrations.close()
        self.sharded_targets.close()
