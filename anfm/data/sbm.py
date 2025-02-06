from anfm.evaluation.spectre_utils import (  # isort: skip
    eval_acc_sbm_graph,  # isort: skip
    is_sbm_graph,  # isort: skip
)

import hashlib
from functools import partial
from typing import Optional

import networkx as nx
import numpy as np

from anfm.data.base.data_generation import sample_with_holdout
from anfm.data.base.dataset import AbstractFiltrationDataset
from anfm.data.base.eval import ratio_vun


def random_sbm(
    random_num_nodes: bool = True,
    rng: Optional[np.random.Generator] = None,
    intra_p=0.3,
    inter_p=0.005,
):
    rng = rng if rng is not None else np.random.default_rng()
    while True:
        # Infinite loop until we find a connected graph
        num_communities = rng.integers(
            2, 6
        )  # Need 6 to be consistent with spectre, which uses 5 but with *inclusive* upper bound
        num_nodes_per_community = rng.integers(
            20, 41, size=num_communities
        )  # Need 41 to be consistent with spectre, which uses 40 but with *inclusive* upper bound
        if not random_num_nodes:
            N = 90
            num_nodes_per_community = np.floor(
                N * num_nodes_per_community / np.sum(num_nodes_per_community)
            ).astype(int)
            assert np.sum(num_nodes_per_community) <= N and np.all(
                num_nodes_per_community > 2
            )
            num_nodes_per_community[0] += N - np.sum(num_nodes_per_community)
        community_labels = np.repeat(
            np.arange(num_communities), num_nodes_per_community
        )
        edge_probs = np.where(
            np.expand_dims(community_labels, 0) == np.expand_dims(community_labels, 1),
            intra_p,
            inter_p,
        )
        adj = (rng.random(edge_probs.shape) < edge_probs).astype(int)
        adj = np.triu(adj, 1)
        adj = adj + adj.transpose()
        g = nx.from_numpy_array(adj)

        for u, v, d in g.edges(data=True):
            if "weight" in d:
                del d["weight"]

        assert random_num_nodes or g.number_of_nodes() == 90
        if nx.is_connected(g):
            return g


class SBMGraphDataset(AbstractFiltrationDataset):
    def __init__(
        self,
        num_samples,
        filtration_kwargs,
        split="train",
        num_repetitions=1,
        seed=0,
        hold_out=None,
    ):
        seed = (
            seed + abs(int(hashlib.md5(split.encode("utf-8")).hexdigest(), 16)) % 923428
        )
        self.hold_out = hold_out
        sampler = partial(random_sbm, rng=np.random.default_rng(seed))
        samples = sample_with_holdout(num_samples, sampler, hold_out)
        super().__init__(
            nx_graphs=samples,
            num_repetitions=num_repetitions,
            filtration_kwargs=filtration_kwargs,
            parent_dir="custom_sbm",
            seed=seed,
        )

    @staticmethod
    def evaluate_graphs(graphs, val_graphs, train_graph_set, comprehensive=False):
        result = AbstractFiltrationDataset.evaluate_graphs(
            graphs, val_graphs, train_graph_set, comprehensive
        )
        result["sbm_accuracy"] = eval_acc_sbm_graph(graphs, refinement_steps=100)
        result["vun"] = ratio_vun(
            graphs, train_graph_set, lambda g: is_sbm_graph(g, refinement_steps=100)
        )
        return result
