import hashlib
from functools import partial
from typing import Optional

import networkx as nx
import numpy as np

from anfm.data.base.data_generation import GraphSet, sample_with_holdout
from anfm.data.base.dataset import AbstractFiltrationDataset
from anfm.data.base.eval import ratio_vun


# Default settings taken from https://github.com/lrjconan/GRAN/blob/master/utils/data_helper.py
def random_lobster(
    expected_num_nodes=80,
    p1=0.7,
    p2=0.7,
    rng=None,
    max_number_of_nodes=100,
    min_number_of_nodes=10,
):
    rng = rng if rng is not None else np.random.default_rng()
    while True:
        g = nx.random_lobster(expected_num_nodes, p1, p2, seed=int(rng.integers(1e9)))
        if (
            max_number_of_nodes is None or g.number_of_nodes() <= max_number_of_nodes
        ) and (
            min_number_of_nodes is None or g.number_of_nodes() >= min_number_of_nodes
        ):
            return g


def is_lobster(nx_graph):
    """Check whether networkx graph is a lobster graph."""
    if not nx.is_tree(nx_graph):
        return False

    pruned_graph = nx_graph.copy()
    for _ in range(2):
        # First iteration should turn it into a caterpillar, second into a path
        leaves = [node for node, degree in pruned_graph.degree() if degree == 1]
        if not leaves:
            break
        pruned_graph.remove_nodes_from(leaves)

    if pruned_graph.number_of_nodes() == 0:
        return True

    has_no_branches = max([d for n, d in pruned_graph.degree()]) <= 2
    # The nx.is_tree check is somewhat redundant because removing leaves should keep it a tree
    is_path = nx.is_tree(pruned_graph) and has_no_branches
    return is_path


class LobsterGraphDataset(AbstractFiltrationDataset):
    def __init__(
        self,
        num_samples: int,
        filtration_kwargs: dict,
        split: str = "train",
        num_repetitions: int = 1,
        seed: int = 0,
        hold_out: Optional[GraphSet] = None,
    ):
        seed = (
            seed + abs(int(hashlib.md5(split.encode("utf-8")).hexdigest(), 16)) % 923428
        )
        self.hold_out = hold_out
        sampler = partial(random_lobster, rng=np.random.default_rng(seed))
        samples = sample_with_holdout(num_samples, sampler, hold_out)
        super().__init__(
            nx_graphs=samples,
            num_repetitions=num_repetitions,
            filtration_kwargs=filtration_kwargs,
            parent_dir="custom_lobster",
            seed=seed,
        )

    @staticmethod
    def evaluate_graphs(graphs, val_graphs, train_graph_set, comprehensive=False):
        result = AbstractFiltrationDataset.evaluate_graphs(
            graphs, val_graphs, train_graph_set, comprehensive
        )
        result["lobster_accuracy"] = sum([is_lobster(g) for g in graphs]) / len(graphs)
        result["vun"] = ratio_vun(graphs, train_graph_set, is_lobster)
        return result
