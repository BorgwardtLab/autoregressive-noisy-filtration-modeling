from anfm.evaluation.spectre_utils import (  # isort: skip
    eval_acc_planar_graph,  # isort: skip
    is_planar_graph,  # isort: skip
)

import hashlib
from functools import partial
from typing import Optional

import networkx as nx
import numpy as np
import scipy

from anfm.data.base.data_generation import GraphSet, sample_with_holdout
from anfm.data.base.dataset import AbstractFiltrationDataset
from anfm.data.base.eval import ratio_vun


def random_planar(num_nodes, rng=None):
    """
    Generate a random planar graph
    :param int num_nodes: Number of nodes in the graph
    :return: planar networkx graph
    """
    rng = rng if rng is not None else np.random.default_rng()
    if isinstance(num_nodes, (list, tuple)):
        num_nodes = rng.choice(num_nodes)
    node_locations = rng.uniform(size=(num_nodes, 2))
    # Create the delaunay triangulation
    triangulation = scipy.spatial.Delaunay(node_locations)
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(
        (s[i], s[j])
        for s in triangulation.simplices
        for i in range(3)
        for j in range(3)
        if i < j
    )
    return graph


class PlanarGraphDataset(AbstractFiltrationDataset):
    def __init__(
        self,
        num_nodes: int,
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
        rng = np.random.default_rng(seed)
        sampler = partial(random_planar, num_nodes=num_nodes, rng=rng)
        samples = sample_with_holdout(num_samples, sampler, hold_out)
        super().__init__(
            nx_graphs=samples,
            num_repetitions=num_repetitions,
            filtration_kwargs=filtration_kwargs,
            parent_dir="custom_planar",
            seed=seed,
        )

    @staticmethod
    def evaluate_graphs(graphs, val_graphs, train_graph_set, comprehensive=False):
        result = AbstractFiltrationDataset.evaluate_graphs(
            graphs, val_graphs, train_graph_set, comprehensive
        )
        result["planar_accuracy"] = eval_acc_planar_graph(graphs)
        result["vun"] = ratio_vun(graphs, train_graph_set, is_planar_graph)
        return result
