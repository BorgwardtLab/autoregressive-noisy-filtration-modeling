from collections import defaultdict

import joblib
import networkx as nx
from tqdm import tqdm


class GraphSet:
    def __init__(self, nx_graphs):
        self.nx_graphs = nx_graphs
        self._hash_set = self.compute_hash_set(nx_graphs)

    def insert(self, g):
        self.nx_graphs.append(g)
        self._hash_set[GraphSet.graph_fingerprint(g)].append(len(self.nx_graphs) - 1)

    def __contains__(self, g):
        fingerprint = self.graph_fingerprint(g)
        if fingerprint not in self._hash_set:
            return False
        potentially_isomorphic = [
            self.nx_graphs[idx] for idx in self._hash_set[fingerprint]
        ]
        for h in potentially_isomorphic:
            if nx.is_isomorphic(g, h):
                return True
        return False

    def __add__(self, other):
        return GraphSet(self.nx_graphs + other.nx_graphs)

    @staticmethod
    def graph_fingerprint(g):
        nodes = list(g.nodes)
        triangle_counts = nx.triangles(g, nodes)
        degrees = [item[1] for item in g.degree(nodes)]
        fingerprint = tuple(
            sorted(
                [
                    joblib.hash((deg, triangle))
                    for deg, triangle in zip(degrees, triangle_counts)
                ]
            )
        )
        return fingerprint

    @staticmethod
    def compute_hash_set(nx_graphs):
        hash_set = defaultdict(list)
        for idx, g in tqdm(enumerate(nx_graphs)):
            hash_set[GraphSet.graph_fingerprint(g)].append(idx)
        return hash_set


def ratio_novel(samples, training_set):
    novel = sum(not sample in training_set for sample in samples)
    return novel / len(samples)


def ratio_unique(samples):
    num_unique = 0
    sample_set = GraphSet([])
    for sample in samples:
        if sample not in sample_set:
            num_unique += 1
        sample_set.insert(sample)
    return num_unique / len(samples)


def ratio_vun(samples, training_set, validity_fn):
    num_vun = 0
    sample_set = GraphSet([])
    for sample in samples:
        if (
            validity_fn(sample)
            and sample not in sample_set
            and not sample in training_set
        ):
            num_vun += 1
        sample_set.insert(sample)
    return num_vun / len(samples)
