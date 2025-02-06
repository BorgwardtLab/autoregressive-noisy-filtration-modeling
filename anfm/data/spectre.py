import os
import tempfile
from typing import Literal

import networkx as nx
import requests
import torch

from anfm.data.base.dataset import AbstractFiltrationDataset
from anfm.data.base.eval import ratio_vun

from anfm.evaluation.spectre_utils import (  # isort: skip
    eval_acc_planar_graph,
    eval_acc_sbm_graph,
    is_planar_graph,
    is_sbm_graph,
)


def get_nx_graphs(dataset: Literal["planar", "sbm"]):
    nx_graphs = []
    if dataset == "planar":
        url = "https://github.com/KarolisMart/SPECTRE/raw/f676c0d55fe0eba2fbf10133291c9aeac89423e6/data/planar_64_200.pt"
    elif dataset == "sbm":
        url = "https://github.com/KarolisMart/SPECTRE/raw/f676c0d55fe0eba2fbf10133291c9aeac89423e6/data/sbm_200.pt"
    else:
        raise ValueError(f"Dataset {dataset} not supported")

    with tempfile.TemporaryDirectory() as tmpdirname:
        print(f"Downloading {dataset} graphs...")
        response = requests.get(url)
        fpath = os.path.join(tmpdirname, "graph_data.pt")
        with open(fpath, "wb") as f:
            f.write(response.content)

        (
            adjs,
            eigvals,
            eigvecs,
            n_nodes,
            max_eigval,
            min_eigval,
            same_sample,
            n_max,
        ) = torch.load(fpath)
        assert len(adjs) == len(n_nodes)
        for adj, n in zip(adjs, n_nodes):
            assert adj.ndim == 2 and adj.shape[0] == adj.shape[1] and adj.shape[0] == n
            G = nx.from_numpy_array(adj.numpy())
            nx_graphs.append(G)

    test_len = int(round(len(adjs) * 0.2))
    train_len = int(round((len(adjs) - test_len) * 0.8))
    val_len = len(adjs) - train_len - test_len
    print(f"Dataset sizes: train {train_len}, val {val_len}, test {test_len}")

    train, val, test = torch.utils.data.random_split(
        nx_graphs,
        [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(1234),
    )
    return list(train), list(val), list(test)


class SpectrePlanarGraphDataset(AbstractFiltrationDataset):
    def __init__(
        self,
        filtration_kwargs,
        split="train",
        num_repetitions=1,
        seed=0,
        hold_out=None,
    ):
        self.hold_out = hold_out
        train, val, test = get_nx_graphs("planar")
        if split == "train":
            nx_graphs = train
        elif split == "val":
            nx_graphs = val
        elif split == "test":
            nx_graphs = test
        else:
            raise ValueError(f"Split {split} not supported")

        super().__init__(
            nx_graphs=nx_graphs,
            num_repetitions=num_repetitions,
            filtration_kwargs=filtration_kwargs,
            parent_dir="spectre_planar",
            seed=seed,
        )

    @staticmethod
    def evaluate_graphs(graphs, val_graphs, train_graph_set, comprehensive=False):
        result = AbstractFiltrationDataset.evaluate_graphs(
            graphs, val_graphs, train_graph_set, comprehensive=comprehensive
        )
        result["planar_accuracy"] = eval_acc_planar_graph(graphs)
        result["vun"] = ratio_vun(graphs, train_graph_set, is_planar_graph)
        return result


class SpectreSBMGraphDataset(AbstractFiltrationDataset):
    def __init__(
        self,
        filtration_kwargs,
        split="train",
        num_repetitions=1,
        seed=0,
        hold_out=None,
    ):
        self.hold_out = hold_out
        train, val, test = get_nx_graphs("sbm")
        if split == "train":
            nx_graphs = train
        elif split == "val":
            nx_graphs = val
        elif split == "test":
            nx_graphs = test
        else:
            raise ValueError(f"Split {split} not supported")

        super().__init__(
            nx_graphs=nx_graphs,
            num_repetitions=num_repetitions,
            filtration_kwargs=filtration_kwargs,
            parent_dir="spectre_sbm",
            seed=seed,
        )

    @staticmethod
    def evaluate_graphs(graphs, val_graphs, train_graph_set, comprehensive=False):
        result = AbstractFiltrationDataset.evaluate_graphs(
            graphs, val_graphs, train_graph_set, comprehensive=comprehensive
        )
        result["sbm_accuracy"] = eval_acc_sbm_graph(graphs, refinement_steps=100)
        result["vun"] = ratio_vun(
            graphs, train_graph_set, lambda g: is_sbm_graph(g, refinement_steps=100)
        )
        return result
