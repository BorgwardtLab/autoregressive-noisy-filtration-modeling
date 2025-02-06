import os

import networkx as nx
import numpy as np
import requests
import torch
from tqdm import tqdm

from anfm import DATA_DIR
from anfm.data.base.dataset import AbstractFiltrationDataset


def get_nx_proteins(min_num_nodes=100, max_num_nodes=500):
    # Based on https://github.com/KarolisMart/SPECTRE/blob/f676c0d55fe0eba2fbf10133291c9aeac89423e6/data.py#L356
    nx_graphs = []

    raw_data_dir = DATA_DIR / "raw_dobson_doig_proteins"

    if not os.path.exists(raw_data_dir):
        os.makedirs(raw_data_dir, exist_ok=False)
        # Download all the data from "https://github.com/lrjconan/GRAN/tree/master/data/DD" to tmpdirname
        url = "https://raw.githubusercontent.com/lrjconan/GRAN/fc9c04a3f002c55acf892f864c03c6040947bc6b/data/DD"
        filenames = [
            "DD_A.txt",
            "DD_graph_indicator.txt",
            "DD_graph_labels.txt",
            "DD_node_labels.txt",
        ]
        for filename in filenames:
            print(f"Downloading {filename}")
            response = requests.get(f"{url}/{filename}")
            with open(os.path.join(raw_data_dir, filename), "wb") as f:
                f.write(response.content)

    G = nx.Graph()
    data_adj = np.loadtxt(os.path.join(raw_data_dir, "DD_A.txt"), delimiter=",").astype(
        int
    )
    data_graph_indicator = np.loadtxt(
        os.path.join(raw_data_dir, "DD_graph_indicator.txt"), delimiter=","
    ).astype(int)
    data_graph_types = np.loadtxt(
        os.path.join(raw_data_dir, "DD_graph_labels.txt"), delimiter=","
    ).astype(int)

    data_tuple = list(map(tuple, data_adj))

    # Add edges
    G.add_edges_from(data_tuple)
    G.remove_nodes_from(list(nx.isolates(G)))

    # remove self-loop
    G.remove_edges_from(nx.selfloop_edges(G))

    # Split into graphs
    graph_num = data_graph_indicator.max()
    node_list = np.arange(data_graph_indicator.shape[0]) + 1

    for i in tqdm(range(graph_num)):
        # Find the nodes for each graph
        nodes = node_list[data_graph_indicator == i + 1]
        G_sub = G.subgraph(nodes)
        G_sub.graph["label"] = data_graph_types[i]
        if (
            G_sub.number_of_nodes() >= min_num_nodes
            and G_sub.number_of_nodes() <= max_num_nodes
        ):
            nx_graphs.append(G_sub)
    return nx_graphs


class GranProteinGraphDataset(AbstractFiltrationDataset):
    def __init__(
        self,
        filtration_kwargs,
        split="train",
        num_repetitions=1,
        seed=0,
        hold_out=None,
    ):
        self.hold_out = hold_out
        nx_graphs = get_nx_proteins()
        test_len = int(round(len(nx_graphs) * 0.2))
        train_len = int(round((len(nx_graphs) - test_len) * 0.8))
        val_len = len(nx_graphs) - train_len - test_len
        print(f"Dataset sizes: train {train_len}, val {val_len}, test {test_len}")

        train, val, test = torch.utils.data.random_split(
            nx_graphs,
            [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(1234),
        )
        if split == "train":
            nx_graphs = train
        elif split == "val":
            nx_graphs = val
        elif split == "test":
            nx_graphs = test
        else:
            raise ValueError(f"Invalid split")

        super().__init__(
            nx_graphs=nx_graphs,
            num_repetitions=num_repetitions,
            filtration_kwargs=filtration_kwargs,
            parent_dir="gran_proteins",
            seed=seed,
        )
