import os
import pickle

import numpy as np

from anfm import DATA_DIR
from anfm.data.base.dataset import AbstractFiltrationDataset


def get_guacamol_nx_graphs(split):
    graph_dir = os.path.join(DATA_DIR, "guacamol_raw_graphs")
    nx_file = os.path.join(graph_dir, f"{split}_nx_graphs.pkl")
    if not os.path.exists(nx_file):
        raise FileNotFoundError(
            f"No nx graphs found for Guacamol {split} set. Run `python anfm/data/install_raw_guacamol.py` to download necessary data."
        )
    with open(nx_file, "rb") as f:
        return pickle.load(f)


class GuacamolGraphDataset(AbstractFiltrationDataset):
    def __init__(
        self,
        filtration_kwargs,
        split="train",
        num_repetitions=1,
        seed=0,
        hold_out=None,
        max_samples=None,
    ):
        self.hold_out = hold_out
        all_nx_graphs = get_guacamol_nx_graphs(split)
        nx_graphs = list(filter(lambda g: g.number_of_nodes() > 5, all_nx_graphs))
        # shuffle
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(nx_graphs))
        nx_graphs = [nx_graphs[i] for i in perm]

        if max_samples is not None:
            rng = np.random.default_rng(0)
            idxs = rng.choice(len(nx_graphs), max_samples, replace=False)
            nx_graphs = [nx_graphs[idx] for idx in idxs]

        super().__init__(
            nx_graphs=nx_graphs,
            num_repetitions=num_repetitions,
            filtration_kwargs=filtration_kwargs,
            parent_dir="guacamol",
            seed=seed,
        )
