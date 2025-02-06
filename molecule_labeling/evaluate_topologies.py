import argparse
import os
import os.path as osp
import pickle
import tempfile

import networkx as nx
import numpy as np
import torch
from guacamol.assess_distribution_learning import assess_distribution_learning
from guacamol.distribution_matching_generator import DistributionMatchingGenerator
from loguru import logger
from rdkit import Chem
from torch_geometric.data import Batch, download_url
from torch_geometric.transforms import (
    AddLaplacianEigenvectorPE,
    AddRandomWalkPE,
    Compose,
)
from torch_geometric.utils import from_networkx
from tqdm import tqdm

from molecule_labeling.autoencoder import LabelAutoencoder
from molecule_labeling.dataset import CustomData, GuacamolDataset
from molecule_labeling.utils import FCDMetric


class DummyGenerator(DistributionMatchingGenerator):
    def __init__(self, smiles_list, seed=0):
        self.smiles_list = smiles_list
        self.available_indices = set(range(len(self.smiles_list)))
        self.rng = np.random.default_rng(seed)

    def generate(self, number_samples):
        assert number_samples <= len(self.available_indices)
        indices = self.rng.choice(
            list(self.available_indices), size=number_samples, replace=False
        )
        samples = [self.smiles_list[i] for i in indices]
        self.available_indices = self.available_indices - set(indices)
        return samples

    def reset(self):
        self.available_indices = set(range(len(self.smiles_list)))


def get_smiles_data(temp_dir):
    # URLs from GuacamolDataset
    train_url = "https://figshare.com/ndownloader/files/13612760"
    # Download files
    train_path = download_url(train_url, temp_dir)
    # Rename files
    train_file = osp.join(temp_dir, "guacamol_v1_train.smiles")
    os.rename(train_path, train_file)
    return train_file


if __name__ == "__main__":
    # Accept path to pickle file
    np.random.seed(42)
    torch.manual_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--output", type=str, required=False, default="distribution_learning.json"
    )
    parser.add_argument("--filter-disconnected", action="store_true")
    args = parser.parse_args()

    laplace_transform = AddLaplacianEigenvectorPE(3, "laplace")
    rwpe_transform = AddRandomWalkPE(8, "rwpe")
    transforms = Compose([laplace_transform, rwpe_transform])

    node_labels = {
        "atom_labels": 12,
        "explicit_hydrogens": 4,
        "charges": 5,
        "radical_electrons": 3,
    }
    edge_labels = {"bond_labels": 4}
    model = LabelAutoencoder(
        node_labels=node_labels,
        edge_labels=edge_labels,
        structural_features={"laplace": 3, "rwpe": 8},
        decoder_layers=5,
        encoder_layers=5,
        latent_dim=8,
    )
    logger.info(f"Loading model from {args.model}")
    with open(args.model, "rb") as f:
        model.load_state_dict(torch.load(f))
    model.eval()

    logger.info(f"Loading samples from {args.samples}")
    # Load pickle file
    with open(args.samples, "rb") as f:
        data = pickle.load(f)

    logger.info("Converting to data list")
    if args.filter_disconnected:
        data_list = [from_networkx(mol) for mol in tqdm(data) if nx.is_connected(mol)]
    else:
        data_list = [
            from_networkx(nx.subgraph(mol, max(nx.connected_components(mol), key=len)))
            for mol in tqdm(data)
        ]

    data_list = [
        CustomData(
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            **{
                name: torch.zeros(data.num_nodes, dim)
                for name, dim in node_labels.items()
            },
            **{
                name: torch.zeros(data.num_edges, dim)
                for name, dim in edge_labels.items()
            },
        )
        for data in tqdm(data_list)
    ]
    data_list = [transforms(data) for data in tqdm(data_list)]

    logger.info("Sampling")
    batch = Batch.from_data_list(data_list)
    with torch.inference_mode():
        node_attr, edge_attr = model.sample(batch, variant="max")
    all_attr = {**node_attr, **edge_attr}
    for name, attr in all_attr.items():
        setattr(batch, name, attr)

    mols = GuacamolDataset.batch_to_mols(batch)
    generated_smiles = [Chem.MolToSmiles(mol) for mol in tqdm(mols)]
    dummy_generator = DummyGenerator(generated_smiles)

    logger.info("Assessing distribution learning")
    with tempfile.TemporaryDirectory() as temp_dir:
        train_path = get_smiles_data(temp_dir)
        assess_distribution_learning(
            dummy_generator,
            chembl_training_file=train_path,
            json_output_file=args.output,
        )
