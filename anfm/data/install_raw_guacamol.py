import contextlib
import hashlib
import os
import os.path as osp
import pickle
from pathlib import Path
from typing import Any, Sequence

import joblib
import torch
import torch.nn.functional as F
from loguru import logger
from rdkit import Chem, RDLogger
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.utils import to_networkx
from tqdm import tqdm

BOND_DICT = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
BOND_ENCODER = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3,
}


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def are_smiles_equivalent(smiles1, smiles2):
    # Convert SMILES to mol objects
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    # Check if either conversion failed
    if mol1 is None or mol2 is None:
        return False

    # Convert to canonical SMILES
    canonical_smiles1 = Chem.MolToSmiles(mol1, canonical=True)
    canonical_smiles2 = Chem.MolToSmiles(mol2, canonical=True)

    return canonical_smiles1 == canonical_smiles2


def compare_hash(output_file: str, correct_hash: str) -> bool:
    """
    Computes the md5 hash of a SMILES file and check it against a given one
    Returns false if hashes are different
    """
    output_hash = hashlib.md5(open(output_file, "rb").read()).hexdigest()
    if output_hash != correct_hash:
        print(
            f"{output_file} file has different hash, {output_hash}, than expected, {correct_hash}!"
        )
        return False

    return True


def files_exist(files) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([osp.exists(f) for f in files])


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


def graph_to_molecule(
    node_labels,
    edge_index,
    edge_labels,
    atom_decoder,
    explicit_hydrogens=None,
    charges=None,
    num_radical_electrons=None,
    pos=None,
):
    assert edge_index.shape[1] == len(edge_labels)
    assert edge_labels.ndim == 1

    node_idx_to_atom_idx = {}
    current_atom_idx = 0
    mol = Chem.RWMol()
    for node_idx, atom in enumerate(node_labels):
        a = Chem.Atom(atom_decoder[atom.item()])
        mol.AddAtom(a)
        node_idx_to_atom_idx[node_idx] = current_atom_idx
        if charges is not None:
            mol.GetAtomWithIdx(node_idx_to_atom_idx[node_idx]).SetFormalCharge(
                charges[node_idx].item()
            )
        if num_radical_electrons is not None:
            mol.GetAtomWithIdx(node_idx_to_atom_idx[node_idx]).SetNumRadicalElectrons(
                num_radical_electrons[node_idx].item()
            )
        current_atom_idx += 1
        if explicit_hydrogens is not None:
            num_hydrogens = explicit_hydrogens[node_idx].item()
            for _ in range(num_hydrogens):
                mol.AddAtom(Chem.Atom("H"))
                mol.AddBond(
                    current_atom_idx,
                    node_idx_to_atom_idx[node_idx],
                    Chem.rdchem.BondType.SINGLE,
                )
                current_atom_idx += 1

    if pos is not None:
        conf = Chem.Conformer(mol.GetNumAtoms())
        for node_idx, atom_pos in enumerate(pos):
            conf.SetAtomPosition(node_idx_to_atom_idx[node_idx], atom_pos.tolist())
        mol.AddConformer(conf)

    added_bonds = set()
    for bond, bond_type in zip(edge_index.T, edge_labels):
        a, b = bond[0].item(), bond[1].item()
        if a != b and (a, b) not in added_bonds:
            added_bonds.add((a, b))
            added_bonds.add((b, a))
            mol.AddBond(
                node_idx_to_atom_idx[a],
                node_idx_to_atom_idx[b],
                BOND_DICT[bond_type.item()],
            )

    if pos is not None:
        Chem.rdmolops.AssignStereochemistryFrom3D(mol)

    return mol


def mol_to_graph(mol, atom_encoder):
    N = mol.GetNumAtoms()

    atom_labels = []
    explicit_hydrogens = []
    implicit_hydrogens = []
    charges = []
    num_radical_electrons = []
    for atom in mol.GetAtoms():
        atom_labels.append(atom_encoder[atom.GetSymbol()])
        explicit_hydrogens.append(atom.GetNumExplicitHs())
        implicit_hydrogens.append(atom.GetNumImplicitHs())
        charges.append(atom.GetFormalCharge())
        num_radical_electrons.append(atom.GetNumRadicalElectrons())

    atom_labels = torch.tensor(atom_labels)
    explicit_hydrogens = torch.tensor(explicit_hydrogens)
    implicit_hydrogens = torch.tensor(implicit_hydrogens)
    charges = torch.tensor(charges)
    num_radical_electrons = torch.tensor(num_radical_electrons)

    row, col, bond_labels = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        bond_labels += 2 * [
            BOND_ENCODER[bond.GetBondType()],
        ]

    assert len(row) > 0, Chem.MolToSmiles(mol)

    edge_index = torch.tensor([row, col], dtype=torch.long)
    bond_labels = torch.tensor(bond_labels, dtype=torch.long)

    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    bond_labels = bond_labels[perm]

    # Access the generated conformer
    if mol.GetNumConformers() > 0:
        conformer = mol.GetConformer()
        pos = torch.tensor(
            [conformer.GetAtomPosition(atom_idx) for atom_idx in range(N)]
        )
    else:
        pos = None

    data = Data(
        edge_index=edge_index,
        bond_labels=bond_labels,
        atom_labels=atom_labels,
        explicit_hydrogens=explicit_hydrogens,
        implicit_hydrogens=implicit_hydrogens,
        radical_electrons=num_radical_electrons,
        charges=charges,
        pos=pos,
        num_nodes=N,
    )
    return data


TRAIN_HASH = "05ad85d871958a05c02ab51a4fde8530"
VALID_HASH = "e53db4bff7dc4784123ae6df72e3b1f0"
TEST_HASH = "677b757ccec4809febd83850b43e1616"

ATOM_ENCODER = {
    "C": 0,
    "N": 1,
    "O": 2,
    "F": 3,
    "B": 4,
    "Br": 5,
    "Cl": 6,
    "I": 7,
    "P": 8,
    "S": 9,
    "Se": 10,
    "Si": 11,
}
ATOM_DECODER = ["C", "N", "O", "F", "B", "Br", "Cl", "I", "P", "S", "Se", "Si"]


class _GuacamolDataset(InMemoryDataset):
    train_url = "https://figshare.com/ndownloader/files/13612760"
    test_url = "https://figshare.com/ndownloader/files/13612757"
    valid_url = "https://figshare.com/ndownloader/files/13612766"
    all_url = "https://figshare.com/ndownloader/files/13612745"

    def __init__(
        self, stage, root, transform=None, pre_transform=None, pre_filter=None
    ):
        self.stage = stage
        if self.stage == "train":
            self.file_idx = 0
        elif self.stage == "val":
            self.file_idx = 1
        elif self.stage == "test":
            self.file_idx = 2
        else:
            raise NotImplementedError
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx])

    @property
    def raw_file_names(self):
        return [
            "guacamol_v1_train.smiles",
            "guacamol_v1_valid.smiles",
            "guacamol_v1_test.smiles",
        ]

    @property
    def split_file_name(self):
        return [
            "guacamol_v1_train.smiles",
            "guacamol_v1_valid.smiles",
            "guacamol_v1_test.smiles",
        ]

    @property
    def split_paths(self):
        r"""The absolute filepaths that must be present in order to skip
        splitting."""
        files = to_list(self.split_file_name)
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_file_names(self):
        return ["old_proc_tr.pt", "old_proc_val.pt", "old_proc_test.pt"]

    def download(self):
        import rdkit  # noqa

        train_path = download_url(self.train_url, self.raw_dir)
        os.rename(train_path, osp.join(self.raw_dir, "guacamol_v1_train.smiles"))
        train_path = osp.join(self.raw_dir, "guacamol_v1_train.smiles")

        test_path = download_url(self.test_url, self.raw_dir)
        os.rename(test_path, osp.join(self.raw_dir, "guacamol_v1_test.smiles"))
        test_path = osp.join(self.raw_dir, "guacamol_v1_test.smiles")

        valid_path = download_url(self.valid_url, self.raw_dir)
        os.rename(valid_path, osp.join(self.raw_dir, "guacamol_v1_valid.smiles"))
        valid_path = osp.join(self.raw_dir, "guacamol_v1_valid.smiles")

        # check the hashes
        # Check whether the md5-hashes of the generated smiles files match
        # the precomputed hashes, this ensures everyone works with the same splits.
        valid_hashes = [
            compare_hash(train_path, TRAIN_HASH),
            compare_hash(valid_path, VALID_HASH),
            compare_hash(test_path, TEST_HASH),
        ]

        if not all(valid_hashes):
            raise SystemExit("Invalid hashes for the dataset files")

        print("Dataset download successful. Hashes are correct.")

        if files_exist(self.split_paths):
            return

    @staticmethod
    def smile_to_graph(smile, pre_transform):
        mol = Chem.MolFromSmiles(smile)
        Chem.SanitizeMol(mol)

        if mol.GetNumBonds() == 0:
            return

        data = mol_to_graph(mol, ATOM_ENCODER)

        reconstruction = graph_to_molecule(
            node_labels=data.atom_labels,
            edge_index=data.edge_index,
            atom_decoder=ATOM_DECODER,
            edge_labels=data.bond_labels,
            explicit_hydrogens=data.explicit_hydrogens,
            charges=data.charges,
            num_radical_electrons=data.radical_electrons,
        )
        # print(smile)
        Chem.SanitizeMol(reconstruction)
        assert are_smiles_equivalent(
            Chem.MolToSmiles(mol, canonical=True),
            Chem.MolToSmiles(reconstruction, canonical=True),
        ), (
            Chem.MolToSmiles(mol, canonical=True),
            Chem.MolToSmiles(reconstruction, canonical=True),
        )

        data.atom_labels = F.one_hot(
            data.atom_labels, num_classes=len(ATOM_ENCODER)
        ).to(torch.float)
        data.bond_labels = F.one_hot(
            data.bond_labels, num_classes=len(BOND_ENCODER)
        ).to(torch.float)
        data.explicit_hydrogens = F.one_hot(data.explicit_hydrogens, num_classes=4).to(
            torch.float
        )
        data.charges = F.one_hot(data.charges + 1, num_classes=5).to(torch.float)
        data.radical_electrons = F.one_hot(data.radical_electrons, num_classes=3).to(
            torch.float
        )

        if pre_transform is not None:
            data = pre_transform(data)
        return data

    def process(self):
        RDLogger.DisableLog("rdApp.*")
        smile_list = open(self.split_paths[self.file_idx]).readlines()

        with tqdm_joblib(
            tqdm(desc="Processing", total=len(smile_list))
        ) as progress_bar:
            data_list = joblib.Parallel(n_jobs=4)(
                joblib.delayed(self.smile_to_graph)(smile.strip(), self.pre_transform)
                for smile in smile_list
            )

        data_list = list(filter(lambda data: data is not None, data_list))
        torch.save(self.collate(data_list), self.processed_paths[self.file_idx])


def install_nx_graphs(split):
    graph_dir = os.path.join(
        Path(__file__).parent.parent.parent, "data", "guacamol_raw_graphs"
    )
    nx_file = os.path.join(graph_dir, f"{split}_nx_graphs.pkl")
    if os.path.exists(nx_file):
        with open(nx_file, "rb") as f:
            return pickle.load(f)
    os.makedirs(graph_dir, exist_ok=True)
    logger.info(f"Creating nx graphs for Guacamol {split} set at {nx_file}...")
    ds = _GuacamolDataset(split, root=graph_dir)
    nx_graphs = []
    for graph in tqdm(ds):
        nx_graphs.append(to_networkx(graph, to_undirected=True))
    with open(nx_file, "wb") as f:
        pickle.dump(nx_graphs, f)
    logger.info(f"Created nx graphs for Guacamol {split} set.")
    return nx_graphs


if __name__ == "__main__":
    install_nx_graphs("train")
    install_nx_graphs("val")
    install_nx_graphs("test")
