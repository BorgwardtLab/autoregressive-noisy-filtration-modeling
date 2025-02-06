import numpy as np
import torch
import torch.nn.functional as F
from fcd import calculate_frechet_distance, get_predictions, load_ref_model
from rdkit import Chem
from torch import nn
from torch_geometric.nn.conv import GINConv, GINEConv, GPSConv
from torch_geometric.nn.pool import global_add_pool

from molecule_labeling.batch_renorm import BatchRenorm1d
from molecule_labeling.dataset import GuacamolDataset


class FCDMetric:
    def __init__(self, ref_molecules):
        self.ref_model = load_ref_model()
        ref_smiles = [Chem.MolToSmiles(mol, canonical=True) for mol in ref_molecules]
        ref_predictions = get_predictions(self.ref_model, ref_smiles)
        self.mu_ref, self.sigma_ref = np.mean(ref_predictions, axis=0), np.cov(
            ref_predictions.T
        )

    def __call__(self, molecules):
        valid_mols = []
        for mol in molecules:
            try:
                Chem.SanitizeMol(mol)
                valid_mols.append(mol)
            except:
                pass
        smiles = [Chem.MolToSmiles(mol, canonical=True) for mol in valid_mols]
        predictions = get_predictions(self.ref_model, smiles)
        mu_pred, sigma_pred = np.mean(predictions, axis=0), np.cov(predictions.T)
        return np.exp(
            -0.2
            * calculate_frechet_distance(
                mu1=self.mu_ref, sigma1=self.sigma_ref, mu2=mu_pred, sigma2=sigma_pred
            )
        )


def count_valid_molecules(structure_batch, attributes, reference_mols=None):
    for name, val in attributes.items():
        setattr(structure_batch, name, val)
    sampled = GuacamolDataset.batch_to_mols(structure_batch)
    assert len(sampled) == structure_batch.num_graphs
    num_valid = 0
    num_correct = 0
    for i, mol in enumerate(sampled):
        if reference_mols is not None:
            Chem.SanitizeMol(reference_mols[i])
            reference_smiles = Chem.MolToSmiles(reference_mols[i], canonical=True)
        else:
            reference_smiles = None
        try:
            Chem.SanitizeMol(mol)
            num_valid += 1
            num_correct += Chem.MolToSmiles(mol, canonical=True) == reference_smiles
        except:
            pass
    return sampled, num_valid, num_correct


def _get_featurizers(feature_dict, hidden_dim):
    featurizers = nn.ModuleDict()
    for feature_name, dim in feature_dict.items():
        mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        norm = BatchRenorm1d(dim)
        ln = nn.LayerNorm(hidden_dim)
        featurizers[feature_name] = nn.Sequential(norm, mlp, ln)
    return nn.ModuleDict(featurizers)


def _get_onehot_featurizers(feature_dict, hidden_dim):
    featurizers = nn.ModuleDict()
    for feature_name, dim in feature_dict.items():
        mlp = nn.Linear(dim, hidden_dim)
        ln = nn.LayerNorm(hidden_dim)
        featurizers[feature_name] = nn.Sequential(mlp, ln)
    return nn.ModuleDict(featurizers)


def flip_vectors(vecs, num_graphs, batch):
    assert vecs.ndim == 2
    signs = (
        2 * torch.randint(0, 2, size=(num_graphs, vecs.shape[-1])).to(vecs.device) - 1
    )
    sign_per_node = signs[batch]
    assert sign_per_node.shape == vecs.shape
    return vecs * sign_per_node


def _get_gps_layers(num_layers, hidden_dim, edge_attributed=False):
    layers = []
    for idx in range(num_layers):
        mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        conv = GPSConv(
            hidden_dim,
            GINEConv(mlp) if edge_attributed else GINConv(mlp),
            heads=4,
            attn_type="multihead",
        )
        layers.append(conv)
    return nn.ModuleList(layers)


class MoleculeDecoder(nn.Module):
    def __init__(
        self,
        node_labels,
        edge_labels,
        structural_features,
        hidden_dim=256,
        layers=4,
        latent_dim=64,
        dropout=0.0,
    ):
        super().__init__()
        self.structure_featurizer = _get_featurizers(structural_features, hidden_dim)
        self.noise_proj = nn.Linear(latent_dim, hidden_dim)
        self.decoder_layers = _get_gps_layers(layers, hidden_dim, edge_attributed=False)
        self.decoder_edge_fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.decoder_node_heads = nn.ModuleDict(
            {feat: nn.Linear(hidden_dim, dim) for feat, dim in node_labels.items()}
        )
        self.decoder_edge_heads = nn.ModuleDict(
            {feat: nn.Linear(hidden_dim, dim) for feat, dim in edge_labels.items()}
        )
        self.latent_dim = latent_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, structure_batch, latents=None):
        x = 0
        for name, model in self.structure_featurizer.items():
            feat = getattr(structure_batch, name)
            if name == "laplace" and self.training:
                feat = flip_vectors(
                    feat, structure_batch.num_graphs, structure_batch.batch
                )
            x = x + model(feat)

        if latents is None:
            latents = torch.randn(
                (
                    x.shape[0],
                    self.latent_dim,
                )
            ).to(structure_batch.edge_index.device)
        x = x + self.noise_proj(latents)

        for layer in self.decoder_layers:
            x = layer(
                x=x, edge_index=structure_batch.edge_index, batch=structure_batch.batch
            )
            x = self.dropout(x)

        edge_activations = self.decoder_edge_fusion(
            x[structure_batch.edge_index[0]] + x[structure_batch.edge_index[1]]
        )
        node_logits = {
            name: F.log_softmax(head(x), dim=-1)
            for name, head in self.decoder_node_heads.items()
        }
        edge_logits = {
            name: F.log_softmax(head(edge_activations), dim=-1)
            for name, head in self.decoder_edge_heads.items()
        }
        return node_logits, edge_logits

    @staticmethod
    def graph_log_prob(node_logits, edge_logits, batch):
        mask = batch.edge_index[0] < batch.edge_index[1]
        masked_edge_index_batch = batch.batch[batch.edge_index[0, mask]]
        edge_samples = {
            key: torch.where(
                mask.unsqueeze(-1),
                getattr(batch, key),
                getattr(batch, key)[batch.to_antiparallel],
            )
            for key in edge_logits
        }
        node_samples = {key: getattr(batch, key) for key in node_logits}
        node_log_likelihoods = {
            key: global_add_pool(
                (node_samples[key] * node_logits[key]).sum(-1), batch.batch
            )
            for key in node_samples
        }
        edge_log_likelihoods = {
            key: global_add_pool(
                (edge_samples[key] * edge_logits[key]).sum(-1)[mask],
                masked_edge_index_batch,
            )
            for key in edge_samples
        }
        graph_log_likelihoods = sum(node_log_likelihoods.values()) + sum(
            edge_log_likelihoods.values()
        )
        return graph_log_likelihoods
