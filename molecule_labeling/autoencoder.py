from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import global_add_pool

from molecule_labeling.utils import (
    MoleculeDecoder,
    _get_featurizers,
    _get_gps_layers,
    _get_onehot_featurizers,
    flip_vectors,
)

GaussianDistribution = namedtuple("GaussianDistribution", ["mu", "logsigma"])


class LabelAutoencoder(nn.Module):
    def __init__(
        self,
        node_labels,
        edge_labels,
        structural_features,
        hidden_dim=256,
        latent_dim=64,
        encoder_layers=4,
        decoder_layers=4,
        dropout=0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder_structure_featurizers = _get_featurizers(
            structural_features, hidden_dim
        )
        self.node_featurizers = _get_onehot_featurizers(node_labels, hidden_dim)
        self.edge_featurizers = _get_onehot_featurizers(edge_labels, hidden_dim)

        self.encoder_layers = _get_gps_layers(
            encoder_layers, hidden_dim, edge_attributed=True
        )
        self.encoder_dropout = nn.Dropout(dropout)
        self.encoder_mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.encoder_logsigma_layer = nn.Linear(hidden_dim, latent_dim)

        self.decoder = MoleculeDecoder(
            node_labels,
            edge_labels,
            structural_features,
            hidden_dim=hidden_dim,
            layers=decoder_layers,
            latent_dim=latent_dim,
            dropout=dropout,
        )

    def encode_prior(self, batch):
        return GaussianDistribution(0, 1)

    def encode_posterior(self, batch):
        x = 0
        for name, model in self.node_featurizers.items():
            x += model(getattr(batch, name))

        for name, model in self.encoder_structure_featurizers.items():
            feat = getattr(batch, name)
            if name == "laplace":
                feat = flip_vectors(feat, batch.num_graphs, batch.batch)
            x += model(feat)

        edge_attr = 0
        for name, model in self.edge_featurizers.items():
            edge_attr += model(getattr(batch, name))

        for layer in self.encoder_layers:
            x = layer(
                x=x, edge_index=batch.edge_index, batch=batch.batch, edge_attr=edge_attr
            )
            x = self.encoder_dropout(x)

        return GaussianDistribution(
            self.encoder_mu_layer(x), self.encoder_logsigma_layer(x)
        )

    def encode(self, batch):
        return self.encode_prior(batch), self.encode_posterior(batch)

    def decode(self, node_latents, structure_batch):
        return self.decoder(structure_batch, node_latents)

    def forward(self, batch):
        prior, posterior = self.encode(batch)
        mu, logsigma = posterior
        samples = logsigma.exp() * torch.randn_like(mu) + mu
        node_logits, edge_logits = self.decode(samples, batch)
        return {
            "prior": prior,
            "posterior": posterior,
            "node_logits": node_logits,
            "edge_logits": edge_logits,
        }

    @staticmethod
    def sample_from_logits(logits):
        return {
            name: F.one_hot(
                torch.multinomial(F.softmax(logit, dim=-1), num_samples=1)[:, 0],
                logit.shape[-1],
            )
            for name, logit in logits.items()
        }

    @staticmethod
    def hard_max_logits(logits):
        return {
            name: F.one_hot(torch.argmax(logit, -1), logit.shape[-1])
            for name, logit in logits.items()
        }

    def sample(self, structure_batch, variant="sample"):
        latents = torch.randn((structure_batch.num_nodes, self.latent_dim)).to(
            structure_batch.edge_index.device
        )
        node_logits, edge_logits = self.decode(latents, structure_batch)
        if variant == "sample":
            node_samples = self.sample_from_logits(node_logits)
            edge_samples = self.sample_from_logits(edge_logits)
        elif variant == "max":
            node_samples = self.hard_max_logits(node_logits)
            edge_samples = self.hard_max_logits(edge_logits)
        return node_samples, edge_samples

    @classmethod
    def evidence_lower_bound(cls, batch, results, normalize_by_size=True):
        mu, logsigma = results["posterior"]
        kl_div = 0.5 * (logsigma.exp() ** 2 + mu**2 - 1 - 2 * logsigma).sum(-1)
        kl_div = global_add_pool(kl_div, batch.batch)
        graph_log_likelihoods = MoleculeDecoder.graph_log_prob(
            results["node_logits"], results["edge_logits"], batch
        )
        assert (
            graph_log_likelihoods.shape == kl_div.shape
        ), "Graph log likelihoods must be the same shape as the number of graphs"
        if normalize_by_size:
            nodes_per_graph = batch.batch.bincount()
            assert (nodes_per_graph > 0).all(), "No nodes in some graphs"
            assert (
                nodes_per_graph.shape == kl_div.shape
            ), "Nodes per graph must be the same shape as the number of graphs"
            graph_log_likelihoods = (graph_log_likelihoods / nodes_per_graph).sum()
            kl_div = (kl_div / nodes_per_graph).sum()
        else:
            graph_log_likelihoods = graph_log_likelihoods.sum()
            kl_div = kl_div.sum()
        return (kl_div - graph_log_likelihoods) / batch.num_graphs


class DiscreteLabelAutoencoder(nn.Module):
    def __init__(
        self,
        node_labels,
        edge_labels,
        structural_features,
        hidden_dim=256,
        latent_dim=64,
        encoder_layers=4,
        decoder_layers=4,
        hard=False,
        temperature=1.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder_structure_featurizers = _get_featurizers(
            structural_features, hidden_dim
        )
        self.prior_structure_featurizers = _get_featurizers(
            structural_features, hidden_dim
        )
        self.node_featurizers = _get_onehot_featurizers(node_labels, hidden_dim)
        self.edge_featurizers = _get_onehot_featurizers(edge_labels, hidden_dim)

        self.encoder_layers = _get_gps_layers(
            encoder_layers, hidden_dim, edge_attributed=True
        )
        self.prior_model_layers = _get_gps_layers(
            encoder_layers, hidden_dim, edge_attributed=False
        )

        self.encoder_final = nn.Linear(hidden_dim, latent_dim)
        self.prior_model_final = nn.Linear(hidden_dim, latent_dim)

        self.latent_embedding = nn.Linear(latent_dim, latent_dim)
        self.decoder = MoleculeDecoder(
            node_labels,
            edge_labels,
            structural_features,
            hidden_dim=hidden_dim,
            layers=decoder_layers,
            latent_dim=latent_dim,
        )
        self.hard = hard
        self.temperature = temperature

    def encode_prior(self, batch):
        x = 0
        for name, model in self.prior_structure_featurizers.items():
            feat = getattr(batch, name)
            if name == "laplace":
                feat = flip_vectors(feat, batch.num_graphs, batch.batch)
            x += model(feat)
        for layer in self.prior_model_layers:
            x = layer(x=x, edge_index=batch.edge_index, batch=batch.batch)
        return F.log_softmax(self.prior_model_final(x), dim=-1)

    def encode_posterior(self, batch):
        x = 0
        for name, model in self.node_featurizers.items():
            x += model(getattr(batch, name))

        for name, model in self.encoder_structure_featurizers.items():
            feat = getattr(batch, name)
            if name == "laplace":
                feat = flip_vectors(feat, batch.num_graphs, batch.batch)
            x += model(feat)

        edge_attr = 0
        for name, model in self.edge_featurizers.items():
            edge_attr += model(getattr(batch, name))

        for layer in self.encoder_layers:
            x = layer(
                x=x, edge_index=batch.edge_index, batch=batch.batch, edge_attr=edge_attr
            )

        return F.log_softmax(self.encoder_final(x), dim=-1)

    def encode(self, batch):
        return self.encode_prior(batch), self.encode_posterior(batch)

    def decode(self, node_latents, structure_batch):
        return self.decoder(structure_batch, node_latents)

    def forward(self, batch):
        prior, posterior = self.encode(batch)
        # Sample from the categorical posterior
        latents = self.latent_embedding(
            F.gumbel_softmax(posterior, hard=self.hard, dim=-1, tau=self.temperature)
        )
        node_logits, edge_logits = self.decode(latents, batch)
        return {
            "prior": prior,
            "posterior": posterior,
            "node_logits": node_logits,
            "edge_logits": edge_logits,
        }

    @staticmethod
    def sample_from_logits(logits):
        return {
            name: F.one_hot(
                torch.multinomial(F.softmax(logit, dim=-1), num_samples=1)[:, 0],
                logit.shape[-1],
            )
            for name, logit in logits.items()
        }

    @staticmethod
    def hard_max_logits(logits):
        return {
            name: F.one_hot(torch.argmax(logit, -1), logit.shape[-1])
            for name, logit in logits.items()
        }

    def sample(self, structure_batch, variant="sample"):
        prior = self.encode_prior(structure_batch)
        latents = self.latent_embedding(
            F.gumbel_softmax(prior, hard=self.hard, dim=-1, tau=self.temperature)
        )

        node_logits, edge_logits = self.decode(latents, structure_batch)
        if variant == "sample":
            node_samples = self.sample_from_logits(node_logits)
            edge_samples = self.sample_from_logits(edge_logits)
        elif variant == "max":
            node_samples = self.hard_max_logits(node_logits)
            edge_samples = self.hard_max_logits(edge_logits)
        return node_samples, edge_samples

    @classmethod
    def evidence_lower_bound(cls, batch, results, normalize_by_size=True):
        prior, posterior = results["prior"], results["posterior"]
        assert prior.ndim == 2 and posterior.ndim == 2, (prior.shape, posterior.shape)
        # kl divergence in torch has arguments in opposite order
        kl_div = global_add_pool(
            F.kl_div(prior, posterior, log_target=True, reduction="none").sum(-1),
            batch.batch,
        )
        assert kl_div.ndim == 1 and len(kl_div) == batch.num_graphs, (
            kl_div.shape,
            batch.num_graphs,
        )
        graph_log_likelihoods = MoleculeDecoder.graph_log_prob(
            results["node_logits"], results["edge_logits"], batch
        )
        assert graph_log_likelihoods.shape == kl_div.shape, graph_log_likelihoods.shape
        if normalize_by_size:
            nodes_per_graph = batch.batch.bincount()
            assert (nodes_per_graph > 0).all(), "No nodes in some graphs"
            assert (
                nodes_per_graph.shape == kl_div.shape
            ), "Nodes per graph must be the same shape as the number of graphs"
            graph_log_likelihoods = (graph_log_likelihoods / nodes_per_graph).sum()
            kl_div = (kl_div / nodes_per_graph).sum()
        else:
            graph_log_likelihoods = graph_log_likelihoods.sum()
            kl_div = kl_div.sum()
        return (kl_div - graph_log_likelihoods) / batch.num_graphs
