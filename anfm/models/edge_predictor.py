import warnings

import numpy as np
import torch
import torch.distributions as D
import torch.nn.functional as F
from einops import einsum, rearrange
from torch import nn


def sample_edges(logits, mixture_logits, mask):
    """For a single step in the filtration, sample edges

    :param logits: Tensor of shape (*, num components, num nodes, num nodes, num edge types)
    :param mixture_logits: Tensor of shape (*, num components)
    :param mask: Tensor of shape (*, number of nodes)
    :return: Tensor of shape (*, num nodes, num nodes, num edge types)
    """
    mixture = D.categorical.Categorical(logits=mixture_logits)
    components = D.independent.Independent(D.Categorical(logits=logits), 2)
    dist = D.mixture_same_family.MixtureSameFamily(mixture, components)
    sample = torch.triu(dist.sample(), diagonal=1)
    sample = sample + sample.transpose(-1, -2)
    sample = F.one_hot(sample, num_classes=logits.shape[-1])
    sample = torch.where(mask.unsqueeze(-1).unsqueeze(-1), sample, 0)
    sample = torch.where(mask.unsqueeze(-2).unsqueeze(-1), sample, 0)
    return sample


def entropy_lower_bound(mixture_logits, logits, mask):
    """
    mixture_logits is of shape (t, *, num components)
    logits is of shape (t, *, num components, num nodes, num nodes, num edge types)
    mask is of shape (t, *, num nodes)
    """
    (
        t,
        *star_shape,
        num_components,
        num_nodes,
        num_nodes_2,
        num_edge_types,
    ) = logits.shape
    assert num_nodes == num_nodes_2
    assert mask.shape == (t, *star_shape, num_nodes)
    assert mixture_logits.shape == (t, *star_shape, num_components)

    logit_mask = (
        (mask.unsqueeze(-1) & mask.unsqueeze(-2))
        .unsqueeze(-1)
        .unsqueeze(-4)
        .expand_as(logits)
    )
    per_component_entropy = -torch.sum(
        (logits.exp() * logits).masked_fill(~logit_mask, 0), dim=(-1, -2, -3)
    )
    lower_bound = torch.sum(mixture_logits.exp() * per_component_entropy, dim=-1)
    return lower_bound


def entropy_upper_bound(mixture_logits, logits, mask):
    (
        t,
        *star_shape,
        num_components,
        num_nodes,
        num_nodes_2,
        num_edge_types,
    ) = logits.shape
    assert num_nodes == num_nodes_2
    assert mask.shape == (t, *star_shape, num_nodes)
    assert mixture_logits.shape == (t, *star_shape, num_components)

    lower_bound = entropy_lower_bound(mixture_logits, logits, mask)
    upper_bound = lower_bound - torch.sum(mixture_logits.exp() * mixture_logits, dim=-1)
    return upper_bound


def cross_entropy(
    mixture_logits,
    logits,
    mask,
    y,
    n_samples=16,
    reduce_batch=True,
    reduce_filtration=True,
    hard_labels=False,
):
    """
    mixture_logits is of shape (t, *, num components)
    logits is of shape (t, *, num components, num nodes, num nodes, num edge types)
    y is of shape (t, *, num nodes, num nodes, num edge types) ?
    mask is of shape (t, *, num nodes)
    """
    (
        t,
        *star_shape,
        num_components,
        num_nodes,
        num_nodes_2,
        num_edge_types,
    ) = logits.shape
    assert num_nodes == num_nodes_2
    t_mix, *star_shape_mix, num_components_mix = mixture_logits.shape
    assert (
        t == t_mix
        and star_shape == star_shape_mix
        and num_components == num_components_mix
    )
    ty, *star_shape_y, num_nodes_y, num_nodes_y_2, num_edge_types_y = y.shape
    assert (
        t == ty
        and star_shape == star_shape_y
        and num_nodes_y == num_nodes_y_2
        and num_edge_types == num_edge_types_y
    ), (logits.shape, y.shape)
    assert mask.shape == (t, *star_shape, num_nodes)

    if num_nodes != num_nodes_y:
        # In this case, there is some unnecessary padding in y
        assert num_nodes_y > num_nodes
        warnings.warn(
            f"Unneccessary padding in target of cross entropy: {num_nodes} != {num_nodes_y}"
        )
        y = y[..., :num_nodes, :num_nodes, :]

    if num_components > 1 and not hard_labels:
        expanded_mask = (
            (mask.unsqueeze(-1) & mask.unsqueeze(-2)).unsqueeze(-1).expand_as(y)
        )

        # We need to fill the padded entries of y with a virtual edge that satisfies the simplex constraint for categorical sampling
        no_edge = torch.zeros(num_edge_types, device=y.device, dtype=y.dtype)
        no_edge[0] = 1
        y_filled = torch.where(expanded_mask, y, no_edge)

        data_dist = D.independent.Independent(D.Categorical(probs=y_filled), 2)
        samples = data_dist.sample_n(n_samples)
        logits_shape = logits.shape
        logits = logits.unsqueeze(0).expand(n_samples, *logits_shape)
        samples = samples.unsqueeze(-3).unsqueeze(-1).expand_as(logits)
        per_edge_log_likelihood = torch.gather(
            logits,
            dim=-1,
            index=samples,
        )[..., 0]
        # per_edge_log_likelihood is of shape (n_samples, t, *, num_components, num nodes, num nodes)
        expanded_mask = mask.unsqueeze(0)
        expanded_mask = expanded_mask.unsqueeze(-2)
        expanded_mask = expanded_mask.unsqueeze(-1) * expanded_mask.unsqueeze(-2)
        expanded_mask = expanded_mask.expand_as(per_edge_log_likelihood)
        assert expanded_mask.shape == per_edge_log_likelihood.shape

        per_component_log_likelihood = torch.sum(
            torch.triu(per_edge_log_likelihood * expanded_mask, diagonal=1),
            dim=(-1, -2),
        )
        # per_component_log_likelihood is of shape (n_samples, t, *, num_components)
        log_likelihood = torch.logsumexp(
            per_component_log_likelihood + mixture_logits.unsqueeze(0), dim=-1
        )
        log_likelihood = torch.mean(log_likelihood, dim=0)
    else:
        # mixture_logits is of shape (t, *, num components)
        # logits is of shape (t, *, num components, num nodes, num nodes, num edge types)
        expanded_mask = mask.unsqueeze(-1) * mask.unsqueeze(
            -2
        )  # Shape (t, *, num nodes, num nodes)
        expanded_mask = expanded_mask.unsqueeze(-1).unsqueeze(
            -4
        )  # Shape (t, *, 1, num nodes, num nodes, 1)
        expanded_mask = expanded_mask.expand_as(logits)
        per_component_log_likelihood = torch.sum(
            torch.triu(
                torch.sum(y.unsqueeze(-4) * logits * expanded_mask, dim=-1), diagonal=1
            ),
            dim=(-1, -2),
        )
        # per_component_likelihood is of shape (t, *, num_components)
        assert per_component_log_likelihood.shape == (t, *star_shape, num_components)
        log_likelihood = torch.logsumexp(
            per_component_log_likelihood + mixture_logits, dim=-1
        )

    # likelihood is of shape (t, *)
    assert log_likelihood.shape == (t, *star_shape)
    if reduce_batch and reduce_filtration:
        return -torch.mean(log_likelihood)
    elif reduce_filtration and not reduce_batch:
        return -torch.mean(log_likelihood, dim=0)
    elif reduce_batch and not reduce_filtration:
        return -torch.mean(log_likelihood, dim=list(range(1, log_likelihood.ndim)))
    return -log_likelihood


class EdgePredictor(nn.Module):
    def __init__(
        self,
        num_edge_types,
        num_mixture_components,
        embedding_dim,
        dot_product_dim,
        num_hidden=1,
    ):
        super().__init__()
        self.num_edge_types = num_edge_types
        self.num_mixture_components = num_mixture_components
        out_dim = dot_product_dim * num_edge_types
        node_mlps = []
        for _ in range(num_mixture_components):
            layers = [nn.Linear(embedding_dim, out_dim), nn.ReLU()]
            for _ in range(num_hidden):
                layers.extend([nn.Linear(out_dim, out_dim), nn.ReLU()])
            node_mlps.append(nn.Sequential(*layers))
        self.node_mlp = nn.ModuleList(node_mlps)

        self.bilinear_form = nn.ModuleList(
            [nn.Linear(out_dim, out_dim) for _ in range(num_mixture_components)]
        )
        if num_mixture_components > 1:
            self.mixture_node_mlp = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim), nn.ReLU()
            )
            self.mixture_graph_mlp = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, num_mixture_components),
            )
        else:
            self.mixture_node_mlp = None
            self.mixture_graph_mlp = None

    def forward(self, x, mask, prev_adj=None):
        """

        :param x: Tensor of shape (*, number of nodes, dim)
        :param mask: Tensor of shape (*, number of nodes)
        :param prev_adj: One-hot tensor
        :return:
        """
        features = [self.node_mlp[i](x) for i in range(self.num_mixture_components)]
        linearly_transformed = [
            self.bilinear_form[i](features[i])
            for i in range(self.num_mixture_components)
        ]
        features = torch.stack(features, dim=-2)
        features = rearrange(
            features,
            "... n m (e d) -> ... n m e d",
            e=self.num_edge_types,
        )
        linearly_transformed = torch.stack(linearly_transformed, dim=-2)
        linearly_transformed = rearrange(
            linearly_transformed,
            "... n m (e d) -> ... n m e d",
            e=self.num_edge_types,
        )
        logits = einsum(
            features,
            linearly_transformed,
            "... i comp type d, ... j comp type d -> ... comp i j type",
        )
        logits = (logits + logits.transpose(-2, -3)) / 2
        if prev_adj is not None:
            logits = logits + prev_adj.unsqueeze(-4)
        logits = torch.log_softmax(logits, dim=-1)

        if self.num_mixture_components > 1:
            # Shape (filtration size, batch size, dim)
            graph_representation = torch.sum(
                self.mixture_node_mlp(x) * mask.unsqueeze(-1), dim=-2
            ) / torch.sum(mask, dim=-1).unsqueeze(-1)
            mixture_logits = self.mixture_graph_mlp(graph_representation)
            mixture_logits = torch.log_softmax(mixture_logits, dim=-1)
        else:
            mixture_logits = torch.zeros((*x.shape[:-2], 1), device=x.device)
        return mixture_logits, logits
