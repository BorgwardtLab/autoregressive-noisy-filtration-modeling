from itertools import chain

import torch
from einops import rearrange
from torch_geometric.data import Batch


def filtration_collate_fn(batch, node_ordering_noise, compute_ordering=True):
    """Given a list of pairs (filtration, target), collate into batch."""
    bs = len(batch)
    max_nodes = max(target.shape[-2] for _, target in batch)
    filtration_size, num_edge_types = batch[0][1].shape[0], batch[0][1].shape[-1]
    batched_target = torch.zeros(
        (bs, filtration_size, max_nodes, max_nodes, num_edge_types),
        device=batch[0][1].device,
    )
    graph_data = []
    for i, (filtration, target) in enumerate(batch):
        if compute_ordering:
            node_values = filtration[0].node_ordering_values
            node_values += torch.randn_like(node_values) * node_ordering_noise
            assert node_values.ndim == 1
            ordering = torch.argsort(node_values)
            rank = torch.argsort(ordering).long()
            for data in filtration:
                data.ordering = rank.clone()

        batched_target[i, :, : target.shape[1], : target.shape[1], :] = target
        graph_data.append(filtration)

    compressed_batch = Batch.from_data_list(list(chain(*zip(*graph_data))))
    batched_target = rearrange(batched_target, "b t i j e -> t b i j e")
    return compressed_batch, batched_target
