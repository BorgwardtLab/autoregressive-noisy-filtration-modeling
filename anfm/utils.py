import filecmp
import json
import os
import shutil
import tarfile
import tempfile

import omegaconf
import torch
from torch_geometric.utils import cumsum, scatter

import wandb
from anfm import CODE_DIR


def dense_to_sparse_adj(dense_adj, mask=None):
    assert dense_adj.ndim == 3 and dense_adj.size(1) == dense_adj.size(2)
    assert (
        mask is None
        or mask.ndim == 2
        and mask.size(1) == dense_adj.size(1)
        and mask.size(0) == dense_adj.size(0)
    )

    flatten_adj = dense_adj.view(-1, dense_adj.size(-1))
    if mask is not None:
        flatten_adj = flatten_adj[mask.view(-1)]
    edge_index = flatten_adj.nonzero().t()
    edge_attr = flatten_adj[edge_index[0], edge_index[1]]

    if mask is None:
        batch = torch.arange(
            dense_adj.size(0), device=dense_adj.device
        ).repeat_interleave(dense_adj.size(1))
        offset = torch.arange(
            start=0,
            end=dense_adj.size(0) * dense_adj.size(2),
            step=dense_adj.size(2),
            device=dense_adj.device,
        )
        offset = offset.repeat_interleave(dense_adj.size(1))
    else:
        batch = torch.arange(dense_adj.size(0), device=dense_adj.device)
        count = mask.sum(dim=-1)
        offset = cumsum(count)[:-1]
        offset = offset.repeat_interleave(count)
        batch = batch.repeat_interleave(count)

        edge_index[1] += offset[edge_index[0]]

    return edge_index, edge_attr, batch


def sparse_to_dense_adj(edge_index, batch, max_num_nodes, edge_attr=None):
    batch_size = int(batch.max()) + 1 if batch.numel() > 0 else 1

    one = batch.new_ones(batch.size(0))
    num_nodes = scatter(one, batch, dim=0, dim_size=batch_size, reduce="sum")
    cum_nodes = cumsum(num_nodes)

    idx0 = batch[edge_index[0]]
    idx1 = edge_index[0] - cum_nodes[batch][edge_index[0]]
    idx2 = edge_index[1] - cum_nodes[batch][edge_index[1]]

    if max_num_nodes is None:
        max_num_nodes = int(num_nodes.max())

    elif (idx1.numel() > 0 and idx1.max() >= max_num_nodes) or (
        idx2.numel() > 0 and idx2.max() >= max_num_nodes
    ):
        mask = (idx1 < max_num_nodes) & (idx2 < max_num_nodes)
        idx0 = idx0[mask]
        idx1 = idx1[mask]
        idx2 = idx2[mask]
        edge_attr = None if edge_attr is None else edge_attr[mask]

    if edge_attr is None:
        edge_attr = torch.ones(idx0.numel(), device=edge_index.device)

    size = [batch_size, max_num_nodes, max_num_nodes]
    size += list(edge_attr.size())[1:]
    flattened_size = batch_size * max_num_nodes * max_num_nodes

    idx = idx0 * max_num_nodes * max_num_nodes + idx1 * max_num_nodes + idx2
    adj = scatter(edge_attr, idx, dim=0, dim_size=flattened_size, reduce="sum")
    adj = adj.view(size)

    node_indices = (
        torch.arange(max_num_nodes, device=adj.device)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )
    mask = node_indices < num_nodes.view(-1, 1)

    return adj, mask


def dense_to_sparse_batch(x, mask=None):
    """Convert a dense batch of node properties to a sparse batch.

    :param x: Tensor of shape bs x num_nodes x num_features
    :param mask: Tensor of shape bs x num_nodes, defaults to None
    """
    assert x.ndim == 3
    assert mask is None or mask.size(0) == x.size(0) and mask.size(1) == x.size(1)

    if mask is None:
        mask = x.new_ones((x.size(0), x.size(1)), dtype=torch.bool)

    num_nodes = torch.sum(mask, dim=1)
    missing_nodes = x.size(1) - num_nodes
    offset = cumsum(missing_nodes)[:-1]
    offset = offset.repeat_interleave(num_nodes)
    node_indices = torch.arange(num_nodes.sum(), device=x.device) + offset

    flat_x = x.reshape(-1, x.size(-1))

    return flat_x[node_indices]


def get_meta_info(output_dir):
    if "meta.json" not in os.listdir(output_dir):
        return {"runid": None, "restarts": 0}

    with open(os.path.join(output_dir, "meta.json"), "r") as f:
        data = json.load(f)
        return data


def save_meta_info(output_dir, data):
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(data, f)


def get_last_ckpt_path(ckpt_dir):
    checkpoints = list(os.listdir(ckpt_dir))
    if len(checkpoints) == 0:
        return None
    ordered_checkpoints = sorted(
        checkpoints, key=lambda fname: int(fname.split(".")[0].split("_")[1])
    )
    return os.path.join(ckpt_dir, ordered_checkpoints[-1])


def ensure_reproducibility(cfg, outdir, restart_idx, assert_code_equality=False):
    """This function saves snapshots of the config and code in the output directory.

    :param cfg: _description_
    :raises RuntimeError: If a config is found in the output directory and it doesn't match
        the current config. Indicates that the run was restarted with a different config.
    :raises RuntimeError: If a code archive is found which does not match the current
        code archive. Indicates modification of code between restarts.
    """
    yaml_config = omegaconf.OmegaConf.to_yaml(cfg, resolve=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        target_path = os.path.join(outdir, "resolved_config.yaml")
        temp_path = os.path.join(tmpdir, "resolved_config.yaml")
        with open(temp_path, "w") as f:
            f.write(yaml_config)
        if os.path.exists(target_path) and not filecmp.cmp(target_path, temp_path):
            raise RuntimeError("Config from previous run differs from current config")
        shutil.move(temp_path, target_path)
        wandb.save(os.path.join(outdir, "resolved_config.yaml"), base_path=outdir)

        target_path = os.path.join(outdir, f"code-{restart_idx}.tar")
        temp_path = os.path.join(tmpdir, "code.tar")
        tar = tarfile.open(temp_path, "w")
        tar.add(
            CODE_DIR,
            filter=lambda x: None if "__pycache__" in x.name else x,
            arcname=os.path.basename(CODE_DIR),
        )
        tar.close()
        initial_code = os.path.join(outdir, "code-1.tar")
        if (
            assert_code_equality
            and os.path.exists(initial_code)
            and not filecmp.cmp(initial_code, temp_path)
        ):
            raise RuntimeError(
                "Code archive from previous run differs from new code archive. Run not reproducible."
            )
        shutil.move(temp_path, target_path)
