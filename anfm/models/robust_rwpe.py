"""Modifications to the RWPE from PyTorch Geometric to make it robust w.r.t. CUDA errors.

Adopted from: https://github.com/pyg-team/pytorch_geometric/blob/aa6cf801ca24d45e8f684025f2045d0ae605102b/torch_geometric/transforms/add_positional_encoding.py

The torch-geometric project is licensed under the MIT license.

Copyright (c) 2023 PyG Team <team@pyg.org>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
from typing import Any, Optional

import torch
import torch_geometric.typing
from loguru import logger
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import (
    get_self_loop_attr,
    is_torch_sparse_tensor,
    scatter,
    to_edge_index,
    to_torch_coo_tensor,
    to_torch_csr_tensor,
)


def add_node_attr(
    data: Data,
    value: Any,
    attr_name: Optional[str] = None,
) -> Data:
    # TODO Move to `BaseTransform`.
    if attr_name is None:
        if data.x is not None:
            x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([x, value.to(x.device, x.dtype)], dim=-1)
        else:
            data.x = value
    else:
        data[attr_name] = value

    return data


class RobustAddRandomWalkPE(BaseTransform):
    r"""Adds the random walk positional encoding from the `"Graph Neural
    Networks with Learnable Structural and Positional Representations"
    <https://arxiv.org/abs/2110.07875>`_ paper to the given graph
    (functional name: :obj:`add_random_walk_pe`).

    Args:
        walk_length (int): The number of random walk steps.
        attr_name (str, optional): The attribute name of the data object to add
            positional encodings to. If set to :obj:`None`, will be
            concatenated to :obj:`data.x`.
            (default: :obj:`"random_walk_pe"`)
    """

    def __init__(
        self,
        walk_length: int,
        attr_name: Optional[str] = "random_walk_pe",
    ) -> None:
        self.walk_length = walk_length
        self.attr_name = attr_name

    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None
        row, col = data.edge_index
        N = data.num_nodes
        assert N is not None

        if data.edge_weight is None:
            value = torch.ones(data.num_edges, device=row.device)
        else:
            value = data.edge_weight
        value = scatter(value, row, dim_size=N, reduce="sum").clamp(min=1)[row]
        value = 1.0 / value

        if N <= 2_000:  # Dense code path for faster computation:
            adj = torch.zeros((N, N), device=row.device)
            adj[row, col] = value
            loop_index = torch.arange(N, device=row.device)
        elif torch_geometric.typing.WITH_WINDOWS:
            adj = to_torch_coo_tensor(data.edge_index, value, size=data.size())
        else:
            adj = to_torch_csr_tensor(data.edge_index, value, size=data.size())

        def get_pe(out: Tensor) -> Tensor:
            if is_torch_sparse_tensor(out):
                return get_self_loop_attr(*to_edge_index(out), num_nodes=N)
            return out[loop_index, loop_index]

        try:
            out = adj
            pe_list = [get_pe(out)]
            for _ in range(self.walk_length - 1):
                out = out @ adj
                pe_list.append(get_pe(out))
        except RuntimeError:
            logger.warning(f"Encountered CUDA exception in RWPE, going to CPU..")
            adj_cpu = adj.cpu()
            out = adj_cpu
            pe_list = [get_pe(out).to(adj.device)]
            for _ in range(self.walk_length - 1):
                out = out @ adj_cpu
                pe_list.append(get_pe(out).to(adj.device))

        pe = torch.stack(pe_list, dim=-1)
        data = add_node_attr(data, pe, attr_name=self.attr_name)

        return data
