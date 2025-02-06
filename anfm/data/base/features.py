try:
    import graph_tool.all as _  # isort: skip
except ImportError:
    pass
import networkx as nx
import numpy as np
import torch

from anfm.utils import dense_to_sparse_batch, sparse_to_dense_adj

torch.set_num_threads(1)
torch.set_num_interop_threads(24)


@torch.jit.script
def mt_eigh(x: torch.Tensor):
    futs = [torch.jit._fork(torch.linalg.eigh, x[i]) for i in range(16)]
    return [torch.jit._wait(fut) for fut in futs]


def fast_eigh(x):
    dev = x.device
    assert x.ndim == 3
    if x.size(0) % 16 != 0:
        raise ValueError("Batch size must be divisible by 16")
    x = x.view((16, -1, *x.shape[1:]))
    result = mt_eigh(x.cpu())
    eigvals = torch.cat([r[0] for r in result], dim=0).to(dev)
    eigvecs = torch.cat([r[1] for r in result], dim=0).to(dev)
    return eigvals, eigvecs


def laplacian_matrix(edge_index, max_num_nodes, batch):
    dense_adj, mask = sparse_to_dense_adj(edge_index, batch, max_num_nodes)
    assert dense_adj.size(-1) == max_num_nodes
    degrees = dense_adj.sum(dim=-1, keepdim=True)
    degrees = torch.where(degrees > 0, degrees, 0.01 * torch.ones_like(degrees))
    normalizing_mat = (degrees**-0.5) * (degrees**-0.5).transpose(1, 2)
    normalized_laplacian = (
        torch.eye(max_num_nodes, device=normalizing_mat.device).unsqueeze(0)
        - normalizing_mat * dense_adj
    )
    return normalized_laplacian, mask


def component_laplacian(nx_graph, k, normalization="unit"):
    if normalization not in ("unit", "sqrt"):
        raise ValueError("Unknown normalization")

    eigenvector_encoding = np.zeros((nx_graph.number_of_nodes(), k))
    eigenvalue_encoding = np.zeros((nx_graph.number_of_nodes(), k))
    component_index = np.zeros(nx_graph.number_of_nodes())
    num_components = 0

    node_list = sorted(list(nx_graph.nodes))
    whole_laplacian = nx.normalized_laplacian_matrix(
        nx_graph, nodelist=node_list
    ).toarray()

    for idx, node_set in enumerate(nx.connected_components(nx_graph)):
        node_set = sorted(list(node_set))
        component_index[node_set] = idx
        # induced = nx.induced_subgraph(nx_graph, node_set)
        # laplacian = nx.normalized_laplacian_matrix(induced).toarray()
        laplacian = whole_laplacian[node_set, :][:, node_set]
        eig_vals, eig_vecs = np.linalg.eigh(laplacian)
        # we randomly flip the eigenvectors
        eig_vecs = eig_vecs * (
            2 * np.random.randint(0, 2, size=(1, eig_vecs.shape[1])) - 1
        )
        if normalization == "sqrt":
            eig_vecs *= np.sqrt(len(node_set))
        if eig_vecs.shape[1] < k + 1:
            eig_vecs = np.pad(
                eig_vecs, [(0, 0), (0, k + 1 - eig_vecs.shape[1])], mode="constant"
            )
            eig_vals = np.pad(
                eig_vals, [(0, k + 1 - eig_vals.shape[0])], mode="constant"
            )
        eigenvector_encoding[node_set] = eig_vecs[:, 1 : k + 1]
        eigenvalue_encoding[node_set] = eig_vals[1 : k + 1]
        num_components += 1

    return eigenvector_encoding, eigenvalue_encoding, component_index, num_components


def global_laplacian(
    edge_index, max_num_nodes, k, batch=None, fast=False, return_dense=False
):
    if batch is None:
        batch = torch.zeros(max_num_nodes, dtype=torch.long)
    laplacian_torch, mask = laplacian_matrix(edge_index, max_num_nodes, batch)
    assert laplacian_torch.ndim == 3
    bs, n, _ = laplacian_torch.shape
    if fast:
        eig_vals, eig_vecs = fast_eigh(laplacian_torch)
    else:
        eig_vals, eig_vecs = torch.linalg.eigh(laplacian_torch)
    assert (eig_vals > -1e-5).all(), eig_vals
    # we randomly flip the eigenvectors
    eig_vecs = eig_vecs * (
        2
        * torch.randint(
            0,
            2,
            size=(
                *laplacian_torch.shape[:-2],
                1,
                eig_vecs.shape[-1],
            ),
            device=eig_vecs.device,
        )
        - 1
    )
    assert eig_vecs.shape[1] >= k + 1
    n_connected_components = (eig_vals < 1e-5).sum(dim=-1)
    # assert (n_connected_components > 0).all(), eig_vals
    to_extend = max(n_connected_components) + k - n
    if to_extend > 0:
        eig_vals = torch.hstack(
            (eig_vals, 2 * torch.ones(bs, to_extend).type_as(eig_vals))
        )
    indices = torch.arange(k, device=eig_vals.device).type_as(
        eig_vals
    ).long().unsqueeze(0) + n_connected_components.unsqueeze(1)
    first_k_eig_vals = torch.gather(eig_vals, dim=1, index=indices)

    n_connected_components = n_connected_components.unsqueeze(-1)
    # Get the eigenvectors corresponding to the first nonzero eigenvalues
    if to_extend > 0:
        vectors = torch.cat(
            (eig_vecs, torch.zeros(bs, n, to_extend).type_as(eig_vecs)), dim=2
        )  # bs, n , n + to_extend
    indices = torch.arange(k, device=eig_vecs.device).type_as(
        eig_vecs
    ).long().unsqueeze(0).unsqueeze(0) + n_connected_components.unsqueeze(
        2
    )  # bs, 1, k
    indices = indices.expand(-1, n, -1)  # bs, n, k
    try:
        first_k_eig_vecs = torch.gather(eig_vecs, dim=2, index=indices)  # bs, n, k
    except Exception as e:
        assert to_extend > 0, (eig_vecs.shape, indices.shape)
        first_k_eig_vecs = torch.gather(vectors, dim=2, index=indices)

    first_k_eig_vals = first_k_eig_vals.unsqueeze(1).expand(-1, n, -1)
    if return_dense:
        return first_k_eig_vecs, first_k_eig_vals
    sparse_eigenvecs = dense_to_sparse_batch(first_k_eig_vecs, mask)
    sparse_eigenvals = dense_to_sparse_batch(first_k_eig_vals, mask)
    return sparse_eigenvecs, sparse_eigenvals


def normalized_fiedler_vector(nx_graph):
    node_set = sorted(list(nx_graph.nodes))
    whole_laplacian = nx.normalized_laplacian_matrix(
        nx_graph, nodelist=node_set
    ).toarray()
    _, eig_vecs = np.linalg.eigh(whole_laplacian)
    fiedler = eig_vecs[:, 1]
    fiedler = fiedler * (2 * np.random.randint(0, 2, size=(1,)) - 1)
    fiedler = fiedler / np.std(fiedler)
    return fiedler


def batch_trace(X):
    """
    Expect a matrix of shape B N N, returns the trace in shape B
    :param X:
    :return:
    """
    diag = torch.diagonal(X, dim1=-2, dim2=-1)
    trace = diag.sum(dim=-1)
    return trace


def batch_diagonal(X):
    """
    Extracts the diagonal from the last two dims of a tensor
    :param X:
    :return:
    """
    return torch.diagonal(X, dim1=-2, dim2=-1)


class KNodeCycles:
    """Builds cycle counts for each node in a graph."""

    def __init__(self):
        super().__init__()

    def calculate_kpowers(self):
        self.k1_matrix = self.adj_matrix.double()
        self.d = self.adj_matrix.sum(dim=-1)
        self.k2_matrix = self.k1_matrix @ self.adj_matrix.double()
        self.k3_matrix = self.k2_matrix @ self.adj_matrix.double()
        self.k4_matrix = self.k3_matrix @ self.adj_matrix.double()
        self.k5_matrix = self.k4_matrix @ self.adj_matrix.double()
        self.k6_matrix = self.k5_matrix @ self.adj_matrix.double()

    def k3_cycle(self):
        """tr(A ** 3)."""
        c3 = batch_diagonal(self.k3_matrix)
        return (c3 / 2).unsqueeze(-1).float(), (torch.sum(c3, dim=-1) / 6).unsqueeze(
            -1
        ).float()

    def k4_cycle(self):
        diag_a4 = batch_diagonal(self.k4_matrix)
        c4 = (
            diag_a4
            - self.d * (self.d - 1)
            - (self.adj_matrix @ self.d.unsqueeze(-1)).sum(dim=-1)
        )
        return (c4 / 2).unsqueeze(-1).float(), (torch.sum(c4, dim=-1) / 8).unsqueeze(
            -1
        ).float()

    def k5_cycle(self):
        """Only global count, from Frank Harary; Bennet Manvel: On the number of cycles in a graph"""
        result = (
            batch_trace(self.k5_matrix)
            - 5 * batch_trace(self.k3_matrix)
            - 5
            * ((self.adj_matrix - 2).sum(-1) * batch_diagonal(self.k3_matrix)).sum(-1)
        )
        return None, (result / 10).unsqueeze(-1).float()

    def k6_cycle(self):
        term_1_t = batch_trace(self.k6_matrix)
        term_2_t = batch_trace(self.k3_matrix**2)
        term3_t = torch.sum(self.adj_matrix * self.k2_matrix.pow(2), dim=[-2, -1])
        d_t4 = batch_diagonal(self.k2_matrix)
        a_4_t = batch_diagonal(self.k4_matrix)
        term_4_t = (d_t4 * a_4_t).sum(dim=-1)
        term_5_t = batch_trace(self.k4_matrix)
        term_6_t = batch_trace(self.k3_matrix)
        term_7_t = batch_diagonal(self.k2_matrix).pow(3).sum(-1)
        term8_t = torch.sum(self.k3_matrix, dim=[-2, -1])
        term9_t = batch_diagonal(self.k2_matrix).pow(2).sum(-1)
        term10_t = batch_trace(self.k2_matrix)

        c6_t = (
            term_1_t
            - 3 * term_2_t
            + 9 * term3_t
            - 6 * term_4_t
            + 6 * term_5_t
            - 4 * term_6_t
            + 4 * term_7_t
            + 3 * term8_t
            - 12 * term9_t
            + 4 * term10_t
        )
        return None, (c6_t / 12).unsqueeze(-1).float()

    def k_cycles(self, adj_matrix, verbose=False):
        assert adj_matrix.ndim == 3
        assert (adj_matrix == adj_matrix.transpose(1, 2)).all()
        assert (torch.abs(torch.diagonal(adj_matrix, dim1=-2, dim2=-1)) < 1e-5).all()
        assert (torch.logical_or(adj_matrix == 0, adj_matrix == 1)).all(), (
            adj_matrix.min(),
            adj_matrix.max(),
        )

        self.adj_matrix = adj_matrix.double()
        self.calculate_kpowers()

        k3x, k3y = self.k3_cycle()
        assert (k3x >= -0.1).all(), (k3x, k3x.min())

        k4x, k4y = self.k4_cycle()
        assert (k4x >= -0.1).all(), (k4x, k4x.min())

        _, k5y = self.k5_cycle()
        assert (k5y >= -0.1).all(), (k5y, k5y.min())

        _, k6y = self.k6_cycle()
        assert (k6y >= -0.1).all(), (k6y, k6y.min())

        kcyclesx = torch.cat([k3x, k4x], dim=-1)
        kcyclesy = torch.cat([k3y, k4y, k5y, k6y], dim=-1)
        return kcyclesx, kcyclesy
