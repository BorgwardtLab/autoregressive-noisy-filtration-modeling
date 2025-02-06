import networkx as nx
import numpy as np

from anfm.filtration.utils import average_incident_weights


def line_fiedler(graph):
    lg = nx.line_graph(graph)
    laplacian = nx.normalized_laplacian_matrix(lg).toarray()
    _, eig_vecs = np.linalg.eigh(laplacian)
    fiedler = eig_vecs[:, 1]
    fiedler = fiedler * (2 * np.random.randint(0, 2) - 1)
    edge_to_idx = {e: i for i, e in enumerate(lg.nodes)}
    edge_weights = [fiedler[edge_to_idx[e]] for e in graph.edges]
    return edge_weights, average_incident_weights(graph, edge_weights)
