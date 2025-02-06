import numpy as np


def average_incident_weights(graph, edge_weights):
    values = [0 for _ in range(len(graph.nodes))]
    counts = [0 for _ in range(len(graph.nodes))]
    assert len(graph.edges) == len(edge_weights)
    for (u, v), curvature_val in zip(graph.edges, edge_weights):
        values[u] += curvature_val
        values[v] += curvature_val
        counts[u] += 1
        counts[v] += 1
    # assert that there are no isolated nodes
    return np.array([v / c for v, c in zip(values, counts)])
