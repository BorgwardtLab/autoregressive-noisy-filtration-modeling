import networkx as nx
import numpy as np


def dfs_edge_weight(graph: nx.Graph):
    perm = np.random.permutation(graph.number_of_nodes())
    nbhd_sorter = {node: perm[i] for i, node in enumerate(graph.nodes)}

    current_idx = 0
    node_order = {}

    for component in nx.connected_components(graph):
        component = graph.subgraph(component)
        initial_node = np.random.choice(list(component.nodes))
        node_order[initial_node] = current_idx
        current_idx += 1

        for u, v in nx.dfs_edges(
            component,
            source=initial_node,
            sort_neighbors=lambda nbhd: sorted(nbhd, key=lambda n: nbhd_sorter[n]),
        ):
            if u not in node_order:
                node_order[u] = current_idx
            if v not in node_order:
                node_order[v] = current_idx

            current_idx += 1

    # Take care of isolated nodes
    for node in graph.nodes:
        if node not in node_order:
            assert (
                graph.degree[node] == 0
            ), f"Node not found in DFS but also not isolated, {graph.degree[node]}, {nx.number_connected_components(graph)}"
            node_order[node] = current_idx
            current_idx += 1

    assert len(set(node_order.values())) == graph.number_of_nodes()
    assert (
        min(node_order.values()) == 0
        and max(node_order.values()) == graph.number_of_nodes() - 1
    )

    node_weight = np.array(
        [float(node_order[i]) for i in range(graph.number_of_nodes())]
    )

    edge_weight = [float(max(node_order[u], node_order[v])) for (u, v) in graph.edges]
    return edge_weight, node_weight
