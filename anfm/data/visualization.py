import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from torch_geometric.utils import to_networkx

from anfm.data.base.features import component_laplacian


def nx_fiedler(nx_graph):
    node_list = sorted(list(nx_graph.nodes))
    whole_laplacian = nx.normalized_laplacian_matrix(
        nx_graph, nodelist=node_list
    ).toarray()
    vals, vecs = np.linalg.eigh(whole_laplacian)
    fiedler = vecs[:, 1]
    return fiedler


def draw_nx_graph(graph, layout, ax, node_color=None):
    if node_color is None:
        fiedler, _, _, _ = component_laplacian(graph, 1, "sqrt")
        node_color = fiedler[:, 0]
    nx.draw(graph, pos=layout, font_size=5, node_size=100, node_color=node_color, ax=ax)


def plot_gridspec_graphs(filtration_graphs, graphs, num_columns=3):
    num_graphs = len(graphs)
    num_rows = (num_graphs - 1) // num_columns + 1  # Number of rows in the grid

    fig_samples = plt.figure(figsize=(5.5 * num_columns, 5.5 * num_rows))
    gs_samples = fig_samples.add_gridspec(num_rows, num_columns)

    fig_density = plt.figure(figsize=(5.5 * num_columns, 5.5 * num_rows))
    gs_density = fig_density.add_gridspec(num_rows, num_columns)

    for i, graph in enumerate(graphs):
        graph = to_networkx(graph, to_undirected=True)
        ax = fig_samples.add_subplot(gs_samples[i])
        layout = nx.fruchterman_reingold_layout(graph)
        draw_nx_graph(graph, layout, ax)
        ax.set_aspect("equal")

        # Plot the density
        density_ax = fig_density.add_subplot(gs_density[i])
        density_ax.plot([sample.num_edges for sample in filtration_graphs[i]])
        density_ax.set_xlabel("Filtration step")
        density_ax.set_ylabel("Number of edges")
        density_ax.set_title("Graph Density over Filtration")

    return fig_samples, fig_density


def plot_filtration(filtration, graph, num_substeps=None, node_color=None):
    graph = to_networkx(graph, to_undirected=True)
    filtration_graphs = [
        to_networkx(g, to_undirected=True) for g in filtration.to_data_list()
    ]
    filtration_graphs.append(graph)
    num_substeps = num_substeps if num_substeps is not None else len(filtration_graphs)

    # draw the filtration process
    num_columns = min(num_substeps, 11)
    num_rows = num_substeps // 11 + (num_substeps % 11 > 0)
    fig_filtration = plt.figure(figsize=(5.5 * num_columns, 5.5 * num_rows))
    axes = fig_filtration.subplots(ncols=num_columns, nrows=num_rows)
    assert (len(filtration_graphs) - num_substeps) % (num_substeps - 1) == 0
    stride = (len(filtration_graphs) - num_substeps) // (num_substeps - 1) + 1
    samples = filtration_graphs[::stride]
    layout = nx.fruchterman_reingold_layout(graph)
    for i in range(num_rows):
        for j in range(num_columns):
            idx = i * num_columns + j
            if idx < len(samples):
                if num_rows > 1:
                    ax = axes[i][j]
                else:
                    ax = axes[j]
                draw_nx_graph(samples[idx], layout, ax, node_color)
                ax.set_aspect("equal")

    return fig_filtration
