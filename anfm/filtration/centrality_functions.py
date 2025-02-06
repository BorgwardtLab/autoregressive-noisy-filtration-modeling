import networkx as nx

from anfm.filtration.utils import average_incident_weights


def edge_betweenness(graph):
    centrality = nx.edge_betweenness_centrality(graph)
    edge_weights = [centrality[e] for e in graph.edges]
    return edge_weights, average_incident_weights(graph, edge_weights)


def edge_remoteness(graph):
    centrality = nx.edge_betweenness_centrality(graph)
    edge_weights = [-centrality[e] for e in graph.edges]
    return edge_weights, average_incident_weights(graph, edge_weights)
