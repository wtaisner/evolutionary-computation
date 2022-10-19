import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_result(nodes: np.ndarray, edges: np.array, objective: int):
    """
    plots obtained result
    :param nodes: numpy array containing (x,y) coordinates and costs for given nodes
    :param edges: numpy array containing edges that form the path
    :param objective: value of objective function for the path found
    :return: plot
    """
    G = nx.Graph()
    G.add_nodes_from(np.arange(len(nodes)))
    G.add_edges_from(edges)

    pos = nodes[:, :2]
    cost = nodes[:, 2]

    fig = plt.figure(1, figsize=(16, 12))
    ec = nx.draw_networkx_edges(G, pos, alpha=0.5)
    nc = nx.draw_networkx_nodes(G, pos, node_color=cost, label=None, node_size=20, cmap='Blues')
    plt.colorbar(nc)
    plt.text(-5, -20, f'objective: {objective}')
    plt.show()
