import time
from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np


class TSP(ABC):
    """
    Abstract class for different algorithms solving the TSP problem
    :param nodes_path: path to a file containing information about (x,y) coordinates and costs for fiven nodes
    """
    def __init__(self, nodes_path: str):
        nodes = np.genfromtxt(nodes_path, dtype=float, delimiter=';')
        self.costs = nodes[:, 2]
        self.n = len(self.costs)

        coords = nodes[:, :2]
        self.dist_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1).round()

    def run_experiment(self):
        """
        performs function run_algorithm starting from each node
        :return: (min, max, avg, edges, start) - min, max , avg value of objective function, additionally for minimal
        value of objective returns edges and starting node
        """
        minv, maxv, avgv = np.inf, -np.inf, 0
        min_edges, starting_node = None, None
        for i in range(self.n):
            objective, edges = self.run_algorithm(i)

            avgv += objective
            if objective > maxv:
                maxv = objective
            if objective < minv:
                minv = objective
                min_edges = edges
                starting_node = i

        return minv, maxv, avgv / self.n, min_edges, starting_node

    @abstractmethod
    def run_algorithm(self, starting_node: int):
        """
        Abstract method for implementation of given algorithm that solves TSP problem
        :param starting_node: index of the starting node
        :return: tuple in form (objective, path), where objective represents the value of objective function for given
        solutions and path is numpy array containing consecutive nodes in the Hamiltonian cycle
        """
        pass


class RandomTSP(TSP):
    """
    A class implementing random selection of nodes and random order of the nodes to solve TSP problem
    """
    def run_algorithm(self, starting_node: int):
        path = [starting_node]
        perc_50 = int(np.ceil(self.n * 0.5))
        nodes = np.arange(0, self.n)
        nodes = np.delete(nodes, starting_node)
        path.extend(list(np.random.choice(nodes, perc_50 - 1, replace=False)))
        distances = np.sum(self.dist_matrix[path, np.roll(path, -1)])
        costs = np.sum(self.costs[path])
        return int(distances + costs), np.concatenate(([path], np.roll([path], -1)), axis=0).T


class NearestNeighbourTSP(TSP):
    """
    A class implementing the nearest neighbour method for solving the TSP problem
    """
    def run_algorithm(self, starting_node: int):
        perc_50 = int(np.ceil(self.n * 0.5))
        path = [starting_node]
        distances = 0
        costs = self.costs[starting_node]
        cost = deepcopy(self.costs)
        dist_matrix = deepcopy(self.dist_matrix)
        for _ in range(perc_50 - 1):
            i = path[-1]
            dist = dist_matrix[i, :]
            dist[path], cost[path] = None, None
            new_node = np.nanargmin(dist + cost)
            path.append(new_node)
            distances += dist_matrix[i, new_node]
            costs += self.costs[new_node]
        distances += self.dist_matrix[path[-1], path[0]]
        return int(distances + costs), np.concatenate(([path], np.roll([path], -1)), axis=0).T


class GreedyCycleTSP(TSP):
    """
    A class implementing greedy cycle method for solving the TSP problem
    """
    def run_algorithm(self, starting_node: int):
        perc_50 = int(np.ceil(self.n * 0.5))
        dist_matrix = deepcopy(self.dist_matrix)
        dist = dist_matrix[starting_node, :]
        cost = deepcopy(self.costs)
        dist[[starting_node]], cost[[starting_node]] = None, None
        new_node = np.nanargmin(dist + cost)
        edges = [[starting_node, new_node], [new_node, starting_node]]
        nodes = [starting_node, new_node]
        costs = self.costs[starting_node] + self.costs[new_node]
        dists = 2 * self.dist_matrix[starting_node, new_node]

        for _ in range(perc_50 - 2):
            dist_matrix[nodes, nodes] = None
            cost[nodes] = None
            i = np.argmin([np.nanmin(dist_matrix[i, :] + dist_matrix[j, :] + cost - self.dist_matrix[i, j]) for i, j in edges])
            i_edge = edges[i]
            dist_i, dist_j = dist_matrix[i_edge[0], :], dist_matrix[i_edge[1], :]
            new_node = np.nanargmin(dist_i + dist_j + cost)
            edges.pop(i)
            edges.append([i_edge[0], new_node])
            edges.append([new_node, i_edge[1]])
            nodes.append(new_node)
            costs += self.costs[new_node]
            dists += self.dist_matrix[i_edge[0], new_node] + self.dist_matrix[i_edge[1], new_node] - self.dist_matrix[i_edge[0], i_edge[1]]

        return int(costs + dists), edges


if __name__ == '__main__':
    nnTSP = GreedyCycleTSP('../data/TSPA.csv')
    #nnTSP = NearestNeighbourTSP('../data/TSPA.csv')
    start = time.time()
    print(nnTSP.run_algorithm(0))
    print(time.time() - start)

