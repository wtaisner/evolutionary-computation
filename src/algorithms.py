import time
from abc import ABC, abstractmethod

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

    @abstractmethod
    def run_algorithm(self, starting_node: int):
        """
        Abstract method for impllementation of given algorithm that solves TSP problem
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
        path.extend(list(np.random.choice(nodes, perc_50 - 1)))
        pairs = np.concatenate((path, np.roll(path, -1)), axis=0).T
        distances = np.sum(self.dist_matrix[pairs])
        costs = np.sum(self.costs[path])
        return int(distances + costs), path


class NearestNeighbourTSP(TSP):
    """
    A class implementing the nearest neighbour method for solving the TSP problem
    """
    def run_algorithm(self, starting_node: int):
        perc_50 = int(np.ceil(self.n * 0.5))
        path = [starting_node]
        distances = 0
        costs = self.costs[starting_node]
        for _ in range(perc_50 - 1):
            i = path[-1]
            dist = self.dist_matrix[i, :]
            dist[path] = None
            cost = self.costs
            cost[path] = None
            new_node = np.nanargmin(dist + cost)
            path.append(new_node)
            distances += self.dist_matrix[i, new_node]
            costs += self.costs[new_node]
        distances += self.dist_matrix[path[-1], path[0]]
        return int(distances + costs), path


class GreedyCycleTSP(TSP):
    """
    A class implementing greedy cycle method for solving the TSP problem
    """
    def run_algorithm(self, starting_node: int):
        pass


if __name__ == '__main__':
    nnTSP = NearestNeighbourTSP('../data/TSPA.csv')
    start = time.time()
    print(nnTSP.run_algorithm(0))
    print(time.time() - start)

