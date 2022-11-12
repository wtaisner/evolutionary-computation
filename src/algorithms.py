import copy
import itertools
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
        return int(distances + costs), path


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
        path = [starting_node, new_node]
        costs = self.costs[starting_node] + self.costs[new_node]
        dists = 2 * self.dist_matrix[starting_node, new_node]

        for _ in range(perc_50 - 2):
            dist_matrix[path, path] = None
            cost[path] = None
            path.append(path[0])
            i = np.argmin([np.nanmin(
                dist_matrix[path[i], :] + dist_matrix[path[i + 1], :] + cost - self.dist_matrix[path[i], path[i + 1]])
                for i in range(len(path) - 1)])
            i_edge = [path[i], path[i + 1]]
            path.pop()
            dist_i, dist_j = dist_matrix[i_edge[0], :], dist_matrix[i_edge[1], :]
            new_node = np.nanargmin(dist_i + dist_j + cost)
            path.insert(i + 1, new_node)
            costs += self.costs[new_node]
            dists += self.dist_matrix[i_edge[0], new_node] + self.dist_matrix[i_edge[1], new_node] - self.dist_matrix[
                i_edge[0], i_edge[1]]

        return int(costs + dists), path


class GreedyCycleRegretTSP(TSP):
    """
    A class implementing greedy cycle method with k-regret for solving TSP problem
    :param k: k-regret, how many elements to take into account while computing regret
    :param weight: weight of the regret, when equal to 1 the next node is chosen only based on the regret, otherwise
    the objective is also taken into account
    """

    def __init__(self, nodes_path: str, k: int, weight: float):
        super().__init__(nodes_path)
        self.k = k
        self.weight = weight

    def run_algorithm(self, starting_node: int):
        perc_50 = int(np.ceil(self.n * 0.5))
        dist_matrix = deepcopy(self.dist_matrix)
        dist = dist_matrix[starting_node, :]
        cost = deepcopy(self.costs)
        dist[[starting_node]], cost[[starting_node]] = None, None
        new_node = np.nanargmin(dist + cost)
        path = [starting_node, new_node]
        costs = self.costs[starting_node] + self.costs[new_node]
        dists = 2 * self.dist_matrix[starting_node, new_node]
        for _ in range(perc_50 - 2):
            dist_matrix[path, path] = None
            cost[path] = None
            path.append(path[0])
            distances = np.array(
                [dist_matrix[path[i], :] + dist_matrix[path[i + 1], :] + cost - self.dist_matrix[path[i], path[i + 1]]
                 for i in range(len(path) - 1)]).T
            distances_sort = np.sort(distances, axis=1)
            distances_diff = np.apply_along_axis(lambda x: self.k * x[0] - np.sum(x[:self.k]), 1, distances_sort)
            new_node = np.nanargmin(self.weight * distances_diff + (1 - self.weight) * distances_sort[:, 0])
            i = np.nanargmin(distances[new_node])
            i_edge = [path[i], path[i + 1]]
            path.pop()
            path.insert(i + 1, new_node)
            costs += self.costs[new_node]
            dists += self.dist_matrix[i_edge[0], new_node] + self.dist_matrix[i_edge[1], new_node] - self.dist_matrix[
                i_edge[0], i_edge[1]]

        return int(costs + dists), path


class GreedyLocalSearchTSP(TSP):
    def __init__(self, nodes_path: str, exchange: str, init_solution: str):
        super().__init__(nodes_path)

        self.exchange = exchange
        pairs = list(itertools.combinations(np.arange(int(np.ceil(self.n * 0.5))), 2))
        pairs = list(zip(['p' for _ in range(len(pairs))], pairs))
        nodes = list(itertools.product(np.arange(int(np.ceil(self.n * 0.5))), np.arange(self.n - int(np.ceil(self.n * 0.5)))))
        nodes = list(zip(['n' for _ in range(len(nodes))], nodes))
        self.all = pairs + nodes
        if init_solution == 'random':
            self.init_solution = RandomTSP(nodes_path)
        elif init_solution == 'greedy_cycle':
            self.init_solution = GreedyCycleTSP(nodes_path)

    def two_nodes_exchange(self, path, cost, pair):
        path.append(path[0])
        a, b = pair
        i0, i, i1, j0, j, j1 = path[a - 1], path[a], path[a + 1], path[b - 1], path[b], path[b + 1]
        if b - a == 1:
            current = self.dist_matrix[i0, i] + self.dist_matrix[j, j1]
            new = self.dist_matrix[i0, j] + self.dist_matrix[i, j1]
        else:
            current = self.dist_matrix[i0, i] + self.dist_matrix[i, i1] + self.dist_matrix[j0, j] + \
                      self.dist_matrix[j, j1]
            new = self.dist_matrix[i0, j] + self.dist_matrix[j, i1] + self.dist_matrix[j0, i] + self.dist_matrix[
                i, j1]
        path.pop()
        if new - current < 0:
            path[a], path[b] = path[b], path[a]
            return True, path, cost + new - current
        return False, path, cost

    def two_edges_exchange(self, path, cost, pair):
        path.append(path[0])
        a, b = pair
        if b - a > 2:
            i1, i2, j1, j2 = path[a], path[a + 1], path[b - 1], path[b]
            current = self.dist_matrix[i1, i2] + self.dist_matrix[j1, j2]
            new = self.dist_matrix[i1, j1] + self.dist_matrix[i2, j2]
        else:
            new, current = 0, 0
        path.pop()
        if new - current < 0:
            path[a + 1:b] = path[b - 1:a:-1]
            return True, path, cost + new - current
        return False, path, cost

    def node_select(self, path, cost, pair, not_selected):
        a, b = pair
        path.insert(0, path[-1])
        path.append(path[1])
        current = self.dist_matrix[path[a], path[a + 1]] + self.dist_matrix[path[a + 1], path[a + 2]] + self.costs[
            path[a + 1]]
        new = self.dist_matrix[path[a], not_selected[b]] + self.dist_matrix[not_selected[b], path[a + 2]] + \
              self.costs[not_selected[b]]
        path.pop(0)
        path.pop()
        if new - current < 0:
            path[a] = not_selected[b]
            return True, path, cost + new - current
        return False, path, cost

    def run_algorithm(self, starting_node: int):
        cost, path = self.init_solution.run_algorithm(starting_node)
        better = True
        while better:
            np.random.shuffle(self.all)
            not_selected = list(set(range(self.n)) - set(path))
            for entry in self.all:
                t, pair = entry
                if t == 'p':
                    if self.exchange == 'nodes':
                        better, path, cost = self.two_nodes_exchange(path, cost, pair)
                    elif self.exchange == 'edges':
                        better, path, cost = self.two_edges_exchange(path, cost, pair)
                    else:
                        better, path, cost = False, path, cost
                elif t == 'n':
                    better, path, cost = self.node_select(path, cost, pair, not_selected)
                if better:
                    break
        return cost, path


class SteepestLocalSearchTSP(TSP):
    def __init__(self, nodes_path: str, exchange: str, init_solution: str):
        super().__init__(nodes_path)

        self.exchange = exchange
        self.pairs = list(itertools.combinations(np.arange(int(np.ceil(self.n * 0.5))), 2))

        if init_solution == 'random':
            self.init_solution = RandomTSP(nodes_path)
        elif init_solution == 'greedy_cycle':
            self.init_solution = GreedyCycleTSP(nodes_path)

    def two_nodes_exchange(self, init_path, cost):
        path = copy.deepcopy(init_path)
        path.append(path[0])
        deltas = []
        for pair in self.pairs:
            a, b = pair
            i0, i, i1, j0, j, j1 = path[a - 1], path[a], path[a + 1], path[b - 1], path[b], path[b + 1]
            if b - a == 1:
                current = self.dist_matrix[i0, i] + self.dist_matrix[j, j1]
                new = self.dist_matrix[i0, j] + self.dist_matrix[i, j1]
            else:
                current = self.dist_matrix[i0, i] + self.dist_matrix[i, i1] + self.dist_matrix[j0, j] + \
                          self.dist_matrix[j, j1]
                new = self.dist_matrix[i0, j] + self.dist_matrix[j, i1] + self.dist_matrix[j0, i] + self.dist_matrix[
                    i, j1]
            deltas.append(new - current)
        best_i = np.argmin(deltas)
        path.pop()
        if deltas[best_i] >= 0:
            return False, path, cost
        best_pair = self.pairs[best_i]
        a, b = path[best_pair[0]], path[best_pair[1]]
        path[best_pair[0]] = b
        path[best_pair[1]] = a
        return True, path, cost + deltas[best_i]

    def two_edges_exchange(self, init_path, cost):
        path = copy.deepcopy(init_path)
        path.append(path[0])
        deltas = []

        for pair in self.pairs:
            a, b = pair
            if b - a > 2:
                i1, i2, j1, j2 = path[a], path[a + 1], path[b - 1], path[b]
                current = self.dist_matrix[i1, i2] + self.dist_matrix[j1, j2]
                new = self.dist_matrix[i1, j1] + self.dist_matrix[i2, j2]
                deltas.append(new - current)
            else:
                deltas.append(None)
        best_i = np.nanargmin(deltas)
        path.pop()
        if deltas[best_i] >= 0:
            return False, path, cost
        a, b = self.pairs[best_i]
        path[a + 1:b] = path[b - 1:a:-1]
        return True, path, cost + deltas[best_i]

    def node_select(self, init_path, cost):
        path = copy.deepcopy(init_path)
        not_selected = list(set(range(self.n)) - set(path))
        pairs = list(itertools.product(np.arange(len(path)), np.arange(len(not_selected))))
        path.insert(0, path[-1])
        path.append(path[1])
        deltas = []
        for pair in pairs:
            a, b = pair
            current = self.dist_matrix[path[a], path[a + 1]] + self.dist_matrix[path[a + 1], path[a + 2]] + self.costs[
                path[a + 1]]
            new = self.dist_matrix[path[a], not_selected[b]] + self.dist_matrix[not_selected[b], path[a + 2]] + \
                  self.costs[not_selected[b]]
            deltas.append(new - current)
        best_i = np.argmin(deltas)
        path.pop(0)
        path.pop()
        if deltas[best_i] >= 0:
            return False, path, cost
        best_pair = pairs[best_i]
        path[best_pair[0]] = not_selected[best_pair[1]]
        return True, path, cost + deltas[best_i]

    def run_algorithm(self, starting_node: int):
        cost, path = self.init_solution.run_algorithm(starting_node)
        better = True
        while better:
            if self.exchange == 'nodes':
                better0, path0, cost0 = self.two_nodes_exchange(path, cost)
            elif self.exchange == 'edges':
                better0, path0, cost0 = self.two_nodes_exchange(path, cost)
            else:
                better0, path0, cost0 = False, path, cost
            better1, path1, cost1 = self.node_select(path, cost)
            if better0 or better1:
                if cost0 < cost1:
                    path, cost = path0, cost0
                else:
                    path, cost = path1, cost1
            else:
                better = False
        return cost, path


if __name__ == '__main__':
    nnTSP = GreedyLocalSearchTSP('../data/TSPA.csv', 'nodes', 'random')
    print(nnTSP.n)
    start = time.time()
    print(nnTSP.run_algorithm(0))
    print(time.time() - start)
