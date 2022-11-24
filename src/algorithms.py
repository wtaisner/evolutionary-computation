import copy
import itertools
import time
from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np


class TSP(ABC):
    """
    Abstract class for different algorithms solving the TSP problem
    :param nodes_path: path to a file containing information about (x,y) coordinates and costs for given nodes
    """

    def __init__(self, nodes_path: str):
        nodes = np.genfromtxt(nodes_path, dtype=float, delimiter=';')
        self.costs = nodes[:, 2]
        self.n = len(self.costs)

        coords = nodes[:, :2]
        self.dist_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1).round()
        self.time_init = 0

    def run_experiment(self):
        """
        performs function run_algorithm starting from each node
        :return: (min, max, avg, edges, start) - min, max , avg value of objective function, additionally for minimal
        value of objective returns edges and starting node
        """
        minv, maxv, avgv, avg_time, min_time, max_time = np.inf, -np.inf, 0, 0, np.inf, -np.inf
        min_path, starting_node = None, None
        for i in range(self.n):
            time_start = time.time()
            objective, path = self.run_algorithm(i)
            time_diff = time.time() - time_start + self.time_init
            avg_time += time_diff
            avgv += objective
            if time_diff > max_time:
                max_time = time_diff
            if time_diff < min_time:
                min_time = time_diff
            if objective > maxv:
                maxv = objective
            if objective < minv:
                minv = objective
                min_path = path
                starting_node = i

        return min_time, max_time, avg_time / self.n, minv, maxv, avgv / self.n, min_path, starting_node

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
        np.random.seed(starting_node)
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


class LocalSearchTSP(TSP):
    def __init__(self, algorithm: str, nodes_path: str, exchange: str, init_solution: str):
        super().__init__(nodes_path)

        time_start = time.time()
        pairs = list(itertools.product(np.arange(int(np.ceil(self.n * 0.5))), np.arange(int(np.ceil(self.n * 0.5)) + 1)))
        pairs = list(zip(['p' for _ in range(len(pairs))], pairs))
        nodes = list(
            itertools.product(np.arange(int(np.ceil(self.n * 0.5))), np.arange(self.n - int(np.ceil(self.n * 0.5)))))
        nodes = list(zip(['n' for _ in range(len(nodes))], nodes))
        self.all = pairs + nodes
        self.exchange = exchange

        if init_solution == 'random':
            self.init_solution = RandomTSP(nodes_path)
        elif init_solution == 'greedy_cycle':
            self.init_solution = GreedyCycleTSP(nodes_path)
        else:
            raise ValueError('init solution should be random or greedy_cycle')

        if algorithm == 'greedy':
            self.loop = self.greedy_loop
        elif algorithm == 'steepest':
            self.loop = self.steepest_loop
        else:
            raise ValueError('init solution should be greedy or steepest')
        self.time_init = time.time() - time_start

    def two_nodes_exchange(self, path, pair):
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
        return new - current

    def two_edges_exchange(self, path, pair):
        path.append(path[0])
        a, b = pair
        if b - a > 2:
            i1, i2, j1, j2 = path[a], path[a + 1], path[b - 1], path[b]
            current = self.dist_matrix[i1, i2] + self.dist_matrix[j1, j2]
            new = self.dist_matrix[i1, j1] + self.dist_matrix[i2, j2]
        else:
            new, current = 0, 0
        path.pop()
        return new - current

    def node_select(self, path, pair, not_selected):
        a, b = pair
        path.insert(0, path[-1])
        path.append(path[1])
        current = self.dist_matrix[path[a], path[a + 1]] + self.dist_matrix[path[a + 1], path[a + 2]] + self.costs[
            path[a + 1]]
        new = self.dist_matrix[path[a], not_selected[b]] + self.dist_matrix[not_selected[b], path[a + 2]] + \
              self.costs[not_selected[b]]
        path.pop(0)
        path.pop()
        return new - current

    def greedy_loop(self, init_path, cost):
        path = copy.deepcopy(init_path)
        np.random.shuffle(self.all)
        not_selected = list(set(range(self.n)) - set(path))
        for entry in self.all:
            t, pair = entry
            if t == 'p':
                if self.exchange == 'nodes':
                    delta = self.two_nodes_exchange(path, pair)
                    if delta < 0:
                        a, b = pair
                        path[a], path[b] = path[b], path[a]
                        cost += delta
                        return True, path, cost
                elif self.exchange == 'edges':
                    delta = self.two_edges_exchange(path, pair)
                    if delta < 0:
                        a, b = pair
                        path[a + 1:b] = path[a + 1:b][::-1]
                        cost += delta
                        return True, path, cost
            else:
                delta = self.node_select(path, pair, not_selected)
                if delta < 0:
                    a, b = pair
                    path[a] = not_selected[b]
                    cost += delta
                    return True, path, cost
        return False, path, cost

    def steepest_loop(self, init_path, cost):
        path = copy.deepcopy(init_path)
        not_selected = list(set(range(self.n)) - set(path))
        best_delta, best_pair, best_t = 0, None, None
        for entry in self.all:
            t, pair = entry
            if t == 'p':
                if self.exchange == 'nodes':
                    delta = self.two_nodes_exchange(path, pair)
                elif self.exchange == 'edges':
                    delta = self.two_edges_exchange(path, pair)
                else:
                    delta = 0
            else:
                delta = self.node_select(path, pair, not_selected)
            if delta < best_delta:
                best_delta, best_pair, best_t = delta, pair, t
        if best_delta < 0:
            a, b = best_pair
            if best_t == 'p':
                if self.exchange == 'nodes':
                    path[a], path[b] = path[b], path[a]
                    return True, path, cost + best_delta
                elif self.exchange == 'edges':
                    path[a + 1:b] = path[a+1:b][::-1]
                    return True, path, cost + best_delta
            else:
                path[a] = not_selected[b]
                return True, path, cost + best_delta
        return False, path, cost

    def run_algorithm(self, starting_node: int):
        cost, path = self.init_solution.run_algorithm(starting_node)
        better = True
        while better:
            better, path, cost = self.loop(path, cost)
        return cost, path


class CandidateSteepestLocalSearchTSP(TSP):
    def __init__(self, nodes_path: str, k: int):
        super().__init__(nodes_path)
        time_start = time.time()
        pairs = list(itertools.product(np.arange(int(np.ceil(self.n * 0.5))), np.arange(k)))
        pairs = list(zip(['p' for _ in range(len(pairs))], pairs))
        nodes = list(itertools.product(np.arange(int(np.ceil(self.n * 0.5))), np.arange(k)))
        nodes = list(zip(['n' for _ in range(len(nodes))], nodes))
        self.all = pairs + nodes

        self.nn_edges, self.nn_nodes = [], []
        for i in range(self.n):
            nn_edges = np.argsort(self.dist_matrix[i, :])[1:k+1]
            nn_nodes = np.argsort(self.dist_matrix[i, :] + self.costs)
            ids = np.argwhere(nn_nodes != i)[:k]
            self.nn_edges.append(nn_edges)
            self.nn_nodes.append(nn_nodes[ids].flatten())
        self.nn_edges = np.array(self.nn_edges)
        self.nn_nodes = np.array(self.nn_nodes)

        self.loop = self.steepest_loop

        self.init_solution = RandomTSP(nodes_path)
        self.time_init = time.time() - time_start

    def two_edges_exchange(self, path, pair):
        path.insert(0, path[-1])
        path.append(path[1])
        a, b = pair[0] + 1, pair[1] + 1
        if b - a > 2 and path[a] != path[b]:
            i1, i2, j1, j2 = path[a], path[a + 1], path[b - 1], path[b]
            current = self.dist_matrix[i1, i2] + self.dist_matrix[j1, j2]
            new = self.dist_matrix[i1, j1] + self.dist_matrix[i2, j2]
        else:
            new, current = 0, 0
        path.pop()
        path.pop(0)
        return new - current

    def node_select(self, path, pair):
        a, b = pair
        path.insert(0, path[-1])
        path.append(path[1])
        current = self.dist_matrix[path[a], path[a + 1]] + self.dist_matrix[path[a + 1], path[a + 2]] + self.costs[
            path[a + 1]]
        new = self.dist_matrix[path[a], b] + self.dist_matrix[b, path[a + 2]] + self.costs[b]
        path.pop(0)
        path.pop()
        return new - current

    def steepest_loop(self, init_path, cost):
        path = copy.deepcopy(init_path)
        best_delta, best_pair, best_t = 0, None, None
        for entry in self.all:
            t, pair = entry
            if t == 'p':
                try:
                    place = path.index(self.nn_edges[path[pair[0]], pair[1]])
                except:
                    place = None
                if place is not None:
                    if pair[0] > place:
                        a1, b1 = place, pair[0] + 1
                        a2, b2 = place - 1, pair[0]
                    else:
                        a1, b1 = pair[0], place + 1
                        a2, b2 = pair[0] - 1, place
                    if a2 < 0 and b1 - a1 > 2:
                        a2, b2 = b2 - 1, len(path)
                    pair1 = [a1, b1]
                    delta1 = self.two_edges_exchange(path, pair1)
                    pair2 = [a2, b2]
                    delta2 = self.two_edges_exchange(path, pair2)
                    if delta1 > delta2:
                        delta, pair = delta2, pair2
                    else:
                        delta, pair = delta1, pair1
                else:
                    delta = 0
            else:
                if self.nn_nodes[path[pair[0]], pair[1]] not in path:
                    pair = [pair[0], self.nn_nodes[path[pair[0]], pair[1]]]
                    delta = self.node_select(path, pair)
                else:
                    delta = 0
            if delta < best_delta:
                best_delta, best_pair, best_t = delta, pair, t
        if best_delta < 0:
            a, b = best_pair
            if best_t == 'p':
                path[a + 1:b] = path[a + 1: b][::-1]
                return True, path, cost + best_delta
            else:
                path[a] = b
                return True, path, cost + best_delta
        return False, path, cost

    def run_algorithm(self, starting_node: int):
        cost, path = self.init_solution.run_algorithm(starting_node)
        better = True
        while better:
            better, path, cost = self.loop(path, cost)
            #print(path)
        return cost, path


if __name__ == '__main__':
    nnTSP = CandidateSteepestLocalSearchTSP('../data/TSPC.csv', 10)
    print(nnTSP.n)
    start = time.time()
    cost, path = nnTSP.run_algorithm(4)
    print(cost, path)
    print(np.sum(nnTSP.dist_matrix[path, np.roll(path, -1)]) + np.sum(nnTSP.costs[path]), path)
    print(time.time() - start)

    nnTSP = LocalSearchTSP('steepest', '../data/TSPC.csv', 'edges', 'random')
    print(nnTSP.n)
    start = time.time()
    cost, path = nnTSP.run_algorithm(4)
    print(cost, path)
    print(np.sum(nnTSP.dist_matrix[path, np.roll(path, -1)]) + np.sum(nnTSP.costs[path]), path)
    print(time.time() - start)
