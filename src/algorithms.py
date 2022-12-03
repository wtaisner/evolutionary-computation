import copy
import itertools
import signal
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List

import numpy as np


class TSP(ABC):
    """
    Abstract class for different algorithms solving the TSP problem
    :param nodes_path: path to a file containing information about (x,y) coordinates and costs for given nodes
    """

    def __init__(self, nodes_path: str):
        nodes = np.genfromtxt(nodes_path, dtype=float, delimiter=';')
        self.costs = nodes[:, 2]
        # self.costs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.n = len(self.costs)
        self.experiments = self.n

        coords = nodes[:, :2]
        self.dist_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1).round()
        # self.dist_matrix = np.array([
        #     [0, 1, 2, 3, 4, 5, 4, 3, 2, 1],
        #     [1, 0, 1, 2, 3, 4, 5, 4, 3, 2],
        #     [2, 1, 0, 1, 2, 3, 4, 5, 4, 3],
        #     [3, 2, 1, 0, 1, 2, 3, 4, 5, 4],
        #     [4, 3, 2, 1, 0, 1, 2, 3, 4, 5],
        #     [5, 4, 3, 2, 1, 0, 1, 2, 3, 4],
        #     [4, 5, 4, 3, 2, 1, 0, 1, 2, 3],
        #     [3, 4, 5, 4, 3, 2, 1, 0, 1, 2],
        #     [2, 3, 4, 5, 4, 3, 2, 1, 0, 1],
        #     [1, 2, 3, 4, 5, 4, 3, 2, 1, 0]
        # ])
        self.time_init = 0

    def run_experiment(self, seed: int = None):
        """
        performs function run_algorithm starting from each node
        :return: (min, max, avg, edges, start) - min, max , avg value of objective function, additionally for minimal
        value of objective returns edges and starting node
        """
        minv, maxv, avgv, avg_time, min_time, max_time = np.inf, -np.inf, 0, 0, np.inf, -np.inf
        min_path, starting_node = None, None
        for i in range(self.experiments):
            time_start = time.time()
            objective, path = self.run_algorithm(i, seed=seed)
            time_diff = time.time() - time_start + self.time_init
            seed += 100
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
        return min_time, max_time, avg_time / self.experiments, minv, maxv, avgv / self.experiments, min_path, starting_node

    @abstractmethod
    def run_algorithm(self, starting_node: int, **kwargs):
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

    def run_algorithm(self, starting_node: int, seed: int = None):
        if seed is None:
            np.random.seed(starting_node)
        else:
            np.random.seed(seed)
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

    def run_algorithm(self, starting_node: int, **kwargs):
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

    def run_algorithm(self, starting_node: int, **kwargs):
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

    def run_algorithm(self, starting_node: int, **kwargs):
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
        pairs = list(
            itertools.product(np.arange(int(np.ceil(self.n * 0.5))), np.arange(int(np.ceil(self.n * 0.5)) + 1)))
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

        self.best_deltas = []

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
                    save = [best_delta, [[path[a], path[(a + 1) % len(path)]], [path[b - 1], path[b % len(path)]]]]
                    path[a + 1:b] = path[a + 1:b][::-1]
                    save.append(path)
                    self.best_deltas.append(save)

                    return True, path, cost + best_delta
            else:
                move = [path[a], not_selected[b]]
                path[a] = not_selected[b]
                self.best_deltas.append([best_delta, move, path])
                return True, path, cost + best_delta
        return False, path, cost

    def run_algorithm(self, starting_node: int, seed: int = None, init_path: List = None):
        if init_path is None:
            cost, path = self.init_solution.run_algorithm(starting_node, seed=seed)
        else:
            path = init_path
            cost = np.sum(self.dist_matrix[path, np.roll(path, -1)]) + np.sum(self.costs[path])
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
            nn_edges = np.argsort(self.dist_matrix[i, :])[1:k + 1]
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

    def run_algorithm(self, starting_node: int, seed: int = None, init_path: List = None):
        if init_path is None:
            cost, path = self.init_solution.run_algorithm(starting_node, seed=seed)
        else:
            path = init_path
            cost = np.sum(self.dist_matrix[path, np.roll(path, -1)]) + np.sum(self.costs[path])
        better = True
        while better:
            better, path, cost = self.loop(path, cost)
        return cost, path


class DeltaListSteepestLocalSearchTSP(TSP):
    def __init__(self, nodes_path: str):
        super().__init__(nodes_path)
        time_start = time.time()

        pairs = list(
            itertools.product(np.arange(int(np.ceil(self.n * 0.5))), np.arange(int(np.ceil(self.n * 0.5)) + 1)))
        pairs = list(zip(['p' for _ in range(len(pairs))], pairs))
        nodes = list(
            itertools.product(np.arange(int(np.ceil(self.n * 0.5))), np.arange(self.n - int(np.ceil(self.n * 0.5)))))
        nodes = list(zip(['n' for _ in range(len(nodes))], nodes))
        self.all = pairs + nodes
        self.improving_moves = []

        self.loop = self.steepest_loop

        self.init_solution = RandomTSP(nodes_path)
        self.time_init = time.time() - time_start

        self.best_deltas = []

    def node_select(self, path, idx_node_in_path, node_not_in_path):
        a = idx_node_in_path
        path.insert(0, path[-1])
        path.append(path[1])
        current = self.dist_matrix[path[a], path[a + 1]] + self.dist_matrix[path[a + 1], path[a + 2]] + self.costs[
            path[a + 1]]
        new = self.dist_matrix[path[a], node_not_in_path] + self.dist_matrix[node_not_in_path, path[a + 2]] + \
              self.costs[node_not_in_path]
        path.pop(0)
        path.pop()
        return new - current

    def two_edges_exchange(self, edges):
        edge1, edge2 = edges
        i1, i2, j1, j2 = edge1[0], edge1[1], edge2[0], edge2[1]
        current = self.dist_matrix[i1, i2] + self.dist_matrix[j1, j2]
        new = self.dist_matrix[i1, j1] + self.dist_matrix[i2, j2]
        if i1 == j1:
            return 0
        return new - current

    def calculate_all_moves(self, init_path):
        path = copy.deepcopy(init_path)
        not_selected = list(set(range(self.n)) - set(path))
        for entry in self.all:
            t, pair = entry
            if t == 'p':
                path.append(path[0])
                a, b = pair
                e1, e2 = [path[a], path[a + 1]], [path[b - 1], path[b]]
                edges = [[e1, e2], [e1[::-1], e2], [e1, e2[::-1]], [e1[::-1], e2[::-1]]]
                for edge in edges:
                    if not (edge[0][0] in edge[1] or edge[0][1] in edge[1]):
                        delta = self.two_edges_exchange(edge)
                        if delta < 0:
                            new_move = [delta, ['p', edge]]
                            self.improving_moves.append(new_move)
                path.pop()
            else:
                a, b = pair
                node_not_in_path = not_selected[b]
                delta = self.node_select(path, a, node_not_in_path)
                if delta < 0:
                    path.insert(0, path[-1])
                    path.append(path[1])
                    e1, e2 = [path[a], path[a + 1], path[a + 2]], [path[a], node_not_in_path, path[a + 2]]
                    self.improving_moves.append([delta, ['n', [e1, e2]]])
                    self.improving_moves.append([delta, ['n', [e1[::-1], e2[::-1]]]])
                    path.pop()
                    path.pop(0)
        self.improving_moves.sort(key=lambda x: x[0])

    @staticmethod
    def check_applicable(path, move):
        e1, e2 = move
        if len(e1) == 2:
            try:
                path.append(path[0])
                a10, b10, a20, b20 = len(path) - 1 - path[::-1].index(e1[0]), len(path) - 1 - path[::-1].index(e1[1]), \
                                     len(path) - 1 - path[::-1].index(e2[0]), len(path) - 1 - path[::-1].index(e2[1])
                a11, b11, a21, b21 = path.index(e1[0]), path.index(e1[1]), path.index(e2[0]), path.index(e2[1])
                path.pop()
            except:
                path.pop()
                return False, False, None
            if b10 - a10 == 1 and b20 - a20 == 1:
                return True, False, [a10, b10, a20, b20]
            if b11 - a11 == 1 and b21 - a21 == 1:
                return True, False, [a11, b11, a21, b21]
            else:
                return False, True, None
        else:
            try:
                path.insert(0, path[-1])
                path.append(path[1])
                a00, a10, a20 = len(path) - 1 - path[::-1].index(e1[0]), len(path) - 1 - path[::-1].index(e1[1]), len(
                    path) - 1 - path[::-1].index(e1[2])
                a01, a11, a21 = path.index(e1[0]), path.index(e1[1]), path.index(e1[2])
                path.pop()
                path.pop(0)
            except:
                path.pop()
                path.pop(0)
                return False, False, None
            if (a20 - a10 == 1 and a10 - a00 == 1 and a20 - a00 == 2) or (
                    a21 - a11 == 1 and a11 - a01 == 1 and a21 - a01 == 2):
                try:
                    b = path.index(e2[1])
                except:
                    return True, False, [a11]
            return False, False, None

    def add_new_moves_edge_exchange(self, path, change):
        not_selected = list(set(range(self.n)) - set(path))
        path.append(path[0])
        edges = np.arange(len(path) - 1)
        for e in edges:
            e1 = [path[e], path[e + 1]]
            for a in change:
                e2 = [path[a], path[a + 1]]
                edges = [[e1, e2], [e1[::-1], e2], [e1, e2[::-1]], [e1[::-1], e2[::-1]]]
                for edge in edges:
                    delta = self.two_edges_exchange(edge)
                    if delta < 0:
                        self.improving_moves.append([delta, ['p', edge]])
        path.pop()
        ids = [change[0], (change[0] + 1) % len(path), change[1], (change[1] + 1) % len(path)]
        for b in not_selected:
            for i in ids:
                delta = self.node_select(path, i, b)
                if delta < 0:
                    path.insert(0, path[-1])
                    path.append(path[1])
                    e0, e1, e2 = path[i], path[i + 1], path[i + 2]
                    m1, m2 = [e0, e1, e2], [e0, b, e2]
                    path.pop()
                    path.pop(0)
                    self.improving_moves.append([delta, ['n', [m1, m2]]])
                    self.improving_moves.append([delta, ['n', [m1[::-1], m2[::-1]]]])

    def add_new_moves_node_select(self, path, change):
        new, removed = change
        not_selected = list(set(range(self.n)) - set(path))
        pred, succ = (new - 1) % len(path), (new + 1) % len(path)
        for b in not_selected:
            for change in [new, pred, succ]:
                delta = self.node_select(path, change, b)
                if delta < 0:
                    path.insert(0, path[-1])
                    path.append(path[1])
                    e0, e1, e2 = path[change], path[change + 1], path[change + 2]
                    m1, m2 = [e0, e1, e2], [e0, b, e2]
                    path.pop()
                    path.pop(0)
                    self.improving_moves.append([delta, ['n', [m1, m2]]])
                    self.improving_moves.append([delta, ['n', [m1[::-1], m2[::-1]]]])

        for i, a in enumerate(path):
            delta = self.node_select(path, i, removed)
            if delta < 0:
                path.insert(0, path[-1])
                path.append(path[1])
                e0, e1, e2 = path[i], path[i + 1], path[i + 2]
                m1, m2 = [e0, e1, e2], [e0, removed, e2]
                path.pop()
                path.pop(0)
                self.improving_moves.append([delta, ['n', [m1, m2]]])
                self.improving_moves.append([delta, ['n', [m1[::-1], m2[::-1]]]])
        new_edges = [pred, new]
        self.add_new_moves_edge_exchange(path, new_edges)

    def steepest_loop(self, path, cost):
        to_delete = []
        best_move_idx = None
        for i, entry in enumerate(self.improving_moves):
            delta, move = entry
            applicable, stay, change = self.check_applicable(path, move[1])
            if applicable:
                to_delete.append(i)
                best_move_idx = i
                best_change = change
                break
            elif not applicable and not stay:
                to_delete.append(i)
        if best_move_idx is not None:
            best_move = self.improving_moves[best_move_idx]
        else:
            return False, path, cost
        self.improving_moves = [self.improving_moves[i] for i, _ in enumerate(self.improving_moves) if
                                i not in to_delete]
        if best_move is not None:
            delta, move = best_move
            move_type, change = move
            if move_type == 'p':
                a1, b1, a2, b2 = best_change
                if a1 > a2:
                    a2, b2, a1, b1 = a1, b1, a2, b2
                path[b1:b2] = path[b1:b2][::-1]
                change = [b1 - 1, b2 - 1]
                self.add_new_moves_edge_exchange(path, change)
            else:
                e1, e2 = change
                a, b = path.index(e1[1]), e2[1]
                removed = path[a]
                path[a] = b
                self.add_new_moves_node_select(path, [a, removed])
            self.improving_moves.sort(key=lambda x: x[0])
            # keys = [m[0] for m in self.improving_moves]
            # for m in new_moves:
            #     place = bisect.bisect(keys, m[0])
            #     self.improving_moves.insert(place, m)
            #     keys.insert(place, m[0])
            return True, path, cost + delta

    def run_algorithm(self, starting_node: int, **kwargs):
        cost, path = self.init_solution.run_algorithm(starting_node)  # 21, [0, 1, 5, 8, 3] #
        self.calculate_all_moves(path)
        better = True
        while better:
            better, path, cost = self.loop(path, cost)
        return cost, path


class MultipleStartLocalSearch(TSP):

    def __init__(self, nodes_path: str, local_search: TSP, experiments: int = 20, iters: int = 200):
        super().__init__(nodes_path)
        time_start = time.time()
        self.local_search = local_search
        self.iters = iters
        self.experiments = experiments
        self.time_init = time.time() - time_start

    def run_algorithm(self, starting_node: int, seed: int = None):
        best_cost, best_path = np.inf, None
        for _ in range(self.iters):
            cost, path = self.local_search.run_algorithm(starting_node, seed=seed)
            starting_node += 1
            starting_node %= self.n
            seed += 1
            if cost < best_cost:
                best_cost, best_path = cost, copy.deepcopy(path)
        return best_cost, best_path


class IteratedLocalSearch(TSP):
    def __init__(self, nodes_path: str, local_search: TSP, max_time: float, experiments: int = 20):
        super().__init__(nodes_path)
        time_start = time.time()
        self.local_search = local_search
        self.max_time = max_time
        self.experiments = experiments
        self.time_init = time.time() - time_start

    def signal_handler(self, signum, frame):
        raise TimeoutError("Timed out!")
    def perturb_solution(self, path: List, seed: int) -> List:
        np.random.seed(seed)
        not_selected = list(set(range(self.n)) - set(path))
        num_nodes = np.random.randint(4, 8)
        new_nodes = np.random.choice(not_selected, size=num_nodes, replace=False)
        old_nodes = np.random.choice(np.arange(len(path)), size=num_nodes, replace=False)
        for b, a in enumerate(old_nodes):
            path[a] = new_nodes[b]
        num_edges = np.random.randint(2, 9)
        for _ in range(num_edges):
            start = np.random.randint(0, len(path) - 4)
            stop = start + np.random.randint(3, len(path) - start)
            path[start:stop] = path[start:stop][::-1]
        return path

    def run_algorithm(self, starting_node: int, seed: int = None):
        time_start = time.time()
        iters = 0
        best_cost, best_path = np.inf, None
        signal.signal(signal.SIGALRM, self.signal_handler)
        signal.setitimer(signal.ITIMER_REAL, self.max_time - self.time_init - time.time() + time_start)
        try:
            cost, path = self.local_search.run_algorithm(starting_node, seed=seed)
            while True:
                path = self.perturb_solution(path, seed)
                cost, path = self.local_search.run_algorithm(starting_node, seed=seed, init_path=path)
                seed += 1
                if cost < best_cost:
                    best_cost, best_path = cost, copy.deepcopy(path)
                iters += 1
        except TimeoutError:
            return best_cost, best_path, iters

    def run_experiment(self, seed: int = None):
        """
        performs function run_algorithm starting from each node
        :return: (min, max, avg, edges, start) - min, max , avg value of objective function, additionally for minimal
        value of objective returns edges and starting node
        """
        minv, maxv, avgv, avg_time, min_time, max_time = np.inf, -np.inf, 0, 0, np.inf, -np.inf
        min_iters, max_iters, avg_iters = np.inf, -np.inf, 0
        min_path, starting_node = None, None
        for i in range(self.experiments):
            time_start = time.time()
            objective, path, iters = self.run_algorithm(i, seed=seed)
            time_diff = time.time() - time_start + self.time_init
            seed += 100
            avg_time += time_diff
            avgv += objective
            avg_iters += iters
            if time_diff > max_time:
                max_time = time_diff
            if time_diff < min_time:
                min_time = time_diff
            if iters > max_iters:
                max_iters = iters
            if iters < min_iters:
                min_iters = iters
            if objective > maxv:
                maxv = objective
            if objective < minv:
                minv = objective
                min_path = path
                starting_node = i

        return min_time, max_time, avg_time / self.experiments, minv, maxv, avgv / self.experiments, min_iters, max_iters, avg_iters / self.experiments, min_path, starting_node


if __name__ == '__main__':
    nnTSP = IteratedLocalSearch('../data/TSPC.csv', LocalSearchTSP('steepest', '../data/TSPC.csv', 'edges', 'random'), max_time=6.0, experiments=4)
    print(nnTSP.n)
    start = time.time()
    c = nnTSP.run_experiment(seed=500)
    print(c)
    print(time.time() - start)
    # print(cost, path)
    # print(np.sum(nnTSP.dist_matrix[path, np.roll(path, -1)]) + np.sum(nnTSP.costs[path]), path)
    # print(time.time() - start)
