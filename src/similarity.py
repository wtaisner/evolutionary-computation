import numpy as np


def edges_similarity(path1, path2):
    all_edges_path1 = np.concatenate(([path1], np.roll([path1], -1)), axis=0).T.tolist()
    all_edges_path2 = np.concatenate(([path2], np.roll([path2], -1)), axis=0).T.tolist()
    common_edges = [e for e in all_edges_path1 if e in all_edges_path2 or e[::-1] in all_edges_path2]
    return len(common_edges)


def nodes_similarity(path1, path2):
    common_nodes = [n for n in path1 if n in path2]
    return len(common_nodes)


def correlation(objective, similarity):
    X = np.array([objective, similarity])
    return np.corrcoef(X)


if __name__ == '__main__':
    path1 = [1, 2, 3, 4, 5]
    path2 = [5, 9, 3, 10, 1]

    print(edges_similarity(path1, path2))
    print(nodes_similarity(path1, path2))
