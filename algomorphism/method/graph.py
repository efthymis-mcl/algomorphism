# author: Efthymis Michalis

import numpy as np
import networkx as nx
from typing import Union, List, Tuple


def vertexes2adjacency(vertexes: Union[List[List[int]], List[tuple], Tuple[list], Tuple[tuple]]) -> np.ndarray:
    """
    Transform vertexes of graph; list/tuple object to adjaceny matrix.

    Args:
        vertexes (`Union[List[List[int]], List[tuple], Tuple[list], Tuple[tuple]]`): vertexes.

    Returns:
        (`np.ndarray`): adjacency matrix.

    Examples:
          >>> vrtxs = [[0,1], [1,2], [1,3], [1,4], [1,5]]
          >>> a = vertexes2adjacency(vrtxs)
          >>> print(a.shape)
          (6, 6)
          >>> print(a)
          [[0. 1. 0. 0. 0. 0.]
           [1. 0. 1. 1. 1. 1.]
           [0. 1. 0. 0. 0. 0.]
           [0. 1. 0. 0. 0. 0.]
           [0. 1. 0. 0. 0. 0.]
           [0. 1. 0. 0. 0. 0.]]
          >>> vrtxs = [[0,1], [2,2], [4,1], [2,4]]
          >>> a = vertexes2adjacency(vrtxs)
          >>> print(a.shape)
          (4, 4)
          >>> print(a)
          [[0. 1. 0. 0.]
           [1. 0. 0. 1.]
           [0. 0. 1. 1.]
           [0. 1. 1. 0.]]
    """
    v_flat = [vij for vi in vertexes for vij in vi]
    nodes = list(set(v_flat))
    n = len(nodes)
    a = np.zeros((n, n))
    for element_i in nodes:
        i = nodes.index(element_i)
        for element_j in nodes:
            j = nodes.index(element_j)
            if [element_i, element_j] in vertexes or [element_j, element_i] in vertexes:
                a[i, j] = 1.0
    return a


def a2g(a: np.ndarray) -> nx.Graph:
    """
    Adjacency matrix to graph object.

    Args:
        a (`np.ndarray`): binary adjacency matrix.

    Returns:
        (`nx.Graph`): networkx graph object.
    """
    g = nx.Graph()
    a = np.where(a == 1)
    a = [[x, y] for x, y in zip(a[0], a[1])]
    g.add_edges_from(a)
    return g


def graphs_stats(a_examples: np.ndarray) -> Tuple[int, dict, dict, dict]:
    """
    Statistics about graph examples. Run bfs tree on every node on every graph getting the sortest path lengest per node
    (depth). After that mesure the max depth, depth distribution per depth observation, max depth distribution (geting
    the maximum depth per graph and update the distribution) and the number of edges distribution for all nodes for all graphs.

    Args:
        a_examples (`np.ndarray`): examples of adjency matrix.

    Returns:
        (`tuple[int, dict, dict, dict]`):
            (max_depth): a int, max depth of all graphs for all nodes of bfs tree,
            (depth_dist): a dict, keys are the observable depth of bfs tree (such as 1, 2, 4, 5, ...),
            (maxdepth_dist): a dict, keys are the observable max-depth per node of bfs tree (such as 1, 2, 4, 5, ...),
            (edge_n): A dict, keys are the observable number of edges per graph.
    """
    max_depth = 0
    depth_dist = {}
    maxdepth_dist = {}
    edge_n = {}
    for a in a_examples:
        g = a2g(a)
        en = len(g.edges())
        if en not in edge_n.keys():
            edge_n[en] = 1
        else:
            edge_n[en] += 1
        for n in g.nodes():
            tree = nx.algorithms.traversal.breadth_first_search.bfs_tree(g, n)
            depths = nx.shortest_path_length(tree, n).values()
            val = 1
            for d in depths:
                if d > -1:
                    if d not in depth_dist.keys():
                        depth_dist[d] = val
                    else:
                        depth_dist[d] += val

            maxd = max(depths)
            if maxd not in maxdepth_dist.keys():
                maxdepth_dist[maxd] = val
            else:
                maxdepth_dist[maxd] += val
            if maxd > max_depth:
                max_depth = maxd
    return max_depth, depth_dist, maxdepth_dist, edge_n
