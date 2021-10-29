import numpy as np
from algomorphism.datasets import GraphBaseDataset
from algomorphism.methods.graphs import vertexes_to_adjacency
from typing import List


class FromEdgesListExamples(GraphBaseDataset):
    """
    Create ndarray Adjency (+renormilized), identiy (featur) matrixes by list of graphs with edges.

    Examples:
            >>> e_list = [[[0,1], [1,2], [1,3], [1,4], [1,5]], [[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]]
            >>> eld = FromEdgesListExamples(e_list)
            >>> print(eld.a[0])
            [[0. 1. 0. 0. 0. 0.]
             [1. 0. 1. 1. 1. 1.]
             [0. 1. 0. 0. 0. 0.]
             [0. 1. 0. 0. 0. 0.]
             [0. 1. 0. 0. 0. 0.]
             [0. 1. 0. 0. 0. 0.]]
            >>> eld.add_graph_by_edges([[0,3], [1,3], [1,4], [3, 4], [2, 4]])
            >>> print(eld.a[-1])
            [[0. 0. 0. 1. 0. 0.]
             [0. 0. 0. 1. 1. 0.]
             [0. 0. 0. 0. 1. 0.]
             [1. 1. 0. 0. 1. 0.]
             [0. 1. 1. 1. 0. 0.]
             [0. 0. 0. 0. 0. 0.]]
    """
    def __init__(self, edges_list):
        """
        Args:
            edges_list (`List[List[list]]`): edges of graphs

        """
        super(FromEdgesListExamples, self).__init__()
        self.x = np.empty(0)
        self.a = np.empty(0)
        self.atld = np.empty(0)
        for edges in edges_list:
            self.add_graph_by_edges(edges)

    def add_graph_by_edges(self, edges):
        """
        Add graph by edges.

        Args:
            edges (`List[list]]`): list with edges of graph.
        """

        a_list = [a for a in self.a]
        atld_list = [atld for atld in self.atld]
        try:
            max_d = self.a.shape[1]
        except IndexError:
            max_d = 0
        a = vertexes_to_adjacency(edges)
        a_tld = self.renormalization(a)
        a_list.append(a)
        atld_list.append(a_tld)
        if a.shape[1] > max_d:
            max_d = a.shape[1]

        x_list = [np.eye(max_d) for _ in range(len(a_list))]

        self.x, self.a = self.numpy_to_mega_batch(x_list, a_list)
        _,  self.atld = self.numpy_to_mega_batch(x_list, atld_list)
