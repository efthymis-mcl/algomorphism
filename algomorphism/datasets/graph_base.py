from typing import Tuple

import numpy as np


class GraphBaseDataset(object):
    def __init__(self):
        pass

    @staticmethod
    def numpy_to_mega_batch(X: list, A: list):
        """
        List of numpy arrays to mega batch array
        Args:
            X: A list, list of feature matrixes,
            A: A list, list of adjency matrixes.

        Returns:
            mega_batch_X: A ndarray, batched feature matrixes,
            mega_batch_A: A ndarray, batched adjency matrixes.

        Examples:
            >>> graph_base = GraphBaseDataset()
            >>> x = [np.random.rand(6,4) for _ in range(6)]+[np.random.rand(3,4) for _ in range(6)]
            >>> a = [np.random.rand(6,6) for _ in range(6)]+[np.random.rand(3,3) for _ in range(6)]
            >>> x, a = graph_base.numpy_to_mega_batch(x,a)
            >>> print(a.shape)
            (12, 6, 6)
            >>> print(x.shape)
            (12, 6, 4)
        """

        def a_post_concat(a):
            a_con = np.concatenate([a, np.zeros((a.shape[0], max_d - a.shape[1]))], axis=1)
            a_con = np.concatenate([a_con, np.zeros((max_d - a_con.shape[0], a_con.shape[1]))],
                                   axis=0)
            return a_con

        def x_post_concat(x):
            x_con = np.concatenate([x, np.zeros((max_d - x.shape[0], x.shape[1]))], axis=0)
            return x_con

        max_d = max([a.shape[0] for a in A])
        mega_batch_A = []
        mega_batch_X = []
        for (x, a) in zip(X, A):
            if a.shape[0] < max_d:
                a = a_post_concat(a)
                x = x_post_concat(x)
            mega_batch_A.append(a)
            mega_batch_X.append(x)
        mega_batch_A = np.array(mega_batch_A)
        mega_batch_X = np.stack(mega_batch_X, axis=0)

        return mega_batch_X, mega_batch_A

    @staticmethod
    def renormalization(a):
        """
        Give an adjacency matrix and returns the renormalized.
        Args:
            a: A ndarray, adjacency matrix.

        Returns:
            atld: A ndarray, renormalized adjacency matrix.

        Examples:
            >>> grapbase = GraphBaseDataset()
            >>> a = np.array([[[0,1,1], [1,0,0], [1,0,0]]])
            >>> atld = grapbase.renormalization(a)
            >>> print(atld)
            [[[0.33333333 0.40824829 0.40824829]
              [0.40824829 0.5        0.        ]
              [0.40824829 0.         0.5       ]]]

        References:
            Thomas N. Kipf, Max Welling. Semi-supervised classification with graph convolutional networks,
            https://arxiv.org/pdf/1609.02907.pdf
        """

        ai = a + np.eye(a.shape[-1])
        degree = np.sum(ai, axis=-1)
        degree = np.eye(a.shape[-1]) * degree
        degree_inv = np.linalg.inv(degree)
        degree_inv = np.power(degree_inv, 0.5)

        atld = np.matmul(degree_inv, ai)
        atld = np.matmul(atld, degree_inv)
        return atld
