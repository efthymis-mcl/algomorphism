from typing import List
import numpy as np


class GraphBaseDataset(object):
    def __int__(self):
        pass

    @staticmethod
    def numpy_to_mega_batch(x_list, a_list):
        """
        List of numpy arrays to mega batch array.

        Args:
            x_list (`list[np.ndarray]`): feature matrixes.
            a_list (`list[np.ndarray]`): adjency matrixes.
        Returns:
            `tuple[np.ndarray, np.ndarray]`: batched x, a lists
        Examples:

            >>> graph_base = GraphBaseDataset()
            >>> x_list = [np.random.rand(6,4) for _ in range(6)]+[np.random.rand(3,4) for _ in range(6)]
            >>> a_list = [np.random.rand(6,6) for _ in range(6)]+[np.random.rand(3,3) for _ in range(6)]
            >>> x, a = graph_base.numpy_to_mega_batch(x,a)
            >>> print(a.shape)
            (12, 6, 6)
            >>> print(x.shape)
            (12, 6, 4)
        """

        def a_post_concat(a):
            a_con = np.concatenate([a, np.zeros((a.shape[0], max_d - a.shape[1]))], axis=1)
            a_con = np.concatenate([a_con, np.zeros((max_d - a_con.shape[0], a_con.shape[1]))], axis=0)
            return a_con

        def x_post_concat(x):
            x_con = np.concatenate([x, np.zeros((max_d - x.shape[0], x.shape[1]))], axis=0)
            return x_con

        max_d = max([a.shape[0] for a in a_list])
        mega_batch_a = []
        mega_batch_x = []
        for (x, a) in zip(x_list, a_list):
            if a.shape[0] < max_d:
                a = a_post_concat(a)
                x = x_post_concat(x)
            mega_batch_a.append(a)
            mega_batch_x.append(x)
        mega_batch_a = np.array(mega_batch_a)
        mega_batch_x = np.stack(mega_batch_x, axis=0)

        return mega_batch_x, mega_batch_a

    @staticmethod
    def numpy_to_disjoint(x_list, a_list):
        """
        Args:
            x_list (`List[np.ndarray]`): feature matrixes,
            a_list (`List[np.ndarray]`): adajence matrixes.

        Returns:
            `tuple[np.ndarray, np.ndarray]`: disjoint matrixes of x_list, a_list.

        Examples:
            >>> x_list = [np.random.rand(6,4) for _ in range(6)]+[np.random.rand(3,4) for _ in range(6)]
            >>> a_list = [np.random.rand(6,6) for _ in range(6)]+[np.random.rand(3,3) for _ in range(6)]
            >>> gbd = GraphBaseDataset()
            >>> x, a = gbd.numpy_to_disjoint(x_list,a_list)
            >>> print(a.shape)
            (54, 54)
            >>> print(x.shape)
            (54, 48)
        """
        disjoint_a = a_list[0]
        disjoint_x = x_list[0]
        for a, x in zip(a_list[1:], x_list[1:]):
            na = a.shape[1]
            nda = disjoint_a.shape[1]
            nx = x.shape[1]
            ndx = disjoint_x.shape[1]

            disjoint_a = np.concatenate([disjoint_a, np.zeros((disjoint_a.shape[0], na))], axis=1)
            a = np.concatenate([np.zeros((a.shape[0], nda)), a], axis=1)
            disjoint_a = np.concatenate([disjoint_a, a], axis=0)

            disjoint_x = np.concatenate([disjoint_x, np.zeros((disjoint_x.shape[0], nx))], axis=1)
            x = np.concatenate([np.zeros((x.shape[0], ndx)), x], axis=1)
            disjoint_x = np.concatenate([disjoint_x, x], axis=0)

        return disjoint_x, disjoint_a

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


