import networkx as nx
import numpy as np
import random
import math
from algomorphism.datasets import GraphBaseDataset
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf


# Classification task
class SimpleGraphsDataset(GraphBaseDataset):

    def __init__(self, n_data, min_n_nodes, max_n_nodes, graph_types):
        super(SimpleGraphsDataset, self).__init__()

        self.__all_graph_types = ['cycle', 'star', 'wheel', 'complete', 'lollipop',
                                  'hypercube', 'circular_ladder', 'grid']

        self.__n_data = n_data
        self.__graph_types = graph_types
        self.__min_n_nodes = min_n_nodes
        self.__max_n_nodes = max_n_nodes

        for graph_type in self.__graph_types:
            assert graph_type in self.__all_graph_types, 'type {} is not included'.format(graph_type)

    def generate_dataset(self, train_per=0.8, val_per=0.1, test_per=0.1):
        a, atld, x, label_list = self.__generate_data()

        self.__lb = LabelBinarizer()
        self.__lb .fit(label_list)
        one_hot_labels = self.__lb .transform(label_list)

        train_idx_min = 0
        train_idx_max = int(self.__n_data*train_per)
        self.__a_train = a[train_idx_min: train_idx_max]
        self.__atld_train = atld[train_idx_min: train_idx_max]
        self.__x_train = x[train_idx_min: train_idx_max]
        self.__y_train = one_hot_labels[train_idx_min: train_idx_max]

        val_idx_min = train_idx_max
        val_idx_max = int(self.__n_data*(train_per + val_per))
        self.__a_val = a[val_idx_min: val_idx_max]
        self.__atld_val = atld[val_idx_min: val_idx_max]
        self.__x_val = x[val_idx_min: val_idx_max]
        self.__y_val = one_hot_labels[val_idx_min: val_idx_max]

        test_idx_min = val_idx_max
        test_idx_max = int(self.__n_data*(train_per + val_per + test_per))
        self.__a_test = a[test_idx_min: test_idx_max]
        self.__atld_test = atld[test_idx_min: test_idx_max]
        self.__x_test = x[test_idx_min: test_idx_max]
        self.__y_test = one_hot_labels[test_idx_min: test_idx_max]

        self.train = tf.data.Dataset.from_tensor_slices((
            tf.cast(self.__atld_train, tf.float32),
            tf.cast(self.__x_train, tf.float32),
            tf.cast(self.__y_train, tf.float32)
        )).batch(128)

        self.val = tf.data.Dataset.from_tensor_slices((
            tf.cast(self.__atld_val, tf.float32),
            tf.cast(self.__x_val, tf.float32),
            tf.cast(self.__y_val, tf.float32)
        )).batch(128)

        self.test = tf.data.Dataset.from_tensor_slices((
            tf.cast(self.__atld_test, tf.float32),
            tf.cast(self.__x_test, tf.float32),
            tf.cast(self.__y_test, tf.float32)
        )).batch(128)

    def get_train_data(self):
        return self.__a_train, self.__atld_train, self.__x_train, self.__y_train

    def get_val_data(self):
        return self.__a_val, self.__atld_val, self.__x_val, self.__y_val

    def get_test_data(self):
        return self.__a_test, self.__atld_test, self.__x_test, self.__y_test

    def get_class_names_by_one_hot_vectors(self, one_hot_vectors):
        class_names = self.__lb.inverse_transform(one_hot_vectors)
        return class_names

    def __generate_data(self):
        a_list = []
        atld_list = []
        x_list = []
        label_list = []
        generated_n_nodes = np.random.randint(self.__min_n_nodes, self.__max_n_nodes, self.__n_data)
        max_n_nodes_observed = 0
        for n_nodes in generated_n_nodes:
            g, label = self.__graph_generator(n_nodes)
            a = nx.adjacency_matrix(g).toarray()
            atld = self.renormalization(a)
            x = np.eye(a.shape[0])

            if max_n_nodes_observed < a.shape[0]:
                max_n_nodes_observed = a.shape[0]

            a_list.append(a)
            atld_list.append(atld)
            x_list.append(x)
            label_list.append(label)

        for i in range(self.__n_data):
            x = x_list[i]
            left_pad = np.zeros((x.shape[0], max_n_nodes_observed - x.shape[1]))

            x = np.concatenate([x, left_pad], axis=1)
            x_list[i] = x

        x, a = self.numpy_to_mega_batch(x_list, a_list)
        _, atld = self.numpy_to_mega_batch(x_list, atld_list)

        return a, atld, x, label_list

    def __graph_generator(self, n_nodes):
        random_idx = random.randint(0, len(self.__graph_types)-1)
        graph_label = self.__graph_types[random_idx]
        if graph_label == 'cycle':
            g = nx.cycle_graph(n_nodes)
        elif graph_label == 'star':
            g = nx.star_graph(n_nodes - 1)
        elif graph_label == 'wheel':
            g = nx.wheel_graph(n_nodes)
        elif graph_label == 'complete':
            g = nx.complete_graph(n_nodes)
        elif graph_label == 'lollipop':
            path_len = random.randint(2, n_nodes // 2)
            g = nx.lollipop_graph(m=n_nodes - path_len, n=path_len)
        elif graph_label == 'hypercube':
            g = nx.hypercube_graph(int(math.log2(n_nodes)))
            g = nx.convert_node_labels_to_integers(g)
        elif graph_label == 'circular_ladder':
            g = nx.circular_ladder_graph(n_nodes // 2)
        elif graph_label == 'grid':
            n_rows = random.randint(2, n_nodes // 2)
            n_cols = n_nodes // n_rows
            g = nx.grid_graph([n_rows, n_cols])
            g = nx.convert_node_labels_to_integers(g)

        return g, graph_label

