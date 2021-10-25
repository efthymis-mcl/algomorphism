import networkx as nx
import numpy as np
import random
import math
from algomorphism.datasets import GraphBaseDataset
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
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


class SeenUnseenBase(object):
    def __init__(self, seen, unseen):
        self.seen = seen
        self.unseen = unseen


# Zero Shot Learning task
class BubbleDataset(object):

    def __init__(self, n_data=5000, train_per=0.6, val_per=0.3, test_per=0.1, seen_classes=None, unseen_classes=None,
                 sigma=1.0):
        class_names = ['top_left', 'top_right', 'middle', 'bottom_left', 'bottom_right']
        class_embeddings = np.array([
            [-1.0, 1.0],
            [1.0, 1.0],
            [0.0, 0.0],
            [-1.0, -1.0],
            [1.0, -1.0]
        ])

        if seen_classes is None:
            seen_classes = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
        if unseen_classes is None:
            unseen_classes = ['middle']

        examples_per_class = n_data//len(class_names)
        self.data_dict = dict.fromkeys(class_names)
        for k, cl_emb in zip(self.data_dict.keys(), class_embeddings):
            self.data_dict[k] = {
                'class_emb': cl_emb,
                'x': cl_emb + sigma*np.random.randn(examples_per_class, 2)
            }

        x_train = []
        y_emb_train = []
        y_train = []

        x_val_seen = []
        y_emb_val_seen = []
        y_val_seen = []
        x_val_unseen = []
        y_emb_val_unseen = []
        y_val_unseen = []

        x_test_seen = []
        y_emb_test_seen = []
        y_test_seen = []
        x_test_unseen = []
        y_emb_test_unseen = []
        y_test_unseen = []

        self.__lb = LabelBinarizer()
        self.__lb.fit(class_names)

        for k, v in self.data_dict.items():
            one_hot_key = self.__lb.transform([k])[0]
            if k in seen_classes:
                x_tr, x_vl_ts = train_test_split(v['x'], train_size=train_per)
                x_vl, x_ts = train_test_split(x_vl_ts, train_size=test_per / train_per)

                x_train.append(x_tr)
                x_val_seen.append(x_vl)
                x_test_seen.append(x_ts)

                y_emb_train.append(np.stack([v['class_emb']]*x_tr.shape[0], axis=0))
                y_emb_val_seen.append(np.stack([v['class_emb']]*x_vl.shape[0], axis=0))
                y_emb_test_seen.append(np.stack([v['class_emb']]*x_ts.shape[0], axis=0))

                y_train.append(np.stack([one_hot_key]*x_tr.shape[0], axis=0))
                y_val_seen.append(np.stack([one_hot_key] * x_vl.shape[0], axis=0))
                y_test_seen.append(np.stack([one_hot_key] * x_ts.shape[0], axis=0))

            elif k in unseen_classes:
                x_vl, x_ts = train_test_split(v['x'], train_size=test_per / val_per)
                x_val_unseen.append(x_vl)
                x_test_unseen.append(x_ts)
                y_emb_val_unseen.append(np.stack([v['class_emb']] * x_vl.shape[0], axis=0))
                y_emb_test_unseen.append(np.stack([v['class_emb']] * x_ts.shape[0], axis=0))

                y_val_unseen.append(np.stack([one_hot_key] * x_vl.shape[0], axis=0))
                y_test_unseen.append(np.stack([one_hot_key] * x_ts.shape[0], axis=0))

        x_train = np.concatenate(x_train, axis=0)
        y_emb_train = np.concatenate(y_emb_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)

        x_val_seen = np.concatenate(x_val_seen, axis=0)
        y_emb_val_seen = np.concatenate(y_emb_val_seen, axis=0)
        y_val_seen = np.concatenate(y_val_seen, axis=0)
        x_val_unseen = np.concatenate(x_val_unseen, axis=0)
        y_emb_val_unseen = np.concatenate(y_emb_val_unseen, axis=0)
        y_val_unseen = np.concatenate(y_val_unseen, axis=0)

        x_test_seen = np.concatenate(x_test_seen, axis=0)
        y_emb_test_seen = np.concatenate(y_emb_test_seen, axis=0)
        y_test_seen = np.concatenate(y_test_seen, axis=0)
        x_test_unseen = np.concatenate(x_test_unseen, axis=0)
        y_emb_test_unseen = np.concatenate(y_emb_test_unseen, axis=0)
        y_test_unseen = np.concatenate(y_test_unseen, axis=0)

        rnd_idx = np.arange(0, x_train.shape[0])
        np.random.shuffle(rnd_idx)
        x_train = x_train[rnd_idx]
        y_emb_train = y_emb_train[rnd_idx]
        y_train = y_train[rnd_idx]

        rnd_idx = np.arange(0, x_val_seen.shape[0])
        np.random.shuffle(rnd_idx)
        x_val_seen = x_val_seen[rnd_idx]
        y_emb_val_seen = y_emb_val_seen[rnd_idx]
        y_val_seen = y_val_seen[rnd_idx]

        rnd_idx = np.arange(0, x_val_unseen.shape[0])
        np.random.shuffle(rnd_idx)
        x_val_unseen = x_val_unseen[rnd_idx]
        y_emb_val_unseen = y_emb_val_unseen[rnd_idx]
        y_val_unseen = y_val_unseen[rnd_idx]

        rnd_idx = np.arange(0, x_test_seen.shape[0])
        np.random.shuffle(rnd_idx)
        x_test_seen = x_test_seen[rnd_idx]
        y_emb_test_seen = y_emb_test_seen[rnd_idx]
        y_test_seen = y_test_seen[rnd_idx]

        rnd_idx = np.arange(0, x_test_unseen.shape[0])
        np.random.shuffle(rnd_idx)
        x_test_unseen = x_test_unseen[rnd_idx]
        y_emb_test_unseen = y_emb_test_unseen[rnd_idx]
        y_test_unseen = y_test_unseen[rnd_idx]

        self.train = tf.data.Dataset.from_tensor_slices((
            tf.cast(x_train, tf.float32),
            tf.cast(y_emb_train, tf.float32),
            tf.cast(y_train, tf.float32),
        )).batch(128)

        self.val = SeenUnseenBase(
            seen=tf.data.Dataset.from_tensor_slices((
                tf.cast(x_val_seen, tf.float32),
                tf.cast(y_emb_val_seen, tf.float32),
                tf.cast(y_val_seen, tf.float32),
            )).batch(128),
            unseen=tf.data.Dataset.from_tensor_slices((
                tf.cast(x_val_unseen, tf.float32),
                tf.cast(y_emb_val_unseen, tf.float32),
                tf.cast(y_val_unseen, tf.float32),
            )).batch(128)
        )
        self.test = SeenUnseenBase(
            seen=tf.data.Dataset.from_tensor_slices((
                tf.cast(x_test_seen, tf.float32),
                tf.cast(y_emb_test_seen, tf.float32),
                tf.cast(y_test_seen, tf.float32),
            )).batch(128),
            unseen=tf.data.Dataset.from_tensor_slices((
                tf.cast(x_test_unseen, tf.float32),
                tf.cast(y_emb_test_unseen, tf.float32),
                tf.cast(y_test_unseen, tf.float32),
            )).batch(128)
        )
