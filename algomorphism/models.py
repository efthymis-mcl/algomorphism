import tensorflow as tf
from .base import BaseNeuralNetwork, MetricBase, LossBase
from .metrics import WeightedCrossEntropyWithLogits as mWCEL
from .layers import GCN, IP, FC


class GAE(tf.Module, BaseNeuralNetwork):
    """
    Graph Auto Encoder.

    References:
        - Variational Graph Auto-Encoders: https://arxiv.org/abs/1611.07308
        - Graph Auto Encoder implantation: https://github.com/tkipf/gae
    """
    def __init__(self, dataset, df_list, w_p, norm=1, early_stop_vars=None, weights_outfile=None, optimizer="SGD",
                 learning_rate=1e-2, ip_weights=False):
        tf.Module.__init__(self, name='gae')
        status = [
            [0],
            [0],
            [1, 2]
        ]

        self.cost_mtr = MetricBase(self,
                                   [mWCEL(w_p, norm)],
                                   status,
                                   [0],
                                   1
                                   )

        self.score_mtr = MetricBase(self,
                                    [tf.keras.metrics.BinaryAccuracy()],
                                    status,
                                    [0]
                                    )

        self.cost_loss = LossBase(self,
                                  [lambda ytr, ypr: norm * tf.reduce_mean(
                                      tf.nn.weighted_cross_entropy_with_logits(ytr, ypr, w_p))],
                                  status,
                                  [0]
                                  )

        BaseNeuralNetwork.__init__(self, status, dataset, early_stop_vars, weights_outfile, optimizer, learning_rate)

        for i, dfi in enumerate(df_list[:-2]):
            setattr(self, 'gcn{}'.format(i),
                    GCN(df_list[i], df_list[i+1], 'relu', name='gcn{}'.format(i))
                    )
        self.gcn_z = GCN(df_list[-2], df_list[-1], name="gcn_z")

        if ip_weights:
            self.ip = IP(df_list[-1])
        else:
            self.ip = IP()

        self.__depth = len(df_list)

    def encoder(self, x, atld):
        for i in range(self.__depth - 2):
            x = getattr(self, 'gcn{}'.format(i))(x, atld)

        z = self.gcn_z(x, atld)
        return z

    def decoder(self, z):
        x = self.ip(z)
        return x

    def __call__(self, inputs, is_score=False):
        z = self.encoder(inputs[0], inputs[1])
        y = self.decoder(z)

        if is_score:
            y = tf.nn.sigmoid(y)

        return tuple((y,))


class GCNClassifier(tf.Module, BaseNeuralNetwork):
    """
    Batch Graph Convolutional Network Classifier. In This architecture of Neural Network, the weights shared for all batch examples.
    So the learning generalized up to maximum number nodes of training examples.

    References:
        - Semi-Supervised Classification with Graph Convolutional Networks: https://arxiv.org/abs/1609.02907
        - Graph Convolutional Network implantation: https://github.com/tkipf/gcn
    """
    def __init__(self, dataset, df_list, nc, optimizer=None, clip_norm=0.0, early_stop_vars=None,
                 name='gcnclf'):

        tf.Module.__init__(self, name=name)
        status = [
            [0],
            [0],
            [1, 2]
        ]

        self.score_mtr = MetricBase(self,
                                    [tf.keras.metrics.CategoricalAccuracy()],
                                    status,
                                    [0]
                                    )
        self.cost_mtr = MetricBase(self,
                                   [tf.keras.metrics.CategoricalCrossentropy()],
                                   status,
                                   [0],
                                   1
                                   )
        self.cost_loss = LossBase(self,
                                  [tf.keras.losses.CategoricalCrossentropy()],
                                  status,
                                  [0]
                                  )

        BaseNeuralNetwork.__init__(self, status, dataset=dataset, optimizer=optimizer, clip_norm=clip_norm,
                                   early_stop_vars=early_stop_vars)

        for i, dfi in enumerate(df_list[:-1]):
            setattr(self, 'gcn{}'.format(i),
                    GCN(df_list[i], df_list[i+1], 'relu', name='gcn{}'.format(i)))

        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = FC(df_list[0] * df_list[-1], 512, 'relu')
        self.out = FC(512, nc, 'softmax')

        self.__depth = len(df_list)

    def __call__(self, inputs):
        x = inputs[0]
        atld = inputs[1]
        for i in range(self.__depth - 1):
            x = getattr(self, 'gcn{}'.format(i))(x, atld)

        x = self.flatten(x)
        x = self.fc1(x)
        y = self.out(x)
        y = tuple([y])
        return y
