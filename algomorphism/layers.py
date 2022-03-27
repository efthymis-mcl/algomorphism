import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense


class FC(tf.Module):
    """
    Full Connected layer.
    """
    def __init__(self, in_features, out_features, activation=None):
        super(FC, self).__init__(name="fc")

        self.weights = tf.Variable(
            tf.keras.initializers.GlorotUniform()(shape=[in_features, out_features]),
            name='weights'
        )
        self.bias = tf.Variable(
            tf.keras.initializers.GlorotUniform()(shape=[out_features]),
            name='bias'
        )
        self.activation = tf.keras.activations.get(activation)

    def __call__(self, inputs):
        x = tf.matmul(inputs, self.weights) + self.bias
        x = self.activation(x)
        return x


class GCN(tf.Module):
    """
    Graph Convolutional Layer
    """
    def __init__(self, in_features, out_features, activation=None, name=None):
        if name is None:
            name = "gcn"
        super(GCN, self).__init__(name=name)

        self.weights = tf.Variable(
            tf.keras.initializers.GlorotUniform()(shape=[in_features, out_features]),
            name='weights'
        )
        self.bias = tf.Variable(
            tf.keras.initializers.GlorotUniform()(shape=[out_features]),
            name='bias'
        )
        self.activation = tf.keras.activations.get(activation)

    def __call__(self, X, Atld):
        x = tf.matmul(Atld, X)
        x = tf.matmul(x, self.weights) + self.bias
        x = self.activation(x)
        return x


class IP(tf.Module):
    """
    Inner Product layer (weights is optional)
    """
    def __init__(self, in_features=None, activation=None):
        super(IP, self).__init__(name='ip')
        self.weights = None
        if in_features is not None:
            self.weights = tf.Variable(
                tf.random.normal([in_features, in_features]), name='weights')

        self.activation = tf.keras.activations.get(activation)

    def __call__(self, inputs):
        x = inputs
        if self.weights is not None:
            x = tf.matmul(x, self.weights)
        x = tf.matmul(x, inputs, transpose_b=True)
        x = self.activation(x)
        return x


class Attention(tf.keras.layers.Layer):
    """
    Attention Mechanism layer.

    References:
        - Attention Is All You Need: https://arxiv.org/abs/1706.03762
    """

    def __init__(self, units):
        super(Attention, self).__init__(name='attention')

        self.fc = Dense(units, activation='tanh')
        self.V = Dense(1)

    def call(self, deep_features):
        at = self.fc(deep_features)
        at = self.V(at)

        at = tf.nn.softmax(at, axis=1)

        at = tf.multiply(at, deep_features)

        at = tf.reduce_sum(at, axis=-2)

        return at


class BahdanauAttention(tf.keras.layers.Layer):
    """

    References:
        - End-to-End Attention-based Large Vocabulary Speech Recognition: https://arxiv.org/abs/1508.04395
        - Bahdanau Attention implantation: https://www.tensorflow.org/text/tutorials/nmt_with_attention
    """
    def __init__(self, units):
        super(BahdanauAttention, self).__init__(name='bahdanau_attention')

        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, deep_features, hidden):
        hidden = tf.expand_dims(hidden, 1)
        at = tf.nn.tanh(tf.add(self.W1(deep_features), self.W1(hidden)))
        at = self.V(at)

        at = tf.nn.softmax(at, axis=1)
        at = tf.multiply(at, deep_features)
        at = tf.reduce_sum(at, axis=-2)

        return at


class LSTM_At(tf.keras.layers.Layer):
    def __init__(self, units, name, activation='tanh'):
        super(LSTM_At, self).__init__(name=name)

        self.lstm1 = LSTM(units, return_sequences=True, activation=activation)
        self.at1 = Attention(units)

    def call(self, input):
        x = self.lstm1(input)
        y = self.at1(x)
        return y
