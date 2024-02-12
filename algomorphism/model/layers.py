# author: Efthymis Michalis

import tensorflow as tf


class FullConnected(tf.Module):
    """
    Full Connected layer.
    """
    def __init__(self, in_features, out_features, activation=None, weights_initializer=None, bias_initializer=None, name="fc"):
        super(FullConnected, self).__init__(name=name)
        if weights_initializer is None:
            weights_initializer = tf.keras.initializers.GlorotNormal()

        if bias_initializer is None:
            bias_initializer = tf.keras.initializers.GlorotNormal()

        weights = weights_initializer(shape=[in_features, out_features])
        bias = bias_initializer(shape=[out_features])

        self.weights = tf.Variable(
            weights,
            name='weights'
        )
        self.bias = tf.Variable(
            bias,
            name='bias'
        )
        self.activation = tf.keras.activations.get(activation)

    def __call__(self, inputs):
        x = tf.matmul(inputs, self.weights) + self.bias
        x = self.activation(x)
        return x


class GraphConv(tf.Module):
    """
    Graph Convolutional Layer
    """
    def __init__(self, in_features, out_features, activation=None, name="gcn"):
        super(GraphConv, self).__init__(name=name)

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


class InnerProduct(tf.Module):
    """
    Inner Product layer (weights is optional)
    """
    def __init__(self, in_features=None, activation=None, name="ip"):
        super(InnerProduct, self).__init__(name=name)
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


class Attention(tf.Module):
    """
    Attention layer.

    References:
        - Attention Is All You Need: https://arxiv.org/abs/1706.03762
    """
    def __init__(self, name="attention"):
        super(Attention, self).__init__(name=name)

    def __call__(self, q, k, v):
        y = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(k.shape[-1], tf.float32)
        norm = tf.sqrt(dk)
        y = y / norm
        y = tf.nn.softmax(y)
        y = tf.matmul(y, v)
        return y


class MultiHeadAttention(tf.Module):
    """
    Multi Head Attention layer.

    References:
        - Attention Is All You Need: https://arxiv.org/abs/1706.03762
    """
    def __init__(self, feature_dim, attention_dims, n_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)

        k_dim = attention_dims[0]
        v_dim = attention_dims[1]
        self.n_heads = n_heads
        self.attention = Attention()
        
        weight_initializer = tf.keras.initializers.GlorotNormal()

        for i in range(self.n_heads):
            wqi = weight_initializer(shape=[feature_dim, k_dim])
            wki = weight_initializer(shape=[feature_dim, k_dim])
            wvi = weight_initializer(shape=[feature_dim, v_dim])
            setattr(self, 'wq{}'.format(i), 
                    tf.Variable(wqi, 'wq{}'.format(i)))
            setattr(self, 'wk{}'.format(i), 
                    tf.Variable(wki, 'wk{}'.format(i)))
            setattr(self, 'wv{}'.format(i), 
                    tf.Variable(wvi, 'wv{}'.format(i)))
        
        self.wo = tf.Variable(weight_initializer(shape=[n_heads*v_dim, feature_dim]), 'wo')
    
    def __call__(self, q, k, v):
        heads = []
        for i in range(self.n_heads):
            wqi = getattr(self, 'wq{}'.format(i))
            wki = getattr(self, 'wk{}'.format(i))
            wvi = getattr(self, 'wv{}'.format(i))
            
            lq = tf.matmul(q, wqi)
            lk = tf.matmul(k, wki)
            lv = tf.matmul(v, wvi)
            attention = self.attention(lq, lk, lv)
            heads.append(attention)
        
        y = tf.concat(heads, axis=-1)
        y = tf.matmul(y, self.wo)
        return y
