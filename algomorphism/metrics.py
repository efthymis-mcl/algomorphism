from tensorflow.keras.metrics import Metric
import tensorflow as tf
from functools import partial


class WeightedCrossEntropyWithLogits(Metric):
    """
  Object based on ` tf.nn.weighted_cross_entropy_with_logits ` with normalization parameter.

  Attributes: __loss: An object, weighted cross entropy with logits loss object where the weight given as partial,
  __norm: A float, normalization parameter. This attribute multiply with the outcome of __loss, __loss_sum: An object,
  the sumation of __loss over batch where in most common usages at the end of epoch reset to $0$.
    """

    def __init__(self, w_p, norm):
        """
    Args:
      w_p: A float, weight of loss (weighted cross entropy),
      norm: A float, normalization parameter.
    """
        super(WeightedCrossEntropyWithLogits, self).__init__(name='weighted_cross_entropy_with_logits')
        self.__loss = partial(tf.nn.weighted_cross_entropy_with_logits, pos_weight=w_p)
        self.__norm = norm
        self.__loss_sum = self.add_weight(name='loss_sum', initializer='zeros')

    def update_state(self, y_true, y_pred):
        """
    Update __loss_sum by averaging the output of __loss.

    Args:
      y_true: A tf_tensor, the true examples,
      y_pred: A tf_tensor, the predicted output of neural network.
    """
        self.__loss_sum.assign_add(tf.reduce_mean(self.__loss(y_true, y_pred)))

    def result(self):
        """
    The result of normalized loss sum. The most common usage is after the end of epoch.

    Returns:
      norm_loss: A tf_float, the normalized loss sum.
    """
        norm_loss = self.__norm * self.__loss_sum
        return norm_loss

    def reset_states(self):
        """
    Reset __loss_sum to $0$
    """
        self.__loss_sum.assign(0)


class MeanSquaredErrorWithLambda(Metric):
    """
    Based on ` tf.keras.metrics.MeanSquaredError ` object using a $lambda$ parameter where reduce or increase the "strength"
    of gradient at backpropagation step. A common usage is where adding two types of loss and this loss have different gradient
    and want to balance the gradients. $Lambda$ could be choice with trials or more wise with hyper-parameter search methods.

    Attributes:
      __loss: An object, Mean Square Error (MSE) tf Metric object,
      __loss_sum: An object, the sumation of __loss over batch where in most common usages at the end of epoch reset to $0$.
      __lambda: A float, a parameter where multiplied with __loss_sum on result step.
    """

    def __init__(self, lamda=1.0):
        """
        Args:
          lamda: float (optional), lambda parameter where using at multiplication with MSE loss.
          Default is $1.0$.
        """
        super(MeanSquaredErrorWithLambda, self).__init__(name='l2mse')
        self.__loss = tf.keras.metrics.MeanSquaredError()
        self.__loss_sum = self.add_weight(name='loss_sum', initializer='zeros')
        self.__lambda = lamda

    def update_state(self, y_true, y_pred):
        """
    Update __loss_sum by averaging the output of __loss.

    Args:
      y_true: A tf_tensor, the true examples,
      y_pred: A tf_tensor, the predicted output of neural network.

    """
        l2loss = self.__loss(y_true, y_pred)
        self.__loss_sum.assign_add(tf.reduce_mean(l2loss))

    def result(self):
        """
    The result of: loss multiplied with lambda. The most common usage is after the end of epoch.

    Returns:
      lambda_loss: A tf_float, loss multiplied with lambda.
    """
        lambda_loss = self.__lambda * self.__loss_sum
        return lambda_loss

    def reset_states(self):
        """
    Reset __loss_sum to $0$

    """
        self.__loss_sum.assign(0)

    def set_lambda(self, lamda: float):
        """
    Lamda setter.
    Args:
      lamda: A float, new lambda

    """
        self.__lambda = lamda


class CategoricalCrossEntropyWithLambda(Metric):
    """
    Based on ` tf.keras.metrics.CategoricalCrossentropy ` object using a $lambda$ parameter where reduce or increase the "strength"
    of gradient at backpropagation step. A common usage is where adding two types of loss and this loss have different gradient
    and want to balance the gradients. $Lambda$ could be choice with trials or more wise with hyper-parameter search methods.

    Attributes:
        __loss: An object, Categorical Cross Entropy (CCE) tf Metric object,
        __loss_sum: An object, the sumation of __loss over batch where in most common usages at the end of epoch reset to $0$,
        __lambda: A float, a parameter where multiplied with __loss_sum on result step.
    """

    def __init__(self, lamda=1.0):
        """
        Args:
            lamda: float (optional), lambda parameter where using at multiplication with CCE loss. Default is 1.0 .
        """
        super(CategoricalCrossEntropyWithLambda, self).__init__(name='cce_l')
        self.__loss = tf.keras.metrics.CategoricalCrossentropy()
        self.__loss_sum = self.add_weight(name='loss_sum', initializer='zeros')
        self.__lambda = lamda

    def update_state(self, y_true, y_pred):
        """
        Update __loss_sum by averaging the output of __loss.

        Args:
            y_true: A tf_tensor, the true examples,
            y_pred: A tf_tensor, the predicted output of neural network.
        """

        l2loss = self.__loss(y_true, y_pred)
        self.__loss_sum.assign_add(tf.reduce_mean(l2loss))

    def result(self):
        """
        The result of: loss multiplied with lambda. The most common usage is after the end of epoch.

        Returns:
            lambda_loss: A tf_float, loss multiplied with lambda.
        """

        lambda_loss = self.__lambda * self.__loss_sum
        return lambda_loss

    def reset_states(self):
        """
        Reset __loss_sum to $0$
        """

        self.__loss_sum.assign(0)

    def set_lambda(self, lamda: float):
        """
        Lambda setter.
            Args:
                lamda: A float, new lambda
        """

        self.__lambda = lamda


class LogCoshMetric(Metric):
    """
    LogCosh based on ` tf.keras.metrics.logcosh `.

    Attributes:
        __loss: An object, Logarithm Cosh tf Metric object,
        __loss_sum: An object, the summation of __loss over batch where in most common usages at the end of epoch reset to $0$,
    """

    def __init__(self):
        super(LogCoshMetric, self).__init__(name="log_cosh_mtr")
        self.__loss = tf.keras.metrics.logcosh
        self.__loss_sum = self.add_weight(name="loss_sum", initializer='zeros')

    def update_state(self, y_true, y_pred):
        """
        Update __loss_sum by averaging the output of __loss.

        Args:
            y_true: A tf_tensor, the true examples,
            y_pred: A tf_tensor, the predicted output of neural network.
        """

        logcosmtr = self.__loss(y_true, y_pred)
        self.__loss_sum.assign_add(tf.reduce_mean(logcosmtr))

    def result(self):
        """
        Result of current loss summation over batches.

        Returns:
           loss_sum: A tf_float, current loss summation result.
        """

        loss_sum = self.__loss_sum
        return loss_sum

    def reset_states(self):
        """
        Reset __loss_sum to $0$.
        """

        self.__loss_sum.assign(0)


class MeanEuclideanError(Metric):
    """
    Mean Euclidean Error over batches. Special case for Post thesis work using min_xy where is minimum value of XY cordinates
    and maximum value of XY cordinates of given Positioning data (see Reference).

    Attributes:
        __loss: A lambda object, given y (true examples) and y_ (prediction of neural network) return the Euclidean Distance,
        __mean: An object, the mean of __loss over batch where in most common usages at the end of epoch reset to $0$,
        __min_xy: A float, minimum value of XY coordinates,
        __max_xy: A float, maximum value of XY coordinates.
    """

    def __init__(self, min_xy: float, max_xy: float):
        """
        Args:
            min_xy: A float, minimum value of XY coordinates,
            max_xy: A float, maximum value of XY coordinates.
        """

        super(MeanEuclideanError, self).__init__(name='mee')
        self.__loss = lambda y, y_: tf.sqrt(tf.reduce_sum((y - y_) ** 2, axis=1))
        self.__mean = tf.keras.metrics.Mean()
        self.__min_xy, self.__max_xy = (min_xy, max_xy)

    def update_state(self, y_true, y_pred):
        """
        Update __loss_sum by averaging the output of __loss.

        Args:
            y_true: A tf_tensor, the true examples,
            y_pred: A tf_tensor, the predicted output of neural network.
        """

        y_true = y_true * (self.__max_xy - self.__min_xy) + self.__min_xy
        y_pred = y_pred * (self.__max_xy - self.__min_xy) + self.__min_xy
        loss = self.__loss(y_true, y_pred)
        self.__mean.update_state(loss)

    def result(self):
        """
        Result of current loss summation over batches.

        Returns:
            loss_mean: A tf_float, current loss summation result.
        """

        loss_mean = self.__mean.result()
        return loss_mean

    def reset_states(self):
        """
        Reset __mean to $0$.
        """

        self.__mean.assign(0)


class ClassWeightedCategoricalCrossEntropy(Metric):
    """
    Class weighted Categorical Cross Entropy metric for imbalance data.

    Attributes:
        __class_weights: A list like, class weights of dataset's classes
        __mean: A tf.keras.metrics.Mean object, mean for batches

    Examples:
        >>> import numpy as np
        >>> from sklearn.preprocessing import LabelBinarizer

        >>> data_class = [0, 1, 2, 4, 1, 0, 2, 1, 3, 3, 4, 2, 1, 0, 2, 4, 3, 0, 0, 1]
        >>> n_data = len(data_class)
        >>> print(n_data)
        20
        >>> n_class = max(data_class) + 1
        >>> print(n_class)
        5
        >>> data_class_one_hot = LabelBinarizer().fit_transform(data_class)
        >>> class_weights = np.sum(data_class_one_hot, axis= 0)/n_data
        >>> print(class_weights)
        [0.25 0.25 0.2 0.15 0.15]
        >>> print(np.sum(class_weights))
        1.0
    """

    def __init__(self, class_weights):
        """
        Args:
            class_weights: A list like, class weights of dataset's classes
        """

        super(ClassWeightedCategoricalCrossEntropy, self).__init__(name='cwccem')
        self.__class_weights = class_weights
        self.__mean = tf.keras.metrics.Mean()

    def update_state(self, y_true, y_pred):
        """
        Compute batch Categorical Cross Entropy loss and update__mean.

        Args:
            y_true: A tf_tensor, the true examples,
            y_pred: A tf_tensor, the predicted output of neural network.
        """

        loss = y_true * tf.math.log(y_pred)
        loss = self.__class_weights * loss
        loss = -tf.reduce_sum(loss, axis=-1)
        self.__mean.update_state(loss)

    def result(self):
        """
        Result of current loss summation over batches.

        Returns:
            mean_loss: A tf_float, current loss summation result.
        """

        mean_loss = self.__mean.result().numpy()
        return mean_loss

    def reset_states(self):
        """
        Reset __mean to $0$.
        """

        self.__mean.reset_state()
