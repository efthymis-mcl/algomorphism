# author: Efthymis Michalis

import tensorflow as tf
from functools import partial


class WeightedCrossEntropyWithLogits(object):
    """
     Object based on ` tf.nn.weighted_cross_entropy_with_logits ` with normalization parameter.

     Attributes:
        loss (`object`): weighted cross entropy with logits loss object where the weight given as partial,
        norm (`float`): A float, normalization parameter. This attribute multiply with the outcome of loss,
    """

    def __init__(self, w_p: float, norm: float):
        """
        Args:
            w_p (`float`): weight of loss (weighted cross entropy),
            norm (`float`): normalization parameter.
        """
        self.__loss = partial(tf.nn.weighted_cross_entropy_with_logits, pos_weight=w_p)
        self.__norm = norm

    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Call function.

        Args:
            y_true (`tf.Tensor`): the true examples,
            y_pred (`tf.Tensor`): the predicted output of neural network.

        Returns:
            `tf.Tensor`:  mean loss multiplied by norm
        """
        loss = self.__norm * tf.reduce_mean(self.__loss(y_true, y_pred))
        return loss


class MeanSquaredErrorWithLambda(object):
    """
    Based on `tf.keras.losses.MeanSquaredError` object using a $lambda$ parameter where reduce or increase the "strength"
    of gradient at backpropagation step. A common usage is where adding two types of loss and this loss have different gradient
    and want to balance the gradients. $Lambda$ could be choice with trials or more wise with hyper-parameter search methods.

    Attributes:
        loss: An object, Mean Square Error (MSE) tf Loss object,
        lambda: A float, a parameter where multiplied with loss on result step.
    """

    def __init__(self, lamda: float = 1.0):
        """
        Args:
            lamda (`float`): lambda parameter where using at multiplication with MSE loss, default is `1.0`.
        """
        self.__lambda = lamda
        self.__loss = tf.keras.losses.MeanSquaredError()

    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Call function.

        Args:
            y_true (`tf.Tensor`): the true examples,
            y_pred (`tf.Tensor`): the predicted output of neural network.

        Returns:
           `tf.Tensor`: mean loss multiplied by lambda
        """
        return self.__lambda * self.__loss(y_true, y_pred)

    def set_lambda(self, lamda):
        self.__lambda = lamda


class CategoricalCrossEntropyWithLambda(object):
    """
    Based on ` tf.keras.losses.CategoricalCrossentropy ` object using a $lambda$ parameter where reduce or increase the "strength"
    of gradient at backpropagation step. A common usage is where adding two types of loss and this loss have different gradient
    and want to balance the gradients. $Lambda$ could be choice with trials or more wise with hyperparameter search methods.

    Attributes:
        __loss: An object, Categorical Cross Entropy (CCE) tf Loss object,
        __lambda: A float, a parameter where multiplied with __loss on result step.
    """

    def __init__(self, lamda: float = 1.0):
        """
        Args:
            lamda (`float`): lambda parameter where using at multiplication with CCE loss, default is `1.0`.
        """
        self.__lambda = lamda
        self.__loss = tf.keras.losses.CategoricalCrossentropy()

    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Call function.

        Args:
            y_true (`tf.Tensor`): the true examples,
            y_pred (`tf.Tensor`): the predicted output of neural network.

        Returns:
            `tf.Tensor`: mean loss multiplied by lambda.
        """
        loss = self.__lambda * self.__loss(y_true, y_pred)
        return loss

    def set_lambda(self, lamda: float):
        """
        Lambda setter.

        Args:
            lamda (`float`): new lambda parameter.
        """
        self.__lambda = lamda


class ClassWeightedCategoricalCrossEntropy(object):
    """
    Class weighted Categorical Cross Entropy loss object for imbalance data.

    Attributes:
        __class_weights: A list like, class weights of dataset's classes
    """

    def __init__(self, class_weights):
        """
        Args:
            class_weights (`list`): class weights of dataset's classes.
        """

        self.__class_weights = class_weights

    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute mean batch Categorical Cross Entropy loss.

        Args:
            y_true (`tf.Tensor`): the true examples,
            y_pred (`tf.Tensor`): the predicted output of neural network.

        Returns:
            `tf.Tensor`: batched class weighted categorical cross entropy.
        """

        loss = y_true * tf.math.log(y_pred)
        loss = self.__class_weights * loss
        loss = -tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
        return loss
