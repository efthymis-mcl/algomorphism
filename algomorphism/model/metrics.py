# author: Efthymis Michalis

import tensorflow as tf
from functools import partial


class ConfusionMatrix(tf.keras.metrics.Metric):
    def __init__(self, name='cm'):
        super(ConfusionMatrix, self).__init__(name=name)
        self.__cm_sum = self.add_weight(name='cm_sum', initializer='zeros')
    
    @tf.autograph.experimental.do_not_convert
    def update_state(self, y_true, y_pred):
        cm = self.current_confusion_matrix(y_true, y_pred)
        self.__cm_sum.assign_add(cm)
    
    @tf.autograph.experimental.do_not_convert
    def result(self):
        return self.__cm_sum
    
    def reset_states(self):
        self.__cm_sum.assign(0)
    
    def current_confusion_matrix(self, y_true, y_pred):
        y_pred_mask = tf.math.reduce_max(y_pred, -1, keepdims=True)
        y_pred_mask = tf.cast(y_pred == y_pred_mask, tf.float32)

        cm = tf.matmul(y_true, y_pred_mask, transpose_a=True)
        return cm


class CategoricalPrecision(ConfusionMatrix):
    def __init__(self, name='cp'):
        super(CategoricalPrecision, self).__init__(name=name)
        self.__pr_sum = self.add_weight(name='pr_sum', initializer='zeros')
    
    @tf.autograph.experimental.do_not_convert
    def update_state(self, y_true, y_pred):
        precision = self.current_precision(y_true, y_pred)
        self.__pr_sum.assign_add(precision)
    
    @tf.autograph.experimental.do_not_convert
    def result(self):
        return self.__pr_sum
    
    def reset_states(self):
        self.__pr_sum.assign(0)

    def current_precision(self, y_true, y_pred):
        cm = self.current_confusion_matrix(y_true, y_pred)
        true_positive = tf.reduce_sum([cm[i, i] for i in range(cm.shape[0])])
        false_positive = tf.reduce_sum([cm[i, j] for i in range(cm.shape[0]) for j in range(i)])
        precision = true_positive / (true_positive + false_positive)
        return precision


class CategoricalRecall(ConfusionMatrix):
    def __init__(self, name='cr'):
        super(CategoricalRecall, self).__init__(name=name)
        self.__rl_sum = self.add_weight(name='rl_sum', initializer='zeros')
    
    @tf.autograph.experimental.do_not_convert
    def update_state(self, y_true, y_pred):
        recall = self.current_recall(y_true, y_pred)
        self.__rl_sum.assign_add(recall)
    
    @tf.autograph.experimental.do_not_convert
    def result(self):
        return self.__rl_sum
    
    def reset_states(self):
        self.__rl_sum.assign(0)

    def current_recall(self, y_true, y_pred):
        cm = self.current_confusion_matrix(y_true, y_pred)
        print(cm)
        true_positive = tf.reduce_sum([cm[i, i] for i in range(cm.shape[0])])
        false_negative = tf.reduce_sum([cm[i, j] for j in range(cm.shape[0]) for i in range(j)])
        recall = true_positive / (true_positive + false_negative)
        return recall


class CategoricalF1Score(ConfusionMatrix):
    def __init__(self, name='f1'):
        super(CategoricalF1Score, self).__init__(name=name)
        self.__f1_sum = self.add_weight(name='f1_sum', initializer='zeros')
        self.__n = 0.0

    @tf.autograph.experimental.do_not_convert
    def update_state(self, y_true, y_pred):
        if len(y_true.shape) > 1 and len(y_pred.shape) > 1:
                 y_true = tf.reshape(y_true, (tf.reduce_prod(y_true.shape[:-1]), y_true.shape[-1]))
                 y_pred = tf.reshape(y_pred, (tf.reduce_prod(y_pred.shape[:-1]), y_pred.shape[-1]))
        f1 = self.current_f1(y_true, y_pred)
        self.__n += 1
        self.__f1_sum.assign_add(f1)
    
    @tf.autograph.experimental.do_not_convert
    def result(self):
        return self.__f1_sum / self.__n

    def reset_states(self):
        self.__n = 0
        self.__f1_sum.assign(0)

    def current_f1(self, y_true, y_pred):
        cm = self.current_confusion_matrix(y_true, y_pred)
        true_positive = tf.reduce_sum([cm[i, i] for i in range(cm.shape[0])])
        false_negative = tf.reduce_sum([cm[i, j] for j in range(cm.shape[0]) for i in range(j)])
        false_positive = tf.reduce_sum([cm[i, j] for i in range(cm.shape[0]) for j in range(i)])
        f1 = 2 * true_positive / (2 * true_positive + false_positive + false_negative)
        return f1


class WeightedCrossEntropyWithLogits(tf.keras.metrics.Metric):
    """
    Object based on `tf.nn.weighted_cross_entropy_with_logits` with normalization parameter.

    Attributes:
        loss: An object, weighted cross entropy with logits loss object where the weight given as partial,
        norm: A float, normalization parameter. This attribute multiply with the outcome of loss, loss_sum: An object,
        the summation of loss over batch where in most common usages at the end of epoch reset to $0$.
    """

    def __init__(self, w_p: float, norm: float):
        """
        Args:
            w_p (`float`): weight of loss (weighted cross entropy),
            norm (`float`): normalization parameter.
        """
        super(WeightedCrossEntropyWithLogits, self).__init__(name='weighted_cross_entropy_with_logits')
        self.__loss = partial(tf.nn.weighted_cross_entropy_with_logits, pos_weight=w_p)
        self.__norm = norm
        self.__loss_sum = self.add_weight(name='loss_sum', initializer='zeros')

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        """
        Update __loss_sum by averaging the output of __loss.

        Args:
            y_true (`tf.Tensor`): the true examples,
            y_pred (`tf.Tensor`): the predicted output of neural network.
        """
        self.__loss_sum.assign_add(tf.reduce_mean(self.__loss(y_true, y_pred)))

    def result(self) -> tf.Tensor:
        """
        The result of normalized loss sum. The most common usage is after the end of epoch.

        Returns:
            (`tf.Tensor`): the normalized loss sum.
        """
        norm_loss = self.__norm * self.__loss_sum
        return norm_loss

    def reset_states(self):
        """
        Reset loss_sum to $0$
        """

        self.__loss_sum.assign(0)


class MeanSquaredErrorWithLambda(tf.keras.metrics.Metric):
    """
    Based on `tf.keras.metrics.MeanSquaredError` object using a $lambda$ parameter where reduce or increase the "strength"
    of gradient at backpropagation step. A common usage is where adding two types of loss and this loss have different gradient
    and want to balance the gradients. $Lambda$ could be choice with trials or more wise with hyper-parameter search methods.

    Attributes:
      loss (`object`): Mean Square Error (MSE) tf Metric object,
      loss_sum (`object`): the summation of loss over batch where in most common usages at the end of epoch reset to $0$.
      lambda (`float`): a parameter where multiplied with loss_sum on result step.
    """

    def __init__(self, lamda: float = 1.0):
        """
        Args:
          lamda (`float`): lambda parameter where using at multiplication with MSE loss. Default 1.0.
        """
        super(MeanSquaredErrorWithLambda, self).__init__(name='l2mse')
        self.__loss = tf.keras.metrics.MeanSquaredError()
        self.__loss_sum = self.add_weight(name='loss_sum', initializer='zeros')
        self.__lambda = lamda

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        """
        Update loss_sum by averaging the output of loss.

        Args:
            y_true (`tf.Tensor`): the true examples,
            y_pred (`tf.Tensor`): the predicted output of neural network.

        """
        l2loss = self.__loss(y_true, y_pred)
        self.__loss_sum.assign_add(tf.reduce_mean(l2loss))

    def result(self) -> tf.Tensor:
        """
        The result of: loss multiplied with lambda. The most common usage is after the end of epoch.

        Returns:
            lambda_loss (`tf.Tensor`): loss multiplied with lambda.
        """
        lambda_loss = self.__lambda * self.__loss_sum
        return lambda_loss

    def reset_states(self):
        """
        Reset loss_sum to $0$
        """

        self.__loss_sum.assign(0)

    def set_lambda(self, lamda: float):
        """
        Lamda setter.

        Args:
            lamda (`float`): set new lambda.

        """
        self.__lambda = lamda


class CategoricalCrossEntropyWithLambda(tf.keras.metrics.Metric):
    """
    Based on ` tf.keras.metrics.CategoricalCrossentropy ` object using a $lambda$ parameter where reduce or increase the "strength"
    of gradient at backpropagation step. A common usage is where adding two types of loss and this loss have different gradient
    and want to balance the gradients. $Lambda$ could be choice with trials or more wise with hyper-parameter search methods.

    Attributes:
        loss (`object`): Categorical Cross Entropy (CCE) tf Metric object,
        loss_sum (`object`): the summation of loss over batch where in most common usages at the end of epoch reset to $0$,
        lambda (`float`): a parameter where multiplied with loss_sum on result step.
    """

    def __init__(self, lamda:float = 1.0):
        """
        Args:
            lamda (`optional`): lambda parameter where using at multiplication with CCE loss. Default is $1.0$.
        """

        super(CategoricalCrossEntropyWithLambda, self).__init__(name='cce_l')
        self.__loss = tf.keras.metrics.CategoricalCrossentropy()
        self.__loss_sum = self.add_weight(name='loss_sum', initializer='zeros')
        self.__lambda = lamda

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        """
        Update loss_sum by averaging the output of loss.

        Args:
            y_true (`tf.Tensor`): the true examples,
            y_pred (`tf.Tensor`): the predicted output of neural network.
        """

        l2loss = self.__loss(y_true, y_pred)
        self.__loss_sum.assign_add(tf.reduce_mean(l2loss))

    def result(self) -> tf.Tensor:
        """
        The result of: loss multiplied with lambda. The most common usage is after the end of epoch.

        Returns:
            (`tf.Tensor`): loss multiplied with lambda.
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
            lamda (`float`): set new lambda
        """

        self.__lambda = lamda


class LogCoshMetric(tf.keras.metrics.Metric):
    """
    LogCosh based on ` tf.keras.metrics.logcosh `.

    Attributes:
        loss: An object, Logarithm Cosh tf Metric object,
        loss_sum: An object, the summation of loss over batch where in most common usages at the end of epoch reset to $0$,
    """

    def __init__(self):
        super(LogCoshMetric, self).__init__(name="log_cosh_mtr")
        self.__loss = tf.keras.metrics.logcosh
        self.__loss_sum = self.add_weight(name="loss_sum", initializer='zeros')

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        """
        Update loss_sum by averaging the output of loss.

        Args:
            y_true (`tf.Tensor`): the true examples,
            y_pred (`tf.Tensor`): the predicted output of neural network.
        """

        logcosmtr = self.__loss(y_true, y_pred)
        self.__loss_sum.assign_add(tf.reduce_mean(logcosmtr))

    def result(self) -> tf.Tensor:
        """
        Result of current loss summation over batches.

        Returns:
           loss_sum (`tf.Tensor`): current loss summation result.
        """

        loss_sum = self.__loss_sum
        return loss_sum

    def reset_states(self):
        """
        Reset __loss_sum to $0$.
        """

        self.__loss_sum.assign(0)


class ClassWeightedCategoricalCrossEntropy(tf.keras.metrics.Metric):
    """
    Class weighted Categorical Cross Entropy metric for imbalance data.

    Attributes:
        class_weights (`list`): class weights of dataset's classes
        mean (`object`): A tf.keras.metrics.Mean object, mean for batches

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
        [0.25 0.25 0.2  0.15 0.15]
        >>> print(np.sum(class_weights))
        1.0
    """

    def __init__(self, class_weights: list):
        """
        Args:
            class_weights (`list`): class weights of dataset's classes
        """

        super(ClassWeightedCategoricalCrossEntropy, self).__init__(name='cwccem')
        self.__class_weights = class_weights
        self.__mean = tf.keras.metrics.Mean()

    @tf.autograph.experimental.do_not_convert
    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        """
        Compute batch Categorical Cross Entropy loss and update_mean.

        Args:
            y_true (`tf.Tensor`): the true examples,
            y_pred (`tf.Tensor`): the predicted output of neural network.
        """

        loss = y_true * tf.math.log(y_pred)
        loss = self.__class_weights * loss
        loss = -tf.reduce_sum(loss, axis=-1)
        self.__mean.update_state(loss)

    @tf.autograph.experimental.do_not_convert
    def result(self) -> tf.Tensor:
        """
        Result of current loss summation over batches.

        Returns:
            (`tf.Tensor`): current loss summation result.
        """

        mean_loss = self.__mean.result().numpy()
        return mean_loss

    def reset_states(self):
        """
        Reset mean to $0$.
        """

        self.__mean.reset_state()
