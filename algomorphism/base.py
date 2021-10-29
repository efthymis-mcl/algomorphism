from itertools import product
from typing import Union

import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
import copy
from functools import partial


class MetricLossBase(object):
    """
    Base of (Metric & Loss)-Base classes. This class uses 3 types for examples (input or true output) where going through
    Metric object (this type of object have a usage of performance of Neural Network). In this case the status is 2. For
    Loss object: this type of object have a usage of calculate the gradient of Cost/Loss. In this case the status
    is 1. Finally, for input examples, the status is 0. The reason of tow type of status (1 & 2) is that in some cases such as
    Zero Shot learning the top output of Neural Network could be a vector where this vector goes through post process into classifier.
    In this case only use loss type for top output of Neural Network and input status.

    - 0: input
    - 1: output loss
    - 2: output metric
    """
    def __init__(self, model=None):
        """

        Args:
            model (`BaseNeuralNetwork`): model to parse loss call function.
        """
        if model is not None:
            if hasattr(model, 'call'):
                self.__callfn = model.call
            else:
                self.__callfn = model.__call__

    def predict_outputs(self, inputs, is_score=False, set_is_score=False):
        """
        Predict outputs by `BaseNeuralNetwork` model.

        Args:
            inputs (`tuple`): inputs of model.
            is_score (`bool`): check if compute score (this is useful for ZS learning).
            set_is_score (`bool`): compute score (this is useful for ZS learning).

        Returns:
            `tuple`: predicted outputs
        """
        if is_score:
            pred_outputs = self.__callfn(inputs, set_is_score)
        else:
            pred_outputs = self.__callfn(inputs)

        return pred_outputs

    @staticmethod
    def status_idxs(type: int, status: list):
        """
        Args:
            type: A int, type of status (0, 1 or 2),
            status: A list, nested list of status per data in example.

        Returns:
            idxs: A list, list with indexes from `status` where mach with `type`

        Examples:
            Giving an example usage where the example have 3 data types (input, output (1st) with loss status type, output (2nd)
            with metric & loss status type).

            >>> mlb = MetricLossBase()
            >>> status = [[0], [1], [1,2]]
            >>> # 1st case
            >>> type = 0
            >>> print(mlb.status_idxs(type, status))
            [0]
            >>> # 2nd case
            >>> type = 1
            >>> print(mlb.status_idxs(type, status))
            [1, 2]
            >>> # 3rd case
            >>> type = 2
            >>> print(mlb.status_idxs(type, status))
            [2]
        """
        idxs = []
        for i, st in enumerate(status):
            if type in st:
                idxs.append(i)
        return idxs

    @staticmethod
    def get_batch_by_indxs(batch: Union[list, tuple], idxs: list):
        """

        Args:
          batch: A list or tuple, subset of data examples,
          idxs: A list, list of indexes.

        Returns:
          batch_by_indxs: A tuple, specific data examples based on idxs.
        """

        batch_by_indxs = tuple([batch[i] for i in idxs])
        return batch_by_indxs


class MetricBase(MetricLossBase):
    """
    Multiple metric avenging manipulation. Create a list of metrics such as `tf.keras.metric.Metric`.

    Attributes:
      __mtr: A list, list of metric objects based on ` tf.keras.metric.Metrics `,
      __mtr_idxs: A list, indexes list of status with for output type status,
      __mtr_select: A list, list of indexes of __mtr,
      __input_idxs: A list, indexes list of input data examples,
    """

    def __init__(self, model, mtrs_obj, status, mtr_select, status_out_type=2):
        """

        Args:
          model: An object, based on ` tf.Module ` or ` tf.keras.model.Model `,
          mtrs_obj: A list, list of metric objects based on ` tf.keras.metric.Metrics `,
          status: A list, nested list of status per data of examples,
          mtr_select: A list, list of indexes of mtrs_obj,
          status_out_type: int (optional), type of output data of examples. Default is 2.
        """
        super(MetricBase, self).__init__(model)
        self.__mtr = mtrs_obj
        self.__mtr_idxs = self.status_idxs(status_out_type, status)
        self.__mtr_select = mtr_select
        self.__input_idxs = self.status_idxs(0, status)

    def metric_dataset(self, dataset_example, is_score=False):
        """
        Calculate the average metric over all metrics depends on dataset (examples) status structure.

        Args:
          dataset_example: A tf.data.Dataset,
        Returns:
          mtr: tf float, average metric of all metrics summation.

        """
        for batch in dataset_example:
            inputs = self.get_batch_by_indxs(batch, self.__input_idxs)
            true_outputs = self.get_batch_by_indxs(batch, self.__mtr_idxs)
            pred_outputs = self.predict_outputs(inputs, is_score, is_score)
            self.metric_batch(true_outputs, pred_outputs)

        mtr = 0
        for metric in self.__mtr:
            mtr += metric.result().numpy()
            metric.reset_states()

        mtr = mtr / tf.cast(len(self.__mtr), tf.float32)
        return mtr

    def metric_batch(self, true_outputs: Union[list, tuple], predict_outputs: Union[list, tuple]):
        """
        Update all metrics over batch.

        Args:
          true_outputs: A list, true outputs list of tensors,
          predict_outputs: A list, predicted outputs list of tensors.

        """
        for i, ms in enumerate(self.__mtr_select):
            self.__mtr[ms].update_state(true_outputs[i], predict_outputs[i])


class LossBase(MetricLossBase):
    """
    Multiple loss avenging manipulation. Create a list of metrics such as `tf.keras.losses.Loss`.

    Attributes:
      __loss: A list, list of losses objects based on ` tf.keras.loeses.Loss `,
      __loss_idxs: A list, indexes list of status with for output type status,
      __loss_select: A list, list of indexes of __loss,
      __input_idxs: A list, indexes list of input data examples,
    """

    def __init__(self, model, losses_obj, status, loss_select):
        """

        Args:
          model: An object, based on ` tf.Module ` or ` tf.keras.model.Model `,
          losses_obj: A list, list of loss objects based on ` tf.keras.losses.Loss `,
          status: A list, nested list of status per data of examples,
          loss_select: A list, list of indexes of losses_obj.
        """
        super(LossBase, self).__init__(model)
        self.__loss = losses_obj
        self.__loss_idxs = self.status_idxs(1, status)
        self.__loss_select = loss_select
        self.__input_idxs = self.status_idxs(0, status)

    def loss_batch(self, batch: Union[list, tuple], is_score=False):
        """
        Compute loss over all losses objects (__loss list) using summation.

        Args:
          is_score: A boolean, for zero shot case
          batch: A list or tuple, subset of data examples.

        Returns:
          loss: tf float, the summation of all losses.
        """
        loss = 0
        inputs = self.get_batch_by_indxs(batch, self.__input_idxs)
        pred_outputs = self.predict_outputs(inputs, is_score, False)
        true_outputs = self.get_batch_by_indxs(batch, self.__loss_idxs)
        for i, ls in enumerate(self.__loss_select):
            loss += self.__loss[ls](true_outputs[i], pred_outputs[i])

        return loss


class EarlyStopping(object):
    """
    Attributes:
      es_strategy: A str, early stopping strategy. {'first_drop' or 'patience'},
      es_metric: A srt, metric where strategy follow. {'{train, val or test}_{cost or score}'},
      __metric_max: A float, this created if strategy is 'first_drop',
      __save_weights_obj: An object (optional), tf.models.Model or tf.Module object save_weights with partial input (path). Default is None.

      Examples:
          # patience strategy example:
          >>> entry = {'es_strategy':'patience', 'es_metric':'val_cost', 'es_min_delta': 1e-3, 'es_patience': 10 }
          >>> ES = EarlyStopping(entry)
          >>> print(ES.es_strategy)
          patience

          # first drop strategy
          >>> entry = {'es_strategy':'first_drop', 'es_metric':'val_score'}
          >>> ES = EarlyStopping(entry)
          >>> print(ES.es_strategy)
          first_drop

          # save weights object example
          >>> # entry = {'es_strategy':'first_drop', 'es_metric':'val_score'}
          >>> # save_weights_obj = partial(self.save_weights, "path/to/weights.tf")
          >>> # ES = EarlyStoping(entry, save_weights_obj)
    """

    def __init__(self, entries, save_weights_obj=None):
        """

        Args:
          entries: A dict, dictionary of class attributes based on early stopping (es) strategy
          save_weights_obj: An object (optional), tf.keras.models.Model or tf.Module object save_weights with partial input (path). Default is None.

        """
        self.es_strategy = None
        self.es_metric = None
        self.__dict__.update(entries)
        if self.es_strategy == 'first_drop' and self.es_metric is not None:
            self.__metric_max = 0
        self.__save_weights_obj = save_weights_obj

    def check_stop(self, h):
        """
        Application of early stopping strategy based on history (h). For `first drop` strategy training stops if the caption
        metric drops for first time. For `patience` strategy training stops when after a number of epochs the metric don't change,
        based on a `delta` range where the metric don't change.

        Args:
          h: A dict, history dictionary

        Returns:
          A bool, True or False based on early stopping strategy.
        """
        if len(h[self.es_metric]) > 0:
            if self.es_strategy is not None and self.es_metric is not None:
                sub_h = list(h[self.es_metric].values())
                if self.es_strategy == 'first_drop':
                    if sub_h[-1] > self.__metric_max:
                        self.__metric_max = sub_h[-1]
                        self.__save_weights()
                    elif self.__metric_max > 0 and sub_h[-1] < self.__metric_max:
                        return True
                    return False
                elif self.es_strategy == 'patience':
                    if len(sub_h) >= self.es_patience:
                        if abs(sub_h[-1] - sub_h[-self.es_patience]) < self.es_min_delta:
                            return True
                        else:
                            return False
                    return False

    def __save_weights(self):
        """
        Activate the save_weights (tf.keras.models.Model or tf.Module).
        """
        if self.__save_weights_obj is not None:
            self.__save_weights_obj()


class History(object):
    """
    History object to track epoch performance of nn model
    """
    def __init__(self, dataset):
        """

        Args:
            dataset: A dataset of `algomorphis.datasets` with `train` , `val` `test` attributes.
        """
        dataset_attrs = self.__get_dataset_atr(dataset)
        self.history = self.__set_up_history(dataset_attrs)
        self.__harmonic_score = lambda seen, unseen: 2*seen*unseen/(seen + unseen)

    @staticmethod
    def __get_dataset_atr(dataset):
        """
        Get dataset attributes.

        Args:
            dataset: A dataset of `algomorphis.datasets` with `train` , `val` `test` attributes.

        Returns:
            `list`: `dataset` attributes with '_' separator
        """
        def append_atr(dataset_attr, example_type):
            if hasattr(dataset, example_type):
                if hasattr(getattr(dataset, example_type), 'seen') and hasattr(getattr(dataset, example_type), 'unseen'):
                    dataset_attr.append('{}_seen'.format(example_type))
                    dataset_attr.append('{}_unseen'.format(example_type))
                    dataset_attr.append('{}_harmonic'.format(example_type))
                else:
                    dataset_attr.append(example_type)
            return dataset_attr

        dataset_attr = []
        example_types = ['val', 'test']
        for ex_type in example_types:
            dataset_attr = append_atr(dataset_attr, ex_type)
        return dataset_attr

    def __set_up_history(self, dataset_attrs):
        """
        Set up `History.history`

        Args:
            dataset_attrs (`list`): `dataset` attributes with '_' separator

        Returns:
            `dict`: empty history with attributes.
        """
        mtrs_attr = ['cost']
        if hasattr(self, 'score_mtr'):
            mtrs_attr.append('score')

        history = {}
        for mtr_attr in mtrs_attr:
            history['train_{}'.format(mtr_attr)] = {}
        for d_attr, mtr_attr in product(dataset_attrs, mtrs_attr):
            if d_attr.split('_')[-1] != 'harmonic':
                history['{}_{}'.format(d_attr, mtr_attr)] = {}
            else:
                history[d_attr] = {}

        return history

    def append_history_print(self, dataset, epoch, print_types=None):
        """
        Append and print epoch performance to history if print types is not None.

        Args:
            dataset: A dataset of `algomorphis.datasets` with `train` , `val` `test` attributes.
            epoch (`int`): index of epoch
            print_types (`list`): e.g. types: ['train'], ['train', 'val'], ['val', 'test']

        """
        def val_test_metrics(dataset_example, key_split):
            if 'seen' in key_split:
                if 'cost' in key_split:
                    value = self.cost_mtr.metric_dataset(dataset_example.seen)
                if 'score' in key_split:
                    value = self.score_mtr.metric_dataset(dataset_example.seen, is_score=True)
            elif 'unseen' in key_split:
                if 'cost' in key_split:
                    value = self.cost_mtr.metric_dataset(dataset_example.unseen)
                if 'score' in key_split:
                    value = self.score_mtr.metric_dataset(dataset_example.unseen, is_score=True)
            elif 'harmonic' in key_split:
                value = self.__harmonic_score(
                    self.score_mtr.metric_dataset(dataset_example.seen, is_score=True),
                    self.score_mtr.metric_dataset(dataset_example.unseen, is_score=True)
                )
            else:
                if 'cost' in key_split:
                    value = self.cost_mtr.metric_dataset(dataset_example)
                if 'score' in key_split:
                    value = self.score_mtr.metric_dataset(dataset_example)

            return value

        for key in self.history.keys():
            key_split = key.split('_')
            if 'train' in key_split and 'train' in print_types:
                if 'cost' in key_split:
                    value = self.cost_mtr.metric_dataset(dataset.train)
                    self.history[key][epoch] = float(value.numpy())
                    key_join = ' '.join(key_split)
                    print('{}: {}'.format(key_join, value))
                if 'score' in key_split:
                    if any([True if 'seen' in k.split('_') else False for k in self.history.keys()]):
                        is_score = True
                    else:
                        is_score = False
                    value = self.score_mtr.metric_dataset(dataset.train, is_score)
                    self.history[key][epoch] = float(value.numpy())
                    key_join = ' '.join(key_split)
                    print('{}: {}'.format(key_join, value))

            elif 'val' in key_split and 'val' in print_types:
                value = val_test_metrics(dataset.val, key_split)
                self.history[key][epoch] = float(value.numpy())
                key_join = ' '.join(key_split)
                print('{}: {}'.format(key_join, value))

            elif 'test' in key_split and 'test' in print_types:
                value = val_test_metrics(dataset.test, key_split)
                self.history[key][epoch] = float(value.numpy())
                key_join = ' '.join(key_split)
                print('{}: {}'.format(key_join, value))


class Trainer(History):
    """
    Training options: early stopping and gradient clipping. Keep learning history and give a verity of optimizers.

    Attributes:
      __optimizer: An optimizer object, options: {'SGD', 'Adagrad', 'Adadelta', 'Adam'},
      __clip_norm: A float, gradient clipping (normalization method),
      __epochs_cnt: An int, epochs counter,
      __early_stop: A dict, early stopping attributes on dictionary. Default is None.
    """

    def __init__(self, dataset, early_stop_vars=None, save_weights_obj=None, optimizer="SGD", learning_rate=1e-4,
                 clip_norm=0.0):
        """
        Args:
          early_stop_vars: A dict (optional), early stopping attributes on dictionary . Default is None,
          save_weights_obj: An object (optional), tf.models.Model or tf.Module object save_weights with partial input (path). Default is None.
          optimizer: A str (optional), optimizer options: {'SGD', 'Adagrad', 'Adadelta', 'Adam'}. Default is `SGD`,
          learning_rate: A float (optional), learning rate of optimizer. Default is 1e-4,
          clip_norm: A float (optional). Default is 0.0 .
        """
        super(Trainer, self).__init__(dataset)

        if optimizer == "SGD":
            self.__optimizer = SGD(learning_rate, momentum=0.98)
        elif optimizer == "Adagrad":
            self.__optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate, initial_accumulator_value=1e-3,
                                                           epsilon=1e-7)
        elif optimizer == "Adadelta":
            self.__optimizer = tf.keras.optimizers.Adadelta(learning_rate=learning_rate, rho=0.98)
        elif optimizer == "Adam":
            self.__optimizer = Adam(learning_rate=learning_rate, amsgrad=True)

        self.__clip_norm = clip_norm

        self.__early_stop = None
        if early_stop_vars is not None:
            self.__early_stop = EarlyStopping(early_stop_vars, save_weights_obj)
        self.__epochs_cnt = 0

    def set_lr_rate(self, lr):
        """
        Learning rate setter.

        Args:
          lr: A float, new learning rate.

        """
        self.__optimizer.lr.assign(lr)

    def set_clip_norm(self, clip_norm):
        """
        Gradient clipping setter.

        Args:
          clip_norm: A float, new gradient clipping (normalization method).
        """
        self.__clip_norm = clip_norm

    def set_early_stop(self, early_stop_vars):
        """
        Early stopping setter.

        Args:
          early_stop_vars: A dict, new early stopping parameters.

        """
        self.__early_stop = EarlyStopping(early_stop_vars, None)

    def train(self, dataset, epochs=10, print_types=None):
        for epoch in range(epochs):
            # super hard code
            self.__epochs_cnt += 1
            for batch in dataset.train:
                with tf.GradientTape() as tape:
                    cost_loss = self.cost_loss.loss_batch(batch)
                grads = tape.gradient(cost_loss, self.trainable_variables)
                if self.__clip_norm > 0:
                    grads = [(tf.clip_by_norm(grad, clip_norm=self.__clip_norm)) for grad in grads]
                self.__optimizer.apply_gradients(zip(grads, self.trainable_variables))

            print('Epoch {} finished'.format(self.__epochs_cnt), end='\n')
            if print_types is not None:
                self.append_history_print(dataset, self.__epochs_cnt, print_types)

                if self.__early_stop is not None:
                    if self.__early_stop.check_stop(copy.deepcopy(self.history)):
                        print('Stop Training')
                        break


class BaseNeuralNetwork(Trainer):
    """
    Base of Neural Network models. See examples on `models.py` .

    Attributes:
      __status: A list, a nested list for input/output of model (see MetricLossBase for details),
      __score_mode: A bool, this boolean have use of switch the mode of zero shot nn model output (categorical (score) 'True' or vector 'False').
    """

    def __init__(self, status, dataset, early_stop_vars=None, weights_outfile=None, optimizer="SGD",
                 learning_rate=1e-4, clip_norm=0.0):
        """
        Args:
          status: A list, a nested list for input/output of model (see MetricLossBase for details),
          early_stop_vars: A dict (optional), early stopping attributes on dictionary . Default is None,
          weights_outfile: A list (optional), a list of sub-paths. The first sub-path is the root folder for weights and
                          the second sub-path is the name of the best weights. All weights file type is `.tf`,
          optimizer: A str (optional), the optimizer where use. Default is `SGD`,
          learning_rate: A float (optional), the learning rate of optimizer. Default is `1e-4`.
        """
        save_weights_obj = None
        if weights_outfile is not None:
            save_weights_obj = partial(self.save_weights,
                                       "{}/weights/weights_best_{}.tf".format(weights_outfile[0], weights_outfile[1]))
        super(BaseNeuralNetwork, self).__init__(dataset, early_stop_vars, save_weights_obj, optimizer, learning_rate, clip_norm)

        self.__status = status
        self.__score_mode = False

    def get_score_mode(self):
        """
        Score mode getter.
        """
        return self.__score_mode

    def get_results(self, dataset, is_score=False):

        def forloop(dataset_exmpl):
            in_idxs = self.cost_mtr.status_idxs(0, self.__status)
            tr_out_idxs = self.cost_mtr.status_idxs(2, self.__status)
            for batch in dataset_exmpl:
                inputs = self.cost_mtr.get_batch_by_indxs(batch, in_idxs)
                outs = self.cost_mtr.get_batch_by_indxs(batch, tr_out_idxs)
                if is_score:
                    if hasattr(self, "__call__"):
                        predict = self.__call__(inputs, is_score)
                    elif hasattr(self, "call"):
                        predict = self.call(inputs, is_score)
                else:
                    if hasattr(self, "__call__"):
                        predict = self.__call__(inputs)
                    elif hasattr(self, "call"):
                        predict = self.call(inputs)

            return inputs, outs, predict

        return_dict = {
            'train': (forloop(dataset.train))
        }

        if hasattr(dataset.val, 'seen'):
            return_dict['val_seen'] = (forloop(dataset.val.seen))
        if hasattr(dataset.val, 'unseen'):
            return_dict['val_unseen'] = (forloop(dataset.val.unseen))
        if not hasattr(dataset.val, 'seen') and not hasattr(dataset.val, 'unseen'):
            return_dict['val'] = (forloop(dataset.val))

        if hasattr(dataset.test, 'seen'):
            return_dict['test_seen'] = (forloop(dataset.test.seen))
        if hasattr(dataset.test, 'unseen'):
            return_dict['test_unseen'] = (forloop(dataset.test.unseen))
        if not hasattr(dataset.val, 'seen') and not hasattr(dataset.test, 'unseen'):
            return_dict['test'] = (forloop(dataset.test))

        return return_dict
