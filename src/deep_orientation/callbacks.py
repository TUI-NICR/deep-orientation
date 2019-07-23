# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

"""
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import Callback


class LRPolyDecay(Callback):
    """
    Creates a Poly Decay learning rate scheduler as a keras callback.

    Parameters
    ----------
    lr_init: float
        The initial learning rate to decay from.
    power: float
        The exponent in the poly decay formula.
    max_iter: int
        The number of iterations over which the initial learning rate is
        decaying to zero.
    lr_min: float
        Minimal learning rate.
    verbose: int
        For printing some information, 0: quiet, 1: update messages.

    """
    def __init__(self, lr_init, power, max_iter, lr_min, verbose=0):
        super(LRPolyDecay, self).__init__()
        self._cur_iter = 0
        self._lr_init = lr_init
        self._power = power
        self._max_iter = max_iter
        self._lr_min = lr_min
        self._verbose = verbose

    def on_train_begin(self, logs):
        # check whether it is possible to change learning rate
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')

    def on_epoch_end(self, epoch, logs):
        # get current learning rate
        cur_lr = float(K.get_value(self.model.optimizer.lr))

        # print learning rate on epoch begin
        if self._verbose > 0:
            print(f"Learning rate at end of epoch {epoch}: {cur_lr}")

        # append learning rate to logs
        # note: for tensorflow callback, we need numpy objects
        logs['lr'] = np.array(cur_lr)

    def on_batch_begin(self, batch, logs):
        # keep track of total batch count
        self._cur_iter += 1

        # calculate current learning rate
        lr = self._lr_init*(1-(self._cur_iter / self._max_iter))**self._power

        # limit learning rate
        lr = max(lr, self._lr_min)

        # set the new learning rate
        K.set_value(self.model.optimizer.lr, lr)


class ValidationCallback(Callback):
    def __init__(self,
                 tf_dataset,
                 dataset_name,
                 output_filepath,
                 validation_steps,
                 verbose=0):
        super(ValidationCallback, self).__init__()

        self._tf_dataset = tf_dataset
        self._dataset_name = dataset_name

        self._output_filepath = output_filepath

        self._validation_steps = validation_steps

        self._verbose = verbose

        self._output_tensors = None
        self._built = False

    def build_graph(self):
        # create iterator
        it = self._tf_dataset.make_one_shot_iterator()
        # get next element
        el = it.get_next()
        # create inputs
        inputs = [Input(tensor=e) for e in el[0]]
        y_true = el[1]
        y_pred = self.model(inputs, training=False)
        loss = self.model.loss_functions[0](y_true, y_pred)
        self._output_tensors = [loss, y_true, y_pred]

    def on_epoch_end(self, epoch, logs=None):
        # build graph
        if not self._built:
            self.build_graph()
            self._built = True

        sess = K.get_session()
        outputs = []
        for i in range(self._validation_steps):
            if self._verbose > 0:
                print(f"Validation on {self._dataset_name}: {i+1:04d}/"
                      f"{self._validation_steps:04d}")
            # run graph
            out = sess.run(self._output_tensors)
            # reshape (vector -> matrix)
            out = [o.reshape(-1, 1) if o.ndim == 1 else o
                   for o in out]
            # concatenate
            out = np.concatenate(out, axis=1)
            # append
            outputs.append(out)

        # concatenate outputs
        outputs = np.concatenate(outputs, axis=0)

        # get mean loss
        mean_loss = outputs[:, 0].mean()

        # add to logs
        logs[self._dataset_name+'_loss'] = mean_loss

        # save file
        # note: unlike in keras callbacks, epoch is in range [0, n_epochs-1]
        fp = self._output_filepath.format(epoch=epoch)
        np.save(fp, outputs)
