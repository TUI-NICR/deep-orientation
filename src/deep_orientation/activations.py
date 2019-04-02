# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

"""
from tensorflow.keras import backend as K
from tensorflow.keras.activations import relu


def biternion(x):
    return x / K.sqrt(K.sum(x ** 2, axis=1, keepdims=True))


def leaky_relu(x, alpha=0.01):
    return relu(x, alpha=alpha)


def relu6(x):
    return relu(x, max_value=6.)
