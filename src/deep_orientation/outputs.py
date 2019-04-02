# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

"""
import warnings

from tensorflow.keras.layers import Dense

from .activations import biternion


OUTPUT_REGRESSION = 'regression'
OUTPUT_CLASSIFICATION = 'classification'
OUTPUT_BITERNION = 'biternion'
OUTPUT_TYPES = (OUTPUT_REGRESSION, OUTPUT_CLASSIFICATION, OUTPUT_BITERNION)


def classification_output(n_classes, **dense_kwargs):
    return Dense(units=n_classes, activation='softmax', **dense_kwargs)


def regression_output(**dense_kwargs):
    return Dense(units=1, activation='linear', **dense_kwargs)


def biternion_output(**dense_kwargs):
    if 'kernel_initializer' not in dense_kwargs:
        warnings.warn("No `kernel_initializer` given. The reference "
                      "implementation uses 'random_normal' with std of 0.01.")
    return Dense(units=2, activation=biternion, **dense_kwargs)
