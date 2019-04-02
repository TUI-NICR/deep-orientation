# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

"""
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input


INPUT_DEPTH = 'depth'
INPUT_RGB = 'rgb'
INPUT_DEPTH_AND_RGB = 'depth_and_rgb'
INPUT_TYPES = (INPUT_DEPTH, INPUT_RGB, INPUT_DEPTH_AND_RGB)


def rgb_input(input_shape, **input_kwargs):
    if K.image_data_format() == 'channels_last':
        shape = input_shape + (3,)
    else:
        shape = (3,) + input_shape
    return Input(shape=shape, name='input_rgb', **input_kwargs)


def depth_input(input_shape, **input_kwargs):
    if K.image_data_format() == 'channels_last':
        shape = input_shape + (1,)
    else:
        shape = (1,) + input_shape
    return Input(shape=shape, name='input_depth', **input_kwargs)
