# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

"""
import warnings

from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten

from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import RandomNormal

from tensorflow.keras.applications import MobileNetV2

from .activations import relu6
from .inputs import depth_input, rgb_input
from .outputs import classification_output, regression_output, biternion_output

from .inputs import INPUT_DEPTH, INPUT_RGB
from .outputs import OUTPUT_TYPES
from .outputs import OUTPUT_REGRESSION, OUTPUT_CLASSIFICATION, OUTPUT_BITERNION


INPUT_SHAPES = ((96, 96),)


def get_model(input_type=INPUT_DEPTH,
              input_shape=(96, 96),
              output_type=OUTPUT_BITERNION,
              weight_decay=0.00005,
              activation=relu6,
              n_classes=None,
              **kwargs):

    # check arguments
    assert input_type in [INPUT_DEPTH, INPUT_RGB]
    assert input_shape in INPUT_SHAPES
    assert output_type in OUTPUT_TYPES
    assert n_classes is not None or output_type != OUTPUT_CLASSIFICATION
    assert K.image_data_format() == 'channels_last'

    if weight_decay is not None:
        warnings.warn("given weight_decay is applied to output stage only")

    if activation is not None:
        warnings.warn("given activation is applied to output stage only")

    for kw in kwargs:
        if kw in ['sampling']:
            warnings.warn("argument '{}' not supported for MobileNet v2"
                          "".format(kw))

    if 'alpha' not in kwargs:
        warnings.warn("no value for alpha given, using default: 1.0")
    alpha = kwargs.get('alpha', 1.0)

    # regularizer
    reg = l2(weight_decay) if weight_decay is not None else None

    # define input ------------------------------------------------------------
    if input_type == INPUT_DEPTH:
        input_ = depth_input(input_shape)
    elif input_type == INPUT_RGB:
        input_ = rgb_input(input_shape)
    else:
        raise ValueError("input type: {} not supported".format(input_type))

    # build model -------------------------------------------------------------
    # load base model with pretrained weights
    mobile_net = MobileNetV2(input_shape=input_shape + (3,),
                             alpha=alpha,
                             # depth_multiplier=1,    # does not exit any more
                             include_top=False,
                             weights='imagenet',
                             input_tensor=None,
                             pooling='avg',
                             classes=None)

    # if the input is a depth image, we have to convert the kernels of the
    # first conv layer
    if input_type == INPUT_DEPTH:
        # easiest way: modify config, recreate model and copy modified weights
        # get config
        cfg = mobile_net.get_config()
        # modify input shape
        batch_input_shape = (None, ) + input_shape + (1, )
        cfg['layers'][0]['config']['batch_input_shape'] = batch_input_shape
        # instantiate a new model
        mobile_net_mod = Model.from_config(cfg)
        # copy (modified) weights
        assert len(mobile_net.layers) == len(mobile_net_mod.layers)
        for l_mod, l in zip(mobile_net_mod.layers, mobile_net.layers):
            # get weights
            weights = l.get_weights()

            # modify kernels for Conv1 (sum over input channels)
            if l.name == 'Conv1':
                assert len(weights) == 1, "Layer without bias expected"
                kernels = weights[0]
                kernels_mod = kernels.sum(axis=2, keepdims=True)
                weights_mod = (kernels_mod, )
            else:
                weights_mod = weights

            # set (modified) weights
            l_mod.set_weights(weights_mod)

        mobile_net = mobile_net_mod

    # build final model
    x = mobile_net(input_)
    x = Flatten(name='output_1_flatten')(x)
    x = Dropout(rate=0.2, name='output_2_dropout')(x)
    x = Dense(units=512, kernel_regularizer=reg, name='output_2_dense')(x)
    x = Activation(activation, name='output_2_act')(x)
    x = Dropout(rate=0.5, name='output_3_dropout')(x)

    if output_type == OUTPUT_BITERNION:
        kernel_initializer = RandomNormal(mean=0.0, stddev=0.01)
        x = biternion_output(kernel_initializer=kernel_initializer,
                             kernel_regularizer=reg,
                             name='output_3_dense_and_act')(x)
    elif output_type == OUTPUT_REGRESSION:
        x = regression_output(kernel_regularizer=reg,
                              name='output_3_dense_and_act')(x)
    elif output_type == OUTPUT_CLASSIFICATION:
        x = classification_output(n_classes, name='output_3_dense_and_act',
                                  kernel_regularizer=reg)(x)

    return Model(inputs=input_, outputs=[x])
