# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

"""
import argparse as ap
import os
import subprocess
from time import time

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from deep_orientation import beyer    # noqa # pylint: disable=unused-import
from deep_orientation import mobilenet_v2    # noqa # pylint: disable=unused-import
from deep_orientation import beyer_mod_relu     # noqa # pylint: disable=unused-import

from deep_orientation.inputs import INPUT_TYPES
from deep_orientation.inputs import INPUT_DEPTH, INPUT_RGB, INPUT_DEPTH_AND_RGB
from deep_orientation.outputs import OUTPUT_TYPES
from deep_orientation.outputs import (OUTPUT_REGRESSION, OUTPUT_CLASSIFICATION,
                                      OUTPUT_BITERNION)
import deep_orientation.preprocessing as pre

from utils.io import read_json


def _parse_args():
    """Parse command-line arguments"""
    desc = 'Simple script to time inference'
    parser = ap.ArgumentParser(description=desc,
                               formatter_class=ap.RawTextHelpFormatter)

    parser.add_argument('model',
                        type=str,
                        help=("Model to use: beyer, beyer_mod_relu or "
                              "mobilenet_v2"),
                        choices=['beyer', 'beyer_mod_relu', 'mobilenet_v2'])

    # input -------------------------------------------------------------------
    parser.add_argument('-it', '--input_type',
                        type=str,
                        default=INPUT_DEPTH,
                        choices=INPUT_TYPES,
                        help=("Input type. One of {}, default: "
                              "{}".format(INPUT_TYPES, INPUT_DEPTH)))

    parser.add_argument('-iw', '--input_width',
                        type=int,
                        default=46,
                        help="Patch width to use, default: 96")

    parser.add_argument('-ih', '--input_height',
                        type=int,
                        default=46,
                        help="Patch height to use, default: 96")

    # output ------------------------------------------------------------------
    parser.add_argument('-ot', '--output_type',
                        type=str,
                        default=OUTPUT_BITERNION,
                        choices=OUTPUT_TYPES,
                        help=("Output type. One of {}, default: "
                              "{})".format(OUTPUT_TYPES, OUTPUT_BITERNION)))

    # parameters --------------------------------------------------------------
    parser.add_argument('-nr', '--n_repetitions',
                        type=int,
                        default=500,
                        help="Number of repetitions, default: 100")

    parser.add_argument('-nir', '--n_initial_repetitions',
                        type=int,
                        default=5,
                        help="Number of repetitions without timing, "
                             "default: 5")

    parser.add_argument('-b', '--batch_size',
                        type=int,
                        default=1,
                        help="Batch size to use, default: 1")

    parser.add_argument('-nc', '--n_classes',
                        type=int,
                        default=8,
                        help=("Number of classes when output_type is "
                              "{}, default: 8".format(OUTPUT_CLASSIFICATION)))

    parser.add_argument('-k', '--kappa',
                        type=float,
                        default=1.0,
                        help=("Kappa to use when output_type is "
                              "{} or {}, "
                              "default: {}: 1.0, "
                              "{}: 0.5".format(OUTPUT_BITERNION,
                                               OUTPUT_REGRESSION,
                                               OUTPUT_BITERNION,
                                               OUTPUT_REGRESSION)))

    parser.add_argument('-ma', '--mobilenet_v2_alpha',
                        type=float,
                        choices=[0.35, 0.5, 0.75, 1.0],
                        default=1.0,
                        help="Alpha value for MobileNet v2 (default: 1.0)")

    # other -------------------------------------------------------------------
    parser.add_argument('-opt', '--optimize',
                        type=str,
                        choices=['trt', 'frozen'],
                        default=None,
                        help="Optimize graph")

    parser.add_argument('-t', '--dtype',
                        type=str,
                        default='float32',
                        choices=['float32', 'float16'],
                        help="Dtype to use for inference")

    parser.add_argument('-d', '--devices',
                        type=str,
                        default='0',
                        help="GPU device id(s) to use. (default: 0)")

    parser.add_argument('-c', '--cpu',
                        action='store_true',
                        default=False,
                        help="CPU only, do not run with GPU support")

    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        default=False,
                        help="Enable verbose output")

    # return parsed args
    return parser.parse_args()


def _get_shape(batch_size, height, width, n_channels):
    if K.image_data_format() == 'channels_last':
        return batch_size, height, width, n_channels
    else:
        return batch_size, n_channels, height, width


def _load_frozen_graph_def(filepath):
    with tf.gfile.GFile(filepath, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def


def main():
    # parse args --------------------------------------------------------------
    args = _parse_args()

    # set device and data format ----------------------------------------------
    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        args.devices = ''
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.devices
    if not args.devices or args.model == 'mobilenet_v2':
        # note: tensorflow supports b01c pooling on cpu only
        K.set_image_data_format('channels_last')
    else:
        K.set_image_data_format('channels_first')

    # set dtype ---------------------------------------------------------------
    K.set_floatx(args.dtype)

    # create random data ------------------------------------------------------
    if args.verbose:
        print("Generating random data ...")

    # helper function
    def generate_data(data_shape):
        return np.random.random(data_shape).astype(args.dtype)

    # data for timing
    data = []
    for _ in range(args.n_repetitions):
        single_input = []
        if args.input_type in [INPUT_DEPTH, INPUT_DEPTH_AND_RGB]:
            shape = _get_shape(args.batch_size, args.input_height,
                               args.input_width, 1)
            single_input.append(generate_data(shape))
        if args.input_type in [INPUT_RGB, INPUT_DEPTH_AND_RGB]:
            shape = _get_shape(args.batch_size, args.input_height,
                               args.input_width, 3)
            single_input.append(generate_data(shape))
        data.append(single_input)

    # data initial runs
    data_initial = []
    for _ in range(args.n_repetitions):
        single_input_initial = []
        if args.input_type in [INPUT_DEPTH, INPUT_DEPTH_AND_RGB]:
            shape = _get_shape(args.batch_size, args.input_height,
                               args.input_width, 1)
            single_input_initial.append(generate_data(shape))
        if args.input_type in [INPUT_RGB, INPUT_DEPTH_AND_RGB]:
            shape = _get_shape(args.batch_size, args.input_height,
                               args.input_width, 3)
            single_input_initial.append(generate_data(shape))
        data_initial.append(single_input_initial)

    # time prediction on random data ------------------------------------------
    fps = []
    if args.optimize is not None:
        # use TensorRT
        # convert to frozen graph
        if args.verbose:
            print("Creating frozen graph ...")
        model_filepath = './model.pb'
        subprocess.check_call('python freeze_graph.py {} {}'
                              ' --input_type {}'
                              ' --input_height {}'
                              ' --input_width {}'
                              ' --output_type {}'
                              ' --n_classes {}'
                              ' --kappa {}'
                              ' --mobilenet_v2_alpha {}'
                              ' --devices {}'
                              ' --dtype {}'.format(args.model,
                                                   model_filepath,
                                                   args.input_type,
                                                   args.input_height,
                                                   args.input_width,
                                                   args.output_type,
                                                   args.n_classes,
                                                   args.kappa,
                                                   args.mobilenet_v2_alpha,
                                                   args.devices or '""',
                                                   args.dtype),
                              shell=True,
                              cwd=os.path.dirname(__file__) or './')

        # load frozen graph
        if args.verbose:
            print("Loading frozen graph ...")

        graph_def = _load_frozen_graph_def(model_filepath)
        names = read_json(model_filepath + '.json')

        if args.optimize == 'trt':
            # optimize using TensorRT
            import tensorflow.contrib.tensorrt as trt

            graph_def = trt.create_inference_graph(
                input_graph_def=graph_def,
                outputs=names['output_names'],
                max_batch_size=args.batch_size,
                max_workspace_size_bytes=1 << 25,
                precision_mode='FP16' if args.dtype == 'float16' else 'FP32',
                minimum_segment_size=50
            )

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        tf.import_graph_def(graph_def, name='')

        input_tensors = []
        for name in names['input_names']:
            input_tensors.append(sess.graph.get_tensor_by_name(name + ':0'))

        output_tensor = sess.graph.get_tensor_by_name(
            names['output_names'][0] + ':0')

        # some dry runs
        for d in data_initial:
            output = sess.run(
                output_tensor,
                feed_dict={t: v for t, v in zip(input_tensors, d)})

        # time inference
        for d in data:
            t1 = time()
            output = sess.run(
                output_tensor,
                feed_dict={t: v for t, v in zip(input_tensors, d)})
            fps.append(1 / (time() - t1))

    else:
        # use Keras and Tensorflow

        # load model
        if args.verbose:
            print("Loading model ...")
        model_module = globals()[args.model]
        model_kwargs = {}
        if args.model == 'mobilenet_v2':
            model_kwargs['alpha'] = args.mobilenet_v2_alpha
        model = model_module.get_model(input_type=args.input_type,
                                       input_shape=(args.input_height,
                                                    args.input_width),
                                       output_type=args.output_type,
                                       n_classes=args.n_classes,
                                       sampling=False,
                                       **model_kwargs)

        # some dry runs
        for d in data_initial:
            model.predict(d, batch_size=args.batch_size)

        # time inference
        for d in data:
            t1 = time()
            model.predict(d, batch_size=args.batch_size)
            fps.append(1/(time()-t1))

    mean = np.mean(fps)
    std = np.std(fps)
    model_str = args.model
    size_str = '{}x{}x{}'.format(args.batch_size, args.input_height,
                                 args.input_width)
    if args.model == 'mobilnet_v2':
        model_str += "_{:0.2f}".format(args.mobilenet_v2_alpha).replace('.',
                                                                        '_')
    print("FPS ({}, {}, {}, {}, {}, opt: {}, gpu id: {}, mean of {} runs): "
          "{:.1f}+-{:.1f}".format(model_str, args.output_type,
                                  args.input_type, size_str, args.dtype,
                                  args.optimize, args.devices or '-1',
                                  args.n_repetitions, mean, std))


if __name__ == '__main__':
    main()
