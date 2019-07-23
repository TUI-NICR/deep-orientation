# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

"""
import argparse as ap
import os

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

from utils.io import write_json


def _parse_args():
    """Parse command-line arguments"""
    desc = 'Simple script to create a frozen graph model'
    parser = ap.ArgumentParser(description=desc,
                               formatter_class=ap.RawTextHelpFormatter)

    parser.add_argument('model',
                        type=str,
                        help=("Model to use: beyer, beyer_mod_relu or "
                              "mobilenet_v2"),
                        choices=['beyer', 'beyer_mod_relu', 'mobilenet_v2'])

    parser.add_argument('output_filepath',
                        type=str,
                        help="Path where to store the frozen graph pb file")

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

    parser.add_argument('-nc', '--n_classes',
                        type=int,
                        default=8,
                        help=("Number of classes when output_type is "
                              "{}, default: 8".format(OUTPUT_CLASSIFICATION)))

    parser.add_argument('-k', '--kappa',
                        type=float,
                        default=None,
                        help=("Kappa to use when output_type is "
                              "{} or {}, "
                              "default: {}: 1.0, "
                              "{}: 0.5".format(OUTPUT_BITERNION,
                                               OUTPUT_REGRESSION,
                                               OUTPUT_BITERNION,
                                               OUTPUT_REGRESSION)))

    # parameters --------------------------------------------------------------
    parser.add_argument('-ma', '--mobilenet_v2_alpha',
                        type=float,
                        choices=[0.35, 0.5, 0.75, 1.0],
                        default=1.0,
                        help="Alpha value for MobileNet v2 (default: 1.0)")

    # other -------------------------------------------------------------------
    parser.add_argument('-t', '--dtype',
                        type=str,
                        default='float32',
                        choices=['float32', 'float16'],
                        help="Dtype to use for inference")

    parser.add_argument('-d', '--devices',
                        type=str,
                        default='0',
                        help="GPU device id(s) to use. Use '' (empty string) "
                             "for CPU only (default: 0)")

    # return parsed args
    return parser.parse_args()


def _freeze_session(session,
                    keep_var_names=None,
                    output_names=None,
                    clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.

    Taken from: https://stackoverflow.com/questions/45466020/how-to-export-keras-h5-to-tensorflow-pb
    """

    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ''
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph


def main():
    # parse args --------------------------------------------------------------
    args = _parse_args()

    # set device --------------------------------------------------------------
    os.environ['CUDA_VISIBLE_DEVICES'] = args.devices

    # set learning phase ------------------------------------------------------
    K.set_learning_phase(0)

    # set data format ---------------------------------------------------------
    if args.devices == '' or args.model == 'mobilenet_v2':
        # note: tensorflow supports b01c pooling on cpu only
        K.set_image_data_format('channels_last')
    else:
        K.set_image_data_format('channels_first')

    # set dtype ---------------------------------------------------------------
    K.set_floatx(args.dtype)

    # load model --------------------------------------------------------------
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

    # create frozen graph
    sess = K.get_session()
    out_name = [out.op.name for out in model.outputs]
    frozen_graph = _freeze_session(sess, output_names=out_name)

    dirname = os.path.dirname(args.output_filepath)
    filename = os.path.basename(args.output_filepath)
    assert os.path.splitext(filename)[1] == '.pb'
    tf.train.write_graph(frozen_graph, dirname, filename, as_text=False)
    # tf.train.write_graph(frozen_graph, dirname, filename, as_text=False)

    # store input and output names as json file
    write_json(args.output_filepath+'.json',
               {'input_names': [input.op.name for input in model.inputs],
                'output_names': [output.op.name for output in model.outputs]})


if __name__ == '__main__':
    main()
