# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

"""
import os
import warnings
from contextlib import redirect_stdout
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from tensorflow.keras import Model

from deep_orientation import beyer, beyer_mod_relu, mobilenet_v2
from deep_orientation.outputs import OUTPUT_TYPES
from deep_orientation.inputs import INPUT_TYPES
from utils.io import create_directory_if_not_exists


def _print_model_summary_to_file(model, filepath):
    with open(filepath, 'w') as f:
        with redirect_stdout(f):
            model.summary()


def _plot_nested_model(model,
                       to_file,
                       show_shapes=False,
                       show_layer_names=True,
                       **kwargs):
    """Small wrapper for keras.utils.plot_model in order to be able to plot
    nested models as well & to catch an error when pdf as extension is used"""

    # since keras > 2.2.4, 'plot_model' tries to return a Jupyter Image object,
    # which fails for pdf files
    def _plot_model(*pm_args, **pm_kwargs):
        try:
            return plot_model(*pm_args, **pm_kwargs)
        except ValueError as e:
            warnings.warn(str(e))
            # ValueError: Cannot embed the 'pdf' image format
            return None

    # plot main model
    _plot_model(model, to_file=to_file, show_shapes=show_shapes,
                show_layer_names=show_layer_names, **kwargs)

    for l in model.layers:
        if isinstance(l, Model):
            fp, ext = os.path.splitext(to_file)
            fp += f'_{l.name}{ext}'
            _plot_model(l, to_file=fp, show_shapes=show_shapes,
                        show_layer_names=show_layer_names, **kwargs)


if __name__ == "__main__":
    output_path = '../model_overviews'

    # create output directory
    create_directory_if_not_exists(output_path)

    # original beyer ----------------------------------------------------------
    K.set_image_data_format('channels_first')
    create_directory_if_not_exists(os.path.join(output_path, 'beyer'))
    for shape in beyer.INPUT_SHAPES:
        base_filepath = os.path.join(output_path, 'beyer', 'beyer')
        for output_type in OUTPUT_TYPES:
            for input_type in INPUT_TYPES:
                model = beyer.get_model(input_type=input_type,
                                        input_shape=shape,
                                        output_type=output_type,
                                        n_classes=8)
                fp = base_filepath + f'_{input_type}_{output_type}_{shape}.pdf'
                _plot_nested_model(model, to_file=fp, show_shapes=True,
                                   show_layer_names=True)
                fp = base_filepath + f'_{input_type}_{output_type}_{shape}.txt'
                _print_model_summary_to_file(model, filepath=fp)
                K.clear_session()

    # beyer mod relu ----------------------------------------------------------
    K.set_image_data_format('channels_first')
    create_directory_if_not_exists(os.path.join(output_path, 'beyer_mod_relu'))
    for shape in beyer_mod_relu.INPUT_SHAPES:
        base_filepath = os.path.join(output_path, 'beyer_mod_relu',
                                     'beyer_mod_relu')
        for output_type in OUTPUT_TYPES:
            for input_type in INPUT_TYPES:
                model = beyer_mod_relu.get_model(input_type=input_type,
                                                 input_shape=shape,
                                                 output_type=output_type,
                                                 n_classes=8)
                fp = base_filepath + f'_{input_type}_{output_type}_{shape}.pdf'
                _plot_nested_model(model, to_file=fp, show_shapes=True,
                                   show_layer_names=True)
                fp = base_filepath + f'_{input_type}_{output_type}_{shape}.txt'
                _print_model_summary_to_file(model, filepath=fp)
                K.clear_session()

    # mobile net v2 -----------------------------------------------------------
    K.set_image_data_format('channels_last')
    create_directory_if_not_exists(os.path.join(output_path, 'mobilenet_v2'))
    for shape in mobilenet_v2.INPUT_SHAPES:
        for alpha in [0.35, 0.5, 0.75, 1.0]:
            for output_type in OUTPUT_TYPES:
                for input_type in INPUT_TYPES[:-1]:
                    alpha_str = f'{alpha}'.replace('.', '_')
                    base_filepath = os.path.join(output_path,
                                                 'mobilenet_v2',
                                                 f'mobilenet_v2_{alpha_str}')

                    model = mobilenet_v2.get_model(input_type=input_type,
                                                   input_shape=shape,
                                                   output_type=output_type,
                                                   alpha=alpha,
                                                   n_classes=8)
                    fp = base_filepath + \
                        f'_{input_type}_{output_type}_{shape}.pdf'
                    _plot_nested_model(model, to_file=fp, show_shapes=True,
                                       show_layer_names=True)
                    fp = base_filepath + \
                        f'_{input_type}_{output_type}_{shape}.txt'
                    _print_model_summary_to_file(model, filepath=fp)

                    K.clear_session()
