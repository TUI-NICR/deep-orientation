# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

"""
import argparse as ap
import json
import os

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import TerminateOnNaN
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop

import nicr_rgb_d_orientation_data_set as dataset

from deep_orientation import beyer    # noqa # pylint: disable=unused-import
from deep_orientation import mobilenet_v2    # noqa # pylint: disable=unused-import
from deep_orientation import beyer_mod_relu     # noqa # pylint: disable=unused-import

from deep_orientation import callbacks
from deep_orientation import losses
from deep_orientation.inputs import INPUT_TYPES
from deep_orientation.inputs import INPUT_DEPTH, INPUT_RGB, INPUT_DEPTH_AND_RGB
from deep_orientation.outputs import OUTPUT_TYPES
from deep_orientation.outputs import (OUTPUT_REGRESSION, OUTPUT_CLASSIFICATION,
                                      OUTPUT_BITERNION)
import deep_orientation.preprocessing as pre
from utils.io import create_directory_if_not_exists, write_json
import utils.img as img_utils


def _parse_args():
    """Parse command-line arguments"""
    desc = 'Train neural network for orientation estimation'
    parser = ap.ArgumentParser(description=desc,
                               formatter_class=ap.RawTextHelpFormatter)

    parser.add_argument('model',
                        type=str,
                        help=("Model to use: beyer, beyer_mod_relu or "
                              "mobilenet_v2"),
                        choices=['beyer', 'beyer_mod_relu', 'mobilenet_v2'])

    # output ------------------------------------------------------------------
    parser.add_argument('-o', '--output_basepath',
                        type=str,
                        default='/results/rotator',
                        help=("Path where to store output files, default: "
                              "'/results/rotator'"))

    # dataset -----------------------------------------------------------------
    parser.add_argument('-db', '--dataset_basepath',
                        type=str,
                        default='/datasets/rotator',
                        help=("Path to downloaded dataset (default: "
                              "'/datasets/rotator')"))

    parser.add_argument('-ds', '--dataset_size',
                        type=str,
                        choices=dataset.SIZES,
                        default=dataset.SIZE_SMALL,
                        help=(f"Dataset image size to use. One of :"
                              f"{dataset.SIZES}, default: "
                              f"{dataset.SIZE_SMALL}"))

    parser.add_argument('-ts', '--training_set',
                        type=str,
                        default=dataset.TRAIN_SET,
                        help=(f"Set to use for training, default: "
                              f"{dataset.TRAIN_SET}"))

    parser.add_argument('-vs', '--validation_sets',
                        type=str,
                        nargs='+',
                        default=[dataset.VALID_SET, dataset.TEST_SET],
                        help=(f"Sets to use for validation, default: "
                              f"[{dataset.VALID_SET}, {dataset.TEST_SET}]"))

    # input -------------------------------------------------------------------
    parser.add_argument('-it', '--input_type',
                        type=str,
                        default=INPUT_DEPTH,
                        choices=INPUT_TYPES,
                        help=(f"Input type. One of {INPUT_TYPES}, default: "
                              f"{INPUT_DEPTH}"))

    parser.add_argument('-iw', '--input_width',
                        type=int,
                        default=46,
                        help="Patch width to use, default: 96")

    parser.add_argument('-ih', '--input_height',
                        type=int,
                        default=46,
                        help="Patch height to use, default: 96")

    parser.add_argument('-ip', '--input_preprocessing',
                        type=str,
                        default='standardize',
                        choices=['standardize', 'scale01', 'none'],
                        help="Preprocessing to apply. One of [standardize, "
                             "scale01, none], default: standardize")

    # output ------------------------------------------------------------------
    parser.add_argument('-ot', '--output_type',
                        type=str,
                        default=OUTPUT_BITERNION,
                        choices=OUTPUT_TYPES,
                        help=(f"Output type. One of {OUTPUT_TYPES}, default: "
                              f"{OUTPUT_BITERNION})"))

    # hyper parameters --------------------------------------------------------
    parser.add_argument('-lr', '--learning_rate',
                        type=float,
                        default=0.01,
                        help="(Base) learning rate, default: 0.01")

    parser.add_argument('-lrd', '--learning_rate_decay',
                        type=str,
                        default='poly',
                        choices=['poly'],
                        help="Learning rate decay to use, default: poly")

    parser.add_argument('-m', '--momentum',
                        type=float,
                        default=0.9,
                        help="Momentum to use, default: 0.9")

    parser.add_argument('-ne', '--n_epochs',
                        type=int,
                        default=800,
                        help="Number of epochs to train, default: 800")

    parser.add_argument('-es', '--early_stopping',
                        type=int,
                        default=100,
                        help=("Number of epochs with no improvement after "
                              "which training will be stopped, default: 100."
                              "To disable early stopping use -1."))

    parser.add_argument('-b', '--batch_size',
                        type=int,
                        default=128,
                        help="Batch size to use, default: 128")

    parser.add_argument('-vb', '--validation_batch_size',
                        type=int,
                        default=512,
                        help="Batch size to use for validation, default: 512")

    parser.add_argument('-nc', '--n_classes',
                        type=int,
                        default=8,
                        help=(f"Number of classes when output_type is "
                              f"{OUTPUT_CLASSIFICATION}, default: 8"))

    parser.add_argument('-k', '--kappa',
                        type=float,
                        default=None,
                        help=(f"Kappa to use when output_type is "
                              f"{OUTPUT_BITERNION} or {OUTPUT_REGRESSION}, "
                              f"default: {OUTPUT_BITERNION}: 1.0, "
                              f"{OUTPUT_REGRESSION}: 0.5"))

    parser.add_argument('-opt', '--optimizer',
                        type=str,
                        choices=['sgd', 'adam', 'rmsprop'],
                        default='sgd',
                        help='Optimizer to use, default: sgd')

    parser.add_argument('-naug', '--no_augmentation',
                        action='store_true',
                        default=False,
                        help='Disable augmentation')

    parser.add_argument('-rid', '--run_id',
                        type=int,
                        default=0,
                        help="Run ID (default: 0)")

    parser.add_argument('-ma', '--mobilenet_v2_alpha',
                        type=float,
                        choices=[0.35, 0.5, 0.75, 1.0],
                        default=1.0,
                        help="Alpha value for MobileNet v2 (default: 1.0)")

    # other -------------------------------------------------------------------
    parser.add_argument('-d', '--devices',
                        type=str,
                        default='0',
                        help="GPU device id(s) to train on. (default: 0)")

    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        default=False,
                        help="Enable verbose output")

    # return parsed args
    return parser.parse_args()


def _create_tensorflow_dataset(dataset,
                               input_type,
                               input_shape,
                               output_type,
                               batch_size=5,
                               n_prefetch_batches=5,
                               n_classes=None,
                               shuffle=True,
                               flip=True,
                               scale01=False,
                               standardize=True,
                               zero_mean=True,
                               unit_variance=True):
    # check inputs
    assert input_type in INPUT_TYPES
    assert output_type in OUTPUT_TYPES

    # define sample generator
    def _sample_generator():
        for s in dataset:
            # return tuple of filepaths and the orientation
            yield s.get_depth_patch_filepath(),\
                  s.get_rgb_patch_filepath(), \
                  s.get_mask_patch_filepath(),\
                  s.orientation

    # define function for input preprocessing
    def _load_and_preprocess_input(depth_filepath,
                                   rgb_filepath,
                                   mask_filepath,
                                   orientation):

        # input
        mask = img_utils.load(mask_filepath.decode())
        mask_resized = pre.resize_mask(mask, input_shape)
        mask = mask > 0
        mask_resized = mask_resized > 0

        if input_type in [INPUT_DEPTH, INPUT_DEPTH_AND_RGB]:
            # load
            depth = img_utils.load(depth_filepath.decode())
            # mask
            depth = pre.mask_img(depth, mask)
            # resize
            depth = pre.resize_depth_img(depth, input_shape)
            # 01 -> 01c
            depth = depth[..., None]
            # preprocess
            depth = pre.preprocess_img(depth,
                                       mask=mask_resized,
                                       scale01=scale01,
                                       standardize=standardize,
                                       zero_mean=zero_mean,
                                       unit_variance=unit_variance)
            # convert to correct data format
            axes = '01c' if K.image_data_format() == 'channels_last' else 'c01'
            depth = img_utils.dimshuffle(depth, '01c', axes)

        if input_type in [INPUT_RGB, INPUT_DEPTH_AND_RGB]:
            # load
            rgb = img_utils.load(rgb_filepath.decode())
            # mask
            rgb = pre.mask_img(rgb, mask)
            # resize
            rgb = pre.resize_rgb_img(rgb, input_shape)
            # preprocess
            rgb = pre.preprocess_img(rgb,
                                     mask=mask_resized,
                                     scale01=scale01,
                                     standardize=standardize,
                                     zero_mean=zero_mean,
                                     unit_variance=unit_variance)
            # convert to correct data format
            axes = '01c' if K.image_data_format() == 'channels_last' else 'c01'
            rgb = img_utils.dimshuffle(rgb, '01c', axes)

        if input_type == INPUT_DEPTH:
            return depth, orientation
        elif input_type == INPUT_RGB:
            return rgb, orientation
        elif input_type == INPUT_DEPTH_AND_RGB:
            return depth, rgb, orientation

    # define function for augmentation
    def _augment(*args):
        if np.random.uniform() > 0.5:
            ret = ()
            for a in args[:-1]:
                if K.image_data_format() == 'channels_last':
                    ret = ret + (a[:, ::-1, :],)
                else:
                    ret = ret + (a[:, :, ::-1],)

            ret = ret + (np.array((360. - args[-1]) % 360., dtype=K.floatx()),)
            return ret
        else:
            return args[:-1] + (np.array(args[-1], dtype=K.floatx()),)

    # define function for output preprocessing
    def _preprocess_output(*args):
        orientation = args[-1]
        if output_type == OUTPUT_BITERNION:
            t = pre.deg2biternion(orientation)
        elif output_type == OUTPUT_REGRESSION:
            t = pre.deg2rad(orientation)
        elif output_type == OUTPUT_CLASSIFICATION:
            t = pre.deg2class(orientation, n_classes)

        return args[:-1] + (t,)

    # define function for shape handling and packaging
    def _set_shapes_and_pack(*args):
        # determine shapes
        s = list(input_shape)
        shapes = []
        if input_type in [INPUT_DEPTH, INPUT_DEPTH_AND_RGB]:
            shapes.append(s+[1] if K.image_data_format() == 'channels_last'
                          else [1]+s)
        if input_type in [INPUT_RGB, INPUT_DEPTH_AND_RGB]:
            shapes.append(s+[3] if K.image_data_format() == 'channels_last'
                          else [3]+s)
        if output_type == OUTPUT_BITERNION:
            shapes.append([2])
        elif output_type == OUTPUT_REGRESSION:
            shapes.append([1])
        elif output_type == OUTPUT_CLASSIFICATION:
            shapes.append([n_classes])
        # set shapes
        for a, s in zip(args, shapes):
            a.set_shape(s)

        # finally, convert to keras format: (tuple of inputs, tuple of outputs)
        return (args[:-1]), args[-1]

    n_samples = len(dataset)

    # create dataset from generator
    ds = tf.data.Dataset.from_generator(_sample_generator,
                                        output_types=(tf.string, tf.string,
                                                      tf.string, tf.float32))

    # apply input preprocessing
    out_dtypes = [tf.float32] * (3 if input_type == INPUT_DEPTH_AND_RGB else 2)
    ds = ds.map(
        lambda d_fp, rgb_fp, m_fp, o: tuple(tf.py_func(
            _load_and_preprocess_input, [d_fp, rgb_fp, m_fp, o], out_dtypes)),
        num_parallel_calls=5)

    # cache
    ds = ds.cache()

    # shuffle
    if shuffle:
        ds = ds.shuffle(buffer_size=n_samples)

    # apply augmentation
    if flip:
        out_dtypes = [tf.float32] * (
            3 if input_type == INPUT_DEPTH_AND_RGB else 2)
        ds = ds.map(
            lambda *args: tuple(tf.py_func(_augment, [*args], out_dtypes)),
            num_parallel_calls=5)

    # apply output preprocessing
    out_dtypes = [tf.float32] * (3 if input_type == INPUT_DEPTH_AND_RGB else 2)
    ds = ds.map(
        lambda *args: tuple(tf.py_func(
            _preprocess_output, [*args], out_dtypes)),
        num_parallel_calls=5)

    # since several python function were applied, we have to set the shapes
    # manually :(
    # finally, convert to keras format: (tuple of inputs, tuple of outputs)
    ds = ds.map(_set_shapes_and_pack, num_parallel_calls=5)

    # batch
    ds = ds.batch(batch_size)

    # prefetch some batches
    ds = ds.prefetch(n_prefetch_batches)

    # create repeated dataset (for Keras)
    ds = ds.repeat()

    return ds


def main():
    # parse args --------------------------------------------------------------
    args = _parse_args()

    # set device --------------------------------------------------------------
    os.environ['CUDA_VISIBLE_DEVICES'] = args.devices

    # output path -------------------------------------------------------------
    lr_str = f'{args.learning_rate:0.6f}'.replace('.', '_')

    model_str = args.model
    if args.model == 'mobilenet_v2':
        model_str += f"_{args.mobilenet_v2_alpha:0.2f}".replace('.', '_')

    exp_identifier = (f'{model_str}__'
                      f'{args.input_type}__'
                      f'{args.input_height}x{args.input_width}__'
                      f'{args.output_type}__'
                      f'{lr_str}__'
                      f'{args.run_id}')
    output_path = os.path.join(args.output_basepath, exp_identifier)
    create_directory_if_not_exists(output_path)

    # dump args ---------------------------------------------------------------
    write_json(os.path.join(output_path, 'config.json'), vars(args))

    # data --------------------------------------------------------------------
    # samples for training
    train_set = dataset.load_set(dataset_basepath=args.dataset_basepath,
                                 set_name=args.training_set,
                                 default_size=args.dataset_size)
    # limit training samples to multiple of batch size
    train_set.strip_to_multiple_of_batch_size(args.batch_size)
    train_steps_per_epoch = len(train_set) // args.batch_size

    # samples for validation
    valid_sets = [dataset.load_set(dataset_basepath=args.dataset_basepath,
                                   set_name=sn,
                                   default_size=args.dataset_size)
                  for sn in args.validation_sets]
    valid_steps_per_epoch = \
        [(len(set)+args.validation_batch_size-1) // args.validation_batch_size
         for set in valid_sets]

    # create tensorflow datasets
    tf_dataset_train = _create_tensorflow_dataset(
        dataset=train_set,
        input_type=args.input_type,
        input_shape=(args.input_height, args.input_width),
        output_type=args.output_type,
        batch_size=args.batch_size,
        n_prefetch_batches=5,
        n_classes=args.n_classes,
        shuffle=True,
        flip=not args.no_augmentation,
        scale01=args.input_preprocessing == 'scale01',
        standardize=args.input_preprocessing == 'standardize',
        zero_mean=True,
        unit_variance=True
    )
    tf_datasets_valid = \
        [_create_tensorflow_dataset(
            dataset=set_,
            input_type=args.input_type,
            input_shape=(args.input_height, args.input_width),
            output_type=args.output_type,
            batch_size=args.validation_batch_size,
            n_prefetch_batches=5,
            n_classes=args.n_classes,
            shuffle=False,
            flip=False,
            scale01=args.input_preprocessing == 'scale01',
            standardize=args.input_preprocessing == 'standardize',
            zero_mean=True,
            unit_variance=True)
         for set_ in valid_sets]

    # model -------------------------------------------------------------------
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
    if args.optimizer == 'adam':
        # adam
        opt = Adam(lr=args.learning_rate, beta_1=0.9, beta_2=0.999,
                   epsilon=None, decay=0.0, amsgrad=False)
    elif args.optimizer == 'rmsprop':
        opt = RMSprop(lr=args.learning_rate, rho=0.9, epsilon=None, decay=0.0)
    else:
        # sgd
        opt = SGD(lr=args.learning_rate, momentum=args.momentum, decay=0.0)

    if args.output_type == OUTPUT_BITERNION:
        kappa = args.kappa or 1.0
        loss = losses.vonmisses_loss_biternion(kappa)
    elif args.output_type == OUTPUT_REGRESSION:
        kappa = args.kappa or 0.5
        loss = losses.vonmisses_loss(kappa)
    else:
        loss = 'categorical_crossentropy'

    model.compile(optimizer=opt, loss=loss)

    # callbacks ---------------------------------------------------------------
    cbs = []
    # Validation callbacks
    # map 'validation' (= dataset.VALID_SET) to 'valid' (keras default for
    # validation set)
    dataset_names = [n.replace(dataset.VALID_SET, 'valid')
                     for n in args.validation_sets]
    for tf_ds, ds_name, steps in zip(tf_datasets_valid, dataset_names,
                                     valid_steps_per_epoch):
        # note: epoch is in range [0, args.n_epochs-1]
        filepath = os.path.join(output_path,
                                f'outputs_{ds_name}'+'_{epoch:04d}.npy')
        cbs.append(callbacks.ValidationCallback(tf_dataset=tf_ds,
                                                dataset_name=ds_name,
                                                output_filepath=filepath,
                                                validation_steps=steps,
                                                verbose=int(args.verbose)))

    # early stopping
    if args.early_stopping > 0:
        cbs.append(EarlyStopping(monitor='valid_loss',
                                 patience=args.early_stopping,
                                 mode='min',
                                 verbose=int(args.verbose)))

    # learning rate poly decay
    max_iter = train_steps_per_epoch*args.n_epochs
    cbs.append(callbacks.LRPolyDecay(lr_init=args.learning_rate,
                                     power=0.9,
                                     max_iter=max_iter,
                                     lr_min=1e-6,
                                     verbose=int(args.verbose)))

    # model checkpoints
    # note: due to keras implementation 'epoch' is in range [1, args.n_epochs]
    filepath = os.path.join(output_path, 'weights_valid_{epoch:04d}.hdf5')
    cbs.append(ModelCheckpoint(filepath=filepath,
                               monitor='valid_loss',
                               mode='min',
                               verbose=int(args.verbose),
                               save_best_only=True,
                               save_weights_only=True))
    filepath = os.path.join(output_path, 'weights_test_{epoch:04d}.hdf5')
    cbs.append(ModelCheckpoint(filepath=filepath,
                               monitor='test_loss',
                               mode='min',
                               verbose=int(args.verbose),
                               save_best_only=True,
                               save_weights_only=True))

    # CSV logger
    cbs.append(CSVLogger(filename=os.path.join(output_path, 'log.csv')))

    # Tensorboard
    cbs.append(TensorBoard(log_dir=output_path,
                           histogram_freq=0,
                           batch_size=32,
                           write_graph=False,
                           write_grads=False,
                           write_images=False,
                           embeddings_freq=0,
                           embeddings_layer_names=None,
                           embeddings_metadata=None,
                           embeddings_data=None))

    # TerminateOnNaN
    cbs.append(TerminateOnNaN())

    # training ----------------------------------------------------------------
    model.fit(tf_dataset_train,
              epochs=args.n_epochs,
              steps_per_epoch=train_steps_per_epoch,
              callbacks=cbs)


if __name__ == '__main__':
    main()
