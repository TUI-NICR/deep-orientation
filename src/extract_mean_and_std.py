# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

"""
import argparse as ap

import numpy as np

from nicr_rgb_d_orientation_data_set import SETS, SIZES, SIZE_SMALL
from nicr_rgb_d_orientation_data_set import load_set
import deep_orientation.preprocessing as pre

import utils.img as img_utils


def _parse_args():
    """Parse command-line arguments"""
    desc = 'Simple script for extracting mean and std statistics'
    parser = ap.ArgumentParser(description=desc,
                               formatter_class=ap.RawTextHelpFormatter)

    parser.add_argument('-d', '--dataset_basepath',
                        type=str,
                        default='/datasets/rotator',
                        help=("Path to downloaded dataset (default: "
                              "'/datasets/rotator')"))

    parser.add_argument('-s', '--size',
                        type=str,
                        choices=SIZES,
                        default=SIZE_SMALL,
                        help=(f"Dataset image size to use. One of :"
                              f"{SIZES} (default: {SIZE_SMALL})"))

    parser.add_argument('-iw', '--image_width',
                        type=int,
                        default=None,
                        help="Desired patch width")

    parser.add_argument('-ih', '--image_height',
                        type=int,
                        default=None,
                        help="Desired patch height")

    parser.add_argument('-sn', '--set_name',
                        type=str,
                        default='training',
                        help=("Sets to extract mean and std for. One of :"
                              f"{SETS}"))

    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        default=False,
                        help='Enable verbose output')

    # return parsed args
    return parser.parse_args()


def extract_mean_and_std(dataset_basepath, set_name, size, patch_shape,
                         with_progressbar=True):
    """
    Extract patch mean and patch std for a given set. Note that the statistics
    are calculated in two consecutive runs through the dataset.

    Parameters
    ----------
    dataset_basepath : str
        Path to dataset root, e.g. '/dataset/orientation/'.
    set_name : str
        Set to extract the patches for. Should be one of 'training',
        'validation' or 'test'.
    size : str
        Image size to use. Should be either 'small' or 'large'.
    patch_shape : {tuple, None}
        Size of the patches as tuple (height, width) or None to skip resizing
        at all.
    with_progressbar : bool
        If `with_progressbar` is True, `tqdm` is used to display a progress
        bar.

    Returns
    -------
    mean_rgb, mean_depth, std_rgb, std_depth : tuple
        Calculated measures as tuple.

    """
    # load dataset
    dataset = load_set(dataset_basepath=dataset_basepath,
                       set_name=set_name,
                       default_size=size)

    # small helper function for tqdm
    def _tqdm_wrapper(iterable, **tqdm_kwargs):
        if with_progressbar:
            from tqdm import tqdm
            return tqdm(iterable, unit="images", **tqdm_kwargs)
        else:
            return iterable

    # small helper function to load, mask and resize all images of a sample
    def _load_masked_patches(sample):
        # load images
        rgb_patch = sample.get_rgb_patch()
        depth_patch = sample.get_depth_patch()
        mask_patch = sample.get_mask_patch() > 0

        # mask images
        rgb_patch[np.logical_not(mask_patch), ...] = 0
        depth_patch[np.logical_not(mask_patch), ...] = 0

        # scale images
        if patch_shape is not None:
            rgb_patch = pre.resize_rgb_img(rgb_patch, patch_shape)

            depth_patch = pre.resize_depth_img(depth_patch, patch_shape)
            mask_patch = pre.resize_mask(mask_patch, patch_shape) > 0
        return rgb_patch, depth_patch, mask_patch

    # get mean
    rgb_sum = np.array([0.0, 0.0, 0.0], dtype='float64')
    d_sum = np.array([0.0], dtype='float64')
    n_files = len(dataset)
    n_pixel = 0
    for s in _tqdm_wrapper(dataset, initial=0, total=2*n_files):
        # load, mask and (opt.) resize images
        rgb_patch, depth_patch, mask_patch = _load_masked_patches(s)

        # accumulate sums
        rgb_sum += rgb_patch[mask_patch, ...].sum(axis=0)
        d_sum += depth_patch[mask_patch].sum()

        n_pixel += mask_patch.sum()
    mean_rgb = rgb_sum/n_pixel
    mean_depth = d_sum/n_pixel

    # get std
    rgb_ssum = np.array([0.0, 0.0, 0.0], dtype='float64')
    d_ssum = np.array([0.0], dtype='float64')
    for s in _tqdm_wrapper(dataset, initial=n_files, total=2*n_files):
        # load, mask and (opt.) resize images
        rgb_patch, depth_patch, mask_patch = _load_masked_patches(s)

        # cast to int64
        rgb_patch = rgb_patch.astype('int64')
        depth_patch = depth_patch.astype('int64')

        # accumulate squared sums
        rgb_ssum += ((rgb_patch[mask_patch, ...] - mean_rgb)**2).sum(axis=0)
        d_ssum += ((depth_patch[mask_patch] - mean_depth)**2).sum()
    std_rgb = np.sqrt(rgb_ssum/n_pixel)
    std_depth = np.sqrt(d_ssum/n_pixel)

    return mean_rgb, std_rgb, mean_depth, std_depth


def main():
    # parse args
    args = _parse_args()

    # get mean and std
    if args.image_width is not None and args.image_height is not None:
        patch_shape = (args.image_height, args.image_width)
    else:
        patch_shape = None

    res = extract_mean_and_std(dataset_basepath=args.dataset_basepath,
                               set_name=args.set_name,
                               size=args.size,
                               patch_shape=patch_shape,
                               with_progressbar=args.verbose)

    print(f"RGB: {res[0]} +- {res[1]}\nDepth: {res[2]} +- {res[3]}")


if __name__ == '__main__':
    main()


# Results Set: training -------------------------------------------------------

# Small without resizing
# -> python extract_mean_and_std.py -v
# RGB: [44.51986784 44.28380003 36.84354289]
#           +- [45.02234949 43.72092759 44.03834093]
# Depth: [2653.79497631]
#           +- [881.3370401]

# Small resized to (123, 54)
# -> python extract_mean_and_std.py -v -ih 123 -iw 54
# RGB: [44.21400569 43.66822962 34.81356634]
#           +- [44.44694949 43.10526    42.42409946]
# Depth: [3264.00968793]
#           +- [984.23187293]

# Small resized to (68, 68)
# -> python extract_mean_and_std.py -v -ih 68 -iw 68
# RGB: [44.22773147 43.68055367 34.82029051]
#           +- [44.4560061  43.11495019 42.42998318]
# Depth: [3264.16421881]
#           +- [984.29443982]

# Small resized to (46, 46)
# -> python extract_mean_and_std.py -v -ih 46 -iw 46
# RGB: [44.21047002 43.66637243 34.8192426 ]
#           +- [44.44384712 43.10110676 42.42469019]
# Depth: [3265.02556636]
#           +- [984.13732871]

# Small resized to (126, 48)
# -> python extract_mean_and_std.py -v -ih 126 -iw 48
# RGB: [44.21755497 43.67245863 34.81711793]
#       +- [44.45056138 43.11000162 42.42929206]
# Depth: [3264.19940802]
#       +- [984.19405027]

# Small resized to (96, 96)
# -> python extract_mean_and_std.py -v -ih 96 -iw 96
# RGB: [44.21673501 43.67051567 34.8150814]
#           +- [44.45012716 43.10778348 42.42515952]
# Depth: [3264.22263638]
#           +- [984.20424869]

# Small resized to (48, 48)
# -> python extract_mean_and_std.py -v -ih 48 -iw 48
# RGB: [44.19723332 43.65215421 34.80570469]
#           +- [44.4348561  43.08977808 42.41423595]
# Depth: [3264.90346171]
#           +- [984.02504252]

# Results Set: validation -----------------------------------------------------
# Small without resizing
# -> python extract_mean_and_std.py -v -sn validation
# RGB: [47.38808171 47.73720528 38.92053315]
#           +- [49.83459429 50.23246324 50.31454072]
# Depth: [2645.71610819]
#           +- [856.64936016]

# Results Set: test -----------------------------------------------------------
# Small without resizing
# -> python extract_mean_and_std.py -v -sn test
# RGB: [47.48801353 46.46069743 41.25795585]
#           +- [46.12427943 41.60361093 44.48936395]
# Depth: [2666.5722938]
#           +- [912.43737393]
