# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

"""
import argparse as ap

from nicr_rgb_d_orientation_data_set import SETS, SIZES, SIZE_SMALL
from nicr_rgb_d_orientation_data_set import load_set


def _parse_args():
    """Parse command-line arguments"""
    desc = 'Simple script for extracting patches without further preprocessing'
    parser = ap.ArgumentParser(description=desc,
                               formatter_class=ap.RawTextHelpFormatter)

    parser.add_argument('-d', '--dataset_basepath',
                        type=str,
                        default='/datasets/rotator',
                        help=("Path to downloaded dataset, default: "
                              "'/datasets/rotator'"))

    parser.add_argument('-s', '--size',
                        type=str,
                        choices=SIZES,
                        default=SIZE_SMALL,
                        help=(f"Image size to extract the patches for. One "
                              f"of :{SIZES}, default: {SIZE_SMALL}"))

    parser.add_argument('-sn', '--set_names',
                        type=str,
                        nargs='+',
                        default=SETS,
                        help=("Sets to extract the patches for. One of :"
                              f"{SETS}"))

    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        default=False,
                        help='Enable verbose output')

    # return parsed args
    return parser.parse_args()


def main():
    # parse args
    args = _parse_args()

    # extract patches
    for set_name in args.set_names:
        if args.verbose:
            print(f"Processing {set_name} set ...")

        # load dataset
        dataset = load_set(dataset_basepath=args.dataset_basepath,
                           set_name=set_name,
                           default_size=args.size)
        # extract all patches
        dataset.extract_all_patches(size=None,
                                    with_progressbar=args.verbose)


if __name__ == '__main__':
    main()
