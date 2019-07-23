# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

"""
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical

from . import img_utils

# see: extract_mean_and_std.py
_RGB_MEANS = {
    # beyer
    (123, 54): (44.21400569, 43.66822962, 34.8135663),
    (68, 68): (44.22773147, 43.68055367, 34.82029051),
    (46, 46): (44.21047002, 43.66637243, 34.8192426),
    # beyer_mod
    (126, 48): (44.21755497, 43.67245863, 34.81711793),
    (96, 96): (44.21673501, 43.67051567, 34.8150814),
    (48, 48): (44.19723332, 43.65215421, 34.80570469)
}

_RGB_STDS = {
    # beyer
    (123, 54): (44.44694949, 43.10526, 42.42409946),
    (68, 68): (44.4560061, 43.11495019, 42.42998318),
    (46, 46): (44.44384712, 43.10110676, 42.42469019),
    # beyer_mod
    (126, 48): (44.45056138, 43.11000162, 42.42929206),
    (96, 96): (44.45012716, 43.10778348, 42.42515952),
    (48, 48): (44.4348561, 43.08977808, 42.41423595)
}

_RGB_MAX = 255.0

_DEPTH_MEANS = {
    # beyer
    (123, 54): 3264.00968793,
    (68, 68): 3264.16421881,
    (46, 46): 3265.02556636,
    # beyer_mod
    (126, 48): 3264.19940802,
    (96, 96): 3264.22263638,
    (48, 48): 3264.90346171
}

_DEPTH_STDS = {
    # beyer
    (123, 54): 984.23187293,
    (68, 68): 984.29443982,
    (46, 46): 984.13732871,
    # beyer_mod
    (126, 48): 984.19405027,
    (96, 96): 984.20424869,
    (48, 48): 984.02504252
}

_DEPTH_MAX = 8000.0


# default interpolation for depth, rgb and mask images
_DEPTH_INTERPOLATION = 'nearest'
_RGB_INTERPOLATION = 'nearest'
_MASK_INTERPOLATION = 'nearest'


def resize_mask(mask, shape_or_scale):
    return img_utils.resize(mask, shape_or_scale,
                            interpolation=_MASK_INTERPOLATION)


def resize_depth_img(depth_img, shape_or_scale):
    return img_utils.resize(depth_img, shape_or_scale,
                            interpolation=_DEPTH_INTERPOLATION)


def resize_rgb_img(rgb_img, shape_or_scale):
    return img_utils.resize(rgb_img, shape_or_scale,
                            interpolation=_RGB_INTERPOLATION)


def mask_img(img, mask, fill_value=0):
    img[np.logical_not(mask), ...] = fill_value
    return img


def preprocess_img(img, mask=None,
                   scale01=False,
                   standardize=True, zero_mean=True, unit_variance=True):
    assert scale01 ^ standardize
    height, width, n_ch = img.shape
    shape = (height, width)

    # convert to floatX
    img = img.astype(K.floatx())

    # normalize
    if standardize:
        if zero_mean:
            mean = _RGB_MEANS[shape] if n_ch == 3 else _DEPTH_MEANS[shape]
            mean = np.array(mean, dtype=K.floatx()).reshape(1, 1, n_ch)
            img -= mean

        if unit_variance:
            std = _RGB_STDS[shape] if n_ch == 3 else _DEPTH_STDS[shape]
            std = np.array(std, dtype=K.floatx()).reshape(1, 1, n_ch)
            img /= std

    # scale
    if scale01:
        max_ = _RGB_MAX if n_ch == 3 else _DEPTH_MAX
        max_ = np.array(max_, dtype=K.floatx())
        img /= max_

    if mask is not None:
        # fix invalid pixels (apply mask again)
        img = mask_img(img, mask)

    return img


def preprocess_img_inverse(img,
                           scale01=False,
                           standardize=True, zero_mean=True,
                           unit_variance=True):
    assert scale01 ^ standardize
    height, width, n_ch = img.shape
    shape = (height, width)

    # derive mask
    mask = img.sum(axis=-1) != 0

    # inverse normalization
    if standardize:
        if unit_variance:
            std = _RGB_STDS[shape] if n_ch == 3 else _DEPTH_STDS[shape]
            std = np.array(std, dtype=K.floatx()).reshape(1, 1, n_ch)
            img *= std

        if zero_mean:
            mean = _RGB_MEANS[shape] if n_ch == 3 else _DEPTH_MEANS[shape]
            mean = np.array(mean, dtype=K.floatx()).reshape(1, 1, n_ch)
            img += mean

    # inverse scaling
    if scale01:
        max_ = _RGB_MAX if n_ch == 3 else _DEPTH_MAX
        max_ = np.array(max_, dtype=K.floatx())
        img *= max_

    # fix invalid pixels (apply mask again)
    img = mask_img(img, mask)

    # convert dtype
    img = img.astype('uint8' if n_ch == 3 else 'uint16')

    return img


def deg2biternion(deg):
    y = np.deg2rad(deg)
    return np.array([np.cos(y), np.sin(y)], dtype=K.floatx())


def deg2rad(deg):
    return np.deg2rad(deg).astype(K.floatx())


def deg2class(deg, n_classes):
    bin_width = 360. / n_classes
    bin_width_2 = 360. / (2 * n_classes)
    angle_class = (((deg + bin_width_2) % 360.) / bin_width).astype('uint')
    return to_categorical(angle_class, num_classes=n_classes)
