# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from operator import attrgetter

import numpy as np
import cv2


def _rint(value):
    return int(np.round(value))


def _is_cv3():
    return cv2.__version__.startswith("3")


def _const(cv2_const, cv3_const):
    return attrgetter(cv3_const if _is_cv3() else cv2_const)(cv2)


_INTERPOLATION_DICT = {
    # bicubic interpolation
    'bicubic': _const('INTER_CUBIC', 'INTER_CUBIC'),
    # nearest-neighbor interpolation
    'nearest': _const('INTER_NEAREST', 'INTER_NEAREST'),
    # bilinear interpolation (4x4 pixel neighborhood)
    'linear': _const('INTER_LINEAR', 'INTER_LINEAR'),
    # resampling using pixel area relation, preferred for shrinking
    'area': _const('INTER_AREA', 'INTER_AREA'),
    # Lanczos interpolation (8x8 pixel neighborhood)
    'lanczos4': _const('INTER_LANCZOS4', 'INTER_LANCZOS4')
}


def resize(img, shape_or_scale, interpolation='linear'):
    # ensure that img is a numpy object
    img = np.asanyarray(img)

    # get current shape
    cur_height, cur_width = img.shape[:2]

    # check shape_or_scale
    if isinstance(shape_or_scale, (tuple, list)) and len(shape_or_scale) == 2:
        if all(isinstance(e, int) for e in shape_or_scale):
            new_height, new_width = shape_or_scale
        elif all(isinstance(e, float) for e in shape_or_scale):
            fy, fx = shape_or_scale
            new_height = _rint(fy*cur_height)
            new_width = _rint(fx*cur_width)
        else:
            raise ValueError("`shape_or_scale` should either be a tuple of "
                             "ints (height, width) or a tuple of floats "
                             "(fy, fx)")
    elif isinstance(shape_or_scale, float):
        new_height = _rint(shape_or_scale * cur_height)
        new_width = _rint(shape_or_scale * cur_width)
    else:
        raise ValueError("`shape_or_scale` should either be a tuple of ints "
                         "(height, width) or a tuple of floats (fy, fx) or a "
                         "single float value")

    # scale image
    if cur_height == new_height and cur_width == new_width:
        return img

    return cv2.resize(img,
                      dsize=(new_width, new_height),
                      interpolation=_INTERPOLATION_DICT[interpolation])
