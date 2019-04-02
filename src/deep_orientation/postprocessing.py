# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

"""
import numpy as np


def rad2deg(rad):
    return np.rad2deg(rad.flatten()) % 360


def biternion2deg(biternion):
    rad = np.arctan2(biternion[:, 1], biternion[:, 0])
    return rad2deg(rad)


def class2deg(c, n_classes):
    bin_width = 360. / n_classes
    return c * bin_width
