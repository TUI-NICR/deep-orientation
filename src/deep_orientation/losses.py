# -*- coding: utf-8 -*
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

"""
from tensorflow.keras import backend as K


def vonmisses_loss_biternion(kappa=1.0):
    def loss(y_true, y_pred):
        cos_angles = K.batch_dot(y_pred, y_true, axes=1)
        cos_angles = K.exp(kappa * (cos_angles - 1))
        score = 1 - cos_angles
        return score
    return loss


def vonmisses_loss(kappa=0.5):
    def loss(y_true, y_pred):
        c = K.exp(2 * kappa)
        y_pred_ = K.flatten(y_pred)
        y_true_ = K.flatten(y_true)
        score = c - K.exp(kappa * (K.cos(y_pred_ - y_true_) + 1))
        return score
    return loss
