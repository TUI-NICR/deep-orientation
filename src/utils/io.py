# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Benjamin Lewandowski <benjamin.lewandowski@tu-ilmenau.de>

"""
from collections import OrderedDict
import errno
import json
import os

import numpy as np


def create_directory_if_not_exists(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def get_files_by_extension(path,
                           extension='.png',
                           flat_structure=False,
                           recursive=False,
                           follow_links=True):
    # check input args
    if not os.path.exists(path):
        raise IOError("No such file or directory: '{}'".format(path))

    if flat_structure:
        filelist = []
    else:
        filelist = {}

    # path is a file
    if os.path.isfile(path):
        basename = os.path.basename(path)
        if extension is None or basename.lower().endswith(extension):
            if flat_structure:
                filelist.append(path)
            else:
                filelist[os.path.dirname(path)] = [basename]
        return filelist

    # get filelist
    filter_func = lambda f: extension is None or f.lower().endswith(extension)
    for root, _, filenames in os.walk(path, topdown=True,
                                      followlinks=follow_links):
        filenames = list(filter(filter_func, filenames))
        if filenames:
            if flat_structure:
                filelist.extend((os.path.join(root, f) for f in filenames))
            else:
                filelist[root] = sorted(filenames)
        if not recursive:
            break

    # return
    if flat_structure:
        return sorted(filelist)
    else:
        return OrderedDict(sorted(filelist.items()))


def read_keras_csv_logfile(filepath):
    # load csv file (note: first row is a header containing the measure names)
    log = np.genfromtxt(filepath, delimiter=',', dtype=np.unicode_)
    # convert to dict and return
    return {log[0, i]: np.array(log[1:, i], dtype='float32')
            for i in range(log.shape[1])}


def read_json(json_path):
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    return json_data


def write_json(json_path, json_data):
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=4, sort_keys=True)


def write_pgm(file_path, img):
    max_ = img.max()
    height, width = img.shape
    with open(file_path, 'wb') as f:
        f.write('P5 {:d} {:d} {:d} '.format(width, height, max_))
        f.write(img.byteswap().tobytes())
