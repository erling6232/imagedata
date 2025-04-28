"""This module provides plugins for various imaging formats.

Standard plugins provides support for DICOM and Nifti image file formats.
"""

# Copyright (c) 2013-2024 Erling Andersen, Haukeland University Hospital, Bergen, Norway

import os
import sys
import time
import uuid
import numpy as np


(SORT_ON_SLICE,
 SORT_ON_TAG) = range(2)
sort_on_set = {SORT_ON_SLICE, SORT_ON_TAG}

INPUT_ORDER_AUTO = 'auto'
INPUT_ORDER_NONE = 'none'
INPUT_ORDER_TIME = 'time'
INPUT_ORDER_TRIGGERTIME = 'triggertime'
INPUT_ORDER_B = 'b'
INPUT_ORDER_BVECTOR = 'bvector'
INPUT_ORDER_RSI = 'rsi'
INPUT_ORDER_FA = 'fa'
INPUT_ORDER_TE = 'te'
INPUT_ORDER_FAULTY = 'faulty'
input_order_set = {INPUT_ORDER_NONE, INPUT_ORDER_TIME, INPUT_ORDER_B, INPUT_ORDER_BVECTOR,
                   INPUT_ORDER_RSI, INPUT_ORDER_FA, INPUT_ORDER_TRIGGERTIME,
                   INPUT_ORDER_TE, INPUT_ORDER_FAULTY}


class NotImageError(Exception):
    pass


class EmptyImageError(Exception):
    pass


class UnknownInputError(Exception):
    pass


class UnknownTag(Exception):
    pass


class NotTimeOrder(Exception):
    pass


class CannotSort(Exception):
    pass


class SOPInstanceUIDNotFound(Exception):
    pass


class FormatPluginNotFound(Exception):
    pass


class WriteNotImplemented(Exception):
    pass


def sort_on_to_str(sort_on):
    if sort_on == SORT_ON_SLICE:
        return "SORT_ON_SLICE"
    elif sort_on == SORT_ON_TAG:
        return "SORT_ON_TAG"
    else:
        raise (UnknownTag("Unknown numerical sort_on {:d}.".format(sort_on)))


def str_to_sort_on(s):
    if s == "slice":
        return SORT_ON_SLICE
    elif s == "tag":
        return SORT_ON_TAG
    else:
        raise (UnknownTag("Unknown sort_on string {}.".format(s)))


def str_to_dtype(s):
    if s == "none":
        return None
    elif s == "uint8":
        return np.uint8
    elif s == "uint16":
        return np.uint16
    elif s == "int16":
        return np.int16
    elif s == "int":
        return np.int16
    elif s == "float":
        return np.float
    elif s == "float32":
        return np.float32
    elif s == "float64":
        return np.float64
    elif s == "double":
        return np.double
    else:
        raise (ValueError("Output data type {} not implemented.".format(s)))


def shape_to_str(shape):
    """Convert numpy image shape to printable string

    Args:
        shape
    Returns:
        printable shape (str)
    Raises:
        ValueError: when shape cannot be converted to printable string
    """
    if len(shape) == 5:
        return "{}x{}tx{}x{}x{}".format(shape[0], shape[1], shape[2], shape[3], shape[4])
    elif len(shape) == 4:
        return "{}tx{}x{}x{}".format(shape[0], shape[1], shape[2], shape[3])
    elif len(shape) == 3:
        return "{}x{}x{}".format(shape[0], shape[1], shape[2])
    elif len(shape) == 2:
        return "{}x{}".format(shape[0], shape[1])
    elif len(shape) == 1:
        return "{}".format(shape[0])
    else:
        raise ValueError("Unknown shape")


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


def get_plugins_list():
    from .. import plugins
    return plugins['format'] if 'format' in plugins else []


def find_plugin(ftype):
    """Return plugin for given format type."""
    plugins = get_plugins_list()
    for pname, ptype, pclass in plugins:
        if ptype == ftype:
            return pclass()
    raise FormatPluginNotFound("Plugin for format {} not found.".format(ftype))


_my_root = "2.16.578.1.37.1.1.2.{}.{}.{}".format(
    hex(uuid.getnode())[2:],
    os.getpid(),
    int(time.time())
)


def get_uid(k=[0]) -> str:
    """Generator function which will return a unique UID.
    """
    while True:
        k[0] += 1
        yield "%s.%d" % (_my_root, k[0])
