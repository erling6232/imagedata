"""This module provides plugins for various imaging formats.

Standard plugins provides support for DICOM and Nifti image file formats.
"""

# Copyright (c) 2013-2022 Erling Andersen, Haukeland University Hospital, Bergen, Norway

import logging
import sys

import numpy as np

logger = logging.getLogger(__name__)

(SORT_ON_SLICE,
 SORT_ON_TAG) = range(2)
sort_on_set = {SORT_ON_SLICE, SORT_ON_TAG}

INPUT_ORDER_NONE = 'none'
INPUT_ORDER_TIME = 'time'
INPUT_ORDER_B = 'b'
INPUT_ORDER_FA = 'fa'
INPUT_ORDER_TE = 'te'
INPUT_ORDER_FAULTY = 'faulty'
input_order_set = {INPUT_ORDER_NONE, INPUT_ORDER_TIME, INPUT_ORDER_B, INPUT_ORDER_FA,
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


def input_order_to_str(input_order):
    if input_order == INPUT_ORDER_NONE:
        return "INPUT_ORDER_NONE"
    elif input_order == INPUT_ORDER_TIME:
        return "INPUT_ORDER_TIME"
    elif input_order == INPUT_ORDER_B:
        return "INPUT_ORDER_B"
    elif input_order == INPUT_ORDER_FA:
        return "INPUT_ORDER_FA"
    elif input_order == INPUT_ORDER_TE:
        return "INPUT_ORDER_TE"
    elif input_order == INPUT_ORDER_FAULTY:
        return "INPUT_ORDER_FAULTY"
    elif issubclass(type(input_order), str):
        return input_order
    else:
        raise (UnknownTag("Unknown numerical input_order {:d}.".format(input_order)))


def input_order_to_dirname_str(input_order):
    if input_order == INPUT_ORDER_NONE:
        return "none"
    elif input_order == INPUT_ORDER_TIME:
        return "time"
    elif input_order == INPUT_ORDER_B:
        return "b"
    elif input_order == INPUT_ORDER_FA:
        return "fa"
    elif input_order == INPUT_ORDER_TE:
        return "te"
    elif input_order == INPUT_ORDER_FAULTY:
        return "faulty"
    elif issubclass(type(input_order), str):
        keepcharacters = ('-', '_', '.', ' ')
        return ''.join([c for c in input_order if c.isalnum() or c in keepcharacters]).rstrip()
    else:
        raise (UnknownTag("Unknown numerical input_order {:d}.".format(input_order)))


def str_to_input_order(s):
    if s == "none":
        return INPUT_ORDER_NONE
    elif s == "time":
        return INPUT_ORDER_TIME
    elif s == "b":
        return INPUT_ORDER_B
    elif s == "fa":
        return INPUT_ORDER_FA
    elif s == "te":
        return INPUT_ORDER_TE
    elif s == "faulty":
        return INPUT_ORDER_FAULTY
    else:
        # raise (UnknownTag("Unknown input order {}.".format(s)))
        return s


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
    from imagedata import plugins
    return plugins['format'] if 'format' in plugins else []


def find_plugin(ftype):
    """Return plugin for given format type."""
    plugins = get_plugins_list()
    for pname, ptype, pclass in plugins:
        if ptype == ftype:
            return pclass()
    raise FormatPluginNotFound("Plugin for format {} not found.".format(ftype))
