"""This module provides plugins for various imaging formats.

Standard plugins provides support for DICOM and Nifti image file formats.
"""

# Copyright (c) 2013-2018 Erling Andersen, Haukeland University Hospital, Bergen, Norway

import logging
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
input_order_set = {INPUT_ORDER_NONE, INPUT_ORDER_TIME, INPUT_ORDER_B, INPUT_ORDER_FA, INPUT_ORDER_TE,
                   INPUT_ORDER_FAULTY}


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
    else:
        raise ValueError("Unknown shape")


# noinspection PyGlobalUndefined
def add_plugin_dir(d):
    from pkgutil import extend_path
    global __path__, plugins
    __path__ = extend_path(__path__, d)
    plugins = load_plugins()


# noinspection PyGlobalUndefined
def load_plugins(plugins_folder_list=None):
    """Searches the plugins folder and imports all valid plugins,
    returning a list of the plugins successfully imported as tuples:
    first element is the plugin name (e.g. MyPlugin),
    second element is the class of the plugin.
    """
    import sys
    import os
    import imp
    import inspect
    from imagedata.formats.abstractplugin import AbstractPlugin

    global __path__
    global plugins

    try:
        if plugins_folder_list is None:
            plugins_folder_list = __path__
        if isinstance(plugins_folder_list, str):
            plugins_folder_list = [plugins_folder_list, ]
    except NameError:
        return

    logger.debug("type(plugins_folder_list) {}".format(type(plugins_folder_list)))
    logger.debug("__name__ {}".format(__name__))
    logger.debug("__file__ {}".format(__file__))
    logger.debug("__path__ {}".format(__path__))

    plugins = []
    for plugins_folder in plugins_folder_list:
        if plugins_folder not in sys.path:
            sys.path.append(plugins_folder)
        for root, dirs, files in os.walk(plugins_folder):
            logger.debug("root %s dirs %s" % (root, dirs))
            for module_file in files:
                module_name, module_extension = os.path.splitext(module_file)
                if module_extension == os.extsep + "py":
                    module_hdl = False
                    try:
                        # print("Attempt {}".format(module_name))
                        logger.debug("Attempt {}".format(module_name))
                        module_hdl, path_name, description = imp.find_module(module_name)
                        plugin_module = imp.load_module(module_name, module_hdl, path_name,
                                                        description)
                        plugin_classes = inspect.getmembers(plugin_module, inspect.isclass)
                        logger.debug("  Plugins {}".format(plugin_classes))
                        for plugin_class in plugin_classes:
                            # print("  Plugin {}".format(plugin_class[1]))
                            logger.debug("  Plugin {}".format(plugin_class[1]))
                            if issubclass(plugin_class[1], AbstractPlugin):
                                # Load only those plugins defined in the current module
                                # (i.e. don't instantiate any parent plugins)
                                if plugin_class[1].__module__ == module_name:
                                    # plugin = plugin_class[1]()
                                    pname, pclass = plugin_class
                                    plugins.append((pname, pclass.name, pclass))
                    # except (ImportError, TypeError, RuntimeError) as e:
                    except (ImportError, TypeError, RuntimeError):
                        # print("  ImportError: {}".format(e))
                        # logger.debug("  ImportError: {}".format(e))
                        pass
                    except Exception as e:
                        # print("  Exception: {}".format(e))
                        logger.debug("  Exception: {}".format(e))
                        raise
                    finally:
                        if module_hdl:
                            module_hdl.close()
    return plugins


def get_plugins_list():
    global plugins
    return plugins


def find_plugin(ftype):
    """Return plugin for given format type."""
    global plugins
    for pname, ptype, pclass in plugins:
        if ptype == ftype:
            return pclass()
    raise FormatPluginNotFound("Plugin for format {} not found.".format(ftype))


plugins = load_plugins()
