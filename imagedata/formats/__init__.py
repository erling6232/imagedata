"""This module provides plugins for various imaging formats.

Standard plugins provides support for DICOM and Nifti image file formats.
"""

# Copyright (c) 2013-2018 Erling Andersen, Haukeland University Hospital, Bergen, Norway

import logging
import numpy as np

logging.getLogger(__name__).addHandler(logging.NullHandler())

(SORT_ON_SLICE,
SORT_ON_TAG)             = range(2)
sort_on_set = {SORT_ON_SLICE, SORT_ON_TAG}

(INPUT_ORDER_NONE,
INPUT_ORDER_TIME,
INPUT_ORDER_B,
INPUT_ORDER_FA,
INPUT_ORDER_TE,
INPUT_ORDER_FAULTY)      = range(6)
input_order_set = {INPUT_ORDER_NONE, INPUT_ORDER_TIME, INPUT_ORDER_B, INPUT_ORDER_FA, INPUT_ORDER_TE, INPUT_ORDER_FAULTY}

class NotImageError(Exception): pass
class EmptyImageError(Exception): pass
class UnknownInputError(Exception): pass
class UnknownTag(Exception): pass
class NotTimeOrder(Exception): pass
class SOPInstanceUIDNotFound(Exception): pass
class FormatPluginNotFound(Exception): pass
class WriteNotImplemented(Exception): pass

def sort_on_to_str(sort_on):
    if sort_on == SORT_ON_SLICE:
        return("SORT_ON_SLICE")
    elif sort_on == SORT_ON_TAG:
        return("SORT_ON_TAG")
    else:
        raise(UnknownTag("Unknown numerical sort_on %d." % sort_on))

def str_to_sort_on(str):
    if str == "slice":
        return SORT_ON_SLICE
    elif str == "tag":
        return SORT_ON_TAG
    else:
        raise(UnknownTag("Unknown sort_on string %s." % str))

def str_to_dtype(str):
    if str == "none":
        return None
    elif str == "uint8":
        return np.uint8
    elif str == "uint16":
        return np.uint16
    elif str == "int16":
        return np.int16
    elif str == "int":
        return np.int
    elif str == "float":
        return np.float
    elif str == "float32":
        return np.float32
    elif str == "float64":
        return np.float64
    elif str == "double":
        return np.double
    else:
        raise(ValueError("Output data type %s not implemented." % str))

def input_order_to_str(input_order):
    if input_order == INPUT_ORDER_NONE:
        return("INPUT_ORDER_NONE")
    elif input_order == INPUT_ORDER_TIME:
        return("INPUT_ORDER_TIME")
    elif input_order == INPUT_ORDER_B:
        return("INPUT_ORDER_B")
    elif input_order == INPUT_ORDER_FA:
        return("INPUT_ORDER_FA")
    elif input_order == INPUT_ORDER_TE:
        return("INPUT_ORDER_TE")
    elif input_order == INPUT_ORDER_FAULTY:
        return("INPUT_ORDER_FAULTY")
    else:
        raise(UnknownTag("Unknown numerical input_order %d." % input_order))

def input_order_to_dirname_str(input_order):
    if input_order == INPUT_ORDER_NONE:
        return("none")
    elif input_order == INPUT_ORDER_TIME:
        return("time")
    elif input_order == INPUT_ORDER_B:
        return("b")
    elif input_order == INPUT_ORDER_FA:
        return("fa")
    elif input_order == INPUT_ORDER_TE:
        return("te")
    elif input_order == INPUT_ORDER_FAULTY:
        return("faulty")
    else:
        raise(UnknownTag("Unknown numerical input_order %d." % input_order))

def str_to_input_order(str):
    if str == "none":
        return INPUT_ORDER_NONE
    elif str == "time":
        return INPUT_ORDER_TIME
    elif str == "b":
        return INPUT_ORDER_B
    elif str == "fa":
        return INPUT_ORDER_FA
    elif str == "te":
        return INPUT_ORDER_TE
    elif str == "faulty":
        return INPUT_ORDER_FAULTY
    else:
        raise(UnknownTag("Unknown input order %s." % str))

def shape_to_str(shape):
    """Convert numpy image shape to printable string

    Input:
    - shape
    Returns:
    - printable shape (str)
    Exceptions:
    - ValueError: when shape cannot be converted to printable string
    """
    if len(shape) == 5:
        return "{}x{}tx{}x{}x{}".format(shape[0],shape[1],shape[2],shape[3],shape[4])
    elif len(shape) == 4:
        return "{}tx{}x{}x{}".format(shape[0],shape[1],shape[2],shape[3])
    elif len(shape) == 3:
        return "{}x{}x{}".format(shape[0],shape[1],shape[2])
    elif len(shape) == 2:
        return "{}x{}".format(shape[0],shape[1])
    else:
        raise ValueError("Unknown shape")

def add_plugin_dir(dir):
    from pkgutil import extend_path
    global __path__, plugins
    __path__ = extend_path(__path__, dir)
    plugins = load_plugins()

def load_plugins(plugins_folder_list=None):
    """Searches the plugins folder and imports all valid plugins,
    returning a list of the plugins successfully imported as tuples:
    first element is the plugin name (e.g. MyPlugin),
    second element is the class of the plugin.
    """
    import sys, os, imp, inspect
    from imagedata.formats.abstractplugin import AbstractPlugin

    global __path__
    global plugins

    try:
        if plugins_folder_list is None:
            plugins_folder_list = __path__
        if isinstance(plugins_folder_list, str):
                plugins_folder_list=[plugins_folder_list,]
    except NameError:
        return

    logging.debug("type(plugins_folder_list) {}".format(type(plugins_folder_list)))
    logging.debug("__name__ {}".format(__name__))
    logging.debug("__file__ {}".format(__file__))
    logging.debug("__path__ {}".format(__path__))

    plugins = []
    for plugins_folder in plugins_folder_list:
        if not plugins_folder in sys.path:
            sys.path.append(plugins_folder)
        for root, dirs, files in os.walk(plugins_folder):
            logging.debug("root %s dirs %s" % (root, dirs))
            for module_file in files:
                module_name, module_extension = os.path.splitext(module_file)
                if module_extension == os.extsep + "py":
                    try:
                        #print("Attempt {}".format(module_name))
                        logging.debug("Attempt {}".format(module_name))
                        module_hdl, path_name, description = imp.find_module(module_name)
                        plugin_module = imp.load_module(module_name, module_hdl, path_name,
                                                                                        description)
                        plugin_classes = inspect.getmembers(plugin_module, inspect.isclass)
                        logging.debug("  Plugins {}".format(plugin_classes))
                        for plugin_class in plugin_classes:
                            #print("  Plugin {}".format(plugin_class[1]))
                            logging.debug("  Plugin {}".format(plugin_class[1]))
                            if issubclass(plugin_class[1], AbstractPlugin):
                                # Load only those plugins defined in the current module
                                # (i.e. don't instantiate any parent plugins)
                                if plugin_class[1].__module__ == module_name:
                                    #plugin = plugin_class[1]()
                                    pname,pclass = plugin_class
                                    plugins.append((pname,pclass.name,pclass))
                    except ImportError as e:
                        #print("  ImportError: {}".format(e))
                        logging.debug("  ImportError: {}".format(e))
                        pass
                    except Exception as e:
                        #print("  Exception: {}".format(e))
                        logging.debug("  Exception: {}".format(e))
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
    for pname,ptype,pclass in plugins:
        if ptype == ftype:
            return pclass()
    raise FormatPluginNotFound("Plugin for format {} not found.".format(ftype))

plugins = load_plugins()
