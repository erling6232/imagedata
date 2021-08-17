"""This module provides plugins for various image transports.

Standard plugins provides support for file, http/https and xnat transports.
"""

# Copyright (c) 2013-2018 Erling Andersen, Haukeland University Hospital, Bergen, Norway

import logging
import urllib

logger = logging.getLogger(__name__)


global __path__


class TransportPluginNotFound(Exception):
    pass


class RootIsNotDirectory(Exception):
    pass


class RootDoesNotExist(Exception):
    pass


class FunctionNotSupported(Exception):
    pass


def add_plugin_dir(directory_name):
    from pkgutil import extend_path
    global __path__, plugins
    __path__ = extend_path(__path__, directory_name)
    plugins = load_plugins()


# noinspection PyDeprecation
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
    from imagedata.transports.abstracttransport import AbstractTransport

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

    plugins = {}
    for plugins_folder in plugins_folder_list:
        if plugins_folder not in sys.path:
            sys.path.append(plugins_folder)
        for root, dirs, files in os.walk(plugins_folder):
            logger.debug("root %s dirs %s" % (root, dirs))
            for module_file in files:
                module_name, module_extension = os.path.splitext(module_file)
                module_hdl = False
                if module_extension == os.extsep + "py":
                    try:
                        logger.debug("Attempt {}".format(module_name))
                        module_hdl, path_name, description = imp.find_module(module_name)
                        plugin_module = imp.load_module(module_name, module_hdl, path_name,
                                                        description)
                        plugin_classes = inspect.getmembers(plugin_module, inspect.isclass)
                        for plugin_class in plugin_classes:
                            if issubclass(plugin_class[1], AbstractTransport):
                                # Load only those plugins defined in the current module
                                # (i.e. don't instantiate any parent plugins)
                                if plugin_class[1].__module__ == module_name:
                                    # plugin = plugin_class[1]()
                                    pname, pclass = plugin_class
                                    plugins[pclass.name] = (pname, pclass)
                    except ImportError as e:
                        logger.debug(e)
                        pass
                    except Exception as e:
                        logger.debug(e)
                        raise
                    finally:
                        if module_hdl:
                            module_hdl.close()
    return plugins


def get_plugins_dict():
    global plugins
    return plugins


def find_plugin(ptype):
    """Return plugin for given transport type."""
    global plugins
    if ptype in plugins:
        pname, pclass = plugins[ptype]
        return pclass()
    raise TransportPluginNotFound("Plugin for transport {} not found.".format(ptype))


def Transport(
        scheme,
        netloc=None,
        root=None,
        mode='r',
        read_directory_only=False,
        opts=None):
    """Return plugin for given transport scheme."""
    if opts is None:
        opts = {}
    if netloc is None and root is None:
        url_tuple = urllib.parse.urlsplit(scheme)
        scheme = url_tuple.scheme
        netloc = url_tuple.hostname
        # netloc = url_tuple.netloc
        try:
            opts['username'] = url_tuple.username
        except AttributeError:
            opts['username'] = None
        try:
            opts['password'] = url_tuple.password
        except AttributeError:
            opts['password'] = None
        root = url_tuple.path
    global plugins
    for ptype in plugins.keys():
        pname, pclass = plugins[ptype]
        if scheme in pclass.schemes:
            return pclass(
                root=root,
                netloc=netloc,
                mode=mode,
                read_directory_only=read_directory_only,
                opts=opts)
    raise TransportPluginNotFound("Plugin for transport scheme {} not found.".format(scheme))


plugins = load_plugins()

if __name__ == "__main__":
    import doctest

    doctest.testmod()
