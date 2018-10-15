#!/usr/bin/env python3

"""This module provides plugins for various image archive formats.

Standard plugins provides support for filesystem, tar, tar.gz, tar.bz2, gzip, zip.
"""

# Copyright (c) 2018 Erling Andersen, Haukeland University Hospital, Bergen, Norway

import os.path
import logging
import urllib.parse

logging.getLogger(__name__).addHandler(logging.NullHandler())

#class NotImageError(Exception): pass
class ArchivePluginNotFound(Exception): pass

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
    from imagedata.archives.abstractarchive import AbstractArchive

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

    plugins = {}
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
                        #print("Attemp2 {}".format(plugin_module))
                        plugin_classes = inspect.getmembers(plugin_module, inspect.isclass)
                        #print("Attemp3 {} {}".format(module_name, plugin_classes))
                        for plugin_class in plugin_classes:
                            #print("plugin_class {} ({})".format(plugin_class,type(plugin_class[1])))
                            if issubclass(plugin_class[1], AbstractArchive):
                                # Load only those plugins defined in the current module
                                # (i.e. don't instantiate any parent plugins)
                                #print("compare plugin_class[1].__module__ ({}) to module_name ({})".format(plugin_class[1].__module__, module_name))
                                if plugin_class[1].__module__ == module_name:
                                    #plugin = plugin_class[1]()
                                    pname,pclass = plugin_class
                                    plugins[pclass.name] = (pname,pclass)
                    except ImportError as e:
                        logging.debug(e)
                        #print(e)
                        pass
                    except Exception as e:
                        logging.debug(e)
                        #print(e)
                        raise
                    finally:
                        if module_hdl:
                            module_hdl.close()
    return plugins

def get_plugins_dict():
    global plugins
    return plugins

def find_plugin(ptype, transport, url, mode="r"):
    """Return plugin for given image archive type."""
    global plugins
    if ptype in plugins:
        pname, pclass = plugins[ptype]
        return pclass(transport=transport, url=url, mode=mode)
    raise ArchivePluginNotFound("Plugin for image archive {} not found.".format(ptype))

def find_mimetype_plugin(mimetype, transport, url, mode="r"):
    """Return plugin for given file type."""
    global plugins
    urldict = urllib.parse.urlsplit(url, scheme="file")
    if mimetype is None:
        logging.debug("imagedata.archives.find_mimetype_plugin: filesystem")
        return find_plugin('filesystem', transport, url, mode)
    logging.debug("imagedata.archive.find_mimetype_plugins: {}".format(plugins.keys()))
    for ptype in plugins.keys():
        pname, pclass = plugins[ptype]
        logging.debug("imagedata.archive.find_mimetype_plugin: compare '{}' to {}".format(mimetype, pclass.mimetypes))
        if mimetype in pclass.mimetypes:
            logging.debug("imagedata.archives.find_mimetype_plugin: {}".format(ptype))
            return pclass(transport=transport, url=url, mode=mode)
    if os.path.isfile(urldict.path):
        logging.debug("imagedata.archives.find_mimetype_plugin: filesystem")
        return find_plugin('filesystem', transport, url, mode)
    raise ArchivePluginNotFound("Plugin for MIME type {} not found.".format(mimetype))

plugins = load_plugins()

if __name__ == "__main__":
    import doctest
    doctest.testmod()
