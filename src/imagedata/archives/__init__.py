#!/usr/bin/env python3

"""This module provides plugins for various image archive formats.

Standard plugins provides support for filesystem, tar, tar.gz, tar.bz2, gzip, zip.
"""

# Copyright (c) 2018 Erling Andersen, Haukeland University Hospital, Bergen, Norway

import os.path
import logging
import urllib.parse
import fnmatch

logger = logging.getLogger(__name__)


# class NotImageError(Exception):
#     pass


class ArchivePluginNotFound(Exception):
    pass


class FileAlreadyExistsError(Exception):
    pass


def add_plugin_dir(directory_name):
    from pkgutil import extend_path
    global __path__, plugins
    __path__ = extend_path(__path__, directory_name)
    plugins = load_plugins()


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
    from imagedata.archives.abstractarchive import AbstractArchive

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
                        # print("Attempt {}".format(module_name))
                        logger.debug("Attempt {}".format(module_name))
                        module_hdl, path_name, description = imp.find_module(module_name)
                        plugin_module = imp.load_module(module_name, module_hdl, path_name,
                                                        description)
                        # print("Attemp2 {}".format(plugin_module))
                        plugin_classes = inspect.getmembers(plugin_module, inspect.isclass)
                        # print("Attemp3 {} {}".format(module_name, plugin_classes))
                        for plugin_class in plugin_classes:
                            # print("plugin_class {} ({})".format(plugin_class,type(plugin_class[1])))
                            if issubclass(plugin_class[1], AbstractArchive):
                                # Load only those plugins defined in the current module
                                # (i.e. don't instantiate any parent plugins)
                                # print("compare plugin_class[1].__module__ ({}) to module_name ({})".format(plugin_class[1].__module__, module_name))
                                if plugin_class[1].__module__ == module_name:
                                    # plugin = plugin_class[1]()
                                    pname, pclass = plugin_class
                                    plugins[pclass.name] = (pname, pclass)
                    except ImportError as e:
                        logger.debug(e)
                        # print(e)
                        pass
                    except Exception as e:
                        logger.debug(e)
                        # print(e)
                        raise
                    finally:
                        if module_hdl:
                            module_hdl.close()
    return plugins


def get_plugins_dict():
    global plugins
    return plugins


def find_plugin(ptype, url, mode="r", opts=None):
    """Return plugin for given image archive type."""
    if opts is None:
        opts = {}
    global plugins
    if ptype in plugins:
        pname, pclass = plugins[ptype]
        return pclass(url=url, mode=mode, opts=opts)
    raise ArchivePluginNotFound("Plugin for image archive {} not found.".format(ptype))


def lookup_mimetype_plugin(mimetype):
    """Return name of plugin that will handle given _mimetypes."""

    if mimetype is None:
        logger.debug("imagedata.archives.lookup_mimetype_plugin: filesystem")
        return 'filesystem'
    for ptype in plugins.keys():
        pname, pclass = plugins[ptype]
        if mimetype in pclass.mimetypes:
            return pname


def find_mimetype_plugin(mimetype, url, mode="r", opts=None):
    """Return plugin for given file type."""
    if opts is None:
        opts = {}
    global plugins
    if os.name == 'nt' and \
            fnmatch.fnmatch(url, '[A-Za-z]:\\*'):
        # Windows: Parse without /x:, then re-attach drive letter
        urldict = urllib.parse.urlsplit(url[2:], scheme="file")
        _path = url[:2] + urldict.path
    else:
        urldict = urllib.parse.urlsplit(url, scheme="file")
        _path = urldict.path if len(urldict.path) > 0 else urldict.netloc
    if urldict.scheme == 'xnat':
        mimetype = 'application/zip'
    if mimetype is None:
        logger.debug("imagedata.archives.find_mimetype_plugin: filesystem")
        return find_plugin('filesystem', url, mode, opts=opts)
    logger.debug("imagedata.archive.find_mimetype_plugins: {}".format(plugins.keys()))
    for ptype in plugins.keys():
        pname, pclass = plugins[ptype]
        logger.debug("imagedata.archive.find_mimetype_plugin: compare '{}' to {}".format(mimetype, pclass.mimetypes))
        if mimetype in pclass.mimetypes:
            logger.debug("imagedata.archives.find_mimetype_plugin: {}, mode: {}".format(ptype, mode))
            return pclass(url=url, mode=mode, opts=opts)
    if os.path.isfile(_path):
        logger.debug("imagedata.archives.find_mimetype_plugin: filesystem")
        try:
            return find_plugin('filesystem', url, mode, opts=opts)
        except ArchivePluginNotFound:
            # Fall-through to fail with ArchivePluginNotFound
            pass
    raise ArchivePluginNotFound("Plugin for MIME type {} not found.".format(mimetype))


plugins = load_plugins()

if __name__ == "__main__":
    import doctest

    doctest.testmod()
