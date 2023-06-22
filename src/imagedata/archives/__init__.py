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


def find_plugin(pfind, url, mode="r", opts=None):
    """Return plugin for given image archive type."""
    if opts is None:
        opts = {}
    from imagedata import plugins
    if 'archive' in plugins:
        for pname, ptype, pclass in plugins['archive']:
            if ptype == pfind:
                return pclass(url=url, mode=mode, opts=opts)
    raise ArchivePluginNotFound("Plugin for image archive {} not found.".format(ptype))


def find_mimetype_plugin(mimetype, url, mode="r", opts=None):
    """Return plugin for given file type."""
    if opts is None:
        opts = {}
    from imagedata import plugins
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
    for pname, ptype, pclass in plugins['archive']:
        logger.debug("imagedata.archive.find_mimetype_plugin: compare '{}' to {}".format(
            mimetype, pclass.mimetypes))
        if mimetype in pclass.mimetypes:
            logger.debug("imagedata.archives.find_mimetype_plugin: {}, mode: {}".format(
                ptype, mode))
            return pclass(url=url, mode=mode, opts=opts)
    if os.path.isfile(_path):
        logger.debug("imagedata.archives.find_mimetype_plugin: filesystem")
        try:
            return find_plugin('filesystem', url, mode, opts=opts)
        except ArchivePluginNotFound:
            # Fall-through to fail with ArchivePluginNotFound
            pass
    raise ArchivePluginNotFound("Plugin for MIME type {} not found.".format(mimetype))
