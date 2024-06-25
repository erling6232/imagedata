"""This module provides plugins for various image archive formats.

Standard plugins provide support for local filesystem and zip archives.
"""

# Copyright (c) 2018-2024 Erling Andersen, Haukeland University Hospital, Bergen, Norway

import os.path
import logging
import urllib.parse
import fnmatch
from ..transports import Transport

logger = logging.getLogger(__name__)


class ArchivePluginNotFound(Exception):
    pass


class FileAlreadyExistsError(Exception):
    pass


def find_plugin(pfind, url, mode="r", opts=None):
    """Return plugin for given image archive type."""
    if opts is None:
        opts = {}
    from .. import plugins
    if 'archive' in plugins:
        for pname, ptype, pclass in plugins['archive']:
            if ptype == pfind:
                return pclass(url=url, mode=mode, opts=opts)
    raise ArchivePluginNotFound("Plugin for image archive {} not found.".format(ptype))


def find_mimetype_plugin(mimetype, url, mode="r", read_directory_only=False, opts=None):
    """Return plugin for given file type."""
    _name: str = '{}.{}'.format(__name__, find_mimetype_plugin.__name__)
    if opts is None:
        opts = {}
    from .. import plugins
    if os.name == 'nt' and \
            fnmatch.fnmatch(url, '[A-Za-z]:\\*'):
        # Windows: Parse without /x:, then re-attach drive letter
        urldict = urllib.parse.urlsplit(url[2:], scheme="file")
        # _path = url[:2] + urldict.path
    else:
        urldict = urllib.parse.urlsplit(url, scheme="file")
        # _path = urldict.path if len(urldict.path) > 0 else urldict.netloc
    # if urldict.scheme == 'xnat':
    #     mimetype = 'application/zip'
    # if mimetype is None:
    #     logger.debug("imagedata.archives.find_mimetype_plugin: filesystem")
    #     return find_plugin('filesystem', url, mode, opts=opts)
    transport = None
    if urldict.scheme:
        transport = Transport(
            urldict.scheme,
            netloc=urldict.netloc,
            root=urldict.path,
            mode=mode,
            read_directory_only=read_directory_only)
    if mimetype is None:
        # Get any transport requirement for mimetype
        try:
            mimetype = transport.mimetype
        except AttributeError:
            pass
    for pname, ptype, pclass in plugins['archive']:
        logger.debug("{}: compare '{}' to {}".format(
            _name, mimetype, pclass.mimetypes))
        if mimetype in pclass.mimetypes:
            logger.debug("{}: {}, mode: {}".format(
                _name, ptype, mode))
            return pclass(url=url, transport=transport, mode=mode, opts=opts)
    # if os.path.isfile(_path):
    # if os.path.exists(_path):
    if urldict.scheme == "file":
        logger.debug("{}: filesystem".format(_name))
        try:
            return find_plugin('filesystem', url, mode, opts=opts)
        except ArchivePluginNotFound:
            # Fall-through to fail with ArchivePluginNotFound
            pass
    raise ArchivePluginNotFound("Plugin for MIME type {} not found.".format(mimetype))


def get_archiver_list():
    from .. import plugins
    return plugins['archive'] if 'archive' in plugins else []
