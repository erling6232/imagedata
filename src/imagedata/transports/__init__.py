"""This module provides plugins for various image transports.

Standard plugins provides support for file, http/https and xnat transports.
"""

# Copyright (c) 2013-2022 Erling Andersen, Haukeland University Hospital, Bergen, Norway

import logging
from urllib import parse
# from .abstracttransport import AbstractTransport

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
        url_tuple = parse.urlsplit(scheme)
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
    from imagedata import plugins
    if 'transport' in plugins:
        for pname, ptype, pclass in plugins['transport']:
            if scheme in pclass.schemes:
                return pclass(
                    root=root,
                    netloc=netloc,
                    mode=mode,
                    read_directory_only=read_directory_only,
                    opts=opts)
    raise TransportPluginNotFound("Plugin for transport scheme {} not found.".format(scheme))
