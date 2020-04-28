#!/usr/bin/env python3

"""Read/write files in xnat database
"""

# Copyright (c) 2018 Erling Andersen, Haukeland University Hospital, Bergen, Norway

import logging
import xnat
from imagedata.transports.abstracttransport import AbstractTransport


class XnatTransport(AbstractTransport):
    """Read/write files in xnat database.
    """

    name = "xnat"
    description = "Read and write files in xnat database."
    authors = "Erling Andersen"
    version = "1.0.0"
    url = "www.helse-bergen.no"
    schemes = ["xnat"]

    def __init__(self, netloc=None, root=None, mode='r', read_directory_only=False, opts=None):
        super(XnatTransport, self).__init__(self.name, self.description,
                                            self.authors, self.version, self.url, self.schemes)
        if opts is None:
            opts = {}
        self.read_directory_only = read_directory_only
        self.netloc = netloc
        self.opts = opts
        logging.debug("XnatTransport __init__ root: {}".format(root))
        self.__root = root
        self.__mode = mode

        self.__session = xnat.connect(self.__root)
        logging.debug("XnatTransport __init__ session: {}".format(self.__session))

    def walk(self, top):
        """Generate the file names in a directory tree by walking the tree.
        Input:
        - top: starting point for walk (str)
        Return:
        - tuples of (root, dirs, files) 
        """
        pass

    def isfile(self, path):
        """Return True if path is an existing regular file.
        """
        pass

    def open(self, path, mode='r'):
        """Extract a member from the archive as a file-like object.
        """
        pass
