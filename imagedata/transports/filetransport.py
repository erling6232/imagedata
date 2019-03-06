"""Read/Write local image files
"""

# Copyright (c) 2018 Erling Andersen, Haukeland University Hospital, Bergen, Norway

import os, os.path, io
import sys
import logging
from imagedata.transports.abstracttransport import AbstractTransport
from imagedata.transports import RootIsNotDirectory, RootDoesNotExist

class FileTransport(AbstractTransport):
    """Read/write local files."""

    name = "file"
    description = "Read and write local image files."
    authors = "Erling Andersen"
    version = "1.0.0"
    url = "www.helse-bergen.no"
    schemes = ["file"]

    def __init__(self, root, mode='r', read_directory_only=False):
        super(FileTransport, self).__init__(self.name, self.description,
            self.authors, self.version, self.url, self.schemes)
        logging.debug("FileTransport __init__ root: {}".format(root))
        if mode[0] == 'r' and read_directory_only and not os.path.isdir(root):
            raise RootIsNotDirectory("Root ({}) should be a directory".format(root))
        if mode[0] == 'r' and not os.path.exists(root):
            raise RootDoesNotExist("Root ({}) does not exist".format(
                root))
        self.__root = root
        self.__mode = mode

    def _get_path(self, path):
        """Return either relative or absolute path.

        If path is relative path, prepend self.__root
        If path is absolute path, return path only
        """
        if os.path.isabs(path):
            return path
        return os.path.join(self.__root, path)

    def walk(self, top):
        """Generate the file names in a directory tree by walking the tree.
        Input:
        - top: starting point for walk (str)
        Return:
        - tuples of (root, dirs, files) 
        """
        walk_list = []
        for root, dirs, files in os.walk(self._get_path(top)):
            if root.startswith(self.__root):
                local_root = root[len(self.__root)+1:]     # Strip off root
            logging.debug('FileTransport.walk: dirs %s, files %s' %
                    (dirs, files))
            walk_list.append((local_root, dirs, files))
        return walk_list

    def isfile(self, path):
        """Return True if path is an existing regular file.
        """
        return os.path.isfile(os.path.join(self.__root, path))

    def open(self, path, mode='r'):
        """Extract a member from the archive as a file-like object.
        """
        logging.debug("FileTransport open: mode {} path {}".format(mode, path))
        fname = os.path.join(self.__root, path)
        if mode[0] == 'w':
            os.makedirs(os.path.dirname(fname), exist_ok=True)
        logging.debug("FileTransport open {}, mode: {}".format(
            fname, mode))
        return io.FileIO(fname, mode)
