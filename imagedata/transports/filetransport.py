"""Read/Write local image files
"""

# Copyright (c) 2018 Erling Andersen, Haukeland University Hospital, Bergen, Norway

import os, os.path, io
import sys
import logging
from imagedata.transports.abstracttransport import AbstractTransport

class RootIsNotDirectory(Exception): pass

class FileTransport(AbstractTransport):
    """Read/write local files."""

    name = "file"
    description = "Read and write local image files."
    authors = "Erling Andersen"
    version = "1.0.0"
    url = "www.helse-bergen.no"
    schemes = ["file"]

    def __init__(self, root):
        super(FileTransport, self).__init__(self.name, self.description,
            self.authors, self.version, self.url, self.schemes)
        logging.debug("FileTransport __init__ root: {}".format(root))
        #if not os.path.isdir(root):
        #    raise RootIsNotDirectory("Root ({}) should be a directory".format(root))
        self.__root = root

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
            walk_list.append((local_root, dirs, files))
        return walk_list

    def isfile(self, path):
        """Return True if path is an existing regular file.
        """
        return os.path.isfile(os.path.join(self.__root, path))

    def open(self, path, mode='r'):
        """Extract a member from the archive as a file-like object.
        """
        fname = os.path.join(self.__root, path)
        logging.debug("open: mode {} path {}".format(mode, fname))
        if mode[0] == 'w':
            os.makedirs(os.path.dirname(fname), exist_ok=True)
        return io.FileIO(fname, mode)
