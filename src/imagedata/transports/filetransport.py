"""Read/Write local image files
"""

# Copyright (c) 2018-2021 Erling Andersen, Haukeland University Hospital, Bergen, Norway

import os
import os.path
import io
import logging
from imagedata.transports.abstracttransport import AbstractTransport
from imagedata.transports import RootIsNotDirectory, RootDoesNotExist

logger = logging.getLogger(__name__)


class FileTransport(AbstractTransport):
    """Read/write local files."""

    name = "file"
    description = "Read and write local image files."
    authors = "Erling Andersen"
    version = "1.1.0"
    url = "www.helse-bergen.no"
    schemes = ["file"]

    def __init__(self, netloc=None, root=None, mode='r', read_directory_only=False, opts=None):
        super(FileTransport, self).__init__(self.name, self.description,
                                            self.authors, self.version, self.url, self.schemes)
        self.netloc = netloc
        self.opts = opts
        logger.debug("FileTransport __init__ root: {} ({})".format(root, mode))
        assert root is not None, "Root should not be None"
        if mode[0] == 'r' and read_directory_only and not os.path.isdir(root):
            logger.debug("FileTransport __init__ RootIsNotDirectory")
            raise RootIsNotDirectory("Root ({}) should be a directory".format(root))
        if mode[0] == 'r' and not os.path.exists(root):
            logger.debug("FileTransport __init__ RootDoesNotExist")
            raise RootDoesNotExist("Root ({}) does not exist".format(root))
        self.__root = root
        self.__mode = mode

    def close(self):
        """Close the transport
        """
        return

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
        for root, dirs, files in os.walk(self._get_path(top)):
            local_root = root
            if local_root.startswith(self.__root):
                local_root = root[len(self.__root) + 1:]  # Strip off root
            yield local_root, dirs, files

    def isfile(self, path):
        """Return True if path is an existing regular file.
        """
        return os.path.isfile(os.path.join(self.__root, path))

    def open(self, path, mode='r'):
        """Extract a member from the archive as a file-like object.
        """
        logger.debug("FileTransport open: {} ({})".format(path, mode))
        filename = os.path.join(self.__root, path)
        if mode[0] == 'w':
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        logger.debug("FileTransport open: {} ({})".format(filename, mode))
        return io.FileIO(filename, mode)

    def info(self, path) -> str:
        """Return info describing the object

        Args:
            path (str): object path

        Returns:
            description (str): Preferably a one-line string describing the object
        """
        return ''
