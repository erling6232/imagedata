"""Read/Write local files
"""

# Copyright (c) 2018-2024 Erling Andersen, Haukeland University Hospital, Bergen, Norway

from typing import List, Optional
import os
import os.path
import io
import logging
from .abstracttransport import AbstractTransport

logger = logging.getLogger(__name__)


class FileTransport(AbstractTransport):
    """Read/write local files.

    Args:
        netloc (str): Not used.
        root (str): Root path.
        mode (str): Filesystem access mode.
        read_directory_only (bool): Whether root should refer to a directory.
        opts (dict): Options

    Returns:
        FileTransport instance

    Raises:
        RootIsNotDirectory: when the root is not a directory when read_directory_only is True.
        FileNotFoundError: Specified root does not exist.
        AssertionError: When root is None.
    """

    name: str = "file"
    description: str = "Read and write local files."
    authors: str = "Erling Andersen"
    version: str = "1.1.0"
    url: str = "www.helse-bergen.no"
    schemes: List[str] = ["file"]
    __root: str = None
    __mode: str = None
    __basename: str = None
    netloc: str = None
    opts: dict = None
    path: str = None

    def __init__(self,
                 netloc: Optional[str] = None,
                 root: Optional[str] = None,
                 mode: Optional[str] = 'r',
                 read_directory_only: Optional[bool] = False,
                 opts: Optional[dict] = None):
        _name: str = '{}.{}'.format(__name__, self.__init__.__name__)
        super(FileTransport, self).__init__(self.name, self.description,
                                            self.authors, self.version, self.url, self.schemes)
        self.netloc = netloc
        self.opts = opts
        self.path = root
        logger.debug("{}: root: {} ({})".format(_name, root, mode))
        assert root is not None, "Root should not be None"
        # if mode[0] == 'r' and read_directory_only and not os.path.isdir(root):
        #     logger.debug("FileTransport __init__ RootIsNotDirectory")
        #     raise RootIsNotDirectory("Root ({}) should be a directory".format(root))
        if mode[0] == 'r' and not os.path.exists(root):
            logger.debug("{}: FileNotFoundError".format(_name))
            raise FileNotFoundError("Root ({}) does not exist".format(root))
        if mode[0] == 'w' and not os.path.exists(root) and \
                os.path.exists(os.path.dirname(root)):
            self.__basename = os.path.basename(root)
            root = os.path.dirname(root)
        self.__root = root
        self.__mode = mode

    def close(self):
        """Close the transport
        """
        return

    def _get_path(self, path):
        """Get absolute path of object.
        If path is relative path, prepend self.__root.
        If path is absolute path, return path only.

        Args:
            path: Absolute or relative path to object.

        Returns:
            Absolute path of object.
        """
        if os.path.isabs(path):
            return path
        return os.path.join(self.__root, path)

    def walk(self, top):
        """Generate the file names in a directory tree by walking the tree.

        Args:
            top: starting point for walk (str)

        Returns:
            tuples of (root, dirs, files)
        """
        # for root, dirs, files in os.walk(self._get_path(top)):
        #     local_root = root
        #     if local_root.startswith(self.__root):
        #         local_root = root[len(self.__root) + 1:]  # Strip off root
        #     yield local_root, dirs, files
        for root, dirs, files in os.walk(top):
            yield root, dirs, files

    def isfile(self, path):
        """Check whether path refers to an existing regular file.

        Args:
            path: Path to file.

        Returns:
            Existense of regular file (bool)
        """
        return os.path.isfile(path)

    def exists(self, path):
        """Determine whether the named path exists.
        """
        return os.path.exists(path)

    def open(self, path: str, mode: str = 'r') -> io.IOBase:
        """Extract a member from the archive as a file-like object.

        Args:
            path (str): Path to file.
            mode (str): Open mode, can be 'r', 'w', 'x' or 'a'
        """
        _name: str = '{}.{}'.format(__name__, self.open.__name__)
        logger.debug("{}: {} ({})".format(_name, path, mode))
        if mode[0] in ['w', 'x', 'a']:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        return io.FileIO(path, mode)

    def info(self, path) -> str:
        """Return info describing the object

        Args:
            path (str): object path

        Returns:
            description (str): Preferably a one-line string describing the object
        """
        return ''
