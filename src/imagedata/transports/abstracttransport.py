"""Abstract class for image transports.

Defines generic functions.
"""

# Copyright (c) 2018-2022 Erling Andersen, Haukeland University Hospital, Bergen, Norway

from abc import ABCMeta, abstractmethod  # , abstractproperty
from typing import List, Optional
import io


class NoOtherInstance(Exception):
    pass


class AbstractTransport(object, metaclass=ABCMeta):
    """Abstract base class definition for imagedata transport plugins.
    Plugins must be a subclass of AbstractPlugin and
    must define the attributes set in __init__() and
    the following methods:

    open() method
    isfile() method
    walk() method
    """

    plugin_type = 'transport'
    mimetype = "*"  # Determines archive plugin

    def __init__(self, name: str, description: str, authors: str, version: str, url: str,
                 schemes: Optional[List[str]] = None):
        object.__init__(self)
        self.__name: str = name
        self.__description: str = description
        self.__authors: str = authors
        self.__version: str = version
        self.__url: str = url
        self.__schemes: List[str] = schemes

    @property
    def name(self) -> str:
        """Plugin name

        Single word string describing the image format.
        Typical names: file, dicom, xnat
        """
        return self.__name

    @property
    def description(self) -> str:
        """Plugin description

        Single line string describing the transport method.
        """
        return self.__description

    @property
    def authors(self) -> str:
        """Plugin authors

        Multi-line string naming the author(s) of the plugin.
        """
        return self.__authors

    @property
    def version(self) -> str:
        """Plugin version

        String giving the plugin version.
        Version scheme: 1.0.0
        """
        return self.__version

    @property
    def url(self) -> str:
        """Plugin URL

        URL string to the site of the plugin or the author(s).
        """
        return self.__url

    @property
    def schemes(self) -> List[str]:
        """List of transport schemes supported by this plugin.

        List of strings.
        """
        return self.__schemes

    @abstractmethod
    def walk(self, top: str):
        """Generate the file names in a directory tree by walking the tree.
        Input:
        - top: starting point for walk (str)
        Return:
        - tuples of (root, dirs, files)
        """
        pass

    @abstractmethod
    def isfile(self, path: str) -> bool:
        """Return True if path is an existing regular file.
        """
        pass

    @abstractmethod
    def exists(self, path: str) -> bool:
        """Determine whether the named path exists.
        """
        pass

    @abstractmethod
    def open(self, path: str, mode: str = 'r') -> io.IOBase:
        """Extract a member from the archive as a file-like object.
        """
        pass

    @abstractmethod
    def close(self):
        """Close the transport
        """
        pass

    @abstractmethod
    def info(self, path) -> str:
        """Return info describing the object

        Args:
            path (str): object path

        Returns:
            description (str): Preferably a one-line string describing the object
        """
        pass
