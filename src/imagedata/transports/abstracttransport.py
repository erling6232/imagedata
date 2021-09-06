"""Abstract class for image transports.

Defines generic functions.
"""

# Copyright (c) 2018 Erling Andersen, Haukeland University Hospital, Bergen, Norway

from abc import ABCMeta, abstractmethod  # , abstractproperty
# import imagedata.transports


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

    def __init__(self, name, description, authors, version, url, schemes):
        object.__init__(self)
        self.__name = name
        self.__description = description
        self.__authors = authors
        self.__version = version
        self.__url = url
        self.__schemes = schemes

    @property
    def name(self):
        """Plugin name
        
        Single word string describing the image format.
        Typical names: file, dicom, xnat
        """
        return self.__name

    @property
    def description(self):
        """Plugin description
        
        Single line string describing the transport method.
        """
        return self.__description

    @property
    def authors(self):
        """Plugin authors
        
        Multi-line string naming the author(s) of the plugin.
        """
        return self.__authors

    @property
    def version(self):
        """Plugin version
        
        String giving the plugin version.
        Version scheme: 1.0.0
        """
        return self.__version

    @property
    def url(self):
        """Plugin URL
        
        URL string to the site of the plugin or the author(s).
        """
        return self.__url

    @property
    def schemes(self):
        """List of transport schemes supported by this plugin.
        
        List of strings.
        """
        return self.__schemes

    @abstractmethod
    def walk(self, top):
        """Generate the file names in a directory tree by walking the tree.
        Input:
        - top: starting point for walk (str)
        Return:
        - tuples of (root, dirs, files) 
        """
        pass

    @abstractmethod
    def isfile(self, path):
        """Return True if path is an existing regular file.
        """
        pass

    @abstractmethod
    def open(self, path, mode='r'):
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