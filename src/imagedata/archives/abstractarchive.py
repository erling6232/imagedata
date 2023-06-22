"""Abstract class for archives.

Defines generic functions.
"""

# Copyright (c) 2018-2022 Erling Andersen, Haukeland University Hospital, Bergen, Norway

from abc import ABCMeta, abstractmethod


class NoOtherInstance(Exception):
    pass


class WriteMultipleArchives(Exception):
    pass


class Member(object):
    """Class definition for filehandle in imagedata archives.
    """

    def __init__(self, filename,
                 info=None,
                 fh=None,
                 local_file=None):
        """Initialize the filehandle object."""
        self.filename = filename
        if info is None:
            self.info = {}
        else:
            self.info = info
        self.fh = fh
        self.local_file = local_file


class AbstractArchive(object, metaclass=ABCMeta):
    """Abstract base class definition for imagedata archive plugins.
    Plugins must be a subclass of AbstractPlugin and
    must define the attributes set in __init__() and
    the following methods:

    __init__() method
    use_query() method
    getnames() method
    basename() method
    open() method
    getmembers() method
    to_localfile() method
    add_localfile() method
    writedata() method
    is_file() method
    """

    plugin_type = 'archive'

    def __init__(self, name, description, authors, version, url, _mimetypes):
        """Initialize the archive object.
        """
        object.__init__(self)
        self.__name = name
        self.__description = description
        self.__authors = authors
        self.__version = version
        self.__url = url
        self.__mimetypes = _mimetypes
        self.__transport = None

    @abstractmethod
    def use_query(self):
        """Does the plugin need the ?query part of the url?"""
        pass

    @property
    def name(self):
        """Plugin name

        Single word string describing the image format.
        Typical names: dicom, nifti, itk.
        """
        return self.__name

    @property
    def description(self):
        """Plugin description

        Single line string describing the image format.
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
    def transport(self):
        """Underlying transport plugin
        """
        return self.__transport

    @property
    def mimetypes(self):
        """MIME types supported by this plugin.

        List of strings.
        """
        return self.__mimetypes

    @abstractmethod
    def getnames(self, files=None):
        """Get name list of the members.

        Args:
            files: List or single str of filename matches.
        Returns:
            The members as a list of their names.
                It has the same order as the members of the archive.
        """
        pass

    @abstractmethod
    def basename(self, filehandle):
        """Basename of file.

        Examples:
            if archive.basename(filehandle) == "DICOMDIR":

        Args:
            filehandle: reference to member object
        Returns:
            Basename of file: str
        """
        pass

    @abstractmethod
    def open(self, member, mode='rb'):
        """Open file.

        Args:
            member: Handle to file.
            mode: Open mode.
        Returns:
             An IO object for the member.
        """
        pass

    @abstractmethod
    def getmembers(self, files=None):
        """Get the members of the archive.

        Args:
            files: List of filename matches.
        Returns:
            The members of the archive as a list of Filehandles.
                The list has the same order as the members of the archive.
        """
        pass

    @abstractmethod
    def to_localfile(self, member):
        """Access a member object through a local file.

        Args:
            member: handle to member file.
        Returns:
            filename to file guaranteed to be local.
        """
        pass

    @abstractmethod
    def add_localfile(self, local_file, filename):
        """Add a local file to the archive.

        Args:
            local_file: named local file
            filename: filename in the archive
        """
        pass

    @abstractmethod
    def writedata(self, filename, data):
        """Write data to a named file in the archive.

        Args:
            filename: named file in the archive
            data: data to write
        """
        pass

    @abstractmethod
    def close(self):
        """Close archive.
        """
        pass

    @abstractmethod
    def is_file(self, member):
        """Determine whether the named file is a single file.

        Args:
            member: file member.
        Returns:
            whether member is a single file (bool)
        """
        pass

    @abstractmethod
    def __enter__(self):
        """Enter context manager.
        """
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Leave context manager, cleaning up any open files.
        """
        pass


# class ArchiveCollection(AbstractArchive):
#     """A collection of one or more archives, providing the same interface as
#     a single archive.
#     """
#
# import os.path
# import collections
# import logging
# import mimetypes
# import urllib.parse
# import imagedata.archives
# import imagedata.transports
#
#     name = "archivecollection"
#     description = "A collection of one or more archives"
#     authors = "Erling Andersen"
#     version = "1.0.0"
#     url = "www.helse-bergen.no"
#     mimetypes = None
#
#     def __init__(self, transport=None, url=None, mode="r", read_directory_only=True,
#                  opts=None):
#         super(ArchiveCollection, self).__init__(self.name, self.description,
#                                                 self.authors, self.version, self.url,
#                                                 self.mimetypes)
#
#         if opts is None:
#             opts = {}
#         # Handle both single url, a url tuple, and a url list
#         if isinstance(urls, list):
#             self.__urls = urls
#         elif isinstance(urls, tuple):
#             self.__urls = list(urls)
#         else:
#             self.__urls = [urls]
#         if len(self.__urls) < 1:
#             raise ValueError("No URL(s) where given")
#
#         self.__archives = []
#         # Collect list of archives
#         for url in self.__urls:
#             logging.debug("ArchiveCollection: url: '{}'".format(url))
#             urldict = urllib.parse.urlsplit(url, scheme="file")
#             # dirname = os.path.dirname(urldict.path)
#             basename = os.path.basename(urldict.path)
#             logging.debug("ArchiveCollection: transport root: '{}'".format(os.curdir))
#             # transport = imagedata.transports.Transport(
#             #     urldict.scheme, root=os.curdir)
#             logging.debug("ArchiveCollection: archive url: '{}'".format(url))
#             archive = imagedata.archives.find_mimetype_plugin(
#                 mimetypes.guess_type(basename)[0],
#                 # transport,
#                 url,
#                 mode=mode)
#             self.__archives.append(archive)
#
#         # Construct the member dict (archive,filehandle) for all archives
#         self.__memberlist = collections.OrderedDict()
#         for archive in self.__archives:
#             logging.debug("ArchiveCollection: construct archive {}".format(archive))
#             filedict = archive.getnames()
#             logging.debug("ArchiveCollection: construct filelist {} {}".format(
#                 type(filedict), filedict))
#             for key in filedict.keys():
#                 self.__memberlist[key] = (archive, filedict[key])
#
#     def getnames(self, files=None):
#         """Return the members as a list of their names.
#         It has the same order as the members of the archive.
#
#         Chain the references (archive,filehandle) from each archive
#         """
#         return self.__memberlist
#
#     def basename(self, filehandle):
#         """Basename of file.
#
#         Typical use:
#             if archive.basename(filehandle) == "DICOMDIR":
#
#         Input:
#         - filehandle: reference to member object
#         """
#         archive, name = filehandle
#         return archive.basename(name)
#
#     def getmember(self, filehandle):
#         """Return the members of the archive as an OrderedDict of member objects.
#         The keys are the member names as given by getnames().
#         """
#         archive, fh = filehandle
#         return archive.getmember(fh)
#
#     def getmembers(self, files=None):
#         """Return the members of the archive as a list of member objects.
#         The list has the same order as the members in the archive.
#         """
#         return self.__memberlist
#
#     def to_localfile(self, filehandle):
#         """Access a member object through a local file.
#         """
#         archive, name = filehandle
#         return archive.to_localfile(name)
#
#     def writedata(self, filename, data):
#         """Write data to a named file in the archive.
#         Input:
#         - filename: named file in the archive
#         - data: data to write
#         """
#         if len(self.__archives) > 1:
#             raise WriteMultipleArchives("Cannot write data to multiple archives")
#         self.__archives[0].writedata(filename, data)
#
#     def close(self):
#         """Close archive.
#         """
#         for archive in self.__archives:
#             archive.close()
#
#     def __enter__(self):
#         """Enter context manager.
#         """
#         return self
#
#     def __exit__(self):
#         """Leave context manager, cleaning up any open files.
#         """
#         self.close()
#
#     @abstractmethod
#     def use_query(self):
#         pass
#
#     @abstractmethod
#     def open(self, filehandle, mode='rb'):
#         pass
#
#     @abstractmethod
#     def add_localfile(self, local_file, filename):
#         pass
#
#     @abstractmethod
#     def is_file(self):
#         pass
