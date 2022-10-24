"""Read/Write local files
"""

# Copyright (c) 2018-2022 Erling Andersen, Haukeland University Hospital, Bergen, Norway

import os
import os.path
import fnmatch
import shutil
import urllib.parse
import logging
from abc import ABC

from .abstractarchive import AbstractArchive, Member
from . import FileAlreadyExistsError
from ..transports import Transport, RootDoesNotExist, RootIsNotDirectory

logger = logging.getLogger(__name__)


class ReadOnlyError(Exception):
    pass


class WriteOnFile(Exception):
    pass


class NoSuchFile(Exception):
    pass


class FilesystemArchive(AbstractArchive, ABC):
    """Read/write local files.

    Args:
        transport: a Transport instance
        url (str): URL to filesystem
        mode (str): filesystem access mode
        read_directory_only (bool): Whether url should refer to a directory.
        opts (dict): Options

    Returns:
        FilesystemArchive instance
    """

    name = "filesystem"
    description = "Read and write local files."
    authors = "Erling Andersen"
    version = "1.1.0"
    url = "www.helse-bergen.no"
    mimetypes = ['*']  # Disregards MIME types

    # self.__dirname: root directory
    # self.__filelist: list of absolute filename

    def __init__(self, transport=None, url=None, mode='r',
                 read_directory_only=True, opts=None):
        super(FilesystemArchive, self).__init__(
            self.name, self.description,
            self.authors, self.version, self.url, self.mimetypes)
        logger.debug("FilesystemArchive.__init__ url: {}".format(url))

        if os.name == 'nt' and \
                fnmatch.fnmatch(url, '[A-Za-z]:\\*'):
            # Windows: Parse without x:, then reattach drive letter
            urldict = urllib.parse.urlsplit(url[2:], scheme="file")
            self.__netloc = ''
            self.__path = url[:2] + urldict.path
        else:
            urldict = urllib.parse.urlsplit(url, scheme="file")
            if os.name == 'nt' and \
                    fnmatch.fnmatch(urldict.netloc, '[A-Za-z]:\\*'):
                self.__netloc = ''
                self.__path = urldict.netloc
            else:
                self.__netloc = urldict.netloc
                self.__path = urldict.path
        if transport is not None:
            self.__transport = transport
        elif url is None:
            raise ValueError('url not given')
        else:
            # Determine transport from url
            logger.debug('FilesystemArchive.__init__: scheme: %s, path: %s' %
                         (urldict.scheme, self.__path))
            self.__transport = Transport(
                urldict.scheme,
                netloc=self.__netloc,
                root=self.__path,
                mode=mode,
                read_directory_only=read_directory_only,
                opts=opts)
        self.__mode = mode

        logger.debug("FilesystemArchive __init__: {}".format(type(transport)))

        logger.debug("FilesystemArchive path: {}".format(self.__path))
        logger.debug("FilesystemArchive open zipfile mode %s" % self.__mode)

        # If the URL refers to a single file, let directory_name refer to the
        # directory and basename to the file
        logger.debug("FilesystemArchive __init__ verify : {}".format(self.__path))
        if os.path.isfile(self.__path):
            self.__dirname = os.path.dirname(self.__path)
            self.__basename = os.path.basename(self.__path)
            logger.debug("FilesystemArchive __init__ directory_name : {}".format(self.__dirname))
            logger.debug("FilesystemArchive __init__ basename: {}".format(self.__basename))
            return

        # The URL refers to a directory. Let directory_name refer to the directory
        self.__dirname = self.__path
        self.__basename = ''
        logger.debug("FilesystemArchive __init__ scan directory_name : {}".format(self.__dirname))
        logger.debug("FilesystemArchive __init__ scan basename: {}".format(self.__basename))

    @staticmethod
    def _get_transport(url, mode, read_directory_only):
        """Get transport plugin from url.

        If the url addresses a missing file in read mode,
        access the parent directory.
        """

        url_tuple = urllib.parse.urlsplit(url, scheme='file')
        logger.debug('FilesystemArchive._get_transport: scheme: %s, netloc: %s' %
                     (url_tuple.scheme, url_tuple.path))

        try:
            _transport = Transport(
                url_tuple.scheme,
                netloc=url_tuple.netloc,
                root=url_tuple.path,
                mode=mode,
                read_directory_only=read_directory_only)
        except RootDoesNotExist:
            # Mode='r': location does not exist
            raise
        except RootIsNotDirectory:
            # Mode='r': Retry with parent directory
            parent, _ = os.path.split(url_tuple.path)
            _transport = Transport(
                url_tuple.scheme,
                netloc=url_tuple.netloc,
                root=parent,
                mode=mode,
                read_directory_only=read_directory_only)
        return _transport

    @property
    def transport(self):
        """Underlying transport plugin
        """
        return self.__transport

    def use_query(self):
        """Does the plugin need the ?query part of the url?"""
        return False

    def _scan_subdirs(self, path: str):
        filelist = list()
        for root, dirs, files in self.__transport.walk(path):
            for filename in files:
                if len(root):
                    filelist.append(os.path.join(root, filename))
                else:
                    filelist.append(filename)
        return sorted(filelist)

    def _search_subdirs(self, path: str, search: str):
        filelist = list()
        for root, dirs, files in self.__transport.walk(path):
            for _file in files:
                if len(root):
                    filename = os.path.join(root, _file)
                else:
                    filename = _file
                if fnmatch.fnmatchcase(filename, os.path.normpath(search)):
                    filelist.append(filename)
                elif fnmatch.fnmatchcase(filename, os.path.normpath(search) + os.sep + '*'):
                    filelist.append(filename)
        return sorted(filelist)

    def getnames(self, files=None):
        """Get name list of the members.

        Args:
            files: List or single str of filename matches.
        Returns:
            The members as a list of their names.
                It has the same order as the members of the archive.
        Raises:
            FileNotFoundError: when no matching file is found.
        """
        if files is not None and issubclass(type(files), str):
            wanted_files = [files]
        else:
            wanted_files = files
        if wanted_files is None or \
                (issubclass(type(wanted_files), list) and (
                        len(wanted_files) == 0 or
                        len(wanted_files) > 0 and wanted_files[0] == '*')):
            return self._scan_subdirs(self.__path)
        else:
            filelist = list()
            found_match = [False for _ in range(len(wanted_files))]
            for i, _file in enumerate(wanted_files):
                if os.path.isfile(_file):
                    add_filelist = [_file]
                else:
                    add_filelist = self._search_subdirs(self.__path, _file)
                if len(add_filelist) > 0:
                    found_match[i] = True
                    filelist += add_filelist
            if len(filelist) < 1:
                raise FileNotFoundError('No such file: {}'.format(wanted_files))
            return filelist

    def basename(self, filehandle):
        """Basename of file.

        Examples:
            if archive.basename(filehandle) == "DICOMDIR":

        Args:
            filehandle: reference to member object

        Returns:
            Basename of file: str
        """
        return os.path.basename(filehandle.filename)

    def open(self, member, mode='rb'):
        """Open file.

        Args:
            member: Handle to file
            mode: Open mode

        Returns:
             An IO object for the member
        """
        # logger.debug("getmember: fname {}".format(filehandle))
        if isinstance(member, str):
            filename = member
        else:
            filename = member.filename
        return self.__transport.open(filename, mode)

    def getmembers(self, files=None):
        """Get the members of the archive.

        Args:
            files: List of filename matches
        Returns:
            The members of the archive as a list of member objects.
                The list has the same order as the members in the archive.

        Raises:
            FileNotFoundError: When no matching file is found.
        """
        # logger.debug("getmembers: files {}".format(files))
        if files is not None and issubclass(type(files), str):
            wanted_files = [files]
        else:
            wanted_files = files
        if wanted_files is None or \
                (issubclass(type(wanted_files), list) and (
                        len(wanted_files) == 0 or
                        len(wanted_files) > 0 and wanted_files[0] == '*')):
            _files = self._scan_subdirs(self.__path)
            filelist = list()
            for _file in _files:
                filelist.append(Member(_file))
        else:
            if issubclass(type(files), list):
                wanted_files = files
            else:
                wanted_files = list((files,))
            filelist = list()
            found_match = [False for _ in range(len(wanted_files))]
            for i, _file in enumerate(wanted_files):
                if os.path.isfile(_file):
                    add_filelist = [_file]
                else:
                    add_filelist = self._search_subdirs(self.__path, _file)
                if len(add_filelist) > 0:
                    found_match[i] = True
                    for item in add_filelist:
                        filelist.append(Member(item))
            # Verify that all wanted files are found
            for i, found in enumerate(found_match):
                if not found:
                    raise FileNotFoundError('No such file: %s' % wanted_files[i])
            if len(filelist) < 1:
                raise FileNotFoundError('No such file: %s' % files)
        return filelist

    def to_localfile(self, member):
        """Access a member object through a local file.

        Args:
            member: handle to member file.

        Returns:
            filename to file guaranteed to be local.
        """
        # logger.debug('FilesystemArchive to_localfile: filename %s' %
        #        filehandle)
        return os.path.join(self.__path, member.filename)

    def add_localfile(self, local_file, filename):
        """Add a local file to the archive.

        Args:
            local_file: named local file
            filename: filename in the archive
        Raises:
            imagedata.archives.FileAlreadyExistsError: When file already exists.
        """
        fname = os.path.join(self.__dirname, filename)
        if not os.path.exists(fname):
            # Ensure the directory exists,
            # create it silently if not.
            os.makedirs(
                os.path.dirname(fname),
                exist_ok=True)
            shutil.copy(local_file, fname)
        else:
            raise FileAlreadyExistsError(
                'File %s already exists' %
                os.path.join(
                    self.__path,
                    filename))

    def writedata(self, filename, data):
        """Write data to a named file in the archive.

        Args:
            filename: named file in the archive
            data: data to write
        Raises:
            ReadOnlyError: when the archive is read-only.
            WriteOnFile: when attempting to write a file to a file.
        """
        if self.__mode[0] == 'r':
            raise ReadOnlyError("Archive is read-only.")
        if len(self.__basename) > 0:
            raise WriteOnFile("Do not know how to write a file to a file.")
        fname = os.path.join(self.__dirname, filename)
        logger.debug("writedata: fname {}".format(fname))
        with self.__transport.open(fname, 'wb') as f:
            f.write(data)

    def close(self):
        """Close function.
        """
        self.__transport.close()

    def is_file(self, member):
        """Determine whether the named file is a single file.

        Args:
            member: file member

        Returns:
            whether named file is a single file (bool)
        """
        return self.__transport.isfile(member.filename)

    def __enter__(self):
        """Enter context manager.
        """
        logger.debug("FilesystemArchive __enter__: {} mode {}".format(
            type(self.__transport), self.__mode))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Leave context manager, cleaning up any open files.
        """
        self.close()
