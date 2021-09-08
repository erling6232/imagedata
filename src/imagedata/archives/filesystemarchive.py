"""Read/Write local image files
"""

# Copyright (c) 2018 Erling Andersen, Haukeland University Hospital, Bergen, Norway

import os
import os.path
import fnmatch
import shutil
import urllib.parse
import logging
from abc import ABC

import imagedata.archives
import imagedata.transports
from imagedata.archives.abstractarchive import AbstractArchive

logger = logging.getLogger(__name__)


class ReadOnlyError(Exception):
    pass


class WriteOnFile(Exception):
    pass


class NoSuchFile(Exception):
    pass


class FilesystemArchive(AbstractArchive, ABC):
    """Read/write local files."""

    name = "filesystem"
    description = "Read and write local image files."
    authors = "Erling Andersen"
    version = "1.0.0"
    url = "www.helse-bergen.no"
    mimetypes = ['*']  # Disregards MIME types

    # self.__dirname: root directory
    # self.__filelist: list of absolute filename

    @staticmethod
    def _get_transport(url, mode, read_directory_only):
        """Get transport plugin from url.

        If the url addressess a missing file in read mode,
        access the parent directory.
        """

        url_tuple = urllib.parse.urlsplit(url, scheme='file')
        logger.debug('FilesystemArchive._get_transport: scheme: %s, netloc: %s' %
                     (url_tuple.scheme, url_tuple.path))

        try:
            _transport = imagedata.transports.Transport(
                url_tuple.scheme,
                netloc=url_tuple.netloc,
                root=url_tuple.path,
                mode=mode,
                read_directory_only=read_directory_only)
        except imagedata.transports.RootDoesNotExist:
            # Mode='r': location does not exist
            raise
        except imagedata.transports.RootIsNotDirectory:
            # Mode='r': Retry with parent directory
            parent, _ = os.path.split(url_tuple.path)
            _transport = imagedata.transports.Transport(
                url_tuple.scheme,
                netloc=url_tuple.netloc,
                root=parent,
                mode=mode,
                read_directory_only=read_directory_only)
        return _transport

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
            # self.__transport = self._get_transport(url, mode, read_directory_only)
            # netloc = urldict.netloc
            # netloc = urldict.path
            # netloc: where is zipfile
            # self.__path: zipfile name
            # netloc, self.__path = os.path.split(urldict.path)
            # logger.debug('FilesystemArchive.__init__: scheme: %s, netloc: %s path: %s' %
            #              (urldict.scheme, netloc, self.__path))
            logger.debug('FilesystemArchive.__init__: scheme: %s, path: %s' %
                         (urldict.scheme, self.__path))
            self.__transport = imagedata.transports.Transport(
                urldict.scheme,
                netloc=self.__netloc,
                root=self.__path,
                mode=mode,
                read_directory_only=read_directory_only,
                opts=opts)
        self.__mode = mode
        self.__files = {}

        logger.debug("FilesystemArchive __init__: {}".format(type(transport)))

        logger.debug("FilesystemArchive path: {}".format(self.__path))
        # self.__fp = self.__transport.open(
        #    self.__path, mode=self.__mode + "b")
        # logger.debug("FilesystemArchive self.__fp: {}".format(type(self.__fp)))
        logger.debug("FilesystemArchive open zipfile mode %s" % self.__mode)

        # If the URL refers to a single file, let directory_name refer to the
        # directory and basename to the file
        logger.debug("FilesystemArchive __init__ verify : {}".format(self.__path))
        if os.path.isfile(self.__path):
            self.__dirname = os.path.dirname(self.__path)
            self.__basename = os.path.basename(self.__path)
            self.__filelist = [self.__basename]
            logger.debug("FilesystemArchive __init__ directory_name : {}".format(self.__dirname))
            logger.debug("FilesystemArchive __init__ basename: {}".format(self.__basename))
            # logger.debug("FilesystemArchive self.__filelist: {}".format(self.__filelist))
            return

        # The URL refers to a directory. Let directory_name refer to the directory
        self.__dirname = self.__path
        self.__basename = None
        logger.debug("FilesystemArchive __init__ scan directory_name : {}".format(self.__dirname))
        logger.debug("FilesystemArchive __init__ scan basename: {}".format(self.__basename))
        self.__filelist = list()
        # logger.debug("FilesystemArchive walk root: {}".format(self.__urldict.path))
        for root, dirs, files in self.__transport.walk(self.__path):
            # logger.debug("FilesystemArchive scan root: {} {}".format(root, files))
            for filename in files:
                # fname = os.path.join(self.__path, root, filename)
                fname = os.path.join(root, filename)
                # logger.debug("FilesystemArchive scan fname: {}".format(fname))
                self.__filelist.append(fname)
                # if transport.isfile(fname):
                #    if root.startswith(self.__dirname):
                #        root = root[len(self.__dirname)+1:] # Strip off directory_name
                #    self.__filelist[fname] = (root,filename)
                #    logger.debug(fname)
        self._sort_filelist()
        # logger.debug("FilesystemArchive self.__filelist: {}".format(self.__filelist))

    @property
    def transport(self):
        """Underlying transport plugin
        """
        return self.__transport

    def use_query(self):
        """Do the plugin need the ?query part of the url?"""
        return False

    def getnames(self, files=None):
        """Get name list of the members.

        Return:
            The members as a list of their names.
                It has the same order as the members of the archive.
        """
        if files is None or \
                (issubclass(type(files), str) and files == '*') or \
                (issubclass(type(files), list) and len(files) > 0 and files[0] == '*'):
            return self.__filelist
        else:
            filelist = list()
            for filename in self.__filelist:
                for required_filename in files:
                    if fnmatch.fnmatchcase(filename, os.path.normpath(required_filename)):
                        filelist.append(filename)
                    elif fnmatch.fnmatchcase(filename, os.path.normpath(required_filename) + os.sep + '*'):
                        filelist.append(filename)
            if len(filelist) < 1:
                raise FileNotFoundError('No such file: %s' % files)
            return filelist

    def basename(self, filehandle):
        """Basename of file.

        Examples:
            if archive.basename(filehandle) == "DICOMDIR":

        Args:
            filehandle: reference to member object
        """
        root, filename = filehandle
        return os.path.basename(filename)

    def open(self, filehandle, mode='rb'):
        """Open file.

        Returns:
             a member object for member with filehandle.
        """
        # logger.debug("getmember: fname {}".format(filehandle))
        return self.__transport.open(filehandle, mode)

    def getmembers(self, files=None):
        """Get the members of the archive.

        Returns:
            The members of the archive as a list of member objects.
                The list has the same order as the members in the archive.
        """
        # logger.debug("getmembers: files {}".format(files))
        if files is None or \
                (issubclass(type(files), str) and files == '*') or \
                (issubclass(type(files), list) and len(files) > 0 and files[0] == '*'):
            return self.__filelist
        else:
            if issubclass(type(files), list):
                wanted_files = files
            else:
                wanted_files = list((files,))
            found_match = [False for _ in range(len(wanted_files))]
            filelist = list()
            for filename in self.__filelist:
                for i, required_filename in enumerate(wanted_files):
                    if fnmatch.fnmatchcase(filename, os.path.normpath(required_filename)):
                        filelist.append(filename)
                        found_match[i] = True
                    elif fnmatch.fnmatchcase(filename, os.path.normpath(required_filename) + os.sep + '*'):
                        filelist.append(filename)
                        found_match[i] = True
            # Verify that all wanted files are found
            for i, found in enumerate(found_match):
                if not found:
                    raise FileNotFoundError('No such file: %s' % wanted_files[i])
            if len(filelist) < 1:
                raise FileNotFoundError('No such file: %s' % files)
            return filelist

    def to_localfile(self, filehandle):
        """Access a member object through a local file.
        """
        # logger.debug('FilesystemArchive to_localfile: filename %s' %
        #        filehandle)
        return os.path.join(self.__path, filehandle)

    def add_localfile(self, local_file, filename):
        """Add a local file to the archive.

        Args:
            local_file: named local file
            filename: filename in the archive
        Returns:
            filehandle to file in the archive
        """

        fname = os.path.join(self.__dirname, filename)
        if fname not in self.__filelist:
            # Ensure the directory exists,
            # create it silently if not.
            os.makedirs(
                os.path.dirname(fname),
                exist_ok=True)
            shutil.copy(local_file, fname)
            self.__filelist.append(fname)
            self._sort_filelist()
        else:
            raise imagedata.archives.FileAlreadyExistsError(
                'File %s already exists' %
                os.path.join(
                    self.__path,
                    filename))

    def writedata(self, filename, data):
        """Write data to a named file in the archive.

        Args:
            filename: named file in the archive
            data: data to write
        """
        if self.__mode[0] == 'r':
            raise ReadOnlyError("Archive is read-only.")
        if self.__basename is not None:
            raise WriteOnFile("Do not know how to write a file to a file.")
        fname = os.path.join(self.__dirname, filename)
        logger.debug("writedata: fname {}".format(fname))
        with self.__transport.open(fname, 'wb') as f:
            f.write(data)
        self.__filelist.append(fname)
        self._sort_filelist()

    def close(self):
        """Close function.
        """
        self.__transport.close()
        return

    def is_file(self, filehandle):
        """Determine whether the named file is a single file.
        """
        return self.__transport.isfile(filehandle)

    def __enter__(self):
        """Enter context manager.
        """
        logger.debug("FilesystemArchive __enter__: {} mode {}".format(type(self.__transport), self.__mode))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Leave context manager, cleaning up any open files.
        """
        self.close()

    def _sort_filelist(self):
        """Sort self.__filelist, usually after creation or insertion"""
        self.__filelist = sorted(self.__filelist)
