#!/usr/bin/env python3

"""Read/Write local image files
"""

# Copyright (c) 2018 Erling Andersen, Haukeland University Hospital, Bergen, Norway

import os, os.path, sys
import collections
import urllib.parse
import logging
import imagedata.archives
from imagedata.archives.abstractarchive import AbstractArchive

class ReadOnlyError(Exception): pass
class WriteOnFile(Exception): pass
class NoSuchFile(Exception): pass

class FilesystemArchive(AbstractArchive):
    """Read/write local files."""

    name = "filesystem"
    description = "Read and write local image files."
    authors = "Erling Andersen"
    version = "1.0.0"
    url = "www.helse-bergen.no"
    mimetypes = ['*'] # Disregards MIME types

    def __init__(self, transport=None, url=None, mode='r'):
        super(FilesystemArchive, self).__init__(self.name, self.description,
            self.authors, self.version, self.url, self.mimetypes)
        self.__transport = transport
        self.__mode = mode

        logging.debug("FilesystemArchive __init__: {}".format(type(transport)))
        logging.debug("FilesystemArchive __init__ url: {}".format(url))
        self.__urldict = urllib.parse.urlsplit(url, scheme="file")

        # If the URL refers to a single file, let dirname refer to the
        # directory and basename to the file
        if os.path.isfile(self.__urldict.path):
            self.__dirname  = os.path.dirname (self.__urldict.path)
            self.__basename = os.path.basename(self.__urldict.path)
            self.__filelist = {self.__urldict.path: self.__basename}
            logging.debug("FilesystemArchive __init__ dirname : {}".format(self.__dirname))
            logging.debug("FilesystemArchive __init__ basename: {}".format(self.__basename))
            logging.debug("FilesystemArchive self.__filelist: {}".format(self.__filelist))
            return

        # The URL refers to a directory. Let dirname refer to the directory
        self.__dirname = self.__urldict.path
        self.__basename = None
        logging.debug("FilesystemArchive __init__ dirname : {}".format(self.__dirname))
        logging.debug("FilesystemArchive __init__ basename: {}".format(self.__basename))
        self.__filelist = collections.OrderedDict()
        for root, dirs, files in self.__transport.walk(self.__urldict.path):
            logging.debug("FilesystemArchive root: {} {}".format(root, files))
            for filename in files:
                fname = os.path.join(root, filename)
                logging.debug("FilesystemArchive fname: {}".format(fname))
                if transport.isfile(fname):
                    if root.startswith(self.__dirname):
                        root = root[len(self.__dirname)+1:] # Strip off dirname
                    self.__filelist[fname] = (root,filename)
                    logging.debug(fname)
        logging.debug("FilesystemArchive self.__filelist: {}".format(self.__filelist))

    def getnames(self):
        """Return the members as a list of their names.
        It has the same order as the members of the archive.
        """
        return list(self.__filelist.keys())

    def basename(self, filehandle):
        """Basename of file.

        Typical use:
            if archive.basename(filehandle) == "DICOMDIR":

        Input:
        - filehandle: reference to member object
        """
        root,filename = filehandle
        return os.path.basename(filename)

    def getmember(self, filehandle):
        """Return a member object for member with filehandle.
        """
        root,filename = filehandle
        if self.__basename is not None and self.__basename != filename:
            raise NoSuchFile("File {} cannot be opened (basename {})".format(filename, self.__basename))
        logging.debug("getmember: join {} {} {}".format(self.__dirname, root,
            filename))
        fname = os.path.join(self.__dirname, root, filename)
        logging.debug("getmember: fname {}".format(fname))
        return self.__transport.open(fname, 'rb')

    def getmembers(self):
        """Return the members of the archive as a list of member objects.
        The list has the same order as the members in the archive.
        """
        return self.__filelist

    def to_localfile(self, filehandle):
        """Access a member object through a local file.
        """
        root,filename = filehandle
        return os.path.join(self.__dirname, root, filehandle)

    def writedata(self, filename, data):
        """Write data to a named file in the archive.
        Input:
        - filename: named file in the archive
        - data: data to write
        """
        if self.__mode[0] == 'r':
            raise ReadOnlyError("Archive is read-only.")
        if self.__basename is not None:
            raise WriteOnFile("Do not know how to write a file to a file.")
        fname = os.path.join(self.__dirname, filename)
        logging.debug("writedata: fname {}".format(fname))
        with self.__transport.open(fname, 'wb') as f:
            f.write(data)
        self.__filelist[fname] = (root,filename)

    def close(self):
        """Dummy close function.
        """
        return

    def __enter__(self):
        """Enter context manager.
        """
        logging.debug("FilesystemArchive __enter__: {} mode {}".format(type(self.__transport), self.__mode))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Leave context manager, cleaning up any open files.
        """
        self.close()
