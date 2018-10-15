#!/usr/bin/env python3

"""Read/Write image files from a zipfile
"""

# Copyright (c) 2018 Erling Andersen, Haukeland University Hospital, Bergen, Norway

import os, os.path, sys, shutil, tempfile
import urllib.parse
import logging
import imagedata.archives
from imagedata.archives.abstractarchive import AbstractArchive
import zipfile

class ReadOnlyError(Exception): pass

class ZipfileArchive(AbstractArchive):
    """Read/write image files from a zipfile."""

    name = "zip"
    description = "Read and write image files from a zipfile."
    authors = "Erling Andersen"
    version = "1.0.0"
    url = "www.helse-bergen.no"
    mimetypes = ['application/zip']

    def __init__(self, transport=None, url=None, mode='r'):
        super(ZipfileArchive, self).__init__(self.name, self.description,
            self.authors, self.version, self.url, self.mimetypes)
        self.__transport = transport
        urldict = urllib.parse.urlsplit(url, scheme="file")
        self.__path = urldict.path
        self.__mode = mode
        self.__extractedfiles = {}

        logging.debug("ZipfileArchive path: {}".format(self.__path))
        fp = self.__transport.open(self.__path, mode=self.__mode)
        logging.debug("ZipfileArchive fp: {}".format(type(fp)))
        filelist = []
        self.__archive = zipfile.ZipFile(fp, self.__mode)
        # Extract the archive
        self.__tmpdir = tempfile.mkdtemp()
        logging.debug("Extract zipfile to {}".format(self.__tmpdir))
        self.__archive.extractall(self.__tmpdir)
        # Get filelist
        logging.debug("ZipFile namelist:\n{}".format(self.__archive.namelist()))
        for fname in self.__archive.namelist():
            logging.debug("ZipfileArchive fname: {}".format(fname))
            if self.__transport.isfile(fname):
                filelist.append(fname)
                logging.debug(fname)
        self.__filelist = sorted(filelist)

    def getnames(self):
        """Return the members as a list of their names.
        It has the same order as the members of the archive.
        """
        return self.__filelist

    def basename(self, filehandle):
        """Basename of file.

        Typical use:
            if archive.basename(filehandle) == "DICOMDIR":

        Input:
        - filehandle: reference to member object
        """
        return os.path.basename(filehandle)

    def getmember(self, filehandle):
        """Return a member object for member with filehandle.

        Extract the member object to local file space.
        This is necessary to allow the seek() operation on open files.
        """

        if filehandle not in self.__extractedfiles:
            self.__archive.extract(filehandle)
            fname = os.path.join(
                    self.__tmpdir,
                    filehandle)
            self.__extractedfiles[filehandle] = fname
        logging.debug("Zipfile getmember: {}".format(self.__extractedfiles[filehandle]))
        return open(self.__extractedfiles[filehandle], mode='rb')

    def getmembers(self):
        """Return the members of the archive as a list of member objects.
        The list has the same order as the members in the archive.
        """
        return self.__filelist

    def to_localfile(self, filehandle):
        """Access a member object through a local file.
        """
        if filehandle not in self.__extractedfiles:
            self.__archive.extract(filehandle)
            fname = os.path.join(
                    self.__tmpdir,
                    filehandle)
            self.__extractedfiles[filehandle] = fname
        return self.__extractedfiles[filehandle]

    def writedata(self, filename, data):
        """Write data to a named file in the archive.
        Input:
        - filename: named file in the archive
        - data: data to write
        """
        if self.__mode[0] == 'r':
            raise ReadOnlyError("Archive is read-only.")
        self.__archive.writestr(filename, data)

    def close(self):
        """Close zip file.
        """
        self.__archive.close()

    def __enter__(self):
        """Enter context manager.
        """
        logging.debug("ZipfileArchive __enter__: {} mode {}".format(type(self.__transport), self.__mode))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Leave context manager, cleaning up any open files.
        """
        shutil.rmtree(self.__tmpdir)
        self.close()
