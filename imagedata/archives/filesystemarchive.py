"""Read/Write local image files
"""

# Copyright (c) 2018 Erling Andersen, Haukeland University Hospital, Bergen, Norway

import os, os.path, sys
import collections
import shutil
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

    # self.__dirname: root directory
    # self.__filelist: list of absolute filename

    def _get_transport(self, url, mode, read_directory_only):
        """Get transport plugin from url.

        If the url addressess a missing file in read mode,
        access the parent directory.
        """

        try:
            url_tuple = urllib.parse.urlsplit(url, scheme='file')
            netloc = url_tuple.path
            logging.debug('FilesystemArchive._get_transport: scheme: %s, netloc: %s' %
                (url_tuple.scheme, netloc))
            transport = imagedata.transports.find_scheme_plugin(
                url_tuple.scheme,
                netloc,
                mode=mode,
                read_directory_only=read_directory_only)
        except imagedata.transports.RootDoesNotExist:
            # Mode='r': location does not exist
            raise
        except imagedata.transports.RootIsNotDirectory:
            # Mode='r': Retry with parent directory
            parent, filename = os.path.split(url_tuple.path)
            transport = imagedata.transports.find_scheme_plugin(
                url_tuple.scheme,
                parent,
                mode=mode,
                read_directory_only=read_directory_only)
        return(transport)

    def __init__(self, transport=None, url=None, mode='r',
            read_directory_only=True):
        super(FilesystemArchive, self).__init__(self.name, self.description,
            self.authors, self.version, self.url, self.mimetypes)
        logging.debug("FilesystemArchive.__init__ url: {}".format(url))
        if transport is not None:
            self.__transport = transport
        elif url is None:
            raise ValueError('url not given')
        else:
            # Determine transport from url
            self.__transport = self._get_transport(url, mode, read_directory_only)
        self.__mode = mode

        logging.debug("FilesystemArchive __init__: {}".format(type(transport)))
        logging.debug("FilesystemArchive __init__ url: {}".format(url))
        self.__urldict = urllib.parse.urlsplit(url, scheme="file")

        # If the URL refers to a single file, let dirname refer to the
        # directory and basename to the file
        logging.debug("FilesystemArchive __init__ verify : {}".format(self.__urldict.path))
        if os.path.isfile(self.__urldict.path):
            self.__dirname  = os.path.dirname (self.__urldict.path)
            self.__basename = os.path.basename(self.__urldict.path)
            self.__filelist = [self.__urldict.path]
            logging.debug("FilesystemArchive __init__ dirname : {}".format(self.__dirname))
            logging.debug("FilesystemArchive __init__ basename: {}".format(self.__basename))
            #logging.debug("FilesystemArchive self.__filelist: {}".format(self.__filelist))
            return

        # The URL refers to a directory. Let dirname refer to the directory
        self.__dirname = self.__urldict.path
        self.__basename = None
        logging.debug("FilesystemArchive __init__ scan dirname : {}".format(self.__dirname))
        logging.debug("FilesystemArchive __init__ scan basename: {}".format(self.__basename))
        self.__filelist = list()
        #logging.debug("FilesystemArchive walk root: {}".format(self.__urldict.path))
        for root, dirs, files in self.__transport.walk(self.__urldict.path):
            #logging.debug("FilesystemArchive scan root: {} {}".format(root, files))
            for filename in files:
                fname = os.path.join(self.__urldict.path, root, filename)
                #logging.debug("FilesystemArchive scan fname: {}".format(fname))
                self.__filelist.append(fname)
                #if transport.isfile(fname):
                #    if root.startswith(self.__dirname):
                #        root = root[len(self.__dirname)+1:] # Strip off dirname
                #    self.__filelist[fname] = (root,filename)
                #    logging.debug(fname)
        #logging.debug("FilesystemArchive self.__filelist: {}".format(self.__filelist))

    def use_query(self):
        """Do the plugin need the ?query part of the url?"""
        return(False)

    def getnames(self, files=None):
        """Return the members as a list of their names.
        It has the same order as the members of the archive.
        """
        #logging.debug('FilesystemArchive getnames: self.__filelist: {}'.format(
        #    self.__filelist))
        if files:
            filelist = list()
            for filename in self.__filelist:
                # logging.debug('ZipfileArchive.getmembers: member {}'.format(filename))
                # filename = member['name']
                for required_filename in files:
                    # if filename.endswith(required_filename):
                    if re.search(required_filename, filename):
                        filelist.append(filename)
            if len(filelist) < 1:
                raise FileNotFoundError('No such file: %s' % files)
            return (filelist)
        else:
            return(self.__filelist)

    def basename(self, filehandle):
        """Basename of file.

        Typical use:
            if archive.basename(filehandle) == "DICOMDIR":

        Input:
        - filehandle: reference to member object
        """
        root,filename = filehandle
        return os.path.basename(filename)

    def open(self, filehandle, mode='rb'):
        """Open file. Return a member object for member with filehandle.
        """
        #logging.debug("getmember: fname {}".format(filehandle))
        return(self.__transport.open(filehandle, mode))

    def getmembers(self, files=None):
        """Return the members of the archive as a list of member objects.
        The list has the same order as the members in the archive.
        """
        #logging.debug("getmembers: files {}".format(files))
        if files:
            filelist = list()
            used = dict()
            for filename in self.__filelist:
                for required_filename in files:
                    if filename.endswith(required_filename):
                        filelist.append(filename)
                        used[required_filename] = True
            #logging.debug("getmembers: filelist {}".format(filelist))
            for required_filename in files:
                if required_filename not in used:
                    raise FileNotFoundError('File %s not found.' %
                            required_filename)
            return(filelist)
        else:
            return(self.__filelist)

    def to_localfile(self, filehandle):
        """Access a member object through a local file.
        """
        #logging.debug('FilesystemArchive to_localfile: filename %s' %
        #        filehandle)
        return(filehandle)

    def add_localfile(self, local_file, filename):
        """Add a local file to the archive.

        Input:
        - local_file: named local file
        - filename: filename in the archive
        Return:
        - filehandle to file in the archive
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
        else:
            raise FileAlreadyExistsError(
                    'File %s alread exists' %
                    filename)

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
        self.__filelist.append(fname)

    def close(self):
        """Dummy close function.
        """
        return

    def is_file(self, filehandle):
        """Determine whether the named file is a single file.
        """
        return(self.__transport.isfile(filehandle))

    def __enter__(self):
        """Enter context manager.
        """
        logging.debug("FilesystemArchive __enter__: {} mode {}".format(type(self.__transport), self.__mode))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Leave context manager, cleaning up any open files.
        """
        self.close()
