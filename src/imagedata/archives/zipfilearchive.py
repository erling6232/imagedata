"""Read/Write image files from a zipfile
"""

# Copyright (c) 2018-2021 Erling Andersen, Haukeland University Hospital, Bergen, Norway

import os
import os.path
import shutil
import tempfile
import io
import fnmatch
import urllib.parse
import logging
from abc import ABC

import imagedata.archives
import imagedata.transports
from imagedata.archives.abstractarchive import AbstractArchive
import zipfile


def list_files(startpath):
    import os
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * level
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))


class WriteFileIO(io.FileIO):
    """Local object making sure the new file is written to zip
    archive before closing."""

    def __init__(self, archive, filename, localfile):
        """Make a WriteFileIO object.

        Args:
            archive: ZipFile object
            filename: path name in zip archive
            localfile: path name to local, temporary file
        """
        super(WriteFileIO, self).__init__(localfile.name, mode='wb')
        self.__archive = archive
        self.__filename = filename
        self.__localfile = localfile

    def close(self):
        """Close file, copy it to archive, then delete local file."""
        logging.debug("ZipfileArchive.WriteFileIO.close:")
        ret = super(WriteFileIO, self).close()
        self.__localfile.close()
        logging.debug("ZipfileArchive.WriteFileIO.close: zip %s as %s" %
                      (self.__localfile.name, self.__filename))
        self.__archive.write(self.__localfile.name, self.__filename)
        logging.debug("ZipfileArchive.WriteFileIO.close: remove %s" %
                      self.__localfile.name)
        os.remove(self.__localfile.name)
        return ret

    def __enter__(self):
        """Enter context manager.
        """
        logging.debug("ZipfileArchive.WriteFileIO __enter__: %s %s" %
                      (self.__filename, self.__localfile.name))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Leave context manager:
        Copy file to zip archive.
        Remove local file.
        """
        self.close()


class ZipfileArchive(AbstractArchive, ABC):
    """Read/write image files from a zipfile."""

    name = "zip"
    description = "Read and write image files from a zipfile."
    authors = "Erling Andersen"
    version = "1.0.0"
    url = "www.helse-bergen.no"
    mimetypes = ['application/zip']

    # Internal data
    # self.__transport: file transport object.
    # self.__fp: zip file object in transport object
    # self.__archive: ZipFile object.
    # self.__path: path to the zip file using given transport.
    # self.__mode: 'r' or 'w': read or write access.
    # self.__tmpdir: Local directory where zip file is unpacked.
    # self.__files: dict of files in the zip archive.
    #   key is path name in the zip archive.
    #   value is a dict of member info:
    #     'unpacked': whether the file is unpacked in tmpdir (boolean)
    #     'name': path name in the zip archive
    #     'fh': file handle when open, otherwise None
    #     'localfile': local filename of unpacked file

    def __init__(self, transport=None, url=None, mode='r', read_directory_only=False, opts=None):
        super(ZipfileArchive, self).__init__(
            self.name, self.description,
            self.authors, self.version, self.url, self.mimetypes)
        self.opts = opts
        logging.debug("ZipfileArchive.__init__ url: {}".format(url))
        urldict = urllib.parse.urlsplit(url, scheme="file")
        if transport is not None:
            self.__transport = transport
            self.__path = urldict.path
        elif url is None:
            raise ValueError('url not given')
        else:
            # Determine transport from url
            # netloc = urldict.netloc
            # netloc = urldict.path
            # netloc: where is zipfile
            # self.__path: zipfile name
            if urldict.scheme == 'xnat':
                netloc = urldict.netloc + urldict.path
                self.__path = urldict.path
                logging.debug('ZipfileArchive.__init__: scheme: %s, netloc: %s' %
                              (urldict.scheme, netloc))
                self.__transport = imagedata.transports.Transport(
                    urldict.scheme,
                    netloc=urldict.netloc,
                    root=urldict.path,
                    mode=mode,
                    read_directory_only=read_directory_only)
            else:
                netloc, self.__path = os.path.split(urldict.path)
                logging.debug('ZipfileArchive.__init__: scheme: %s, netloc: %s' %
                              (urldict.scheme, netloc))
                self.__transport = imagedata.transports.Transport(
                    urldict.scheme,
                    root=netloc,
                    mode=mode,
                    read_directory_only=read_directory_only)
        self.__mode = mode
        self.__files = {}

        logging.debug("ZipfileArchive path: {}".format(self.__path))
        self.__fp = self.__transport.open(
            self.__path, mode=self.__mode + "b")
        logging.debug("ZipfileArchive self.__fp: {}".format(type(self.__fp)))
        logging.debug("ZipfileArchive open zipfile mode %s" % self.__mode)
        self.__archive = zipfile.ZipFile(
            self.__fp,
            mode=self.__mode,
            compression=zipfile.ZIP_DEFLATED)
        # Extract the archive
        self.__tmpdir = tempfile.mkdtemp()
        logging.debug("Extract zipfile {} to {}".format(
            self.__archive, self.__tmpdir))
        # Get filelist in self.__files
        for fname in self.__archive.namelist():
            try:
                _is_dir = self.__archive.getinfo(fname).is_dir()  # Works with Python >= 3.6
            except AttributeError:
                _is_dir = fname[-1] == '/'
            if not _is_dir:
                member = {'unpacked': False, 'name': fname, 'fh': None}
                self.__files[fname] = member
        # logging.debug("ZipFile self.__files: {}".format(self.__files))

    @property
    def transport(self):
        """Underlying transport plugin
        """
        return self.__transport

    def use_query(self):
        """Do the plugin need the ?query part of the url?"""
        return True

    def getnames(self, files=None):
        """Get name list of the members.

        Returns:
            The members as a list of their names.
                It has the same order as the members of the archive.
        """
        if files:
            filelist = list()
            for filename in self.__files:
                # logging.debug('ZipfileArchive.getmembers: member {}'.format(filename))
                for required_filename in files:
                    if fnmatch.fnmatchcase(filename, os.path.normpath(required_filename)):
                        filelist.append(filename)
                    elif fnmatch.fnmatchcase(filename, os.path.normpath(required_filename)+'/*'):
                        filelist.append(filename)
            if len(filelist) < 1:
                raise FileNotFoundError('No such file: %s' % files)
            return filelist
        else:
            return sorted(self.__files.keys())

    def basename(self, filehandle):
        """Basename of file.

        Examples:
            if archive.basename(filehandle) == "DICOMDIR":

        Args:
            filehandle: reference to member object
        """
        return os.path.basename(filehandle['name'])

    @staticmethod
    def _longest_prefix(keys, required):
        prefix = ''
        for folder in keys:
            new_prefix = os.path.commonprefix([folder, os.path.normpath(required)])
            if len(new_prefix) > len(prefix):
                prefix = new_prefix
        return prefix

    def _filehandle_in_files(self, filehandle):
        fname = filehandle['name']
        prefix = self._longest_prefix(self.__files.keys(), fname)
        return prefix in self.__files

    def open(self, filehandle, mode='rb'):
        """Open file.

        Returns:
             A member object for member with filehandle.

        Extract the member object to local file space.
        This is necessary to allow the seek() operation on open files.
        """

        logging.debug('ZipfileArchive.open: mode %s' % mode)
        logging.debug('ZipfileArchive.open: filehandle %s' % filehandle)
        if mode[0] == 'r':
            if filehandle['name'] not in self.__files:
                raise FileNotFoundError(
                    'No such file: %s' % filehandle['name'])
            filehandle['localfile'] = self.__archive.extract(
                filehandle['name'], path=self.__tmpdir)
            filehandle['unpacked'] = True
            filehandle['fh'] = open(filehandle['localfile'], mode=mode)
            return filehandle['fh']
        elif mode[0] == 'w':
            if self.__mode[0] == 'r':
                raise PermissionError(
                    'Cannot write on an archive opened for read')
            # Open local file for write
            localfile = tempfile.NamedTemporaryFile(delete=False)
            logging.debug('ZipfileArchive.open: mode %s file %s' % (
                mode, localfile))
            fh = WriteFileIO(self.__archive, filehandle, localfile)
            member = {'unpacked': True,
                      'name': filehandle,
                      'fh': fh,
                      'localfile': localfile}
            self.__files[filehandle] = member
            return fh
        else:
            raise ValueError('Unknown mode "%s"' % mode)

    def getmembers(self, files=None):
        """Get the members of the archive.

        Returns:
            The members of the archive as a list of member objects.
                The list has the same order as the members in the archive.
        """
        if files:
            if issubclass(type(files), list):
                wanted_files = files
            else:
                wanted_files = list((files,))
            found_match = [False for _ in range(len(wanted_files))]
            filelist = list()
            for filename in self.__files:
                for i, required_filename in enumerate(wanted_files):
                    if fnmatch.fnmatchcase(filename, os.path.normpath(required_filename)):
                        filelist.append(self.__files[filename])
                        found_match[i] = True
                    elif fnmatch.fnmatchcase(filename, os.path.normpath(required_filename)+'/*'):
                        filelist.append(self.__files[filename])
                        found_match[i] = True
            # Verify that all wanted files are found
            for i, found in enumerate(found_match):
                if not found:
                    raise FileNotFoundError('No such file: %s' % wanted_files[i])
            if len(filelist) < 1:
                raise FileNotFoundError('No such file: %s' % files)
            return filelist
        else:
            return self.__files

    def to_localfile(self, filehandle):
        """Access a member object through a local file.
        """
        if not self._filehandle_in_files(filehandle):
            raise FileNotFoundError(
                'No such file: %s' % filehandle['name'])
        if not filehandle['unpacked']:
            filehandle['localfile'] = \
                self.__archive.extract(filehandle['name'])
            filehandle['unpacked'] = True
            self.__files[filehandle['name']] = filehandle
        return filehandle['localfile']

    def add_localfile(self, local_file, filename):
        """Add a local file to the archive.

        Args:
            local_file: named local file
            filename: filename in the archive
        Returns:
            filehandle to file in the archive
        """
        if self.__mode[0] == 'r':
            raise PermissionError(
                'Cannot write on an archive opened for read')
        member = {'unpacked': True,
                  'name': filename,
                  'fh': None,
                  'localfile': local_file}
        self.__archive.write(local_file, arcname=filename)
        logging.debug('ZipfileArchive.add_localfile: local {} as {}'.format(
            local_file, filename))
        self.__files[filename] = member
        logging.debug('{}'.format(self.__archive.namelist()))

    def writedata(self, filename, data):
        """Write data to a named file in the archive.

        Args:
            filename: named file in the archive
            data: data to write
        """
        if self.__mode[0] == 'r':
            raise PermissionError(
                'Cannot write on an archive opened for read')
        member = {'unpacked': False,
                  'name': filename,
                  'fh': None}
        self.__archive.writestr(filename, data)
        self.__files[filename] = member

    def close(self):
        """Close zip file.
        """
        self.__archive.close()
        self.__fp.close()
        shutil.rmtree(self.__tmpdir)
        logging.debug('ZipfileArchive.close: {}'.format(self.__tmpdir))
        self.__transport.close()

    def is_file(self, filehandle):
        """Determine whether the named file is a single file.
        """
        pass

    def __enter__(self):
        """Enter context manager.
        """
        logging.debug("ZipfileArchive __enter__: {} mode {}".format(type(self.__transport), self.__mode))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Leave context manager, cleaning up any open files.
        """
        logging.debug('ZipfileArchive.__exit__:')
        self.close()
