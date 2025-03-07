"""Read/Write files from a zipfile
"""

# Copyright (c) 2018-2024 Erling Andersen, Haukeland University Hospital, Bergen, Norway

from typing import Tuple, Union
import os
import os.path
import shutil
import tempfile
import io
import fnmatch
import urllib.parse
import logging
from abc import ABC

from .abstractarchive import AbstractArchive, Member
from ..transports import Transport
import zipfile

logger = logging.getLogger(__name__)


def list_files(startpath):
    import os
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count('/')
        indent = ' ' * 4 * level
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))


class WriteFileIO(io.FileIO):
    """Local object making sure the new file is written to zip
    archive before closing."""

    def __init__(self, archive, member, local_file):
        """Make a WriteFileIO object.

        Args:
            archive: ZipFile object
            member: member of the zip archive
            local_file: local temporary file
        """

        if isinstance(local_file, str):
            super(WriteFileIO, self).__init__(local_file, mode='wb')
        else:
            super(WriteFileIO, self).__init__(local_file.name, mode='wb')
        self.__archive = archive
        self.__filename = member.filename
        self.__local_file = local_file

    @property
    def local_file(self):
        return self.__local_file

    def close(self):
        """Close file, copy it to archive, then delete local file."""
        _name: str = '{}.{}'.format(__name__, self.close.__name__)
        logger.debug("{}:".format(_name))
        ret = super(WriteFileIO, self).close()
        if isinstance(self.__local_file, str):
            self.__archive.write(self.__local_file, self.__filename)
            try:
                os.remove(self.__local_file)
            except PermissionError:
                pass
        else:
            self.__local_file.close()
            logger.debug("{}: zip {} as {}".format(
                _name, self.__local_file.name, self.__filename)
            )
            self.__archive.write(self.__local_file.name, self.__filename)
            logger.debug("{}: remove {}".format(
                _name, self.__local_file.name)
            )
            try:
                os.remove(self.__local_file.name)
            except PermissionError:
                pass
        return ret

    def __enter__(self):
        """Enter context manager.
        """
        # logger.debug("ZipfileArchive.WriteFileIO __enter__: %s %s" %
        #              (self.__filename, self.__local_file.name))
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
    version = "1.1.0"
    url = "www.helse-bergen.no"
    mimetypes = ['application/zip', 'application/x-zip-compressed']

    # Internal data
    # self.transport: file transport object.
    # self.__fp: zip file object in transport object
    # self.__archive: ZipFile object.
    # self.__path: path to the zip file using given transport.
    # self.__mode: 'r' or 'w': read or write access.
    # self.__tmpdir: Local directory where zip file is unpacked.
    # self.__files: dict of files in the zip archive.
    #   key is path name in the zip archive.
    #   value is a Member object:
    #     info['unpacked']: whether the file is unpacked in tmpdir (boolean)
    #     filename: path name in the zip archive
    #     fh: file handle when open, otherwise None
    #     local_file: local filename of unpacked file

    def __init__(self, transport=None, url=None, mode='r', read_directory_only=False, opts=None):
        super(ZipfileArchive, self).__init__(
            self.name, self.description,
            self.authors, self.version, self.url, self.mimetypes)
        _name: str = '{}.{}'.format(__name__, self.__init__.__name__)
        self.opts = opts
        logger.debug("{}: url: {}".format(_name, url))
        if os.name == 'nt' and fnmatch.fnmatch(url, '[A-Za-z]:\\*'):
            # Windows: Parse without x:, then reattach drive letter
            urldict = urllib.parse.urlsplit(url[2:], scheme="file")
            self.__path = url[:2] + urldict.path
        else:
            urldict = urllib.parse.urlsplit(url, scheme="file")
            self.__path = urldict.path if len(urldict.path) > 0 else urldict.netloc
        if transport is not None:
            self.transport = transport
        elif url is None:
            raise ValueError('url not given')
        else:
            # Determine transport from url
            # netloc = urldict.netloc
            # netloc = urldict.path
            # netloc: where is zipfile
            # self.__path: zipfile name
            try:
                netloc = urldict.netloc + self.__path
                logger.debug('{}: scheme: {}, netloc: {}'.format(
                    _name, urldict.scheme, netloc
                ))
                self.transport = Transport(
                    urldict.scheme,
                    netloc=urldict.netloc,
                    root=urldict.path,
                    mode=mode,
                    read_directory_only=read_directory_only)
            except Exception:
                raise
        self.__mode = mode
        self.__files = {}

        logger.debug("{}: path: {}".format(_name, self.__path))
        self.__fp = self.transport.open(
            self.__path, mode=self.__mode + "b")
        logger.debug("{}: self.__fp: {}".format(_name, type(self.__fp)))
        logger.debug("{}: open zipfile mode {}".format(_name, self.__mode))
        self.__archive = zipfile.ZipFile(
            self.__fp,
            mode=self.__mode,
            compression=zipfile.ZIP_DEFLATED)
        # Extract the archive
        self.__tmpdir = tempfile.mkdtemp()
        logger.debug("{}: Extract zipfile {} to {}".format(
            _name, self.__archive, self.__tmpdir))
        # Get filelist in self.__files
        for fname in self.__archive.namelist():
            try:
                _is_dir = self.__archive.getinfo(fname).is_dir()
            except AttributeError:
                _is_dir = fname[-1] == '/'
            except Exception as e:
                logger.error('{}: {}'.format(_name, e))
                raise
            if not _is_dir:
                self.__files[fname] = Member(fname,
                                             info={'unpacked': False}
                                             )
        # logger.debug("ZipFile self.__files: {}".format(self.__files))

    def use_query(self):
        """Does the plugin need the ?query part of the url?"""
        return True

    def getnames(self, files=None):
        """Get name list of the members.

        Args:
            files: List or single str of filename matches

        Returns:
            The members as a list of their names.
                It has the same order as the members of the archive.
        Raises:
            FileNotFoundError: When no matching file is found.
        """
        _name: str = '{}.{}'.format(__name__, self.getnames.__name__)
        if files is not None and issubclass(type(files), str):
            wanted_files = [files]
        else:
            wanted_files = files
        if wanted_files is None or\
            (issubclass(type(wanted_files), list) and (
                len(wanted_files) == 0 or
                len(wanted_files) > 0 and wanted_files[0] == '*')):
            logger.debug('{}: found files {}'.format(_name, len(self.__files)))
            return sorted(self.__files.keys())
        else:
            filelist = list()
            for filename in self.__files:
                logger.debug('{}: member {}'.format(_name, filename))
                for required_filename in wanted_files:
                    logger.debug('{}: required {}'.format(_name, required_filename))
                    if required_filename[-1] == '/':
                        required_filename = required_filename[:-1]
                    if fnmatch.fnmatchcase(filename, required_filename):
                        filelist.append(filename)
                    elif fnmatch.fnmatchcase(filename, required_filename + '/*'):
                        filelist.append(filename)
            logger.debug('{}: found files {}'.format(_name, len(filelist)))
            if len(filelist) < 1:
                raise FileNotFoundError('No such file: {}'.format(wanted_files))
            return filelist

    def basename(self, filehandle: Member):
        """Basename of file.

        Examples:
            if archive.basename(filehandle) == "DICOMDIR":

        Args:
            filehandle: reference to member object

        Returns:
            Basename of file: str
        """
        return os.path.basename(filehandle.filename)

    @staticmethod
    def _longest_prefix(keys, required):
        prefix = ''
        for folder in keys:
            # new_prefix = os.path.commonprefix([folder, os.path.normpath(required)])
            new_prefix = os.path.commonprefix([folder, required])
            if len(new_prefix) > len(prefix):
                prefix = new_prefix
        return prefix

    def _filehandle_in_files(self, filehandle):
        fname = filehandle.filename
        prefix = self._longest_prefix(self.__files.keys(), fname)
        return prefix in self.__files

    def open(self, member: Member, mode: str = 'rb'):
        """Open file.

        Extract the member object to local file space.
        This is necessary to allow the seek() operation on open files.

        Args:
            member (imagedata.archives.abstractarchive.Member): Handle to file.
            mode (str): Open mode.
        Returns:
            An IO object for the member.
        Raises:
            FileNotFoundError: when file is not found.
            PermissionError: When archive is read-only.
        """

        _name: str = '{}.{}'.format(__name__, self.open.__name__)
        if isinstance(member, str):
            member = Member(member)
        logger.debug('{}: mode {}'.format(_name, mode))
        logger.debug('{}: member {}'.format(_name, member.filename))
        if mode[0] == 'r':
            if member.filename not in self.__files:
                raise FileNotFoundError(
                    'No such file: %s' % member.filename)
            member.local_file = self.__archive.extract(
                member.filename, path=self.__tmpdir)
            member.info['unpacked'] = True
            member.fh = open(member.local_file, mode=mode)
            return member.fh
        elif mode[0] == 'w':
            if self.__mode[0] == 'r':
                raise PermissionError(
                    'Cannot write on an archive opened for read')
            # Open local file for write
            suffix = None
            ext = member.filename.find('.')
            if ext >= 0:
                suffix = member.filename[ext:]
            local_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            logger.debug('{}: mode {} file {}'.format(_name, mode, local_file))
            fh = WriteFileIO(self.__archive, member, local_file)
            # Update info on member file
            self.__files[member.filename] = Member(member.filename,
                                                   info={'unpacked': True},
                                                   fh=fh,
                                                   local_file=local_file
                                                   )
            return fh
        else:
            raise ValueError('Unknown mode "%s"' % mode)

    def getmembers(self, files=None):
        """Get the members of the archive.

        Args:
            files: List of filename matches

        Returns:
            The members of the archive as a list of Filehandles.
                The list same order as the members in the archive.
        """
        if files is not None and issubclass(type(files), str):
            wanted_files = [files]
        else:
            wanted_files = files
        if wanted_files is None or \
            (issubclass(type(wanted_files), list) and (
                len(wanted_files) == 0 or len(wanted_files) > 0 and
                wanted_files[0] == '*')):
            return list(self.__files.values())
        else:
            # logger.debug('ZipfileArchive.getmembers: files {}'.format(len(files)))
            if issubclass(type(files), list):
                wanted_files = []
                for file in files:
                    if file[-1] == '/':
                        file = file[:-1]
                    wanted_files.append(file)
            else:
                if files[-1] == '/':
                    files = files[:-1]
                wanted_files = list((files,))
            # logger.debug('ZipfileArchive.getmembers: wanted_files {}'.format(len(wanted_files)))
            found_match = [False for _ in range(len(wanted_files))]
            filelist = list()
            for filename in self.__files:
                for i, required_filename in enumerate(wanted_files):
                    if fnmatch.fnmatchcase(filename, required_filename):
                        filelist.append(self.__files[filename])
                        found_match[i] = True
                    elif fnmatch.fnmatchcase(filename, required_filename + '/*'):
                        filelist.append(self.__files[filename])
                        found_match[i] = True
            # Verify that all wanted files are found
            for i, found in enumerate(found_match):
                if not found:
                    raise FileNotFoundError('No such file: %s' % wanted_files[i])
            if len(filelist) < 1:
                raise FileNotFoundError('No such file: %s' % files)
            return filelist

    def construct_filename(self,
                           tag: Union[Tuple, None],
                           query: str = None,
                           ) -> str:
        """Construct a filename with given scheme.

        Args:
            tag: a tuple giving the present position of the filename (tuple).
            query: from url query (str).
        Returns:
            A filename compatible with the given archive (str).
        """
        if query is None:
            query = self.fallback
        # ext = self._get_extension(query)
        if tag is None:
            tag = (0,)
            if self.level:
                tag = tuple(0 for _ in range(self.level))
        if '%' in query:
            query = query % tag
        else:
            query = query.format(*tag)
        return query

    def new_local_file(self,
                       filename: str) -> Member:
        """Create new local file.

        Args:
            filename: Preferred filename (str)
        Returns:
            member object (Member). The local_file property has the local filename.
        """
        member = Member(filename)
        if self._filehandle_in_files(member):
            raise FileExistsError('File {} already exists')
        # member.fh = tempfile.NamedTemporaryFile(delete=False)
        # member.local_file = member.fh.name
        suffix = None
        ext = filename.find('.')
        if ext >= 0:
            suffix = filename[ext:]
        member.fh, member.local_file = tempfile.mkstemp(suffix=suffix)
        member.info['unpacked'] = True
        self.__files[member.filename] = member
        return WriteFileIO(self.__archive, member, member.local_file)

    def to_localfile(self, member):
        """Access a member object through a local file.

        Args:
            member: handle to member file.
        Returns:
            filename to file guaranteed to be local.
        Raises:
            FileNotFoundError: when file is not found.
        """
        if not self._filehandle_in_files(member):
            raise FileNotFoundError('No such file: {}'.format(member.filename))
        if not member.info['unpacked']:
            member.local_file = self.__archive.extract(member.filename)
            member.info['unpacked'] = True
            self.__files[member.filename] = member
        return member.local_file

    def add_localfile(self, local_file, filename):
        """Add a local file to the archive.

        Args:
            local_file: named local file
            filename: filename in the archive
        """
        _name: str = '{}.{}'.format(__name__, self.add_localfile.__name__)
        if self.__mode[0] == 'r':
            raise PermissionError(
                'Cannot write on an archive opened for read')
        member = Member(filename, info={'unpacked': True},
                        local_file=local_file)
        self.__archive.write(local_file, arcname=filename)
        logger.debug('{}: local {} as {}'.format(
            _name, local_file, filename))
        self.__files[filename] = member
        logger.debug('{}: {}'.format(_name, self.__archive.namelist()))

    def writedata(self, filename, data):
        """Write data to a named file in the archive.

        Args:
            filename: named file in the archive
            data: data to write
        """
        if self.__mode[0] == 'r':
            raise PermissionError(
                'Cannot write on an archive opened for read')
        member = Member(filename, info={'unpacked': False})
        self.__archive.writestr(filename, data)
        self.__files[filename] = member

    def close(self):
        """Close zip file.
        """
        _name: str = '{}.{}'.format(__name__, self.close.__name__)
        self.__archive.close()
        self.__fp.close()
        shutil.rmtree(self.__tmpdir)
        logger.debug('{}: {}'.format(_name, self.__tmpdir))
        self.transport.close()

    def is_file(self, member):
        """Determine whether the named file is a single file.

        Args:
            member: file member

        Returns:
            whether named file is a single file (bool)
        """
        return member.filename in self.__files and self._filehandle_in_files(member)

    def exists(self, member):
        """Determine whether the named path exists.

        Args:
            member: member name.
        Returns:
            whether member exists (bool)
        """
        return member.filename in self.__files

    @property
    def root(self) -> str:
        """Archive root name.
        """
        return os.path.sep

    @property
    def base(self) -> str:
        """Archive base name.
        """
        return None

    @property
    def path(self) -> str:
        """Archive path.
        """
        return ''

    def __enter__(self):
        """Enter context manager.
        """
        _name: str = '{}.{}'.format(__name__, self.__enter__.__name__)
        logger.debug("{}: {} mode {}".format(
            _name, type(self.transport), self.__mode))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Leave context manager, cleaning up any open files.
        """
        _name: str = '{}.{}'.format(__name__, self.__exit__.__name__)
        logger.debug('{}:'.format(_name))
        self.close()
