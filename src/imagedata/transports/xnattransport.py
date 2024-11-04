"""Read/write files in xnat database
"""

# Copyright (c) 2021-2024 Erling Andersen, Haukeland University Hospital, Bergen, Norway

from typing import List, Optional
import os
import os.path
import io
import fnmatch
import logging
import shutil
from zipfile import ZipFile
import tempfile
import urllib.parse
import xnat
from .abstracttransport import AbstractTransport
from . import FunctionNotSupported

logger = logging.getLogger(__name__)


class XnatTransport(AbstractTransport):
    """Read/write files in xnat database.
    """

    name: str = "xnat"
    description: str = "Read and write files in xnat database."
    authors: str = "Erling Andersen"
    version: str = "2.0.0"
    url: str = "www.helse-bergen.no"
    schemes: List[str] = ["xnat"]
    mimetype: str = "application/zip"  # Determines archive plugin
    read_directory_only: bool = None
    opts: [dict] = None
    netloc: str = None
    __root: str = None
    __mode: str = None
    __local: bool = False
    __must_upload: bool = False
    __tmpdir: [str] = None
    __session: xnat.XNATSession = None
    __project: xnat.session.XNATSession = None
    __subject = None
    __experiment = None
    __scan = None

    def __init__(self,
                 netloc: Optional[str] = None,
                 root: Optional[str] = None,
                 mode: Optional[str] = 'r',
                 read_directory_only: Optional[bool] = False,
                 opts: Optional[dict] = None):
        _name: str = '{}.{}'.format(__name__, self.__init__.__name__)
        super(XnatTransport, self).__init__(self.name, self.description,
                                            self.authors, self.version, self.url, self.schemes)
        if opts is None:
            opts = {}
        self.read_directory_only = read_directory_only
        self.opts = opts
        # Does netloc include username and password?
        if '@' in netloc:
            # Add fake scheme to satisfy urlsplit()
            url_tuple = urllib.parse.urlsplit('xnat://' + netloc)
            self.netloc = url_tuple.hostname
            try:
                opts['username'] = url_tuple.username
            except AttributeError:
                opts['username'] = None
            try:
                opts['password'] = url_tuple.password
            except AttributeError:
                opts['password'] = None
        else:
            self.netloc = netloc
        logger.debug("{}: root: {}".format(_name, root))
        root_split = root.split('/')
        try:
            project = root_split[1]
        except IndexError:
            raise ValueError('No project given in URL {}'.format(root))
        subject = root_split[2] if len(root_split) > 2 and len(root_split[2]) else None
        experiment = root_split[3] if len(root_split) > 3 and len(root_split[3]) else None
        scan = root_split[4] if len(root_split) > 4 and len(root_split[4]) else None
        self.__mode = mode
        self.__local = False
        self.__must_upload = False
        self.__tmpdir = None

        kwargs = {'verify': False}
        if 'username' in opts:
            kwargs['user'] = opts['username']
        if 'password' in opts:
            kwargs['password'] = opts['password']
        self.__session = xnat.connect('https://' + self.netloc, **kwargs)
        logger.debug("{}: session: {}".format(_name, self.__session))
        self.__project = self.__session.projects[project] if project is not None else None
        self.__root = '/' + project
        logger.debug("{}: project: {}".format(_name, self.__project))

        self.__subject = self.__project.subjects[subject] if subject is not None else None
        if subject is not None:
            self.__root += '/' + subject
            logger.debug("{}: Subject: {}".format(_name, self.__subject.label))
        self.__experiment = self.__subject.experiments[experiment]\
            if experiment is not None else None
        if experiment is not None:
            self.__root += '/' + experiment
            logger.debug("{}: Experiment: {}".format(_name, experiment))
        if mode == 'r':
            self.__scan = None
            if scan is not None:
                scans, labels = self._get_scans(self.__experiment, scan)
                if len(scans) == len(labels) == 1:
                    self.__scan = self.__experiment.scans[labels[0]] \
                        if labels[0] is not None else None
        else:
            self.__scan = None
        if scan is not None:
            self.__root += '/' + scan
            logger.debug("{}: Scan: {}".format(_name, scan))

    def close(self):
        """Close the transport
        """
        _name: str = '{}.{}'.format(__name__, self.close.__name__)
        if self.__must_upload:
            # Upload zip file to xnat
            logger.debug("{}: Upload to {}".format(_name, self.__subject.label))
            self.__session.services.import_(self.__zipfile,
                                            project=self.__project.id,
                                            subject=self.__subject.id,
                                            experiment=self.__experiment.label,
                                            trigger_pipelines=False,
                                            overwrite='delete')
        if self.__tmpdir is not None:
            shutil.rmtree(self.__tmpdir)

    def walk(self, top):
        """Generate the file names in a directory tree by walking the tree.
        Input:
        - top: starting point for walk (str)
        Return:
        - tuples of (root, dirs, files)
        """
        _name: str = '{}.{}'.format(__name__, self.walk.__name__)
        logger.debug('{}: root {}, top {}'.format(_name, self.__root, top))
        if len(top) < 1 or top[0] != '/':
            # Add self.__root to relative tree top
            top = self.__root + '/' + top
            logger.debug('{}: new top {}'.format(_name, top))
        url_tuple = urllib.parse.urlsplit(top)
        url = url_tuple.path.split('/')
        subject_search = url[2] if len(url) >= 3 else None
        experiment_search = url[3] if len(url) >= 4 else None
        scan_search = url[4] if len(url) >= 5 else None
        instance_search = url[5] if len(url) >= 6 else None
        logger.debug('{}: subject_search {}'.format(_name, subject_search))
        logger.debug('{}: experiment_search {}'.format(_name, experiment_search))
        logger.debug('{}: scan_search {}'.format(_name, scan_search))
        logger.debug('{}: instance_search {}'.format(_name, instance_search))

        # Walk the patient list
        subjects, labels = self._get_subjects(subject_search)
        if experiment_search is None:
            yield '/{}'.format(self.__project.id), labels, []
        for subject in subjects:
            # Walk the experiment list
            experiments, labels = self._get_experiments(subject, experiment_search)
            if scan_search is None:
                yield '/{}/{}'.format(self.__project.id, subject.label), labels, []
            for experiment in experiments:
                # Walk the scan list
                scans, labels = self._get_scans(experiment, scan_search)
                if instance_search is None:
                    yield '/{}/{}/{}'.format(self.__project.id, subject.label, experiment.id), \
                          labels, []
                for scan in scans:
                    # Walk the file list
                    files = self._get_files(scan, instance_search)
                    yield '/{}/{}/{}/{}'.format(self.__project.id, subject.label,
                                                experiment.id, scan.id), \
                          [], files

    def _get_subjects(self, search):
        if len(search) < 1:
            search = '*'
        subjects = []
        labels = []
        for id in self.__project.subjects:
            subject = self.__project.subjects[id]
            if fnmatch.fnmatch(subject.label, search) or fnmatch.fnmatch(subject.id, search):
                subjects.append(subject)
                labels.append(subject.label)
        return subjects, sorted(labels)

    def _get_experiments(self, subject, search):
        if search is None or len(search) < 1:
            search = '*'
        experiments = []
        labels = []
        for id in subject.experiments:
            experiment = subject.experiments[id]
            if fnmatch.fnmatch(experiment.id, search) or fnmatch.fnmatch(experiment.label, search):
                experiments.append(experiment)
                labels.append(experiment.id)
        return experiments, sorted(labels)

    def _get_scans(self, experiment, search):
        if search is None or len(search) < 1:
            search = '*'
        scans = []
        labels = []
        for id in experiment.scans:
            scan = experiment.scans[id]
            if scan.quality == 'usable' and (
                    fnmatch.fnmatch(scan.type, search) or fnmatch.fnmatch(scan.id, search)):
                scans.append(scan)
                labels.append(scan.id)
        try:
            return scans, sorted(labels, key=int)
        except Exception:
            return scans, sorted(labels)

    def _get_files(self, scan, search):
        if search is None or len(search) < 1:
            search = '*'
        files = []
        for filename in scan.files:
            if fnmatch.fnmatch(filename, search):
                files.append(filename)
        return files

    def isfile(self, path):
        """Return True if path is an existing regular file.
        """
        raise FunctionNotSupported('Accessing the XNAT server is not supported.')

    def exists(self, path):
        """Return True if the named path exists.
        """
        return False

    def _search_subjects(self, path):
        """Search for subject(s) from the archive.
        """
        _name: str = '{}.{}'.format(__name__, self._search_subjects.__name__)
        assert len(path.split('/')) == 3, "{} with wrong level".format(_name)
        subject_list = []
        subject_id = path.split('/')[-1]
        for id in self.__project.subjects:
            logger.debug('{}: locate subject id {}'.format(_name, id))
            subject = self.__project.subjects[id]
            if fnmatch.fnmatch(subject.label, subject_id):
                subject_list.append(subject)
        object_list = []
        for subject in subject_list:
            object_list += self._search_experiments(path + '/*')
        return object_list

    def _search_experiments(self, path):
        """Search for experiment(s) from the archive.
        """
        _name: str = '{}.{}'.format(__name__, self._search_experiments.__name__)
        assert len(path.split('/')) == 4, "{} with wrong level".format(_name)
        object_list = []
        experiment_id = path.split('/')[-1]
        for id in self.__subject.experiments:
            logger.debug('{}: locate experiment id {}'.format(_name, id))
            experiment = self.__subject.experiments[id]
            if fnmatch.fnmatch(experiment.id, experiment_id):
                object_list.append(experiment)
        return object_list

    def _search_scans(self, path):
        """Search for scan(s) from the archive.
        """
        _name: str = '{}.{}'.format(__name__, self._search_scans.__name__)
        assert len(path.split('/')) == 5, "{} with wrong level".format(_name)
        object_list = []
        scan_id = path.split('/')[-1]
        for id in self.__experiment.scans:
            logger.debug('{}: locate scan id {}'.format(_name, id))
            scan = self.__experiment.scans[id]
            logger.debug('{}: locate scan series description {}'.format(
                _name, scan.series_description))
            # if scan.quality == 'usable' and fnmatch.fnmatch(scan.id, scan_id):
            if scan.quality == 'usable' and fnmatch.fnmatch(scan.series_description, scan_id):
                object_list.append(scan)
        return object_list

    def open(self, path, mode='r'):
        """Extract a member from the archive as a file-like object.
        """
        def _combine_zipfiles(orig, temp):
            """Move contents of temp into orig
            """
            z1 = ZipFile(orig, 'a')
            z2 = ZipFile(temp, 'r')
            [z1.writestr(t[0], t[1].read()) for t in ((n, z2.open(n)) for n in z2.namelist())]
            z1.close()
            z2.close()

        _name: str = '{}.{}'.format(__name__, self.open.__name__)
        logger.debug('{}: path "{}", mode {}'.format(_name, path, mode))
        if mode[0] == 'r' and not self.__local:
            level = len(path.split('/'))
            if level == 3:
                object_list = self._search_subjects(path)
            elif level == 4:
                object_list = self._search_experiments(path)
            elif level == 5:
                object_list = self._search_scans(path)
            if len(object_list) == 0:
                raise FileNotFoundError('File {} not found.'.format(path))
            self.__tmpdir = tempfile.mkdtemp()
            self.__zipfile = os.path.join(self.__tmpdir, 'scan.zip')
            self.__ziptemp = os.path.join(self.__tmpdir, 'temp.zip')
            for i, _object in enumerate(object_list):
                # scan.download_dir(self.__tmpdir)
                if i == 0:
                    _object.download(self.__zipfile)
                else:
                    _object.download(self.__ziptemp)
                    _combine_zipfiles(self.__zipfile, self.__ziptemp)
                    os.remove(self.__ziptemp)
            self.__local = True
        elif mode[0] == 'w' and not self.__local:
            self.__tmpdir = tempfile.mkdtemp()
            self.__zipfile = os.path.join(self.__tmpdir, 'upload.zip')
            self.__local = True
            self.__must_upload = True
        if self.__local:
            return io.FileIO(self.__zipfile, mode)
        else:
            raise IOError('Could not download scan {}'.format(path))

    def info(self, path) -> str:
        """Return info describing the object

        Args:
            path (str): object path

        Returns:
            description (str): Preferably a one-line string describing the object
        """
        if path[0] != '/':
            # Add self.__root to relative path
            path = self.__root + '/' + path
        url_tuple = urllib.parse.urlsplit(path)
        url = url_tuple.path.split('/')
        if len(url) < 2:
            raise ValueError('Too few terms in directory tree {}'.format(path))
        elif len(url) == 2:
            # Describe project
            return self.__project
        elif len(url) == 3:
            # Describe subject
            subject_id = url[2]
            subject = self.__project.subjects[subject_id]
            if len(subject.experiments) == 1:
                exp_str = 'experiment'
            else:
                exp_str = 'experiments'
            return '{}, {} {}'.format(subject.label, len(subject.experiments), exp_str)
        elif len(url) == 4:
            # Describe experiment
            subject_id, experiment_label = url[2:4]
            subject = self.__project.subjects[subject_id]
            experiment = subject.experiments[experiment_label]
            scan_str = 'scan' if len(experiment.scans) == 1 else 'scans'
            return '{} {} {}, {}, {} {}'.format(
                experiment.id, experiment.date, experiment.time, experiment.modality,
                len(experiment.scans), scan_str
            )
        elif len(url) == 5:
            # Describe scan
            subject_id, experiment_label, scan_id = url[2:5]
            subject = self.__project.subjects[subject_id]
            experiment = subject.experiments[experiment_label]
            scan = experiment.scans[scan_id]
            frame_str = 'frame' if scan.frames == 1 else 'frames'
            return '{} {} {} {}'.format(scan.id, scan.series_description, scan.frames, frame_str)
        elif len(url) == 6:
            # Describe file
            subject_id, experiment_label, scan_id, file_id = url[2:6]
            subject = self.__project.subjects[subject_id]
            experiment = subject.experiments[experiment_label]
            scan = experiment.scans[scan_id]
            file_descriptor = scan.files[file_id]
            return '{}, {}, {}'.format(
                file_descriptor.file_format,
                file_descriptor.collection,
                file_descriptor.file_size
            )
