"""Read/write files in xnat database
"""

# Copyright (c) 2021 Erling Andersen, Haukeland University Hospital, Bergen, Norway

import os
import os.path
import io
import fnmatch
import logging
import shutil
import tempfile
import urllib
import xnat
from imagedata.transports.abstracttransport import AbstractTransport


class XnatTransport(AbstractTransport):
    """Read/write files in xnat database.
    """

    name = "xnat"
    description = "Read and write files in xnat database."
    authors = "Erling Andersen"
    version = "2.0.0"
    url = "www.helse-bergen.no"
    schemes = ["xnat"]

    def __init__(self, netloc=None, root=None, mode='r', read_directory_only=False, opts=None):
        super(XnatTransport, self).__init__(self.name, self.description,
                                            self.authors, self.version, self.url, self.schemes)
        if opts is None:
            opts = {}
        self.read_directory_only = read_directory_only
        self.netloc = netloc
        self.opts = opts
        logging.debug("XnatTransport __init__ root: {}".format(root))
        root_split = root.split('/')
        try:
            project = root_split[1]
        except IndexError:
            raise ValueError('Too few values in URL {}'.format(root))
        subject = root_split[2] if len(root_split) > 2 else None
        experiment = root_split[3] if len(root_split) > 3 else None
        self.__mode = mode
        self.__local = False
        self.__must_upload = False
        self.__tmpdir = None

        self.__session = xnat.connect('https://'+self.netloc, verify=False)
        logging.debug("XnatTransport __init__ session: {}".format(self.__session))
        self.__project = self.__session.projects[project] if project is not None else None
        self.root = '/' + project
        logging.debug("XnatTransport __init__ project: {}".format(self.__project))

        self.__subject = self.__project.subjects[subject] if subject is not None else None
        if subject is not None:
            self.root += '/' + subject
            logging.debug("Subject: {}".format(self.__subject.label))
        self.__experiment = self.__subject.experiments[experiment] if experiment is not None else None
        if experiment is not None:
            self.root += '/' + experiment
            logging.debug("Experiment: {}".format(experiment))

    def close(self):
        """Close the transport
        """
        if self.__must_upload:
            # Upload zip file to xnat
            logging.debug("Upload to {}".format(self.__subject.label))
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
        if top[0] != '/':
            # Add self.root to relative tree top
            top = self.root + '/' + top
        url_tuple = urllib.parse.urlsplit(top)
        url = url_tuple.path.split('/')
        if len(url) < 3:
            raise ValueError('Too few terms in directory tree {}'.format(top))
        elif len(url) == 3:
            # Walk the subject list
            subject_search = url[2]
            for id in self.__project.subjects:
                subject = self.__project.subjects[id]
                if fnmatch.fnmatch(subject.label, subject_search):
                    yield '/{}'.format(self.__project.id), [subject.label], []
        elif len(url) == 4:
            # Walk the experiment list
            subject_id, experiment_search = url[2:4]
            subject = self.__project.subjects[subject_id]
            experiments = self._get_experiments(subject, experiment_search)
            for experiment in experiments:
                yield '/{}/{}'.format(self.__project.id, subject_id), [experiment], []
        elif len(url) == 5:
            # Walk the scan list
            subject_id, experiment_search, scan_search = url[2:5]
            subject = self.__project.subjects[subject_id]
            experiments = self._get_experiments(subject, experiment_search)
            for experiment_label in experiments:
                experiment = subject.experiments[experiment_label]
                scans = self._get_scans(experiment, scan_search)
                yield '/{}/{}/{}'.format(self.__project.id, subject_id, experiment_label), scans, []

    def _get_experiments(self, subject, search):
        experiments = []
        for id in subject.experiments:
            experiment = subject.experiments[id]
            if fnmatch.fnmatch(experiment.label, search):
                experiments.append(experiment.label)
        return experiments

    def _get_scans(self, experiment, search):
        scans = []
        for id in experiment.scans:
            scan = experiment.scans[id]
            if scan.quality == 'usable' and fnmatch.fnmatch(scan.type, search):
                scans.append(scan.id)
        return scans

    def isfile(self, path):
        """Return True if path is an existing regular file.
        """
        pass

    def open(self, path, mode='r'):
        """Extract a member from the archive as a file-like object.
        """
        if mode[0] == 'r' and not self.__local:
            scan_id = path.split('/')[4]
            if scan_id in self.__experiment.scans:
                scans = [self.__experiment.scans[scan_id]]
            else:
                scans = []
                for id in self.__experiment.scans:
                    scan = self.__experiment.scans[id]
                    if scan.quality == 'usable' and fnmatch.fnmatch(scan.type, scan_id):
                        scans.append(scan)
            if len(scans) == 0:
                raise FileNotFoundError('Scan {} not found.'.format(scan_id))
            elif len(scans) > 1:
                raise ValueError('Too many scans match search "{}": {}'.format(
                    scan_id, scans
                ))
            scan = scans[0]
            self.__tmpdir = tempfile.mkdtemp()
            if scan.quality == 'usable':
                # scan.download_dir(self.__tmpdir)
                self.__zipfile = os.path.join(self.__tmpdir, 'scan.zip')
                scan.download(self.__zipfile)
                self.__local = True
        elif mode[0] == 'w' and not self.__local:
            self.__tmpdir = tempfile.mkdtemp()
            self.__zipfile = os.path.join(self.__tmpdir, 'upload.zip')
            self.__local = True
            self.__must_upload = True
        if self.__local:
            return io.FileIO(self.__zipfile, mode)
        else:
            raise IOError('Could not download scan {}'.format(scan_id))
