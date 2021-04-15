"""Read/write files in xnat database
"""

# Copyright (c) 2021 Erling Andersen, Haukeland University Hospital, Bergen, Norway

import os
import os.path
import io
import logging
import shutil
import tempfile
import xnat
from imagedata.transports.abstracttransport import AbstractTransport


class XnatTransport(AbstractTransport):
    """Read/write files in xnat database.
    """

    name = "xnat"
    description = "Read and write files in xnat database."
    authors = "Erling Andersen"
    version = "1.0.0"
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
        project, subject, experiment = root.split('/')[1:4]
        self.__mode = mode
        self.__local = False
        self.__must_upload = False
        self.__tmpdir = None

        self.__session = xnat.connect('https://'+self.netloc, verify=False)
        # self.__session = xnat.connect(self.__root, verify=False)
        logging.debug("XnatTransport __init__ session: {}".format(self.__session))
        self.__project = self.__session.projects[project]
        logging.debug("XnatTransport __init__ project: {}".format(self.__project))

        self.__subject = self.__project.subjects[subject]
        logging.debug("Subject: {}".format(self.__subject.label))
        self.__experiment = self.__subject.experiments[experiment]
        logging.debug("Experiment: {}".format(experiment))

    def close(self):
        """Close the transport
        """
        if self.__must_upload:
            # Upload zip file to xnat
            logging.debug("Upload to {}".format(self.__subject.label))
            self.__session.services.import_(self.__zipfile,
                                            project=self.__project,
                                            subject=self.__subject.label,
                                            experiment=self.__experiment,
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
        scan_id = top.split('/')[4]

        filelist = []
        scan = self.__experiment.scans[scan_id]
        logging.debug("Scan: {}".format(scan))
        # for file in scan.files:
        #     filelist.append(file)
        # return [(top, scan_id, filelist)]
        return [(top, scan_id, scan.files)]

    def isfile(self, path):
        """Return True if path is an existing regular file.
        """
        pass

    def open(self, path, mode='r'):
        """Extract a member from the archive as a file-like object.
        """
        scan_id = path.split('/')[4]
        scan = self.__experiment.scans[scan_id]
        if mode[0] == 'r' and not self.__local:
            self.__tmpdir = tempfile.mkdtemp()
            if scan.quality == 'usable':
                # scan.download_dir(self.__tmpdir)
                self.__zipfile = os.path.join(self.__tmpdir, 'scan.zip')
                scan.download(self.__zipfile)
                self.__local = True
        elif mode[0] == 'w' and not self.__local:
            pass
            self.__must_upload = True
        if self.__local:
            return io.FileIO(self.__zipfile, mode)
        else:
            raise IOError('Could not download scan {}'.format(scan_id))
