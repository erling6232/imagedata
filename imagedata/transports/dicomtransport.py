"""Send DICOM images to DICOM Storage SCP
"""

# Copyright (c) 2019 Erling Andersen, Haukeland University Hospital, Bergen, Norway

import os, os.path, io
import sys, platform
import logging
import pydicom, pynetdicom
from imagedata.transports.abstracttransport import AbstractTransport
from imagedata.transports import RootIsNotDirectory, RootDoesNotExist, \
            FunctionNotSupported

class AssociationNotEstablished(Exception): pass
class AssociationFailed(Exception): pass
class AETitleNotGiven(Exception): pass

class DicomTransport(AbstractTransport):
    """Send DICOM images to DICOM Storage SCP
    """

    name = "dicom"
    description = "Send DICOM images to DICOM Storage SCP."
    authors = "Erling Andersen"
    version = "1.0.0"
    url = "www.helse-bergen.no"
    schemes = ["dicom"]

    def __init__(self, netloc=None, root=None, mode='r', read_directory_only=False, opts={}):
        super(DicomTransport, self).__init__(self.name, self.description,
            self.authors, self.version, self.url, self.schemes)
        logging.debug("DicomTransport __init__ root: {} ({})".format(root, mode))
        if mode[0] == 'r':
            raise FunctionNotSupported('DICOM receive is not supported.')
        if len(root) < 1:
            raise AETitleNotGiven('AE Title not given')
        if root[0] == '/':
            root = root[1:]
        self.__root = root
        self.__mode = mode
        # Open DICOM Storage Association as SCU
        if 'calling_aet' in opts and opts['calling_aet'] is not None:
            localAE = opts['calling_aet']
        else:
            try:
                hostname = platform.node()
                localAE = hostname.split('.')[0]
            except Exception:
                    localAE = 'IMAGEDATA'
        logging.debug("DicomTransport __init__ calling AET: {}".format(localAE))
        self.__ae = pynetdicom.AE(ae_title=localAE)
        self.__ae.requested_contexts = \
            pynetdicom.StoragePresentationContexts
        host, port = netloc.split(':')
        port = int(port)
        self.__assoc = self.__ae.associate(host, port, ae_title=root)
        if self.__assoc.is_established:
            return
        else:
            raise AssociationNotEstablished(
                'Association rejected, aborted or never connected')

    def close(self):
        """Close the DICOM association transport
        """
        self.__assoc.release()

    def walk(self, top):
        """Generate the file names in a directory tree by walking the tree.
        Input:
        - top: starting point for walk (str)
        Return:
        - tuples of (root, dirs, files) 
        """
        #raise FunctionNotSupported('Walking the DICOM server is not supported.')
        return []

    def isfile(self, path):
        """Return True if path is an existing regular file.
        """
        raise FunctionNotSupported('Accessing the DICOM server is not supported.')

    def open(self, path, mode='r'):
        """Extract a member from the archive as a file-like object.
        """
        raise FunctionNotSupported('Open the DICOM server is not supported.')

    def store(self, ds):
        """Store DICOM dataset using DICOM Storage SCU protocol.
        """
        logging.debug('DicomTransport.store: send dataset')
        status = self.__assoc.send_c_store(ds)
        if status:
            logging.debug('DicomTransport.store: C-STORE request status: 0x{0:04x}'.format(status.Status))
        else:
            raise AssociationFailed('C-STORE request status: 0x{0:04x}'.format(status.Status))
