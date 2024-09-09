"""Transfer DICOM images to and from DICOM Storage SCP
"""

# Copyright (c) 2019-2024 Erling Andersen, Haukeland University Hospital, Bergen, Norway

from typing import Optional
import platform
import urllib
import logging
import pydicom
import pynetdicom
from .abstracttransport import AbstractTransport
from . import FunctionNotSupported

logger = logging.getLogger(__name__)

presentation_contexts = [pynetdicom.sop_class.CTImageStorage, pynetdicom.sop_class.MRImageStorage,
                         pynetdicom.sop_class.ComputedRadiographyImageStorage,
                         pynetdicom.sop_class.DigitalXRayImagePresentationStorage,
                         pynetdicom.sop_class.DigitalXRayImageProcessingStorage,
                         pynetdicom.sop_class.DigitalMammographyXRayImagePresentationStorage,
                         pynetdicom.sop_class.DigitalMammographyXRayImageProcessingStorage,
                         pynetdicom.sop_class.UltrasoundMultiframeImageStorage,
                         pynetdicom.sop_class.UltrasoundImageStorage,
                         pynetdicom.sop_class.SecondaryCaptureImageStorage,
                         pynetdicom.sop_class.XRayAngiographicImageStorage,
                         pynetdicom.sop_class.XRayRadiofluoroscopicImageStorage,
                         pynetdicom.sop_class.NuclearMedicineImageStorage,
                         pynetdicom.sop_class.ParametricMapStorage
                         ]


# Future enhancements:
# EnhancedCTImageStorage,LegacyConvertedEnhancedCTImageStorage,
# EnhancedMRImageStorage, EnhancedMRColorImageStorage, LegacyConvertedEnhancedMRImageStorage,
# EnhancedUSVolumeStorage,
# MultiFrameSingleBitSecondaryCaptureImageStorage,
# MultiFrameGrayscaleByteSecondaryCaptureImageStorage,
# MultiFrameGrayscaleWordSecondaryCaptureImageStorage,
# MultiFrameTrueColorSecondaryCaptureImageStorage,
# StandaloneOverlayStorage, StandaloneCurveStorage,
# EnhancedXAImageStorage, EnhancedXRFImageStorage, XRayAngiographicBiPlaneImageStorage,
# XRay3DAngiographicImageStorage, XRay3DCraniofacialImageStorage, BreastTomosynthesisImageStorage,
# SpatialRegistrationStorage, SpatialFiducialsStorage, DeformableSpatialRegistrationStorage,
# SegmentationStorage, SurfaceSegmentationStorage, TractographyResultsStorage,
# PositronEmissionTomographyImageStorage, LegacyConvertedEnhancedPETImageStorage,
# StandalonePETCurveStorage, EnhancedPETImageStorage,
# RTImageStorage,


class AssociationNotEstablished(Exception):
    pass


class AssociationFailed(Exception):
    pass


class DicomTransport(AbstractTransport):
    """Send DICOM images to DICOM Storage SCP
    """

    name = "dicom"
    description = "Receive/Send DICOM images to DICOM Storage SCP."
    authors = "Erling Andersen"
    version = "2.0.0"
    url = "www.helse-bergen.no"
    schemes = ["dicom"]

    __catalog = {}

    def __init__(self,
                 netloc: Optional[str] = None,
                 root: Optional[str] = None,
                 mode: Optional[str] = 'r',
                 read_directory_only: Optional[bool] = False,
                 opts: Optional[dict] = None):
        _name: str = '{}.{}'.format(__name__, self.__init__.__name__)
        super(DicomTransport, self).__init__(self.name, self.description,
                                             self.authors, self.version, self.url, self.schemes)
        if opts is None:
            opts = {}
        self.read_directory_only = read_directory_only
        logger.debug("{}: root: {} ({})".format(_name, root, mode))
        try:
            root_split = root.split('/')
            aet = root_split[1]
            if len(aet) < 1:
                raise ValueError('AE Title not given')
        except IndexError:
            raise ValueError('AE Title not given')
        self.__root = root
        self.__mode = mode
        self.__aet = aet
        self.__local = False
        self.__files = {}
        if len(root_split) == 2:
            self.patID, self.study_or_accession, self.series = None, None, None
        elif len(root_split) == 3:
            self.patID, self.study_or_accession, self.series = None, None, root_split[2]
        elif len(root_split) == 4:
            self.patID, self.study_or_accession, self.series = None, None, root_split[2]
        elif len(root_split) == 5:
            self.patID, self.study_or_accession, self.series = root_split[2:]
        else:
            raise ValueError('Wrong URL {}'.format(root))
        # Open DICOM Storage Association as SCU
        if 'calling_aet' in opts and opts['calling_aet'] is not None:
            self.__local_aet = opts['calling_aet']
        else:
            try:
                hostname = platform.node()
                self.__local_aet = hostname.split('.')[0]
            except IndexError:
                self.__local_aet = 'IMAGEDATA'
        logger.debug("{}: calling AET: {}".format(_name, self.__local_aet))
        self.__ae = pynetdicom.AE(ae_title=self.__local_aet)
        self.__ae.requested_contexts = pynetdicom.presentation.QueryRetrievePresentationContexts
        # self.__ae.requested_contexts = [
        #    pynetdicom.presentation.QueryRetrievePresentationContexts,
        #    pynetdicom.presentation.StoragePresentationContexts.MRImageStorage,
        #    pynetdicom.presentation.StoragePresentationContexts.CTImageStorage,
        #    ]
        #    pynetdicom.StoragePresentationContexts
        for context in [pynetdicom.sop_class.MRImageStorage, pynetdicom.sop_class.CTImageStorage]:
            self.__ae.add_requested_context(context)
        # self.__ae.add_requested_context(pynetdicom.sop_class.PatientRootQueryRetrieveInformationModelFind)
        # self.__ae.add_requested_context(pynetdicom.sop_class.StudyRootQueryRetrieveInformationModelFind)
        self.__host, port = netloc.split(':')
        self.__port = int(port)
        self.__assoc = self.__ae.associate(self.__host, self.__port, ae_title=self.__aet)
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
        if self.__mode[0] == 'w':
            # Do not search the DICOM archive on write
            return

        if top[0] != '/':
            # Add self.root to relative tree top
            top = self.__root + '/' + top
        url_tuple = urllib.parse.urlsplit(top)
        url = url_tuple.path.split('/')
        patient_search = url[2] if len(url) >= 3 else None
        study_search = url[3] if len(url) >= 4 else None
        series_search = url[4] if len(url) >= 5 else None
        instance_search = url[5] if len(url) >= 6 else None

        # Walk the patient list, should be one only
        if patient_search is None:
            # Do not allow to search multiple patients - could result in too many matches
            raise ValueError('Patient ID must be provided: {}'.format(top))
        patients = self._cfind_patient(patient_search)
        if len(patients) < 1:
            raise ValueError('Patient ID {} not found'.format(patient_search))
        elif len(patients) > 1:
            raise ValueError('Patient ID {} match multiple patients'.format(patient_search))
        patient_id = patients[0]
        if study_search is None:
            yield '/{}'.format(self.__aet), [patient_id], []

        # Walk the study list
        studies = self._cfind_studies(patient_id, study_search)
        # catalog = {}
        if series_search is None:
            yield '/{}/{}'.format(self.__aet, patient_id), studies, []
        for study_instance_uid in studies:
            # catalog[study_instance_uid] = {}
            # Walk the series list
            series = self._cfind_series(study_instance_uid, series_search)
            if instance_search is None:
                yield '/{}/{}/{}'.format(self.__aet, patient_id, study_instance_uid), series, []

            for series_instance_uid in series:
                # catalog[study_instance_uid][series_instance_uid] = {}
                # Walk the instance list
                instances = self._cfind_instances(study_instance_uid, series_instance_uid,
                                                  instance_search)
                yield '/{}/{}/{}/{}'.format(self.__aet, patient_id, study_instance_uid,
                                            series_instance_uid), [], instances

    def isfile(self, path):
        """Return True if path is an existing regular file.
        """
        raise FunctionNotSupported('Accessing the DICOM server is not supported.')

    def exists(self, path):
        """Determine whether the named path exists.
        """
        return False

    def open(self, path, mode='r'):
        """Extract a member from the archive as a file-like object.
        """
        # raise FunctionNotSupported('Open the DICOM server is not supported.')
        if mode[0] == 'r':
            study_instance_uid, series_instance_uid, sop_instance_uid = path.split('/')[3:6]
            if sop_instance_uid not in self.__files:
                self._cget_series('/tmp', study_instance_uid, series_instance_uid)
            if sop_instance_uid not in self.__files:
                raise FileNotFoundError('File not found: {}'.format(path))
            return self.__files[sop_instance_uid]

    def store(self, ds):
        """Store DICOM dataset using DICOM Storage SCU protocol.
        """
        _name: str = '{}.{}'.format(__name__, self.store.__name__)
        logger.debug('{}: send dataset'.format(_name))
        status = self.__assoc.send_c_store(ds)
        if status:
            logger.debug('{}: C-STORE request status: '
                         '0x{:04x}'.format(_name, status.Status))
        else:
            raise AssociationFailed('C-STORE request status: 0x{0:04x}'.format(status.Status))

    def info(self, path) -> str:
        """Return info describing the object

        Args:
            path (str): object path

        Returns:
            description (str): Preferably a one-line string describing the object
        """
        if path[0] != '/':
            # Add self.root to relative path
            path = self.root + '/' + path
        url_tuple = urllib.parse.urlsplit(path)
        url = url_tuple.path.split('/')
        if len(url) < 2:
            raise ValueError('Too few terms in directory tree {}'.format(path))
        elif len(url) == 2:
            # Describe AET
            return url[1]
        elif len(url) == 3:
            # Describe patient
            return url[2]
        elif len(url) == 4:
            # Describe study
            patient_id, study_instance_uid = url[2:]
            if study_instance_uid in self.__catalog:
                study = self.__catalog[study_instance_uid]
                study_date = study.StudyDate
                if len(study_date) == 8:
                    study_date = '{}.{}.{}'.format(
                        study_date[:4], study_date[4:6], study_date[6:8])
                study_time = study.StudyTime
                if len(study_time) == 6:
                    study_time = '{}:{}:{}'.format(
                        study_time[:2], study_time[2:4], study_time[4:6])
                return '{} {} {} {}'.format(
                    study_date, study_time, study.AccessionNumber, study.StudyDescription)
            return ''
        elif len(url) == 5:
            # Describe series
            patient_id, study_instance_uid, series_instance_uid = url[2:]
            if series_instance_uid in self.__catalog:
                series = self.__catalog[series_instance_uid]
                try:
                    series_number = int(series.SeriesNumber)
                except AttributeError:
                    series_number = 0
                try:
                    series_description = series.SeriesDescription
                except AttributeError:
                    series_description = ''
                return '#{}: {} {} {}'.format(
                    series_number, series.NumberOfSeriesRelatedInstances,
                    series.Modality, series_description)
            return ''
        elif len(url) == 6:
            # Describe instance
            patient_id, study_instance_uid, series_instance_uid, sop_instance_uid = url[2:]
            if sop_instance_uid in self.__catalog:
                instance = self.__catalog[sop_instance_uid]
                try:
                    return '{} {}x{}x{}'.format(
                        instance.InstanceNumber, instance.NumberOfFrames,
                        instance.Rows, instance.Columns)
                except AttributeError:
                    return ''
            return ''

    # Implement the handler for evt.EVT_C_STORE
    def _handle_store(self, event):
        """Handle a C-STORE request event."""
        ds = event.dataset
        ds.file_meta = event.file_meta

        # Save dataset in self.__files
        sop_instance_uid = ds.SOPInstanceUID
        self.__files[sop_instance_uid] = ds

        # Return a 'Success' status
        return 0x0000

    def _cget_series(self, destdir, study_instance_uid, series_instance_UID):
        handlers = [(pynetdicom.evt.EVT_C_STORE, self._handle_store)]

        # # Initialise the Application Entity
        ae = pynetdicom.AE(ae_title=self.__local_aet)

        ae.add_requested_context(pynetdicom.sop_class.StudyRootQueryRetrieveInformationModelGet)
        # Add the requested presentation contexts (Storage SCP)
        roles = []
        for storage_class in presentation_contexts:
            # Add the requested presentation contexts (QR SCU)
            ae.add_requested_context(storage_class)
            # Create an SCP/SCU Role Selection Negotiation item for CT Image Storage
            roles.append(pynetdicom.build_role(storage_class, scp_role=True))

        # Create our Identifier (query) dataset
        # We need to supply a Unique Key Attribute for each level above the
        #   Query/Retrieve level
        ds = pydicom.dataset.Dataset()
        ds.QueryRetrieveLevel = 'SERIES'
        # Unique key for SERIES level
        ds.SeriesInstanceUID = series_instance_UID

        # Associate with peer AE at IP 127.0.0.1 and port 11112
        assoc = ae.associate(self.__host, self.__port, ae_title=self.__aet,
                             ext_neg=roles, evt_handlers=handlers)

        if assoc.is_established:
            # Use the C-GET service to send the identifier
            responses = assoc.send_c_get(
                ds, pynetdicom.sop_class.StudyRootQueryRetrieveInformationModelGet)
            for (status, identifier) in responses:
                if status:
                    pass
                else:
                    raise ConnectionError(
                        'Connection timed out, was aborted or received invalid response')

            # Release the association
            assoc.release()
        else:
            raise ConnectionError('Association rejected, aborted or never connected')

    def _cfind_patient(self, patient_id):
        # Create our Identifier (query) dataset
        ds = pydicom.dataset.Dataset()
        ds.PatientID = patient_id
        ds.QueryRetrieveLevel = 'PATIENT'
        ds.PatientName = ''
        ds.PatientBirthDate = ''
        ds.PatientSex = ''

        return self._cfind(ds,
                           pynetdicom.sop_class.PatientRootQueryRetrieveInformationModelFind,
                           'PatientID')

    def _cfind_studies(self, patient_id, search):
        instances = []
        # Create our Identifier (query) dataset
        for keyword in 'StudyInstanceUID', 'AccessionNumber', 'StudyDescription':
            tag = pydicom.dataset.tag_for_keyword(keyword)
            ds = pydicom.dataset.Dataset()
            ds.PatientID = patient_id
            ds.StudyInstanceUID = ''
            ds.QueryRetrieveLevel = 'STUDY'
            ds.StudyDate = ''
            ds.StudyTime = ''
            ds.AccessionNumber = ''
            ds.StudyDescription = ''
            ds.NumberOfStudyRelatedSeries = ''
            ds.NumberOfStudyRelatedInstances = ''
            ds[tag] = pydicom.dataset.DataElement(tag, pydicom.datadict.dictionary_VR(tag), search)

            instances2 =\
                self._cfind(ds,
                            pynetdicom.sop_class.PatientRootQueryRetrieveInformationModelFind,
                            'StudyInstanceUID')
            for instance in instances2:
                if instance not in instances:
                    instances.append(instance)
        return instances

    def _cfind_series(self, study_instance_uid, search):
        instances = []
        # Create our Identifier (query) dataset
        for keyword in 'SeriesInstanceUID', 'SeriesNumber':
            tag = pydicom.dataset.tag_for_keyword(keyword)
            ds = pydicom.dataset.Dataset()
            ds.StudyInstanceUID = study_instance_uid
            ds.SeriesInstanceUID = ''
            ds.QueryRetrieveLevel = 'SERIES'
            ds.SeriesNumber = ''
            ds.SeriesDescription = ''
            ds.Modality = ''
            ds.BodyPartExamined = ''
            ds.NumberOfSeriesRelatedInstances = ''
            VR = pydicom.datadict.dictionary_VR(tag)
            if VR == 'IS':
                try:
                    ds[tag] = pydicom.dataset.DataElement(
                        tag, pydicom.datadict.dictionary_VR(tag), int(search))
                except (ValueError, TypeError):
                    continue
            else:
                ds[tag] = pydicom.dataset.DataElement(
                    tag, pydicom.datadict.dictionary_VR(tag), search)

            instances2 =\
                self._cfind(ds,
                            pynetdicom.sop_class.StudyRootQueryRetrieveInformationModelFind,
                            'SeriesInstanceUID')
            for instance in instances2:
                if instance not in instances:
                    instances.append(instance)
        return instances

    def _cfind_instances(self, study_instance_uid, series_instance_uid, search):
        instances = []
        # Create our Identifier (query) dataset
        for keyword in 'SOPInstanceUID', 'InstanceNumber':
            tag = pydicom.dataset.tag_for_keyword(keyword)
            ds = pydicom.dataset.Dataset()
            ds.StudyInstanceUID = study_instance_uid
            ds.SeriesInstanceUID = series_instance_uid
            ds.QueryRetrieveLevel = 'IMAGE'
            ds.SOPInstanceUID = ''
            ds.InstanceNumber = ''
            ds.NumberOfFrames = ''
            ds.Rows = ''
            ds.Columns = ''
            VR = pydicom.datadict.dictionary_VR(tag)
            if VR == 'IS':
                try:
                    ds[tag] = pydicom.dataset.DataElement(
                        tag, pydicom.datadict.dictionary_VR(tag), int(search))
                except (ValueError, TypeError):
                    continue
            else:
                ds[tag] = pydicom.dataset.DataElement(
                    tag, pydicom.datadict.dictionary_VR(tag), search)
            instances2 =\
                self._cfind(ds,
                            pynetdicom.sop_class.StudyRootQueryRetrieveInformationModelFind,
                            'SOPInstanceUID')
            for instance in instances2:
                if instance not in instances:
                    instances.append(instance)
        return instances

    def _cfind(self, ds, model, tag):
        # Associate with the peer AE
        if self.__assoc.is_established:
            # Send the C-FIND request
            responses = self.__assoc.send_c_find(ds, model)
            instances = []
            for (status, identifier) in responses:
                if status:
                    if identifier is not None:
                        uid = identifier[tag].value
                        instances.append(uid)
                        self.__catalog[uid] = identifier
                else:
                    raise ConnectionError(
                        'Connection timed out, was aborted or received invalid response')
            return instances
        else:
            raise ConnectionError('Association rejected, aborted or never connected')
