"""Read/Write DICOM files
"""

# Copyright (c) 2013-2025 Erling Andersen, Haukeland University Hospital, Bergen, Norway

import os
import sys
import logging
import traceback
import mimetypes
import math
from numbers import Number
from collections import defaultdict, namedtuple, Counter
from functools import partial
from typing import List, Union
from datetime import date, datetime, timedelta, timezone
import numpy as np
import pydicom
import pydicom.valuerep
import pydicom.config
import pydicom.errors
import pydicom.uid
from pydicom.datadict import tag_for_keyword
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset

from ..formats import CannotSort, NotImageError, INPUT_ORDER_FAULTY, \
    SORT_ON_SLICE, \
    INPUT_ORDER_NONE, INPUT_ORDER_TIME, INPUT_ORDER_B, INPUT_ORDER_FA, INPUT_ORDER_TE, \
    INPUT_ORDER_BVECTOR, INPUT_ORDER_TRIGGERTIME, \
    INPUT_ORDER_AUTO
from ..series import Series
from ..axis import VariableAxis, UniformLengthAxis
from .abstractplugin import AbstractPlugin
from ..archives.abstractarchive import AbstractArchive, Member
from ..header import Header
from ..apps.diffusion import get_ds_b_vectors, get_ds_b_value, set_ds_b_value, set_ds_b_vector

logger = logging.getLogger(__name__)
try:
    # pydicom >= 2.3
    pydicom.config.settings.reading_validation_mode = pydicom.config.IGNORE
    # pydicom.config.settings.writing_validation_mode = pydicom.config.IGNORE
    pydicom.config.settings.writing_validation_mode = pydicom.config.WARN
    # pydicom.config.settings.writing_validation_mode = pydicom.config.RAISE
except AttributeError:
    # pydicom < 2.3
    pydicom.config.enforce_valid_values = True

mimetypes.add_type('application/dicom', '.ima')

SeriesUID = namedtuple('SeriesUID', 'patientID, studyInstanceUID, seriesInstanceUID, ' +
                       'acquisitionNumber, echoNumber', defaults=(None, None))

# Type definitions
SourceList = list[dict]


# Class definitions
class ObjectList(list):
    """ObjectList is list[tuple[AbstractArchive, Member]]"""

    def __init__(self):
        super().__init__()

    def append(self, *args):
        for arg in args:
            assert isinstance(arg, tuple), self.__doc__
            assert len(arg) == 2, self.__doc__
            assert isinstance(arg[0], AbstractArchive), self.__doc__
            assert isinstance(arg[1], Member), self.__doc__
        super().append(*args)


class DatasetList(list):
    """DatasetList is list[Dataset]"""

    def __init__(self):
        super().__init__()

    def append(self, *args):
        for arg in args:
            assert isinstance(arg, Dataset), self.__doc__
        super().append(*args)


class DatasetDict(defaultdict):
    """DatasetDict is defaultdict[SeriesUID, DatasetList]"""

    def __init__(self):
        super().__init__(DatasetList)

    def __setitem__(self, key, value):
        assert isinstance(key, SeriesUID), self.__doc__
        assert isinstance(value, DatasetList), self.__doc__
        super().__setitem__(key, value)


class SortedDatasetList(defaultdict):
    """SortedDatasetList is defaultdict[float, DatasetList]"""

    def __init__(self):
        super().__init__(DatasetList)
        self.spacing = None
        self.transformationMatrix = None
        self.imagePositions = None

    def __setitem__(self, key, value):
        assert isinstance(key, float), self.__doc__
        assert isinstance(value, DatasetList), self.__doc__
        super().__setitem__(key, value)


class SortedDatasetDict(defaultdict):
    """SortedDatasetDict is defaultdict[SeriesUID, SortedDatasetList]"""

    def __init__(self):
        super().__init__(lambda: SortedDatasetList)

    def __setitem__(self, key, value):
        assert isinstance(key, SeriesUID), self.__doc__
        assert isinstance(value, SortedDatasetList), self.__doc__
        super().__setitem__(key, value)


class SortedHeaderDict(dict):
    """SortedHeaderDict is dict[SeriesUID, Header]"""

    def __init__(self):
        super().__init__()

    def __setitem__(self, key, value):
        assert isinstance(key, SeriesUID), self.__doc__
        assert isinstance(value, Header), self.__doc__
        super().__setitem__(key, value)


class PixelDict(dict):
    """PixelDict is dict[SeriesUID, np.ndarray]"""

    def __init__(self):
        super().__init__()

    def __setitem__(self, key, value):
        assert isinstance(key, SeriesUID), self.__doc__
        assert isinstance(value, np.ndarray), self.__doc__
        super().__setitem__(key, value)


image_uids = [pydicom.uid.MRImageStorage,
              pydicom.uid.CTImageStorage,
              pydicom.uid.DICOSCTImageStorage,
              pydicom.uid.RTImageStorage,
              pydicom.uid.UltrasoundImageStorage,
              pydicom.uid.UltrasoundMultiFrameImageStorage,
              pydicom.uid.ComputedRadiographyImageStorage,
              pydicom.uid.XRayAngiographicImageStorage,
              pydicom.uid.XRay3DAngiographicImageStorage,
              pydicom.uid.XRay3DCraniofacialImageStorage,
              pydicom.uid.XRayRadiofluoroscopicImageStorage,
              pydicom.uid.SecondaryCaptureImageStorage,
              pydicom.uid.PositronEmissionTomographyImageStorage,
              pydicom.uid.BreastTomosynthesisImageStorage,
              pydicom.uid.NuclearMedicineImageStorage,
              pydicom.uid.ParametricMapStorage,
              pydicom.uid.EddyCurrentImageStorage,
              pydicom.uid.EddyCurrentMultiFrameImageStorage,
              pydicom.uid.VLEndoscopicImageStorage,
              pydicom.uid.VideoEndoscopicImageStorage,
              pydicom.uid.VLMicroscopicImageStorage,
              pydicom.uid.VideoMicroscopicImageStorage,
              pydicom.uid.VLPhotographicImageStorage,
              pydicom.uid.VideoPhotographicImageStorage
              ]

# sr_uids = [pydicom.uid.BasicTextSRStorage,
#            pydicom.uid.EnhancedSRStorage,
#            pydicom.uid.ComprehensiveSRStorage,]


attributes: List[str] = [
    'patientName', 'patientID', 'patientBirthDate',
    'studyInstanceUID', 'studyID',
    'seriesInstanceUID', 'frameOfReferenceUID',
    'seriesDate', 'seriesTime', 'seriesNumber', 'seriesDescription',
    'imageType', 'accessionNumber',
    'modality', 'laterality',
    'echoNumbers', 'acquisitionNumber',
    'protocolName', 'bodyPartExamined', 'patientPosition',
    'windowCenter', 'windowWidth',
    'SOPClassUID'
]


def _get_float(im: Dataset, tag: str) -> float:
    if im.data_element(tag).VR == 'TM':
        time_str = im.data_element(tag).value
        try:
            if '.' in time_str:
                tm = datetime.strptime(time_str, "%H%M%S.%f")
            else:
                tm = datetime.strptime(time_str, "%H%M%S")
        except ValueError:
            raise CannotSort("Unable to extract time value from header.")
        td = timedelta(hours=tm.hour,
                       minutes=tm.minute,
                       seconds=tm.second,
                       microseconds=tm.microsecond)
        return td.total_seconds()
    else:
        try:
            return float(im.data_element(tag).value)
        except ValueError:
            raise CannotSort("Unable to extract value from header.")


def _get_no_value(im: Dataset) -> Number:
    return 0


def _get_acquisition_time(im: Dataset) -> Number:
    return _get_float(im, 'AcquisitionTime')


def _get_trigger_time(im: Dataset) -> Number:
    return _get_float(im, 'TriggerTime') / 1000.


def _get_b_value(im: Dataset) -> Number:
    try:
        return get_ds_b_value(im)
    except IndexError:
        raise CannotSort("Unable to extract b value from header.")


def _get_b_vector(im: Dataset) -> np.ndarray:
    try:
        bvec = get_ds_b_vectors(im)
        if bvec.ndim == 0:
            bvec = np.array([])
        return bvec
    except IndexError:
        raise CannotSort("Unable to extract b vector from header.")


def _get_echo_time(im: Dataset) -> Number:
    return _get_float(im, 'EchoTime')


def _get_flip_angle(im: Dataset) -> Number:
    fa_tag = 'FlipAngle'
    return _get_float(im, 'FlipAngle')


class DoNotIncludeFile(Exception):
    pass


class NoDICOMAttributes(Exception):
    pass


class ValueErrorWrapperPrecisionError(Exception):
    pass


class UnknownTag(Exception):
    pass


class DICOMPlugin(AbstractPlugin):
    """Read/write DICOM files.

    Attributes:
        input_order
        instanceNumber
        today
        now
        serInsUid
        input_options
        output_sort
        output_dir
        seriesTime
    """

    name = "dicom"
    description = "Read and write DICOM files."
    authors = "Erling Andersen"
    version = "2.1.0"
    url = "www.helse-bergen.no"
    extensions = [".dcm", ".ima"]

    root = "2.16.578.1.37.1.1.4"
    smallint = ('bool8', 'byte', 'ubyte', 'ushort', 'uint16', 'int8', 'uint8', 'int16')
    keep_uid = False
    slice_tolerance = 1e-4
    dir_cosine_tolerance = 0.0

    def __init__(self):
        super(DICOMPlugin, self).__init__(self.name, self.description,
                                          self.authors, self.version, self.url)
        self.input_order = None
        self.DicomHeaderDict = None
        self.dicomTemplate = None
        self.instanceNumber = 0
        self.today = date.today().strftime("%Y%m%d")
        self.now = datetime.now().strftime("%H%M%S.%f")
        self.serInsUid = None
        self.input_options = {}
        self.output_sort = None
        self.output_dir = None
        self.seriesTime = None

    def read(self, sources: SourceList, pre_hdr: Header, input_order: str, opts: dict) -> (
            tuple[SortedHeaderDict, PixelDict]):
        """Read image data

        Args:
            self: DICOMPlugin instance
            sources: list of sources to image data
            pre_hdr: Pre-filled header dict. Can be None
            input_order: sort order
            opts: input options (dict)
        Returns:
            Tuple of
                - hdr: Header
                    - input_format
                    - input_order
                    - slices
                    - sliceLocations
                    - dicomTemplate
                    - keep_uid
                    - tags
                    - seriesNumber
                    - seriesDescription
                    - imageType
                    - spacing
                    - orientation
                    - imagePositions
                - si[tag,slice,rows,columns]: multi-dimensional numpy array
        """

        _name: str = '{}.{}'.format(__name__, self.read.__name__)

        self.input_order = input_order
        self.input_options = {
            INPUT_ORDER_NONE: _get_no_value,
            INPUT_ORDER_TIME: _get_acquisition_time,
            INPUT_ORDER_TRIGGERTIME: _get_trigger_time,
            INPUT_ORDER_B: _get_b_value,
            INPUT_ORDER_BVECTOR: _get_b_vector,
            INPUT_ORDER_TE: _get_echo_time,
            INPUT_ORDER_FA: _get_flip_angle,
            'auto_sort': ['time', 'triggertime', 'b', 'fa', 'te']
        }
        for key, value in opts.items():  # Copy opts to self.input_options
            self.input_options[key] = value

        skip_pixels = False
        if 'headers_only' in opts and opts['headers_only']:
            skip_pixels = True
        if 'slice_tolerance' in self.input_options:
            self.slice_tolerance = float(opts['slice_tolerance'])
        if 'dir_cosine_tolerance' in self.input_options:
            self.dir_cosine_tolerance = float(opts['dir_cosine_tolerance'])

        # Read DICOM headers
        logger.debug('{}: sources {}'.format(_name, sources))
        # pydicom.config.debug(True)
        object_list: ObjectList = self._get_dicom_files(sources)

        dataset_dict: DatasetDict
        dataset_dict = self._catalog_on_instance_uid(object_list, opts, skip_pixels)

        imaging_dataset_dict: DatasetDict
        imaging_dataset_dict = self._select_imaging_datasets(dataset_dict, opts)
        non_imaging_dataset_dict: DatasetDict
        non_imaging_dataset_dict = self._select_non_imaging_datasets(dataset_dict, opts)
        logger.debug('{}: imaging_datasets {}'.format(_name, len(imaging_dataset_dict)))
        logger.debug('{}: non_imaging_datasets {}'.format(_name, len(non_imaging_dataset_dict)))

        sorted_header_dict: SortedHeaderDict = SortedHeaderDict()
        pixel_dict: PixelDict = PixelDict()

        if imaging_dataset_dict:
            sorted_dataset_dict: SortedDatasetDict
            sorting: dict[str]
            sorted_dataset_dict, sorting = self._sort_datasets(imaging_dataset_dict, input_order, opts)

            logger.debug('{}: going to _get_headers {}'.format(_name, sources))
            sorted_header_dict = self._get_headers(sorted_dataset_dict, sorting, opts)

            if not skip_pixels:
                logger.debug('{}: going to _construct_pixel_arrays'.format(_name))
                pixel_dict = self._construct_pixel_arrays(sorted_dataset_dict, sorted_header_dict,
                                                          opts, skip_pixels)

                if 'correct_acq' in opts and opts['correct_acq']:
                    for seriesUID in sorted_dataset_dict:
                        pixel_dict[seriesUID] = self._correct_acqtimes_for_dynamic_series(
                            sorted_header_dict[seriesUID], pixel_dict[seriesUID]
                        )

        if non_imaging_dataset_dict:
            logger.debug('{}: going to _get_non_image_headers {}'.format(_name, sources))
            non_image_header_dict: SortedHeaderDict
            non_image_header_dict = self._get_non_image_headers(non_imaging_dataset_dict, opts)
            if not skip_pixels:
                logger.debug('{}: going to _construct_pixel_arrays'.format(_name))
                non_image_pixel_dict = self._construct_pixel_arrays(non_imaging_dataset_dict,
                                                                    non_image_header_dict,
                                                                    opts, skip_pixels)
            for seriesUID in non_image_header_dict:
                if seriesUID in sorted_header_dict:
                    sorted_header_dict[seriesUID].datasets = non_imaging_dataset_dict[seriesUID]
                else:
                    sorted_header_dict[seriesUID] = non_image_header_dict[seriesUID]
                    sorted_header_dict[seriesUID].datasets = non_imaging_dataset_dict[seriesUID]
                if seriesUID in pixel_dict:
                    raise IndexError('Duplicate pixel data')
                elif seriesUID in non_image_pixel_dict:
                    pixel_dict[seriesUID] = non_image_pixel_dict[seriesUID]

        logger.debug('{}: ending'.format(_name))
        return sorted_header_dict, pixel_dict

    def _get_dicom_files(self,
                         sources: SourceList
                         ) -> ObjectList:
        """Get DICOM objects.

        Args:
            self: DICOMPlugin instance
            sources: list of sources to image data
        Returns:
            List of tuples of
                - archive
                - member
        """
        _name: str = '{}.{}'.format(__name__, self._get_dicom_files.__name__)

        logger.debug("{}: sources: {} {}".format(
            _name, type(sources), sources))

        object_list: ObjectList = ObjectList()
        for source in sources:
            archive = source['archive']
            scan_files = source['files']
            logger.debug("{}: archive: {}".format(_name, archive))
            if scan_files is None or len(scan_files) == 0:
                if archive.base is not None:
                    scan_files = [archive.base]
                else:
                    scan_files = ['*']
            elif archive.base is not None:
                raise ValueError('When is archive.base with source[files]')
            logger.debug("{}: source: {} {}".format(_name, type(source), source))
            logger.debug("{}: scan_files: {}".format(_name, scan_files))
            for path in archive.getnames(scan_files):
                logger.debug("{}: member: {}".format(_name, path))
                if os.path.basename(path) == "DICOMDIR":
                    continue
                member = archive.getmembers([path, ])
                if len(member) != 1:
                    raise IndexError('Should not be multiple files for a filename')
                member = member[0]
                object_list.append((archive, member))
        return object_list

    def _catalog_on_instance_uid(self,
                                 object_list: ObjectList,
                                 opts: dict = None,
                                 skip_pixels: bool = False) \
            -> DatasetDict:
        """Sort files on Series Instance UID

        Args:
            self: DICOMPlugin instance
            object_list: List of (archive, member) tuples
            opts: input options (dict)
            skip_pixels: Do not read pixel data (default: False)
        Returns:
            Dict of List of Dataset
        """

        _name: str = '{}.{}'.format(__name__, self._catalog_on_instance_uid.__name__)
        logger.debug('{}:'.format(_name))

        dataset_dict: DatasetDict = DatasetDict()
        last_message = ''
        for archive, member in object_list:
            try:
                with archive.open(member, mode='rb') as f:
                    logger.debug('{}: process_member {}'.format(_name, member))
                    self._extract_member(dataset_dict, f, opts, skip_pixels=skip_pixels)
            except DoNotIncludeFile as e:
                last_message = '{}'.format(e)
            except Exception as e:
                logger.debug('{}: Exception {}'.format(_name, e))
                last_message = '{}'.format(e)
        if len(object_list) > 0 and len(dataset_dict) < 1:
            raise NotImageError(last_message)
        return dataset_dict

    def _select_imaging_datasets(self,
                                 dataset_dict: DatasetDict,
                                 opts: dict = None
                                 ) \
            -> DatasetDict:
        """Select imaging datasets only

        Args:
            self: DICOMPlugin instance
            dataset_dict: Dict of List of Dataset (DatasetDict)
            opts: input options (dict)
        Returns:
            Dict of List of Imaging Dataset
        """
        _name: str = '{}.{}'.format(__name__, self._select_imaging_datasets.__name__)

        # Select datasets on SOPClassUID
        selected_dataset_dict: DatasetDict = DatasetDict()
        for seriesUID in dataset_dict:
            dataset_list = dataset_dict[seriesUID]
            dataset: Dataset
            dataset = dataset_list[0]
            if dataset.SOPClassUID in image_uids:
                # Keep imaging datasets
                selected_dataset_dict[seriesUID] = dataset_list
        logger.debug('{}: end with {}'.format(_name, selected_dataset_dict.keys()))
        return selected_dataset_dict

    def _select_non_imaging_datasets(self,
                                     dataset_dict: DatasetDict,
                                     opts: dict = {}
                                     ) \
            -> DatasetDict:
        """Select non-imaging datasets only

        Args:
            self: DICOMPlugin instance
            dataset_dict: Dict of List of Dataset (DatasetDict)
            opts: input options (dict)
        Returns:
            Dict of List of non-imaging Dataset
        """
        _name: str = '{}.{}'.format(__name__, self._select_non_imaging_datasets.__name__)

        # Select datasets on SOPClassUID
        selected_dataset_dict: DatasetDict = DatasetDict()
        for seriesUID in dataset_dict:
            dataset_list = dataset_dict[seriesUID]
            dataset: Dataset
            dataset = dataset_list[0]
            if dataset.SOPClassUID not in image_uids:
                # Keep non-imaging datasets
                selected_dataset_dict[seriesUID] = dataset_list
        logger.debug('{}: end with {}'.format(_name, selected_dataset_dict.keys()))
        return selected_dataset_dict

    def _extract_member(self,
                        image_list: DatasetDict,
                        member: Union[Dataset, Member, str],
                        opts: dict = None,
                        skip_pixels: bool = False):
        im: Dataset
        if issubclass(type(member), Dataset):
            im = member
        else:
            # Read the DICOM object
            try:
                im = pydicom.filereader.dcmread(member, stop_before_pixels=skip_pixels)
            except pydicom.errors.InvalidDicomError as e:
                raise DoNotIncludeFile('Invalid Dicom Error: {}'.format(e))
            # Verify that the DICOM object has pixel data
            if not skip_pixels:
                try:
                    _ = len(im.pixel_array)
                except AttributeError:
                    pass
                    # raise DoNotIncludeFile('No pixel data in DICOM object')

        if 'input_serinsuid' in opts and opts['input_serinsuid'] is not None:
            if im.SeriesInstanceUID != opts['input_serinsuid']:
                raise DoNotIncludeFile('Series Instance UID not selected')
        if 'input_echo' in opts and opts['input_echo'] is not None:
            if int(im.EchoNumbers) != int(opts['input_echo']):
                raise DoNotIncludeFile('Echo Number not selected')
        if 'input_acquisition' in opts and opts['input_acquisition'] is not None:
            if int(im.AcquisitionNumber) != int(opts['input_acquisition']):
                raise DoNotIncludeFile('Acquisition Number not selected')

        # Catalog images with ref as key
        acquisition_number = echo_number = None
        series_instance_uid = im.SeriesInstanceUID
        if 'ignore_series_uid' in opts and opts['ignore_series_uid']:
            series_instance_uid = None
        if 'split_acquisitions' in opts and opts['split_acquisitions']:
            acquisition_number = im.AcquisitionNumber
        if 'split_echo_numbers' in opts and opts['split_echo_numbers']:
            echo_number = im.EchoNumbers
        ref = SeriesUID(im.PatientID, im.StudyInstanceUID, series_instance_uid,
                        acquisition_number, echo_number)
        image_list[ref].append(im)

    def _sort_datasets(self,
                       image_dict: DatasetDict,
                       input_order: str,
                       opts: dict = None
                       ) -> (SortedDatasetDict, dict[str]):

        def _get_sloc(ds: Dataset) -> float:
            _name: str = '{}.{}'.format(__name__, _get_sloc.__name__)
            try:
                return float(ds.SliceLocation)
            except AttributeError:
                logger.debug('{}: Calculate SliceLocation'.format(_name))
                try:
                    return self._calculate_slice_location(ds)
                except ValueError:
                    pass
            return 0.0

        def _get_tag_value(im: Dataset, input_order: str, opts: dict = None) -> Number:
            _object = self._get_tag(im, input_order, opts)
            if issubclass(type(_object), tuple):
                _sum = 0
                for _item in _object:
                    if issubclass(type(_item), np.ndarray):
                        # Typical array value is the MRI diffusion b-vector
                        # To ensure consistent sorting of b-vectors, the different directions are
                        # weighted (arbitrarily) by the position index in the vector
                        _sum += np.dot(_item, np.array(np.arange(_item.size) + 1))
                    else:
                        _sum += _item
                return _sum
            else:
                if issubclass(type(_object), np.ndarray):
                    return np.dot(_object, np.array(np.arange(_object.size) + 1))
                else:
                    return _object

        _name: str = '{}.{}'.format(__name__, self._sort_datasets.__name__)

        skip_broken_series = False
        if 'skip_broken_series' in opts:
            skip_broken_series = opts['skip_broken_series']

        # Sort datasets on sloc
        sorted_dataset_dict: SortedDatasetDict = SortedDatasetDict()  # defaultdict(lambda: defaultdict(list))
        sorting: dict[SeriesUID, str]
        sorting = {}
        for seriesUID in image_dict:
            sorting[seriesUID] = 'none'
            dataset_dict: DatasetList
            dataset_dict = image_dict[seriesUID]
            try:
                message = '{} ({})'.format(dataset_dict[0].SeriesDescription, dataset_dict[0].SeriesNumber)
            except AttributeError:
                try:
                    message = '{} ({})'.format('', dataset_dict[0].SeriesNumber)
                except AttributeError:
                    message = '{}'.format(dataset_dict[0].SeriesInstanceUID)
            try:
                sorted_dataset = self._sort_dataset_geometry(dataset_dict, message, opts)
            except CannotSort as e:
                logger.debug('{}: _sort_dataset_geometry CannotSort: {}'.format(_name, e))
                if skip_broken_series:
                    continue
            except Exception as e:
                logger.debug('{}: _sort_dataset_geometry {} {}'.format(_name, type(e).__name__, e))
                import traceback
                traceback.print_exc()
                raise

            # Determine (automatic) sorting
            try:
                sorting[seriesUID] = self._determine_sorting(
                    sorted_dataset, input_order, opts
                )
            except CannotSort:
                logger.debug('{}: opts {}'.format(_name, opts))
                logger.debug('{}: skip_broken_series {}'.format(
                    _name, opts['skip_broken_series']
                ))
                if skip_broken_series:
                    logger.debug(
                        '{}: skip_broken_series continue {}'.format(
                            _name, seriesUID
                        ))
                    continue  # Next series
                else:
                    logger.debug('{}: skip_broken_series raise'.format(_name))
                    raise
            # Sort the dataset on selected key for each sloc
            for sort_key in reversed(sorting[seriesUID].split(sep=',')):
                for sloc in sorted(sorted_dataset.keys()):
                    try:
                        sorted_dataset[sloc].sort(
                            key=partial(_get_tag_value, input_order=sort_key, opts=opts)
                        )
                    except ValueError:
                        pass
                    except Exception as e:
                        print(e)
            # Catalog images with seriesUID and sloc as keys
            sorted_dataset_dict[seriesUID] = sorted_dataset
        logger.debug('{}: end with {}'.format(_name, sorted_dataset_dict.keys()))
        return sorted_dataset_dict, sorting

    def _determine_sorting(self,
                           sorted_dataset_dict: SortedDatasetList,
                           input_order: str,
                           opts: dict = None) -> str:

        def _single_slice_over_time(tags):
            """If time and slice both varies, the time stamps address slices of a single volume
            """
            count_time = {}
            count_sloc = {}
            for time, sloc in tags:
                if time not in count_time:
                    count_time[time] = 0
                if sloc not in count_sloc:
                    count_sloc[sloc] = 0
                count_time[time] += 1
                count_sloc[sloc] += 1
            max_time = max(count_time.values())
            max_sloc = max(count_sloc.values())
            return max_time == 1 and max_sloc == 1

        if input_order != 'auto':
            return input_order
        extended_tags = {}
        found_tags = {}
        im = None
        for sloc in sorted_dataset_dict.keys():
            for im in sorted_dataset_dict[sloc]:
                for order in self.input_options['auto_sort']:
                    try:
                        tag = self._get_tag(im, order, opts)
                        if tag is None:
                            continue
                        if order not in found_tags:
                            found_tags[order] = []
                            extended_tags[order] = []
                        if tag not in found_tags[order]:
                            found_tags[order].append(tag)
                            extended_tags[order].append((tag, sloc))
                    except (KeyError, TypeError, CannotSort):
                        pass

        # Determine how to sort
        actual_order = None
        for order in found_tags:
            if len(found_tags[order]) > 1:
                if actual_order in ('time', 'triggertime') and order in ['b', 'te']:
                    # DWI images will typically have varying time.
                    # Let b values override time stamps.
                    actual_order = order
                elif actual_order is None:
                    actual_order = order
                else:
                    raise CannotSort('Cannot auto-sort: {}\n'.format(extended_tags) +
                                     '  actual_order: {}, order: {},'.format(actual_order, order) +
                                     ' Series #{}: {}'.format(im.SeriesNumber, im.SeriesDescription)
                                     )
        if actual_order is None:
            actual_order = INPUT_ORDER_NONE
        elif actual_order in (INPUT_ORDER_TIME, INPUT_ORDER_TRIGGERTIME) and \
            _single_slice_over_time(extended_tags[actual_order]):
            actual_order = INPUT_ORDER_NONE
        return actual_order

    def _get_headers(self,
                     sorted_dataset_dict: SortedDatasetDict,
                     input_order: dict[str],
                     opts: dict = None
                     ) -> SortedHeaderDict:
        """Get DICOM headers"""

        def _verify_consistent_slices(series: SortedDatasetList, message: str) -> Counter:
            _name: str = '{}.{}'.format(__name__, _verify_consistent_slices.__name__)
            # Verify same number of images for each slice
            slice_count = Counter()
            last_sloc = None
            for islice, sloc in enumerate(series):
                slice_count[islice] = len(series[sloc])
                last_sloc = sloc
            logger.debug("{}: tags per slice: {}".format(_name, slice_count))
            accept_uneven_slices = False
            if 'accept_uneven_slices' in opts and opts['accept_uneven_slices']:
                accept_uneven_slices = True
            min_slice_count = min(slice_count.values())
            max_slice_count = max(slice_count.values())
            if min_slice_count != max_slice_count and not accept_uneven_slices:
                logger.error("{}: tags per slice: {}".format(message, slice_count))
                raise CannotSort(
                    "{}: ".format(message) +
                    "Different number of images in each slice. Tags per slice:\n{}".format(slice_count) +
                    "\nLast file: {}".format(series[last_sloc][0].filename) +
                    "\nCould try 'split_acquisitions=True' or 'split_echo_numbers=True'."
                )
            return slice_count

        def _structure_tags_2(tag_list: defaultdict[list])\
                -> defaultdict[tuple]:

            def _count_tags(tag_list, index) -> defaultdict:
                # Count the tags
                dict_depth = lambda: defaultdict(dict_depth)
                d = dict_depth()
                tags = [_ for _ in range(len(tag_list[0]))]
                tags.remove(index)
                tags.insert(0, index)  # Ensure index comes first
                for tag in tag_list:
                    _ = d  # Pointer to present level in dict structure
                    for t in tags:
                        try:
                            _ = _[tag[t]]
                        except TypeError:
                            _ = _[tuple(tag[t].tolist())]
                        except Exception:
                            raise
                    try:
                        _[0] += 1
                    except (KeyError, TypeError):
                        _[0] = 1
                return d

            def _catalog_tags(tag_list: list[tuple]) -> dict[defaultdict]:
                # Catalog the tags and count them
                d = {}
                for index in range(tags):
                    d[index] = _count_tags(tag_list, index)
                return(
                    dict(sorted(d.items(), key=lambda item: len(item[1])))
                )

            def _find_tag_element(catalog: defaultdict, shape: tuple, tag: tuple, p: tuple, index: int) -> int:

                def find_nearest_vector(array, value):
                    idx = None
                    min_diff = np.inf
                    for i in range(len(array)):
                        if array[i].size == value.size:
                            vdiff = np.linalg.norm(abs(array[i] - value))
                            if vdiff < min_diff:
                                idx = i
                                min_diff = vdiff
                    return idx
                    idx = np.searchsorted(array, value, side="left")
                    if idx > 0 and (
                            idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
                        return array[idx - 1]
                    else:
                        return array[idx]

                _name: str = '{}.{}'.format(__name__, _find_tag_element.__name__)
                c_tuple = (catalog[index],)
                try:
                    _ = catalog[0]
                    for i, _p in enumerate(p):
                        if _p in _:
                            _ = _[_p]
                        elif tag[i] in _:
                            _ = _[tag[i]]
                        else:
                            pass
                    c_tuple += (_,)
                except (IndexError, TypeError):
                    pass
                for c in c_tuple:
                    if issubclass(type(tag[index]), np.ndarray):
                        keys = np.empty(len(c), dtype=np.ndarray)
                        nonzero = 0
                        for i, k in enumerate(c.keys()):
                            keys[i] = np.array(list(k))
                            if keys[i].size > 0:
                                nonzero += 1
                        nonzero_keys = np.empty(nonzero, dtype=np.ndarray)
                        i = 0
                        for key in keys:
                            if key.size > 0:
                                nonzero_keys[i] = key
                                i += 1
                        _pos = find_nearest_vector(keys, tag[index])
                        return _pos % shape[index]
                    else:
                        keys = list(c.keys())
                        try:
                            return keys.index(tag[index]) % shape[index]
                        except ValueError:
                            continue
                raise IndexError('{}: Cannot find tag {}'.format(_name, tag))

            def _position(catalog: defaultdict, shape: tuple, tag: tuple) -> tuple[int]:
                # catalog[tuple index][tag0]...[tagN]
                p = tuple()
                for index in range(len(tag)):
                    p += (_find_tag_element(catalog, shape, tag, p, index),)
                return p

            def _calculate_size(catalog: dict) -> int:
                p = 1
                for _ in catalog:
                    p *= len(catalog[_])
                return p

            def _calculate_shape(struct_tag_list: defaultdict) -> tuple[int]:
                s = ()
                for tag in sorted(struct_tag_list):
                    s += (len(struct_tag_list[tag]),)
                return s

            def _recalculate_catalog(catalog: dict, tag_list: list[tuple]) -> dict:
                for _ in catalog:
                    if len(catalog[_]) >= len(tag_list):
                        catalog[_] = _find_candidate(catalog, _, tag_list)
                return catalog

            def _recalculate_catalog_2(catalog: dict, tag_list: list[tuple]) -> dict:
                for _ in catalog:
                    if len(catalog[_]) >= len(tag_list):
                        longest = {}
                        for tag in tag_list:
                            _f = _find_candidate(catalog, _, tag_list, base=tag)
                            if len(_f) > len(longest):
                                longest = _f
                        catalog[_] = longest
                return catalog

            def _find_candidate(catalog: dict, index: int, tag_list: list[tuple], base: tuple = None) -> defaultdict[dict]:
                # Vary index tag, keep other tags constant, and find candidate for reduced tag size
                find = tag_list[0]
                if base is not None:
                    find = base
                new_list = []
                dict_depth = lambda: defaultdict(dict_depth)
                new_tag_dict = dict_depth()
                for tag in tag_list:
                    found = True
                    for i, t in enumerate(tag):
                        if i != index:
                            found = found and t == find[i]
                    if found:
                        new_list.append(tag[index])
                        try:
                            new_tag_dict[tag[index]] = catalog[index][tag[index]]
                        except TypeError:
                            if issubclass(type(tag[index]), np.ndarray):
                                _ = tuple(tag[index].tolist())
                                new_tag_dict[_] = catalog[index][_]
                            else:
                                raise
                return new_tag_dict

            if len(tag_list) == 0:
                return defaultdict(list)
            tags = len(tag_list[0])
            # Catalog the tags, count and sort them
            catalog = _catalog_tags(tag_list)
            size = _calculate_size(catalog)
            previous_size = size + 1
            i = 0
            _catalog = catalog.copy()
            while len(tag_list) < size < previous_size:
                # Some tag values are not unique, need to do a better sort
                _catalog = _recalculate_catalog(_catalog, tag_list)
                previous_size = size  # Did recalculation improve sorting?
                size = _calculate_size(_catalog)
                i += 1
                if i > 2 * len(tag_list):  # Stop criteria
                    raise CannotSort('{}: Too many iterations to sort headers'.format(_name))
            struct_tag_list = defaultdict(list)
            for index in catalog.keys():
                values = catalog[index].keys()
                struct_tag_list[index] = values
            shape = _calculate_shape(struct_tag_list)
            tag_dict = defaultdict(tuple)
            for tag in tag_list:
                _pos = _position(catalog, shape, tag)
                try:
                    tag_dict[tag] = _pos
                except TypeError:
                    _ = tuple()
                    for _t in tag:
                        if issubclass(type(_t), np.ndarray):
                            _ += (tuple(_t.tolist()),)
                        else:
                            _ += (_t,)
                    tag_dict[_] = _pos
            return struct_tag_list, tag_dict

        def _append_tag(tag_list: list[tuple], tag: tuple,
                        accept_duplicate_tag: bool=False,
                        accept_uneven_slices: bool=False) -> None:
            _name: str = '{}.{}'.format(__name__, _append_tag.__name__)
            if len(tag_list) == 0:
                tag_list.append(tag)
                return
            exist = True
            try:
                for t in tag_list:
                    for _, _t in enumerate(t):
                        if issubclass(type(_t), np.ndarray):
                            if _t.size == 0:
                                exist = exist and tag[_].size == 0
                            else:
                                exist = (exist and _t.shape == tag[_].shape and
                                         np.allclose(_t, tag[_], rtol=1e-3, atol=1e-2))
                        else:
                            exist = exist and _t == tag[_]
            except Exception as e:
                print(e)
            if not exist or accept_duplicate_tag:
                tag_list.append(tag)
            elif accept_uneven_slices:
                # Drop duplicate images
                logger.warning("{}: dropping duplicate image: {}".format(_name, tag))
            else:
                raise CannotSort("{}: duplicate tag ({}): {}".format(_name, input_order, tag))

        def _extract_all_tags(hdr: Header,
                              series: SortedDatasetList,
                              input_order: str,
                              slice_count: Counter,
                              message: str
                              ) -> None:

            _name: str = '{}.{}'.format(__name__, _extract_all_tags.__name__)

            accept_duplicate_tag = accept_uneven_slices = False
            if 'accept_duplicate_tag' in opts and opts['accept_duplicate_tag']:
                accept_duplicate_tag = True
            if 'accept_uneven_slices' in opts and opts['accept_uneven_slices']:
                accept_uneven_slices = True
            tag_list = defaultdict(list)
            faulty = 0
            for islice, sloc in enumerate(sorted(series)):
                for im in series[sloc]:
                    tag = self._extract_tag_tuple(im, faulty, input_order, opts)
                    faulty += 1
                    try:
                        _append_tag(tag_list[islice], tag, accept_duplicate_tag, accept_uneven_slices)
                    except CannotSort:
                        raise
                    except Exception as e:
                        raise
            struct_tag_list = defaultdict()
            tag_dict = defaultdict()
            for islice in tag_list.keys():
                struct_tag_list[islice], tag_dict[islice] = _structure_tags_2(tag_list[islice])
            shape = tuple()
            for _ in sorted(struct_tag_list[0].keys()):
                shape += (len(struct_tag_list[0][_]),)
            hdr.tags = {}
            for _slice in tag_dict.keys():
                hdr.tags[_slice] = np.empty(shape, dtype=tuple)
                for _tag in tag_dict[_slice]:
                    idx = tag_dict[_slice][_tag]
                    new_value = tuple()
                    for _t in _tag:
                        if issubclass(type(_t), tuple):
                            new_value += (np.array(_t),)
                        else:
                            new_value += (_t,)
                    hdr.tags[_slice][idx] = new_value
            # Sort images based on position in tag_list
            sorted_headers = {}
            SOPInstanceUIDs = {}
            last_im = None
            # Allow for variable sized slices
            frames = None
            rows = columns = 0
            i = 0
            for islice, sloc in enumerate(sorted(series)):
                # Pre-fill sorted_headers
                sorted_headers[islice] = {}
                for im in series[sloc]:
                    tag = self._extract_tag_tuple(im, i, input_order, opts)
                    idx = self._index_from_tag(tag, hdr.tags[islice])
                    try:
                        if sorted_headers[islice][idx]:
                            # Duplicate tag
                            if accept_duplicate_tag:
                                pass
                            else:
                                print("WARNING: {}: Duplicate tag {}".format(message, tag))
                    except KeyError:
                        pass
                    sorted_headers[islice][idx] = (tag, im)
                    SOPInstanceUIDs[idx + (islice,)] = im.SOPInstanceUID
                    rows = max(rows, im.Rows)
                    columns = max(columns, im.Columns)
                    if 'NumberOfFrames' in im:
                        frames = im.NumberOfFrames
                    last_im = im
                    i += 1
            self.DicomHeaderDict = sorted_headers
            hdr.dicomTemplate = series[next(iter(series))][0]
            hdr.SOPInstanceUIDs = SOPInstanceUIDs
            nz = len(series)
            if frames is not None and frames > 1:
                nz = frames
            ipp = self.getDicomAttribute(self.DicomHeaderDict, tag_for_keyword('ImagePositionPatient'))
            if ipp is not None:
                ipp = np.array(list(map(float, ipp)))[::-1]  # Reverse xyz
            else:
                ipp = np.array([0, 0, 0])
            hdr.spacing = series.spacing
            slice_axis = UniformLengthAxis('slice', ipp[0], nz, hdr.spacing[0])
            row_axis = UniformLengthAxis('row', ipp[1], rows, hdr.spacing[1])
            column_axis = UniformLengthAxis('column', ipp[2], columns, hdr.spacing[2])
            if len(tag_list[0]) > 1:
                tag_axes = []
                for i, order in enumerate(input_order.split(sep=',')):
                    tag_axes.append(
                        VariableAxis(order, list(struct_tag_list[0][i]))
                    )
                axis_names = input_order.split(sep=',')
                axis_names.extend(['slice', 'row', 'column'])
                Axes = namedtuple('Axes', axis_names)
                axes = Axes(*tag_axes, slice_axis, row_axis, column_axis)
            elif nz > 1:
                Axes = namedtuple('Axes', [
                    'slice', 'row', 'column'
                ])
                axes = Axes(slice_axis, row_axis, column_axis)
            else:
                Axes = namedtuple('Axes', [
                    'row', 'column'
                ])
                axes = Axes(row_axis, column_axis)
            hdr.color = False
            if 'SamplesPerPixel' in last_im and last_im.SamplesPerPixel == 3:
                hdr.color = True
            hdr.axes = axes
            self._extract_dicom_attributes(series, hdr, message, opts=opts)

        def _get_printable_description(series: SortedDatasetList) -> str:
            """Get printable description of series"""
            dataset = series[next(iter(series))][0]
            try:
                message = '{} ({})'.format(dataset.SeriesDescription, dataset.SeriesNumber)
            except AttributeError:
                try:
                    message = '{} ({})'.format('', dataset.SeriesNumber)
                except AttributeError:
                    message = '{}'.format(dataset.SeriesInstanceUID)
            return message

        _name: str = '{}.{}'.format(__name__, self._get_headers.__name__)

        skip_broken_series = False
        if 'skip_broken_series' in opts:
            skip_broken_series = opts['skip_broken_series']
        sorted_header_dict: SortedHeaderDict = SortedHeaderDict()
        for seriesUID in sorted_dataset_dict:
            series_dataset: SortedDatasetList = sorted_dataset_dict[seriesUID]
            hdr = Header()
            hdr.input_format = 'dicom'
            hdr.input_order = input_order[seriesUID]
            hdr.sliceLocations = np.array(sorted(series_dataset.keys()))

            if len(series_dataset) == 0:
                raise ValueError("No DICOM images found.")

            message = _get_printable_description(series_dataset)

            try:
                slice_count = _verify_consistent_slices(series_dataset, message)
                _extract_all_tags(hdr, series_dataset, input_order[seriesUID], slice_count, message)
                sorted_header_dict[seriesUID] = hdr
            except CannotSort:
                if skip_broken_series:
                    logger.debug(
                        '{}: skip_broken_series continue {}'.format(
                            _name, seriesUID
                        ))
                    continue  # Next series
                else:
                    logger.debug('{}: skip_broken_series raise'.format(_name))
                    raise
        logger.debug('{}: end with {}'.format(_name,
                                              sorted_header_dict.keys()
                                              ))
        return sorted_header_dict

    def _get_non_image_headers(self,
                               dataset_dict: DatasetDict,
                               opts: dict = None
                               ) -> SortedHeaderDict:
        """Get DICOM headers for non-image datasets"""

        _name: str = '{}.{}'.format(__name__, self._get_non_image_headers.__name__)
        skip_broken_series = False
        if 'skip_broken_series' in opts:
            skip_broken_series = opts['skip_broken_series']
        sorted_header_dict: SortedHeaderDict = SortedHeaderDict()
        for seriesUID in dataset_dict:
            series_dataset: DatasetList = dataset_dict[seriesUID]
            hdr = Header()
            hdr.input_format = 'dicom'
            hdr.input_order = 'none'

            if len(series_dataset) == 0:
                raise ValueError("No DICOM images found.")

            try:
                self._extract_non_image_dicom_attributes(series_dataset, hdr, opts=opts)
                sorted_header_dict[seriesUID] = hdr
            except CannotSort:
                if skip_broken_series:
                    logger.debug(
                        '{}: skip_broken_series continue {}'.format(
                            _name, seriesUID
                        ))
                    continue  # Next series
                else:
                    logger.debug('{}: skip_broken_series raise'.format(_name))
                    raise
        logger.debug('{}: end with {}'.format(_name,
                                              sorted_header_dict.keys()
                                              ))
        return sorted_header_dict

    def _construct_pixel_arrays(self,
                                sorted_dataset_dict: SortedDatasetDict,
                                sorted_header_dict: SortedHeaderDict,
                                opts: dict = None,
                                skip_pixels: bool = False
                                ) -> PixelDict:

        _name: str = '{}.{}'.format(__name__, self._construct_pixel_arrays.__name__)
        skip_broken_series = False
        if 'skip_broken_series' in opts:
            skip_broken_series = opts['skip_broken_series']
        pixel_dict: PixelDict = PixelDict()
        for seriesUID in sorted_header_dict:
            dataset_dict: SortedDatasetList = sorted_dataset_dict[seriesUID]
            header: Header
            header = sorted_header_dict[seriesUID]
            setattr(header, 'keep_uid', True)
            si = None
            if not skip_pixels:
                # Extract pixel data
                try:
                    si = self._construct_pixel_array(
                        dataset_dict, header, header.shape, opts=opts
                    )
                except TypeError:
                    pass
                except Exception:
                    if skip_broken_series:
                        logger.debug(
                            '{}: skip_broken_series continue {}'.format(
                                _name, seriesUID
                            ))
                        continue
                    else:
                        logger.debug('{}: skip_broken_series raise'.format(_name))
                        raise

            if si is not None:
                pixel_dict[seriesUID] = si
        return pixel_dict

    def _construct_pixel_array(self,
                               image_dict: SortedDatasetList,
                               hdr: Header,
                               shape: tuple,
                               opts: dict = None
                               ) -> np.ndarray:

        def _copy_pixels(_si, _hdr, _image_dict):

            _name: str = '{}.{}'.format(__name__, _copy_pixels.__name__)
            faulty = 0
            for _slice, _sloc in enumerate(sorted(_image_dict)):
                _done = {}
                tgs = _hdr.tags[_slice]
                for im in _image_dict[_sloc]:
                    tag = self._extract_tag_tuple(im, faulty, _hdr.input_order, opts)
                    idx = self._index_from_tag(tag, tgs)
                    if idx in _done:
                        logger.warning("Overwriting data at index {}, tag {}".format(idx, tag))
                    _done[idx] = True
                    idx += (_slice,)
                    # Simplify index when image is 3D, remove tag index
                    logger.debug("{}: si.ndim {}, idx {}".format(_name, _si.ndim, idx))
                    if _si.ndim == 3:
                        idx = idx[len(tag):]
                    try:
                        im.decompress()
                    except NotImplementedError as e:
                        logger.error("{}: Cannot decompress pixel data: {}".format(_name, e))
                        raise
                    except ValueError:
                        pass  # Already decompressed
                    try:
                        logger.debug("{}: get idx {} shape {}".format(_name, idx, _si[idx].shape))
                        if _si.ndim > 2:
                            _si[idx] = self._get_pixels_with_shape(im, _si[idx].shape)
                        else:
                            _si[...] = self._get_pixels_with_shape(im, _si.shape)
                    except Exception as e:
                        logger.warning("{}: Cannot read pixel data: {}".format(_name, e))
                        raise
                    del im
                    faulty += 1

        def _copy_pixels_from_frames(_si, _hdr, _image_dict):
            _name: str = '{}.{}'.format(__name__, _copy_pixels_from_frames.__name__)
            assert len(_image_dict) == 1, "Do not know how to unpack frames and slices"
            for im in _image_dict[next(iter(_image_dict))]:
                try:
                    im.decompress()
                except NotImplementedError as e:
                    logger.error("{}: Cannot decompress pixel data: {}".format(_name, e))
                    raise
                try:
                    logger.debug("{}: get shape {}".format(_name, _si.shape))
                    _si = self._get_pixels_with_shape(im, _si.shape)
                except Exception as e:
                    logger.warning("{}: Cannot read pixel data: {}".format(_name, e))
                    raise
                del im

        _name: str = '{}.{}'.format(__name__, self._construct_pixel_array.__name__)

        opts = {} if opts is None else opts
        accept_duplicate_tag = False
        if 'accept_duplicate_tag' in opts:
            accept_duplicate_tag = opts['accept_duplicate_tag']
        # Look-up first image to determine pixel type
        im: Dataset = image_dict[next(iter(image_dict))][0]
        hdr.photometricInterpretation = 'MONOCHROME2'
        if 'PhotometricInterpretation' in im:
            hdr.photometricInterpretation = im.PhotometricInterpretation
        matrix_dtype = np.uint16
        if 'PixelRepresentation' in im:
            if im.PixelRepresentation == 1:
                matrix_dtype = np.int16
        if 'RescaleSlope' in im and 'RescaleIntercept' in im and \
                (abs(im.RescaleSlope - 1) > 1e-4 or abs(im.RescaleIntercept) > 1e-4):
            matrix_dtype = float
        elif im.BitsAllocated == 8:
            if hdr.color:
                matrix_dtype = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])
            else:
                matrix_dtype = np.uint8
        logger.debug('{}: matrix_dtype {}'.format(_name, matrix_dtype))

        # Load DICOM image data
        logger.debug('{}: shape {}'.format(_name, shape))
        si = np.zeros(shape, matrix_dtype)

        if 'NumberOfFrames' in im and im.NumberOfFrames > 1:
            _copy_pixels_from_frames(si, hdr, image_dict)
        else:
            _copy_pixels(si, hdr, image_dict)

        # Simplify shape
        self._reduce_shape(si, hdr.axes)
        logger.debug('{}: si {}'.format(_name, si.shape))

        return si

    def _extract_non_image_dicom_attributes(self,
                                            series: DatasetList,
                                            hdr: Header,
                                            opts: dict = None
                                            ) -> None:
        """Extract DICOM attributes

        Args:
            self: DICOMPlugin instance
            series: DatasetList
            hdr: existing header (Header)
            opts:
        Returns:
            hdr: header
                - seriesNumber
                - seriesDescription
                - imageType
                - modality, laterality, protocolName, bodyPartExamined
                - seriesDate, seriesTime
        """

        dataset = series[0]
        DICOMPlugin._copy_attributes_to_header(dataset, hdr)

    def _extract_dicom_attributes(self,
                                  series: SortedDatasetList,
                                  hdr: Header,
                                  message: str,
                                  opts: dict = None
                                  ) -> None:
        """Extract DICOM attributes

        Args:
            self: DICOMPlugin instance
            series: SortedDatasetList
            hdr: existing header (Header)
            message: series description
            opts:
        Returns:
            hdr: header
                - seriesNumber
                - seriesDescription
                - imageType
                - spacing
                - orientation
                - imagePositions
                - axes
                - modality, laterality, protocolName, bodyPartExamined
                - seriesDate, seriesTime, patientPosition
        """

        dataset = series[next(iter(series))][0]
        DICOMPlugin._copy_attributes_to_header(dataset, hdr)

        # Image position (patient)
        # Reverse orientation vectors from (x,y,z) to (z,y,x)
        try:
            iop = DICOMPlugin._get_attribute(dataset, tag_for_keyword("ImageOrientationPatient"))
        except ValueError:
            iop = [0, 0, 1, 0, 1, 0]
        if iop is not None:
            hdr.orientation = np.array((iop[2], iop[1], iop[0],
                                        iop[5], iop[4], iop[3]))

        # Extract imagePositions and transformationMatrix
        hdr.imagePositions = series.imagePositions
        hdr.transformationMatrix = series.transformationMatrix

        # Testing IPP and transformationMatrix
        T0 = hdr.transformationMatrix[:3, 3]
        ipp = np.array(T0)
        warned = False
        for i in range(len(series)):
            if not warned and not np.allclose(ipp, hdr.imagePositions[i], rtol=1e-3):
                logger.warning('{}: DICOM ImagePosition is inconsistent with ImageOrientation'.format(message))
                warned = True
            ipp += hdr.transformationMatrix[:3, 0]

    @staticmethod
    def _get_attribute(im: Dataset, tag):
        if tag in im:
            return im[tag].value
        else:
            raise ValueError('Tag {:08x} ({}) not found'.format(
                tag, pydicom.datadict.keyword_for_tag(tag)
            ))

    @staticmethod
    def _copy_attributes_to_header(dataset: Dataset, hdr: Header):
        for attribute in attributes:
            dicom_attribute = attribute[0].upper() + attribute[1:]
            try:
                setattr(hdr, attribute,
                        DICOMPlugin._get_attribute(dataset, tag_for_keyword(dicom_attribute))
                        )
            except ValueError:
                pass

    def _sort_dataset_geometry(self, dictionary: DatasetList, message: str, opts: dict = None) -> SortedDatasetList:
        _name: str = '{}.{}'.format(__name__, self._sort_dataset_geometry.__name__)
        def _get_spacing(dictionary: DatasetList) -> np.ndarray:
            _name: str = '{}.{}'.format(__name__, _get_spacing.__name__)
            # Spacing
            dr = dc = 1.0
            try:
                pixel_spacing = self.getDicomAttribute(dictionary, tag_for_keyword("PixelSpacing"))
                if pixel_spacing is not None:
                    # Notice that DICOM row spacing comes first, column spacing second!
                    dr = float(pixel_spacing[0])
                    dc = float(pixel_spacing[1])
            except (AttributeError, TypeError) as e:
                logger.debug('{}: {}'.format(_name, e))
                pass
            try:
                slice_spacing = float(self.getDicomAttribute(dictionary, tag_for_keyword("SpacingBetweenSlices")))
            except TypeError:
                try:
                    slice_spacing = float(self.getDicomAttribute(dictionary, tag_for_keyword("SliceThickness")))
                except TypeError:
                    slice_spacing = 1.0
            return np.array([slice_spacing, dr, dc])

        def _verify_no_gantry_tilt(dictionary: DatasetList):
            try:
                gantry = self.getDicomAttributeValues(dictionary, tag_for_keyword("GantryDetectorTilt"))
                if len(gantry) > 1:
                    raise CannotSort('{}: More than one Gantry/Detector Tilt'.format(message))
                elif len(gantry) == 1:
                    if gantry[0] != 0.0:
                        raise CannotSort('{}: Gantry/Detector Tilt is not zero'.format(message))
            except Exception:
                raise

        def _get_orientation(dictionary: DatasetList) -> list[np.ndarray]:
            # iops = self.getDicomAttribute(dictionary, tag_for_keyword("ImageOrientationPatient"))
            orients = []
            for s in range(len(dictionary)):
                try:
                    iop = self.getDicomAttribute(dictionary, tag_for_keyword("ImageOrientationPatient"))
                except ValueError:
                    iop = [0, 0, 1, 0, 1, 0]
                if iop is not None:
                    orient = np.array((iop[2], iop[1], iop[0],
                                       iop[5], iop[4], iop[3]))
                    orients.append(orient)

            if self.dir_cosine_tolerance == 0.0:
                if len(orients) != 1:
                    found = None
                    for it in orients:
                        if found is None:
                            found = it
                        elif (it!=found).all():
                            raise CannotSort('{}: More than one IOP. Try changing dir_cosine_tolerance'.format(message))
                    if found is None:
                        raise CannotSort('{}: No IOP.'.format(message))
            return orients[0]

        def _verify_single_frame_of_reference(dictionary: DatasetList):
            frames = self.getDicomAttributeValues(dictionary, tag_for_keyword("FrameOfReferenceUID"))
            frames = sorted(set(frames))
            if len(frames) != 1:
                logger.warning('{}: Multiple values of FrameOfReferenceUID'.format(message))

        def _calculate_distances(dictionary: DatasetList, orient: np.ndarray, spacing: np.ndarray,
                                 opts: dict = None)\
                -> list[np.ndarray, np.ndarray]:
            _name: str = '{}.{}'.format(__name__, _calculate_distances.__name__)
            sort_on_slice_location = False
            if 'sort_on_slice_location' in opts:
                sort_on_slice_location = opts['sort_on_slice_location']
            # Calculate slice normal from IOP, will be the same for all slices
            colr = np.array(orient[:3]).reshape(3, 1)
            colc = np.array(orient[3:]).reshape(3, 1)
            colr = colr / np.linalg.norm(colr)
            colc = colc / np.linalg.norm(colc)
            normal = np.cross(colc, colr, axis=0).reshape(3)
            # For each slice, calculate distance along the slice normal using IPP
            distances = []
            ipps = []
            for _slice in range(len(dictionary)):
                ipp = self.getOriginForSlice(dictionary, _slice)
                if self.dir_cosine_tolerance != 0.0:
                    orient2 = orient[_slice]
                    colr2 = np.array(orient2[:3]).reshape(3, 1)
                    colc2 = np.array(orient2[3:]).reshape(3, 1)
                    colr2 = colr2 / np.linalg.norm(colr2)
                    colc2 = colc2 / np.linalg.norm(colc2)
                    normal2 = np.cross(colr2, colc2, axis=0)
                    cd = sum(normal[:] * normal2[:])[0]
                    if np.fabs(1 - cd) > self.dir_cosine_tolerance:
                        raise CannotSort('{}: Problem with dir_cosine_tolerance'.format(message))
                dist = np.dot(normal, ipp)
                distances.append(dist)
                ipps.append(ipp)
            # Determine sorting of the slices based on distance
            distances = np.array(distances)
            distance_idx = np.argsort(distances)
            unique_distances = np.unique(distances)

            # Construct transformationMatrix
            slices = len(unique_distances)
            T0 = ipps[distance_idx[0]]
            Tn = ipps[distance_idx[-1]]
            k = ((Tn - T0) / (slices - 1))
            transform = np.eye(4)
            transform[:3, :4] = np.hstack([
                k.reshape(3, 1),
                colc.reshape(3, 1) * spacing[1],
                colr.reshape(3, 1) * spacing[2],
                T0.reshape(3, 1)])

            if sort_on_slice_location:
                # If we do not trust sorting on ipp, repeat with slice locations
                distances = []
                for _slice in range(len(dictionary)):
                    try:
                        dist = float(self.getDicomAttribute(dictionary, tag_for_keyword("SliceLocation"), _slice))
                    except TypeError:
                        raise CannotSort('{}: Missing SliceLocation'.format(message))
                    distances.append(dist)
                distances = np.array(distances)
                distance_idx = np.argsort(distances)
                unique_distances = np.unique(distances)

            if len(unique_distances) != slices:
                raise CannotSort('{}: Problem with sorting, {} unique distances do not match {} slices'.format(
                    message, len(unique_distances), slices
                ))

            # Sort imagePositions
            imagePositions = {}
            for i in range(slices):
                pos = np.where(distances == unique_distances[i])[0][0]
                imagePositions[i] = ipps[pos]
            return distances, distance_idx, transform, imagePositions

        def _verify_spacing(distances: np.ndarray):
            # Verify spacing
            spacings = []
            spacing_is_good = True
            has_warned = False
            d = np.unique(distances)
            prev = d[0]
            if len(d) > 1:
                current = d[1]
                slice_spacing = current - prev
                for it in range(1, len(d)):
                    current = d[it]
                    spacings.append(abs(current - prev))
                    if abs(current - prev) - slice_spacing > self.slice_tolerance:
                        if not has_warned:
                            logger.warning('{}: Slice spacing differs too much, {} vs {}. Decrease slice_tolerance.'.format(
                                message,
                                abs(current - prev), slice_spacing
                                ))
                            has_warned = True
                        spacing_is_good = False
                    prev = current
            if not spacing_is_good:
                raise CannotSort('{}: Slice spacing varies:\n  Distances: {}\n  Spacing: {}'.format(message, distances, spacings))

        spacing = _get_spacing(dictionary)
        _verify_no_gantry_tilt(dictionary)
        orient = _get_orientation(dictionary)
        _verify_single_frame_of_reference(dictionary)
        distances, distance_idx, transform, ipps = _calculate_distances(dictionary, orient, spacing, opts)
        _verify_spacing(distances)

        # Sort dataset on distances
        sorted_dataset: SortedDatasetList = SortedDatasetList()
        sorted_dataset.spacing = spacing
        sorted_dataset.transformationMatrix = transform
        sorted_dataset.imagePositions = ipps
        for idx in distance_idx:
            distance = distances[idx]
            # Catalog images with distance (sloc) as key
            sorted_dataset[distance].append(dictionary[idx])

        return sorted_dataset

    def getOriginForSlice(self, dictionary, slice):
        """Get origin of given slice.

        Args:
            self: DICOMPlugin instance
            dictionary: image dictionary
            slice: slice number (int)
        Returns:
            z,y,x: coordinate for origin of given slice (np.array)
        """

        try:
            origin = self.getDicomAttribute(dictionary, tag_for_keyword("ImagePositionPatient"), slice)
            if origin is not None:
                x = float(origin[0])
                y = float(origin[1])
                z = float(origin[2])
                return np.array([z, y, x])
        except TypeError:
            pass
        if issubclass(type(slice), Dataset):
            d = slice
            s = 0
        else:
            d = dictionary
            s = slice
        while not issubclass(type(d), Dataset):
            if issubclass(type(d), dict):
                d = d[next(iter(d))]
            elif issubclass(type(d), (tuple, list)):
                for d in d:
                    if issubclass(type(d), Dataset):
                        break
        try:
            origin = self.getDicomAttribute(d, tag_for_keyword("ImagePositionPatient"), s)
        except TypeError:
            origin = self.getDicomAttribute(slice, tag_for_keyword("ImagePositionPatient"), 0)
        if origin is not None:
            x = float(origin[0])
            y = float(origin[1])
            z = float(origin[2])
            return np.array([z, y, x])
        return None

    # noinspection PyPep8Naming
    def setDicomAttribute(self, dictionary, tag, value):
        """Set a given DICOM attribute to the provided value.

        Ignore if no real dicom header exists.

        Args:
            self: DICOMPlugin instance
            dictionary: image dictionary
            tag: DICOM tag of addressed attribute.
            value: Set attribute to this value.
        """
        if dictionary is not None:
            for _slice in dictionary:
                for tg, im in dictionary[_slice]:
                    if tag not in im:
                        VR = pydicom.datadict.dictionary_VR(tag)
                        im.add_new(tag, VR, value)
                    else:
                        im[tag].value = value

    def getDicomAttributeValues(self, dictionary, tag) -> list:
        values = []
        for s in range(len(dictionary)):
            value = self.getDicomAttribute(dictionary, tag, s)
            if value is not None:
                values.append(value)
        return values

    def getDicomAttribute(self, dictionary, tag, slice=0):
        """Get DICOM attribute from first image for given slice.

        Args:
            self: DICOMPlugin instance
            dictionary: image dictionary
            tag: DICOM tag of requested attribute.
            slice: which slice to access. Default: slice=0
        """
        assert dictionary is not None, "dicomplugin.getDicomAttribute: dictionary is None"
        try:
            _, im = dictionary[slice][next(iter(dictionary[slice]))]
        except TypeError:
            try:
                _, im = dictionary[slice][0]
            except TypeError:
                im = dictionary[slice]
        except KeyError:
            try:
                im = dictionary[slice]
            except KeyError:
                im = dictionary
        if tag in im:
            return im[tag].value
        else:
            return None

    def removePrivateTags(self, dictionary):
        """Remove private DICOM attributes.

        Ignore if no real dicom header exists.

        Args:
            self: DICOMPlugin instance
            dictionary: image dictionary
        """
        if dictionary is not None:
            for _slice in dictionary:
                for tg, im in dictionary[_slice]:
                    im.remove_private_tags()

    @staticmethod
    def _get_pixels_with_shape(im, shape):
        """Get pixels from image object. Reshape image to given shape

        Args:
            im: dicom image
            shape: requested image shape
        Returns:
            si: numpy array of given shape
        """

        _name: str = '{}.{}'.format(__name__, '_get_pixels_with_shape')
        _use_float = False
        try:
            if 'RescaleSlope' in im and 'RescaleIntercept' in im:
                _use_float = abs(im.RescaleSlope - 1) > 1e-4 or abs(im.RescaleIntercept) > 1e-4
            if _use_float:
                pixels = float(im.RescaleSlope) * im.pixel_array.astype(float) + \
                    float(im.RescaleIntercept)
            else:
                pixels = im.pixel_array
            if shape != pixels.shape:
                if im.PhotometricInterpretation == 'RGB':
                    # RGB image
                    rgb_dtype = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])
                    si = pixels.copy().view(dtype=rgb_dtype).reshape(pixels.shape[:-1])
                elif 'NumberOfFrames' in im:
                    logger.debug('{}: NumberOfFrames: {}'.format(_name, im.NumberOfFrames))
                    if (im.NumberOfFrames,) + shape == pixels.shape:
                        logger.debug('{}: NumberOfFrames {} copy pixels'.format(_name, im.NumberOfFrames))
                        si = pixels
                    else:
                        logger.debug('{}: NumberOfFrames pixels differ {} {}'.format(
                            _name, (im.NumberOfFrames,) + shape, pixels.shape))
                        raise IndexError(
                            'NumberOfFrames pixels differ {} {}'.format(
                                (im.NumberOfFrames,) + shape, pixels.shape)
                        )
                else:
                    # This happens only when images in a series have varying shape
                    # Place the pixels in the upper left corner of the matrix
                    assert len(shape) == len(pixels.shape), \
                        "Shape of matrix ({}) differ from pixel shape ({})".format(
                            shape, pixels.shape)
                    # Assume that pixels can be expanded to match si shape
                    si = np.zeros(shape, pixels.dtype)
                    roi = []
                    for d in pixels.shape:
                        roi.append(slice(d))
                    roi = tuple(roi)
                    si[roi] = pixels
            else:
                si = pixels
        except UnboundLocalError:
            # A bug in pydicom appears when reading binary images
            if im.BitsAllocated == 1:
                logger.debug(
                    "{}: Binary image, image.shape={}, image shape=({},{},{})".format(
                        _name, im.shape, im.NumberOfFrames, im.Rows, im.Columns))
                _myarr = np.frombuffer(im.PixelData, dtype=np.uint8)
                # Reverse bit order, and copy the array to get a
                # contiguous array
                bits = np.unpackbits(_myarr).reshape(-1, 8)[:, ::-1].copy()
                si = np.fliplr(
                    bits.reshape(
                        1, im.NumberOfFrames, im.Rows, im.Columns))
                if _use_float:
                    si = float(im.RescaleSlope) * si + float(im.RescaleIntercept)
            else:
                raise
        # Delete pydicom's pixel data to save memory
        # image._pixel_array = None
        # if 'PixelData' in image:
        #    image[0x7fe00010].value = None
        #    image[0x7fe00010].is_undefined_length = True
        return si

    def _read_image(self, f, opts, hdr):
        """Read image data from given file handle

        Args:
            self: format plugin instance
            f: file handle or filename (depending on self._need_local_file)
            opts: Input options (dict)
            hdr: Header
        Returns:
            Tuple of
                - hdr: Header
                    Return values:
                    - info: Internal data for the plugin
                          None if the given file should not be included (e.g. raw file)
                - si: numpy array (multi-dimensional)
        """

        pass

    def _set_tags(self, image_list, hdr, si):
        """Set header tags.

        Args:
            self: format plugin instance
            image_list: list with (info,img) tuples
            hdr: Header
            si: numpy array (multi-dimensional)
        Returns:
            hdr: Header
        """

        pass

    def _process_image_members(self,
                               image_dict: DatasetDict,
                               opts: dict = None,
                               skip_pixels: bool = False
                               ) -> SortedDatasetDict:
        """Sort files on Series Instance UID

        Args:
            self: DICOMPlugin instance
            image_dict:
            opts: input options (dict)
            skip_pixels: Do not read pixel data (default: False)
        Returns:
            Dict
                - key: SeriesUID
                - value: dict
                    - key: float
                    - value: list of Dataset
        """

        _name: str = '{}.{}'.format(__name__, self._process_image_members.__name__)

        logger.debug('{}:'.format(_name))

        sorted_dataset_dict: SortedDatasetDict = SortedDatasetDict()
        # Sort datasets on sloc
        for seriesUID in image_dict:
            dataset_list = image_dict[seriesUID]
            for dataset in dataset_list:
                try:
                    logger.debug('{}: process_member {}'.format(_name, dataset))
                    self._sort_datasets(sorted_dataset_dict, seriesUID, dataset, opts, skip_pixels=skip_pixels)
                except Exception as e:
                    logger.debug('{}: Exception {}'.format(_name, e))
            # Sort datasets on tag
            sorted_dataset_dict[seriesUID] = self._sort_images

        return sorted_dataset_dict

    def _correct_acqtimes_for_dynamic_series(self, hdr: Header, si: np.ndarray):
        # si[t,slice,rows,columns]

        _name: str = '{}.{}'.format(__name__, self._correct_acqtimes_for_dynamic_series.__name__)

        # Extract acqtime for each image
        slices = len(hdr.sliceLocations)
        timesteps = self._count_timesteps(hdr)
        logger.info(
            "{}: Slices: {}, apparent time steps: {}, actual time steps: {}".format(
                _name, slices, len(hdr.tags), timesteps))
        new_shape = (timesteps, slices, si.shape[2], si.shape[3])
        newsi = np.zeros(new_shape, dtype=si.dtype)
        acq = np.zeros([slices, timesteps])
        for _slice in self.DicomHeaderDict:
            t = 0
            for tg, im in self.DicomHeaderDict[_slice]:
                acq[_slice, t] = tg
                t += 1

        # Correct acqtimes by setting acqtime for each slice of a volume to
        # the smallest time
        for t in range(acq.shape[1]):
            min_acq = np.min(acq[:, t])
            for _slice in range(acq.shape[0]):
                acq[_slice, t] = min_acq

        # Set new acqtime for each image
        for _slice in self.DicomHeaderDict:
            t = 0
            for tg, im in self.DicomHeaderDict[_slice]:
                im.AcquisitionTime = "%f" % acq[_slice, t]
                newsi[t, _slice, :, :] = si[t, _slice, :, :]
                t += 1

        # Update taglist in hdr
        hdr.tags = {}
        for _slice in self.DicomHeaderDict:
            hdr.tags[_slice] = np.empty((acq.shape[1],))
            for t in range(acq.shape[1]):
                hdr.tags[_slice][t] = acq[0, t]
        return newsi

    @staticmethod
    def _count_timesteps(hdr):
        slices = len(hdr.sliceLocations)
        timesteps = np.zeros([slices], dtype=int)
        for _slice in hdr.DicomHeaderDict:
            timesteps[_slice] = len(hdr.DicomHeaderDict[_slice])
            if timesteps.min() != timesteps.max():
                raise ValueError("Number of time steps ranges from %d to %d." % (
                    timesteps.min(), timesteps.max()))
        return timesteps.max()

    def write_3d_numpy(self, si: Series, destination, opts):
        """Write 3D Series image as DICOM files

        Args:
            self: DICOMPlugin instance
            si: Series array (3D or 4D)
            destination: dict of archive and filenames
            opts: Output options (dict)
        """

        _name: str = '{}.{}'.format(__name__, self.write_3d_numpy.__name__)

        logger.debug('{}: destination {}'.format(_name, destination))
        archive = destination['archive']
        archive.set_member_naming_scheme(
            fallback='Image_{:05d}.dcm',
            level=max(0, si.ndim - 2),
            default_extension='.dcm',
            extensions=self.extensions
        )
        self.keep_uid = False if 'keep_uid' not in opts else opts['keep_uid']

        self.instanceNumber = 0

        logger.debug('{}: orig shape {}, slices {} len {}'.format(
            _name, si.shape, si.slices, si.ndim))
        assert si.ndim == 2 or si.ndim == 3, \
            "write_3d_series: input dimension %d is not 2D/3D." % si.ndim

        self._calculate_rescale(si)
        logger.info("{}: Smallest/largest pixel value in series: {}/{}".format(
            _name, self.smallestPixelValueInSeries, self.largestPixelValueInSeries))
        if 'window' in opts and opts['window'] == 'original':
            raise ValueError('No longer supported: opts["window"] is set')
        self.center = si.windowCenter
        self.width = si.windowWidth
        self.today = date.today().strftime("%Y%m%d")
        self.now = datetime.now().strftime("%H%M%S.%f")
        # Set series instance UID when writing
        self.serInsUid = si.header.seriesInstanceUID if self.keep_uid else si.header.new_uid()
        logger.debug("{}: {}".format(_name, self.serInsUid))
        for key, value in opts.items():  # Copy opts to self.input_options
            self.input_options[key] = value

        if pydicom.uid.UID(si.SOPClassUID).keyword == 'EnhancedMRImageStorage' or \
                pydicom.uid.UID(si.SOPClassUID).keyword == 'EnhancedCTImageStorage':
            # Write Enhanced CT/MR
            self.write_enhanced(si, destination)
        else:
            # Either legacy CT/MR, or another modality
            if si.ndim < 3:
                logger.debug('{}: write 2D ({})'.format(_name, si.ndim))
                if self.keep_uid:
                    sop_ins_uid = si.SOPInstanceUIDs[(0, 0)]
                else:
                    sop_ins_uid = si.header.new_uid()
                self.write_slice('none', None, si, destination, 0,
                                 sop_ins_uid=sop_ins_uid)
            else:
                logger.debug('{}: write 3D slices {}'.format(_name, si.slices))
                for _slice in range(si.slices):
                    if self.keep_uid:
                        sop_ins_uid = si.SOPInstanceUIDs[(0, _slice)]
                    else:
                        sop_ins_uid = si.header.new_uid()
                    try:
                        self.write_slice('none', (_slice,), si[_slice], destination, _slice,
                                         sop_ins_uid=sop_ins_uid)
                    except Exception as e:
                        print('DICOMPlugin.write_slice Exception: {}'.format(e))
                        traceback.print_exc(file=sys.stdout)
                        raise

    def write_4d_numpy(self, si: Series, destination, opts):
        """Write 4D Series image as DICOM files

        si.series_number is inserted into each dicom object

        si.series_description is inserted into each dicom object

        si.image_type: Dicom image type attribute

        opts['output_sort']: Which tag will sort the output images (slice or tag)

        opts['output_dir']: Store all images in a single or multiple directories

        Args:
            self: DICOMPlugin instance
            si: Series array si[tag,slice,rows,columns]
            destination: dict of archive and filenames
            opts: Output options (dict)

        """

        _name: str = '{}.{}'.format(__name__, self.write_4d_numpy.__name__)

        logger.debug('{}: destination {}'.format(_name, destination))
        archive = destination['archive']
        self.keep_uid = False if 'keep_uid' not in opts else opts['keep_uid']

        # Defaults
        self.output_sort = SORT_ON_SLICE
        if 'output_sort' in opts:
            self.output_sort = opts['output_sort']
        self.output_dir = 'single'
        if 'output_dir' in opts:
            self.output_dir = opts['output_dir']

        self.instanceNumber = 0

        logger.debug('{}: orig shape {}, len {}'.format(_name, si.shape, si.ndim))
        assert si.ndim >= 4, "write_4d_series: input dimension %d is less than 4D." % si.ndim

        tags = si.tags[0].ndim
        steps = si.shape[:tags]
        self._calculate_rescale(si)
        logger.info("{}: Smallest/largest pixel value in series: {}/{}".format(
            _name, self.smallestPixelValueInSeries, self.largestPixelValueInSeries))
        self.today = date.today().strftime("%Y%m%d")
        self.now = datetime.now().strftime("%H%M%S.%f")
        # Not used # self.seriesTime = obj.getDicomAttribute(tag_for_keyword("AcquisitionTime"))
        # Set series instance UID when writing
        if not self.keep_uid:
            si.header.seriesInstanceUID = si.header.new_uid()
        self.serInsUid = si.header.seriesInstanceUID
        self.input_options = opts

        if pydicom.uid.UID(si.SOPClassUID).keyword == 'EnhancedMRImageStorage' or \
                pydicom.uid.UID(si.SOPClassUID).keyword == 'EnhancedCTImageStorage':
            # Write Enhanced CT/MR
            self.write_enhanced(si, destination)
            return

        # Either legacy CT/MR, or another modality
        if self.output_sort == SORT_ON_SLICE:
            if self.output_dir == 'single':
                # Filenames: Image_00000.dcm, sort slice fastest
                archive.set_member_naming_scheme(
                    fallback='Image_{:05d}.dcm',
                    level=1,
                    default_extension='.dcm',
                    extensions=self.extensions
                )
            else:  # self.output_dir == 'multi'
                # Filenames: Tag0/../TagN/Image_00000.dcm, sort slice fastest
                dirn = []
                for i, order in enumerate(si.input_order.split(',')):
                    digits = len("{}".format(steps[i]))
                    dirn.append(
                        "{0}{{{1}:0{2}}}".format(
                            order,
                            i,
                            digits)
                    )
                archive.set_member_naming_scheme(
                    fallback=os.path.join(
                        *dirn,
                        'Image_{' + '{}'.format(len(dirn)) + ':05d}.dcm'),
                    level=max(0, si.ndim - 2),
                    default_extension='.dcm',
                    extensions=self.extensions
                )
            ifile = 0
            for tag in np.ndindex(steps):
                for _slice in range(si.slices):
                    _tag = tag + (_slice,)
                    if self.keep_uid:
                        sop_ins_uid = si.SOPInstanceUIDs[tag + (_slice,)]
                    else:
                        sop_ins_uid = si.header.new_uid()
                    if self.output_dir == 'multi' and _slice == 0:
                        # Restart file number in each subdirectory
                        ifile = 0
                    if self.output_dir == 'multi':
                        _file_tag = _tag
                    else:
                        _file_tag = (ifile,)
                    try:
                        _t = si.header.tags[_slice][tag]
                        if _t is None:
                            continue
                        self.write_slice(si.input_order, _file_tag, si[_tag],
                                         destination, ifile,
                                         tag_value=si.header.tags[_slice][tag],
                                         sop_ins_uid=sop_ins_uid)
                    except Exception as e:
                        print('DICOMPlugin.write_slice Exception: {}'.format(e))
                        traceback.print_exc(file=sys.stdout)
                        raise
                    ifile += 1
        else:  # self.output_sort == SORT_ON_TAG:
            if self.output_dir == 'single':
                # Filenames: Image_00000.dcm, sort tags fastest
                archive.set_member_naming_scheme(
                    fallback='Image_{:05d}.dcm',
                    level=1,
                    default_extension='.dcm',
                    extensions=self.extensions
                )
            else:  # self.output_dir == 'multi'
                # Filenames: slice/tag0/../tagN/Image_00000.dcm, sort tags fastest
                digits = len("{}".format(si.slices))
                dirn = ["slice{{0:0{0}}}".format(digits)]
                for i, order in enumerate(si.input_order.split(',')[:-1]):
                    digits = len("{}".format(steps[i]))
                    dirn.append(
                        "{0}{{{1}:0{2}}}".format(
                            order,
                            i+1,
                            digits
                        )
                    )
                order = si.input_order.split(',')[-1]
                digits = len("{}".format(steps[-1]))
                archive.set_member_naming_scheme(
                    fallback=os.path.join(
                        *dirn,
                        order + '{' + '{}'.format(len(dirn)) + ':0{}'.format(digits) + '}.dcm'),
                    level=max(0, si.ndim - 2),
                    default_extension='.dcm',
                    extensions=self.extensions
                )
            ifile = 0
            for _slice in range(si.slices):
                for tag in np.ndindex(steps):
                    _tag = (_slice,) + tag
                    if self.keep_uid:
                        sop_ins_uid = si.SOPInstanceUIDs[tag + (_slice,)]
                    else:
                        sop_ins_uid = si.header.new_uid()
                    if self.output_dir == 'multi' and tag == 0:
                        # Restart file number in each subdirectory
                        ifile = 0
                    if self.output_dir == 'multi':
                        _file_tag = _tag
                    else:
                        _file_tag = (ifile,)
                    try:
                        _t = si.header.tags[_slice][tag]
                        if _t is None:
                            continue
                        self.write_slice(si.input_order, _file_tag, si[tag + (_slice,)],
                                         destination, ifile,
                                         tag_value=si.header.tags[_slice][tag],
                                         sop_ins_uid=sop_ins_uid)
                    except Exception as e:
                        print('DICOMPlugin.write_slice Exception: {}'.format(e))
                        traceback.print_exc(file=sys.stdout)
                        raise
                    ifile += 1

    def write_enhanced(self, si, archive, filename_template, opts):
        """Write enhanced CT/MR object to DICOM file

        Args:
            self: DICOMPlugin instance
            si: Series instance, including these attributes:
            archive: archive object
            filename_template: file name template, possible without '.dcm' extension
            opts: Output options (dict)
        Raises:

        """
        _name: str = '{}.{}'.format(__name__, self.write_enhanced.__name__)

        filename = 'dummy'
        logger.debug("{}: {} {}".format(_name, filename, self.serInsUid))

        try:
            tg, member_name, im = si.DicomHeaderDict[0][0]
        except (KeyError, IndexError):
            raise IndexError("Cannot address dicom_template.DicomHeaderDict[0][0]")
        except ValueError:
            raise NoDICOMAttributes("Cannot write DICOM object when no DICOM attributes exist.")
        logger.debug("{}: member_name {}".format(_name, member_name))
        self.keep_uid = False if 'keep_uid' not in opts else opts['keep_uid']
        if not self.keep_uid:
            si.header.seriesInstanceUID = si.header.new_uid()
        self.serInsUid = si.header.seriesInstanceUID

        ds = self.construct_enhanced_dicom(filename_template, im, si)

        # Add header information
        try:
            ds.SliceLocation = si.sliceLocations[0]
        except (AttributeError, ValueError):
            # Dont know the SliceLocation, attempt to calculate from image geometry
            try:
                ds.SliceLocation = self._calculate_slice_location(im)
            except ValueError:
                # Dont know the SliceLocation, so will set this to be the slice index
                ds.SliceLocation = slice
        try:
            dz, dy, dx = si.spacing
        except ValueError:
            dz, dy, dx = 1, 1, 1
        ds.PixelSpacing = [str(dy), str(dx)]
        ds.SliceThickness = str(dz)
        try:
            ipp = si.imagePositions
            if len(ipp) > 0:
                ipp = ipp[0]
            else:
                ipp = np.array([0, 0, 0])
        except ValueError:
            ipp = np.array([0, 0, 0])
        if ipp.shape == (3, 1):
            ipp.shape = (3,)
        z, y, x = ipp[:]
        ds.ImagePositionPatient = [str(x), str(y), str(z)]
        # Reverse orientation vectors from zyx to xyz
        try:
            ds.ImageOrientationPatient = [
                si.orientation[2], si.orientation[1], si.orientation[0],
                si.orientation[5], si.orientation[4], si.orientation[3]]
        except ValueError:
            ds.ImageOrientationPatient = [0, 0, 1, 0, 1, 0]
        try:
            ds.SeriesNumber = si.seriesNumber
        except ValueError:
            ds.SeriesNumber = 1
        try:
            ds.SeriesDescription = si.seriesDescription
        except ValueError:
            ds.SeriesDescription = ''
        try:
            ds.ImageType = "\\".join(si.imageType)
        except ValueError:
            ds.ImageType = 'DERIVED\\SECONDARY'
        try:
            ds.FrameOfReferenceUID = si.frameOfReferenceUID
        except ValueError:
            pass

        ds.SmallestPixelValueInSeries = np.uint16(self.smallestPixelValueInSeries)
        ds.LargestPixelValueInSeries = np.uint16(self.largestPixelValueInSeries)
        ds[0x0028, 0x0108].VR = 'US'
        ds[0x0028, 0x0109].VR = 'US'
        ds.WindowCenter = self.center
        ds.WindowWidth = self.width
        if si.dtype in self.smallint or np.issubdtype(si.dtype, np.bool_):
            ds.SmallestImagePixelValue = np.uint16(si.min().astype('uint16'))
            ds.LargestImagePixelValue = np.uint16(si.max().astype('uint16'))
            if 'RescaleSlope' in ds:
                del ds.RescaleSlope
            if 'RescaleIntercept' in ds:
                del ds.RescaleIntercept
        else:
            ds.SmallestImagePixelValue = np.uint16((si.min().item() - self.b) / self.a)
            ds.LargestImagePixelValue = np.uint16((si.max().item() - self.b) / self.a)
            try:
                ds.RescaleSlope = "%f" % self.a
            except OverflowError:
                ds.RescaleSlope = "%d" % int(self.a)
            ds.RescaleIntercept = "%f" % self.b
        ds[0x0028, 0x0106].VR = 'US'
        ds[0x0028, 0x0107].VR = 'US'
        # General Image Module Attributes
        ds.InstanceNumber = 1
        ds.ContentDate = self.today
        ds.ContentTime = self.now
        # ds.AcquisitionTime = self.add_time(self.seriesTime, timeline[tag])
        ds.Rows = si.rows
        ds.Columns = si.columns
        self._insert_pixel_data(ds, si)
        # logger.debug("write_enhanced: filename {}".format(filename))

        # Set tag
        # si will always have only the present tag
        self._set_dicom_tag(ds, si.input_order, si.tags[0])

        if len(os.path.splitext(filename)[1]) > 0:
            fn = filename
        else:
            fn = filename + '.dcm'
        logger.debug("{}: filename {}".format(_name, fn))
        # if archive.transport.name == 'dicom':
        #     # Store dicom set ds directly
        #     archive.transport.store(ds)
        # else:
        #     # Store dicom set ds as file
        #     with archive.open(fn, 'wb') as f:
        #         ds.save_as(f)
        raise ValueError("write_enhanced: to be implemented")

    # noinspection PyPep8Naming,PyArgumentList
    def write_slice(self, input_order, tag, si, destination, ifile,
                    tag_value=None,
                    sop_ins_uid=None):
        """Write single slice to DICOM file

        Args:
            self: DICOMPlugin instance
            input_order: input order
            tag: tag index
            si: Series instance, including these attributes:
            -   slices
            -   sliceLocations
            -   dicomTemplate
            -   dicomToDo
            -   seriesNumber
            -   seriesDescription
            -   imageType
            -   frame
            -   spacing
            -   orientation
            -   imagePositions
            -   photometricInterpretation

            destination: destination object
            ifile: instance number in series
            tag_value: set tag value
            sop_ins_uid: set SOP Instance UID
        """

        _name: str = '{}.{}'.format(__name__, self.write_slice.__name__)

        archive: AbstractArchive = destination['archive']
        query = None
        if destination['files'] and len(destination['files']):
            query = destination['files'][0]
        filename = archive.construct_filename(
            tag=tag,
            query=query
        )
        logger.debug("{}: {} {}".format(_name, filename, self.serInsUid))

        try:
            ds = self.construct_dicom(filename, si.dicomTemplate, si, sop_ins_uid=sop_ins_uid)
        except ValueError:
            ds = self.construct_basic_dicom(si, sop_ins_uid=sop_ins_uid)
            ds.SeriesInstanceUID = si.header.seriesInstanceUID

        # Add header information
        try:
            ds.SliceLocation = pydicom.valuerep.format_number_as_ds(float(si.sliceLocations[0]))
        except (AttributeError, ValueError):
            # Do not know the SliceLocation, so will set this to be the slice index
            if tag is None:
                ds.SliceLocation = 0
            else:
                ds.SliceLocation = tag[-1]
        try:
            dz, dy, dx = si.spacing
        except ValueError:
            dz, dy, dx = 1, 1, 1
        ds.PixelSpacing = [pydicom.valuerep.format_number_as_ds(float(dy)),
                           pydicom.valuerep.format_number_as_ds(float(dx))]
        ds.SliceThickness = pydicom.valuerep.format_number_as_ds(float(dz))
        try:
            ipp = si.imagePositions
            if len(ipp) > 0:
                ipp = ipp[0]
            else:
                ipp = np.array([0, 0, 0])
        except ValueError:
            ipp = np.array([0, 0, 0])
        if ipp.shape == (3, 1):
            ipp.shape = (3,)
        z, y, x = ipp[:]
        ds.ImagePositionPatient = [pydicom.valuerep.format_number_as_ds(float(x)),
                                   pydicom.valuerep.format_number_as_ds(float(y)),
                                   pydicom.valuerep.format_number_as_ds(float(z))]
        # Reverse orientation vectors from zyx to xyz
        try:
            ds.ImageOrientationPatient = [
                pydicom.valuerep.format_number_as_ds(float(si.orientation[2])),
                pydicom.valuerep.format_number_as_ds(float(si.orientation[1])),
                pydicom.valuerep.format_number_as_ds(float(si.orientation[0])),
                pydicom.valuerep.format_number_as_ds(float(si.orientation[5])),
                pydicom.valuerep.format_number_as_ds(float(si.orientation[4])),
                pydicom.valuerep.format_number_as_ds(float(si.orientation[3]))]
        except ValueError:
            ds.ImageOrientationPatient = [0, 0, 1, 0, 0, 1]
        try:
            ds.SeriesNumber = si.seriesNumber
        except ValueError:
            ds.SeriesNumber = 1
        try:
            ds.SeriesDescription = si.seriesDescription
        except ValueError:
            ds.SeriesDescription = ''
        try:
            ds.ImageType = "\\".join(si.imageType)
        except ValueError:
            ds.ImageType = 'DERIVED\\SECONDARY'
        try:
            ds.FrameOfReferenceUID = si.frameOfReferenceUID
        except ValueError:
            pass

        # Add DICOM To Do items to present slice
        for _attr, _value, _slice, _tag in si.header.dicomToDo:
            _this_slice = True if _slice is None else _slice == tag[-1]
            _this_tag = True if _tag is None else _tag == tag
            if _this_slice and _this_tag:
                # Set Dicom Attribute
                if _attr not in ds:
                    VR = pydicom.datadict.dictionary_VR(_attr)
                    ds.add_new(_attr, VR, _value)
                else:
                    ds[_attr].value = _value

        self._set_pixel_rescale(ds, si)

        # General Image Module Attributes
        ds.InstanceNumber = ifile + 1
        ds.ContentDate = self.today
        ds.ContentTime = self.now
        # ds.AcquisitionTime = self.add_time(self.seriesTime, timeline[tag])
        ds.Rows = si.rows
        ds.Columns = si.columns
        self._insert_pixel_data(ds, si)

        # Set tag
        # si will always have only the present tag
        self._set_dicom_tag(ds, input_order, tag_value)

        logger.debug("{}: filename {}".format(_name, filename))
        if archive.transport.name == 'dicom':
            # Store dicom set ds directly
            archive.transport.store(ds)
        else:
            # Store dicom set ds as file
            with archive.open(filename, 'wb') as f:
                ds.save_as(f)

    def construct_basic_dicom(self,
                              template: Series = None,
                              filename: str = 'NA',
                              sop_ins_uid: str = None
                              ) -> FileDataset:

        if sop_ins_uid is None:
            raise ValueError('SOPInstanceUID is undefined.')
        # Populate required values for file meta information
        file_meta = FileMetaDataset()
        sop_class_uid = getattr(template, 'SOPClassUID', None)
        if sop_class_uid is None:
            sop_class_uid = '1.2.840.10008.5.1.4.1.1.7'
        file_meta.MediaStorageSOPClassUID = sop_class_uid
        if sop_ins_uid is not None:
            file_meta.MediaStorageSOPInstanceUID = sop_ins_uid
        else:
            file_meta.MediaStorageSOPInstanceUID = template.header.new_uid()
        file_meta.ImplementationClassUID = "%s.1" % self.root
        file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

        # Create the FileDataset instance
        # (initially no data elements, but file_meta supplied)
        ds = FileDataset(
            filename,
            {},
            file_meta=file_meta,
            preamble=b"\0" * 128)
        ds.SOPClassUID = sop_class_uid
        ds.SOPInstanceUID = sop_ins_uid
        ds.PatientName = 'NA'
        ds.PatientID = 'NA'
        ds.PatientBirthDate = '00000000'
        ds.PatientSex = 'O'
        ds.StudyDate = self.today
        ds.StudyTime = '000000'
        try:
            ds.StudyInstanceUID = template.header.studyInstanceUID
            ds.SeriesInstanceUID = template.header.seriesInstanceUID
        except Exception as e:
            print(e)
        ds.StudyID = '0'
        ds.ReferringPhysicianName = 'NA'
        ds.AccessionNumber = 'NA'
        ds.Modality = 'SC'
        return ds

    def construct_dicom(self,
                        filename: str,
                        template: Series,
                        si: Series,
                        sop_ins_uid=None) -> FileDataset:

        self.instanceNumber += 1
        if sop_ins_uid is None:
            sop_ins_uid = si.header.new_uid()

        # Populate required values for file meta information
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = si.SOPClassUID
        file_meta.MediaStorageSOPInstanceUID = sop_ins_uid
        file_meta.ImplementationClassUID = "%s.1" % self.root
        file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

        # Create the FileDataset instance
        # (initially no data elements, but file_meta supplied)
        ds = FileDataset(
            filename,
            {},
            file_meta=file_meta,
            preamble=b"\0" * 128)

        # Add the data elements
        # -- not trying to set all required here. Check DICOM standard
        # copy_general_dicom_attributes(template, ds)
        for element in template.iterall():
            if element.tag == 0x7fe00010:
                continue  # Do not copy pixel data, will be added later
            ds.add(element)

        ds.StudyInstanceUID = si.header.studyInstanceUID
        ds.StudyID = si.header.studyID
        ds.SeriesInstanceUID = self.serInsUid
        ds.SOPClassUID = si.SOPClassUID
        ds.SOPInstanceUID = sop_ins_uid

        ds.AccessionNumber = si.header.accessionNumber
        ds.PatientName = si.header.patientName
        ds.PatientID = si.header.patientID
        ds.PatientBirthDate = si.header.patientBirthDate

        return ds

    @staticmethod
    def _copy_dicom_group(groupno, ds_in, ds_out):
        sub_dataset = ds_in.group_dataset(groupno)
        for data_element in sub_dataset:
            if data_element.VR != "SQ":
                ds_out[data_element.tag] = ds_in[data_element.tag]

    def _insert_pixel_data(self, ds, arr):
        """Insert pixel data into dicom object

        If float array, scale to uint16
        """

        ds.SamplesPerPixel = 1
        ds.PixelRepresentation = 1 if np.issubdtype(arr.dtype, np.signedinteger) else 0
        try:
            ds.PhotometricInterpretation = arr.photometricInterpretation
            if arr.photometricInterpretation == 'RGB':
                ds.SamplesPerPixel = 3
                ds.PlanarConfiguration = 0
        except ValueError:
            ds.PhotometricInterpretation = 'MONOCHROME2'

        if arr.dtype in self.smallint:
            # No scaling of pixel values
            ds.PixelData = arr.tobytes()
            if arr.itemsize == 1:
                ds[0x7fe0, 0x0010].VR = 'OB'
                ds.BitsAllocated = 8
                ds.BitsStored = 8
                ds.HighBit = 7
            elif arr.itemsize == 2:
                ds[0x7fe0, 0x0010].VR = 'OW'
                ds.BitsAllocated = 16
                ds.BitsStored = 16
                ds.HighBit = 15
            else:
                raise TypeError('Cannot store {} itemsize {} without scaling'.format(
                    arr.dtype, arr.itemsize))
        elif arr.dtype == np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')]):
            # RGB image
            ds.PixelData = arr.tobytes()
            ds[0x7fe0, 0x0010].VR = 'OB'
            ds.BitsAllocated = 8
            ds.BitsStored = 8
            ds.HighBit = 7
        elif np.issubdtype(arr.dtype, np.bool_):
            # No scaling. Pack bits in 16-bit words
            ds.PixelData = arr.astype('uint16').tobytes()
            ds[0x7fe0, 0x0010].VR = 'OW'
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
        else:
            # Other high precision data type, like float:
            # rescale to uint16
            rescaled = (np.asarray(arr) - self.b) / self.a
            ds.PixelData = rescaled.astype('uint16').tobytes()
            ds[0x7fe0, 0x0010].VR = 'OW'
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15

    def _calculate_rescale(self, arr):
        """Calculate rescale parameters for series.

        y = ax + b
        x in 0:65535 correspond to y in ymin:ymax
        2^16 = 65536 possible steps in 16 bits dicom
        Returns:
            self.a: Rescale slope
            self.b: Rescale intercept
            self.center: Window center
            self.width: Window width
            self.smallestPixelValueInSeries: arr.min()
            self.largestPixelValueInSeries: arr.max()
            self.range_VR: The VR to use for DICOM elements (SS or US)
        """
        _name: str = '{}.{}'.format(__name__, self._calculate_rescale.__name__)

        self.range_VR = 'SS' if np.issubdtype(arr.dtype, np.signedinteger) else 'US'
        self.range_VR = 'US' if arr.color else self.range_VR
        _range = 65536. if self.range_VR == 'US' else 32768.
        # Window center/width
        try:
            ymin = np.min(arr).item()
            ymax = np.max(arr).item()
        except AttributeError:
            ymin = np.min(arr)
            ymax = np.max(arr)
        if issubclass(type(ymin), tuple):
            ymin = 0
            ymax = 255
            self.center = 127
            self.width = 256
        else:
            self.center = (ymax + ymin) / 2
            self.width = max(1, ymax - ymin)
        # y = ax + b,
        if arr.dtype in self.smallint or \
                np.issubdtype(arr.dtype, np.bool_) or \
                arr.dtype == np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')]):
            # No need to rescale
            self.a = None
            self.b = None
        else:
            # Other high precision data type, like float
            # Must rescale data
            self.b = ymin
            if math.fabs(ymax - ymin) > 1e-6:
                self.a = (ymax - ymin) / (_range - 1)
            else:
                self.a = 1.0
            logger.debug("{}: Rescale slope {}, rescale intercept {}".format(
                _name, self.a, self.b
            ))
        self.smallestPixelValueInSeries = ymin
        self.largestPixelValueInSeries = ymax

    def _set_pixel_rescale(self, ds, arr):
        """Set pixel rescale elements:
        - RescaleSlope
        - RescaleIntercept
        - WindowCenter
        - WindowWidth
        - SmallestPixelValueInSeries
        - LargestPixelValueInSeries
        Args:
            self.a: Rescale slope
            self.b: Rescale intercept
            self.center: Window center
            self.width: Window width
            self.smallestPixelValueInSeries: arr.min()
            self.largestPixelValueInSeries: arr.max()
            self.range_VR: The VR to use for DICOM elements (SS or US)
            ds: DICOM dataset
            arr: pixel series
        """
        ds.WindowCenter = pydicom.valuerep.format_number_as_ds(float(self.center))
        ds.WindowWidth = pydicom.valuerep.format_number_as_ds(float(self.width))
        # Remove existing elements
        for element in ['SmallestImagePixelValue', 'LargestImagePixelValue',
                        'SmallestPixelValueInSeries', 'LargestPixelValueInSeries',
                        'RescaleSlope', 'RescaleIntercept']:
            if element in ds:
                del ds[element]
        if self.a is None:
            # No rescale slope
            _min = 0 if arr.color else arr.min()
            _max = 255 if arr.color else arr.max()
            _series_min = 0 if arr.color else self.smallestPixelValueInSeries
            _series_max = 255 if arr.color else self.largestPixelValueInSeries
        else:
            try:
                ds.RescaleSlope = pydicom.valuerep.format_number_as_ds(self.a)
            except OverflowError:
                ds.RescaleSlope = "%d" % int(self.a)
            ds.RescaleIntercept = pydicom.valuerep.format_number_as_ds(float(self.b))
            _min = np.array((arr.min() - self.b) / self.a).astype('uint16')
            _max = np.array((arr.max() - self.b) / self.a).astype('uint16')
            _series_min = np.array(
                (self.smallestPixelValueInSeries - self.b) / self.a).astype('uint16')
            _series_max = np.array(
                (self.largestPixelValueInSeries - self.b) / self.a).astype('uint16')
        ds.add_new(tag_for_keyword('SmallestImagePixelValue'), self.range_VR, _min)
        ds.add_new(tag_for_keyword('LargestImagePixelValue'), self.range_VR, _max)
        ds.add_new(tag_for_keyword('SmallestPixelValueInSeries'), self.range_VR, _series_min)
        ds.add_new(tag_for_keyword('LargestPixelValueInSeries'), self.range_VR, _series_max)

    @staticmethod
    def _add_time(now, add):
        """Add time to present time now

        Args:
            now: string hhmmss.ms
            add: float [s]
        Returns:
            newtime: string hhmmss.ms
        """
        tnow = datetime.strptime(now, "%H%M%S.%f")
        s = int(add)
        ms = (add - s) * 1000.
        tadd = timedelta(seconds=s, milliseconds=ms)
        tnew = tnow + tadd
        return tnew.strftime("%H%M%S.%f")

    def _extract_tag_tuple(self, im: Dataset, faulty: int, input_order: str, opts: dict[str]) -> tuple:
        tag_list = []
        for order in input_order.split(sep=','):
            try:
                tag = self._get_tag(im, order, opts)
            except KeyError:
                if order == INPUT_ORDER_FAULTY:
                    tag = faulty
                else:
                    raise CannotSort('Tag {} not found in dataset'.format(
                        order
                    ))
            except CannotSort:
                raise
            except Exception:
                raise
            if tag is None:
                raise CannotSort("Tag {} not found in data".format(order))
            tag_list.append(tag)
        return tuple(tag_list)

    def _get_tag(self, im: Dataset, input_order: str, opts: dict = None) -> Number:

        try:
            return self.input_options[input_order](im)
        except (KeyError, TypeError):
            try:
                return _get_float(im, self.input_options[input_order])
            except (AttributeError, KeyError, TypeError):
                raise CannotSort('Tag {} not found in data'.format(input_order))
            except (IndexError, ValueError):
                raise CannotSort('Tag {} cannot be extracted from data'.format(input_order))

    def _choose_tag(self, tag, default):
        # Example: _tag = choose_tag('b', 'csa_header')
        if tag in self.input_options:
            return self.input_options[tag]
        else:
            return default

    def _set_dicom_tag(self, im, input_order, values):
        if input_order is None or values is None:
            return
        try:
            _ = len(values)
        except TypeError:
            values = [values]
        for order, value in zip(input_order.split(sep=','), values):
            if order == INPUT_ORDER_NONE:
                pass
            elif order == INPUT_ORDER_TIME:
                # AcquisitionTime
                time_tag = self._choose_tag("time", "AcquisitionTime")
                if time_tag not in im:
                    VR = pydicom.datadict.dictionary_VR(time_tag)
                    if VR == 'TM':
                        im.add_new(time_tag, VR,
                                   datetime.fromtimestamp(
                                       float(0.0), timezone.utc
                                   ).strftime("%H%M%S.%f")
                                   )
                    else:
                        im.add_new(time_tag, VR, 0.0)
                if im.data_element(time_tag).VR == 'TM':
                    time_str = datetime.fromtimestamp(float(value), timezone.utc).strftime("%H%M%S.%f")
                    im.data_element(time_tag).value = time_str
                else:
                    im.data_element(time_tag).value = float(value)
            elif order == INPUT_ORDER_B:
                set_ds_b_value(im, value)
            elif order == INPUT_ORDER_BVECTOR:
                set_ds_b_vector(im, value)
            elif order == INPUT_ORDER_FA:
                fa_tag = self._choose_tag('fa', 'FlipAngle')
                if fa_tag not in im:
                    VR = pydicom.datadict.dictionary_VR(fa_tag)
                    im.add_new(fa_tag, VR, float(value))
                else:
                    im.data_element(fa_tag).value = float(value)
            elif order == INPUT_ORDER_TE:
                te_tag = self._choose_tag('te', 'EchoTime')
                if te_tag not in im:
                    VR = pydicom.datadict.dictionary_VR(te_tag)
                    im.add_new(te_tag, VR, float(value))
                else:
                    im.data_element(te_tag).value = float(value)
            else:
                # User-defined tag
                if order in self.input_options:
                    _tag = self.input_options[order]
                    if _tag not in im:
                        VR = pydicom.datadict.dictionary_VR(_tag)
                        im.add_new(_tag, VR, float(value))
                    else:
                        im.data_element(_tag).value = float(value)
                else:
                    raise (UnknownTag("Unknown input_order {}.".format(order)))

    @staticmethod
    def _calculate_slice_location(image: Dataset) -> float:
        """Function to calculate slicelocation from imageposition and orientation.

        Args:
            image: image (pydicom dicom object)
        Returns:
            calculated slice location for this slice (float)
        Raises:
            ValueError: when sliceLocation cannot be calculated
        """

        def get_attribute(im, tag):
            if tag in im:
                return im[tag].value
            else:
                raise ValueError('Tag {:08x} ({}) not found'.format(
                    tag, pydicom.datadict.keyword_for_tag(tag)
                ))

        def get_normal(im):
            iop = np.array(get_attribute(im, tag_for_keyword('ImageOrientationPatient')))
            normal = np.zeros(3)
            normal[0] = iop[1] * iop[5] - iop[2] * iop[4]
            normal[1] = iop[2] * iop[3] - iop[0] * iop[5]
            normal[2] = iop[0] * iop[4] - iop[1] * iop[3]
            return normal

        try:
            ipp = np.array(get_attribute(image, tag_for_keyword('ImagePositionPatient')),
                           dtype=float)
            _normal = get_normal(image)
            return np.inner(_normal, ipp)
        except ValueError as e:
            raise ValueError('Cannot calculate slice location: %s' % e)

    @staticmethod
    def _index_from_tag(tag, tags) -> tuple[int]:
        _name: str = '{}.{}'.format(__name__, '_index_from_tag')
        for idx in np.ndindex(tags.shape):
            if tags[idx] is not None:
                found = True
                for i in range(len(tag)):
                    if issubclass(type(tag[i]), np.ndarray):
                        found = found and (tag[i] == tags[idx][i]).all()
                    else:
                        found = found and (tag[i] == tags[idx][i])
                if found:
                    return idx
        raise ValueError('{}: Tag {:08x} ({}) not found'.format(_name, tag, tags))
