"""Read/Write DICOM files
"""

# Copyright (c) 2013-2024 Erling Andersen, Haukeland University Hospital, Bergen, Norway

import os
import sys
import logging
import traceback
import warnings
import mimetypes
import math
from numbers import Number
from collections import defaultdict, namedtuple, Counter
from functools import partial
from typing import Union
from datetime import date, datetime, timedelta, timezone
import numpy as np
import pydicom
import pydicom.valuerep
import pydicom.config
import pydicom.errors
import pydicom.uid
from pydicom.datadict import tag_for_keyword
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset

from ..formats import CannotSort, NotImageError, INPUT_ORDER_FAULTY, input_order_to_dirname_str, \
    SORT_ON_SLICE, \
    INPUT_ORDER_NONE, INPUT_ORDER_TIME, INPUT_ORDER_B, INPUT_ORDER_FA, INPUT_ORDER_TE, \
    INPUT_ORDER_AUTO
from ..series import Series
from ..axis import VariableAxis, UniformLengthAxis
from .abstractplugin import AbstractPlugin
from ..archives.abstractarchive import AbstractArchive, Member
from ..header import Header
from ..apps.diffusion import get_ds_b_value, set_ds_b_value

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
ObjectList = list[tuple[AbstractArchive, Member]]
DatasetDict = defaultdict[SeriesUID, list[Dataset]]
SortedDatasetList = defaultdict[float, list[Dataset]]
SortedDatasetDict = defaultdict[SeriesUID, SortedDatasetList]
SortedHeaderDict = dict[SeriesUID, Header]
PixelDict = dict[SeriesUID, np.ndarray]


class DoNotIncludeFile(Exception):
    pass


class NoDICOMAttributes(Exception):
    pass


class UnevenSlicesError(Exception):
    pass


class ValueErrorWrapperPrecisionError(Exception):
    pass


class UnknownTag(Exception):
    pass


# noinspection PyPep8Naming
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
    version = "2.0.0"
    url = "www.helse-bergen.no"
    extensions = [".dcm", ".ima"]

    root = "2.16.578.1.37.1.1.4"
    smallint = ('bool8', 'byte', 'ubyte', 'ushort', 'uint16', 'int8', 'uint8', 'int16')
    keep_uid = False

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

    # def read(self, sources: list[dict], pre_hdr: Header, input_order: str , opts: dict) ->(
    def read(self, sources: SourceList, pre_hdr: Header, input_order: str , opts: dict) ->(
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

        skip_pixels = False
        if 'headers_only' in opts and opts['headers_only']:
            skip_pixels = True

        # Read DICOM headers
        logger.debug('{}: sources {}'.format(_name, sources))
        # pydicom.config.debug(True)
        # object_list: list[tuple[AbstractArchive, Member]]
        object_list: ObjectList
        object_list = self._get_dicom_files(sources)

        # dataset_dict: defaultdict[SeriesUID, list[Dataset]]
        dataset_dict: DatasetDict
        dataset_dict = self._catalog_on_instance_uid(object_list, opts, skip_pixels)

        # sorted_dataset_dict: defaultdict[SeriesUID, defaultdict[float, list[Dataset]]]
        sorted_dataset_dict: SortedDatasetDict
        sorting: dict[str]
        sorted_dataset_dict, sorting = self._sort_datasets(dataset_dict, input_order, opts)

        # sorted_header_dict: dict[SeriesUID, Header]
        sorted_header_dict: SortedHeaderDict
        logger.debug('{}: going to _get_headers {}'.format(_name, sources))
        sorted_header_dict = self._get_headers(sorted_dataset_dict, sorting, opts)

        # pixel_dict: dict[SeriesUID, np.ndarray]
        pixel_dict: PixelDict
        if skip_pixels:
            pixel_dict = {}
        else:
            logger.debug('{}: going to _construct_pixel_arrays'.format(_name))
            pixel_dict = self._construct_pixel_arrays(sorted_dataset_dict, sorted_header_dict,
                                                      opts, skip_pixels)

            if 'correct_acq' in opts and opts['correct_acq']:
                for seriesUID in sorted_dataset_dict:
                    pixel_dict[seriesUID] = self._correct_acqtimes_for_dynamic_series(
                        sorted_header_dict[seriesUID], pixel_dict[seriesUID]
                    )

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

        object_list: ObjectList
        object_list = []
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

        dataset_dict: DatasetDict
        dataset_dict = defaultdict(list)
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
                # raise
        if len(object_list) > 0 and len(dataset_dict) < 1:
            raise NotImageError(last_message)
        return dataset_dict

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
                    _pixels = len(im.pixel_array)
                except AttributeError:
                    raise DoNotIncludeFile('No pixel data in DICOM object')

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

        _name: str = '{}.{}'.format(__name__, self._sort_datasets.__name__)

        skip_broken_series = False
        if 'skip_broken_series' in opts:
            skip_broken_series = opts['skip_broken_series']

        # Sort datasets on sloc
        sorted_dataset_dict: SortedDatasetDict
        sorted_dataset_dict = defaultdict(lambda: defaultdict(list))
        sorting = {}
        for seriesUID in image_dict:
            sorting[seriesUID] = 'none'
            dataset_dict = image_dict[seriesUID]
            dataset: Dataset
            sorted_dataset: SortedDatasetList = defaultdict(list)
            for dataset in dataset_dict:
                # logger.debug('{}: process_member {}'.format(_name, dataset))
                sloc = _get_sloc(dataset)

                # Catalog images with sloc as key
                sorted_dataset[sloc].append(dataset)
            # Determine (automatic) sorting
            try:
                sorting[seriesUID] = self._determine_sorting(
                    sorted_dataset, input_order, opts
                )
            except (CannotSort, UnevenSlicesError):
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
            for sloc in sorted_dataset.keys():
                sorted_dataset[sloc].sort(
                    key=partial(self._get_tag, input_order=sorting[seriesUID], opts=opts)
                )
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
                for order in ['time', 'b', 'fa', 'te']:
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
                if actual_order == 'time' and order in ['b', 'te']:
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
        elif actual_order == INPUT_ORDER_TIME and _single_slice_over_time(extended_tags['time']):
            actual_order = INPUT_ORDER_NONE
        return actual_order

    def _get_headers(self,
                     sorted_dataset_dict: SortedDatasetDict,
                     input_order: dict[str],
                     opts: dict = None
                     ) -> SortedHeaderDict:
        """Get DICOM headers"""

        def _verify_consistent_slices(series: SortedDatasetList) -> Counter:
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
                logger.error("{}: tags per slice: {}".format(_name, slice_count))
                raise UnevenSlicesError(
                    "Different number of images in each slice. Tags per slice:\n{}".format(slice_count) +
                    "\nLast file: {}".format(series[last_sloc][0].filename) +
                    "\nCould try 'split_acquisitions=True' or 'split_echo_numbers=True'."
                )
            return slice_count

        def _extract_all_tags(hdr: Header,
                              series: SortedDatasetList,
                              input_order: str,
                              slice_count: Counter
                              ) -> None:
            _name: str = '{}.{}'.format(__name__, _extract_all_tags.__name__)
            accept_duplicate_tag = accept_uneven_slices = False
            if 'accept_duplicate_tag' in opts and opts['accept_duplicate_tag']:
                accept_duplicate_tag = True
            if 'accept_uneven_slices' in opts and opts['accept_uneven_slices']:
                accept_uneven_slices = True
            tag_list = defaultdict(list)
            for islice, sloc in enumerate(sorted(series)):
                i = 0
                for im in series[sloc]:
                    try:
                        tag = self._get_tag(im, input_order, opts)
                    except KeyError:
                        if input_order == INPUT_ORDER_FAULTY:
                            tag = i
                        else:
                            raise CannotSort('Tag {} not found in dataset'.format(
                                input_order
                            ))
                    except CannotSort:
                        raise
                    except Exception:
                        raise
                    if tag is None:
                        raise CannotSort("Tag {} not found in data".format(input_order))
                    if tag not in tag_list[islice] or accept_duplicate_tag:
                        tag_list[islice].append(tag)
                    elif accept_uneven_slices:
                        # Drop duplicate images
                        logger.warning("{}: dropping duplicate image: {} {}".format(
                            _name, islice, sloc))
                    else:
                        raise CannotSort("Duplicate tag ({}): {:08x} ({})".format(
                            input_order, tag, pydicom.datadict.keyword_for_tag(tag)
                        ))
                    i += 1
            for islice in tag_list.keys():
                tag_list[islice] = sorted(tag_list[islice])
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
                sorted_headers[islice] = [False for _ in range(slice_count[islice])]
                for im in series[sloc]:
                    if input_order == INPUT_ORDER_FAULTY:
                        tag = i
                    else:
                        try:
                            tag = self._get_tag(im, input_order, opts)
                        except CannotSort:
                            raise
                    idx = tag_list[islice].index(tag)
                    if sorted_headers[islice][idx]:
                        # Duplicate tag
                        if accept_duplicate_tag:
                            while sorted_headers[islice][idx]:
                                idx += 1
                        else:
                            print("WARNING: Duplicate tag", tag)
                    sorted_headers[islice][idx] = (tag, im)
                    SOPInstanceUIDs[(idx, islice)] = im.SOPInstanceUID
                    rows = max(rows, im.Rows)
                    columns = max(columns, im.Columns)
                    if 'NumberOfFrames' in im:
                        frames = im.NumberOfFrames
                    last_im = im
                    i += 1
            self.DicomHeaderDict = sorted_headers
            hdr.dicomTemplate = series[next(iter(series))][0]
            hdr.SOPInstanceUIDs = SOPInstanceUIDs
            hdr.tags = {}
            for _slice in tag_list.keys():
                hdr.tags[_slice] = np.array(tag_list[_slice])
            nz = len(series)
            if frames is not None and frames > 1:
                nz = frames
            hdr.spacing = self.__get_voxel_spacing(sorted_headers)
            ipp = self.getDicomAttribute(self.DicomHeaderDict, tag_for_keyword('ImagePositionPatient'))
            if ipp is not None:
                ipp = np.array(list(map(float, ipp)))[::-1]  # Reverse xyz
            else:
                ipp = np.array([0, 0, 0])
            axes = list()
            if len(tag_list[0]) > 1:
                axes.append(
                    VariableAxis(
                        input_order_to_dirname_str(input_order),
                        tag_list[0])
                )
            axes.append(UniformLengthAxis('slice', ipp[0], nz, hdr.spacing[0]))
            axes.append(UniformLengthAxis('row', ipp[1], rows, hdr.spacing[1]))
            axes.append(UniformLengthAxis('column', ipp[2], columns, hdr.spacing[2]))
            hdr.color = False
            if 'SamplesPerPixel' in last_im and last_im.SamplesPerPixel == 3:
                hdr.color = True
            hdr.axes = axes
            self._extract_dicom_attributes(series, hdr)

        _name: str = '{}.{}'.format(__name__, self._get_headers.__name__)
        skip_broken_series = False
        if 'skip_broken_series' in opts:
            skip_broken_series = opts['skip_broken_series']
        sorted_header_dict: SortedHeaderDict
        sorted_header_dict = dict()
        for seriesUID in sorted_dataset_dict:
            series_dataset: SortedDatasetList
            series_dataset = sorted_dataset_dict[seriesUID]
            hdr = Header()
            hdr.input_format = 'dicom'
            hdr.input_order = input_order[seriesUID]
            sliceLocations = sorted(series_dataset.keys())
            # hdr.slices = len(sliceLocations)
            hdr.sliceLocations = np.array(sliceLocations)

            if len(series_dataset) == 0:
                raise ValueError("No DICOM images found.")

            try:
                slice_count = _verify_consistent_slices(series_dataset)
                _extract_all_tags(hdr, series_dataset, input_order[seriesUID], slice_count)
                sorted_header_dict[seriesUID] = hdr
            except (CannotSort, UnevenSlicesError):
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
        pixel_dict: PixelDict
        pixel_dict = {}
        for seriesUID in sorted_header_dict:
            dataset_dict: SortedDatasetList
            dataset_dict = sorted_dataset_dict[seriesUID]
            header: Header
            header = sorted_header_dict[seriesUID]
            setattr(header, 'keep_uid', True)
            if skip_pixels:
                si = None
            else:
                # Extract pixel data
                si = self._construct_pixel_array(
                    dataset_dict, header, header.shape, opts=opts)

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
            for _slice, _sloc in enumerate(sorted(_image_dict)):
                _done = [False for x in range(len(_image_dict[_sloc]))]
                for im in _image_dict[_sloc]:
                    tag = self._get_tag(im, _hdr.input_order, opts)
                    tgs = _hdr.tags[_slice]
                    idx = np.where(tgs == tag)[0][0]
                    if _done[idx] and accept_duplicate_tag:
                        while _done[idx]:
                            idx += 1
                    _done[idx] = True
                    idx = (idx, _slice)
                    # Simplify index when image is 3D, remove tag index
                    logger.debug("{}: si.ndim {}, idx {}".format(_name, _si.ndim, idx))
                    if _si.ndim == 3:
                        idx = idx[1:]
                    try:
                        im.decompress()
                    except NotImplementedError as e:
                        logger.error("{}: Cannot decompress pixel data: {}".format(_name, e))
                        raise
                    try:
                        logger.debug("{}: get idx {} shape {}".format(_name, idx, _si[idx].shape))
                        _si[idx] = self._get_pixels_with_shape(im, _si[idx].shape)
                    except Exception as e:
                        logger.warning("{}: Cannot read pixel data: {}".format(_name, e))
                        raise
                    del im

        def _copy_pixels_from_frames(_si, _hdr, _image_dict):
            _name: str = '{}.{}'.format(__name__, _copy_pixels_from_frames.__name__)
            assert len(_image_dict) == 1, "Do not know how to unpack frames and slices"
            for im in _image_dict[next(iter(_image_dict))]:
                # tag = self._get_tag(im, _hdr.input_order, opts)
                # tgs = _hdr.tags[0]
                # idx = np.where(tgs == tag)[0][0]
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

    def _extract_dicom_attributes(self,
                                  series: SortedDatasetList,
                                  hdr: Header
                                  ) -> None:
        """Extract DICOM attributes

        Args:
            self: DICOMPlugin instance
            series:
            hdr: existing header (Header)
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
        attributes = [
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

        def get_attribute(im: Dataset, tag):
            if tag in im:
                return im[tag].value
            else:
                raise ValueError('Tag {:08x} ({}) not found'.format(
                    tag, pydicom.datadict.keyword_for_tag(tag)
                ))

        dataset = series[next(iter(series))][0]
        for attribute in attributes:
            dicom_attribute = attribute[0].upper() + attribute[1:]
            try:
                setattr(hdr, attribute,
                        get_attribute(dataset, tag_for_keyword(dicom_attribute))
                        )
            except ValueError:
                pass

        # Image position (patient)
        # Reverse orientation vectors from (x,y,z) to (z,y,x)
        try:
            iop = get_attribute(dataset, tag_for_keyword("ImageOrientationPatient"))
        except ValueError:
            iop = [0, 0, 1, 0, 1, 0]
        if iop is not None:
            hdr.orientation = np.array((iop[2], iop[1], iop[0],
                                        iop[5], iop[4], iop[3]))

        # Extract imagePositions
        hdr.imagePositions = {}
        for i, _slice in enumerate(series):
            hdr.imagePositions[i] = self.getOriginForSlice({i: [(0, series[_slice][0])]}, i)

    def __get_voxel_spacing(self, dictionary):
        # Spacing
        pixel_spacing = self.getDicomAttribute(dictionary, tag_for_keyword("PixelSpacing"))
        dy = 1.0
        dx = 1.0
        if pixel_spacing is not None:
            # Notice that DICOM row spacing comes first, column spacing second!
            dy = float(pixel_spacing[0])
            dx = float(pixel_spacing[1])
        try:
            dz = float(self.getDicomAttribute(dictionary, tag_for_keyword("SpacingBetweenSlices")))
        except TypeError:
            try:
                dz = float(self.getDicomAttribute(dictionary, tag_for_keyword("SliceThickness")))
            except TypeError:
                dz = 1.0
        return np.array([dz, dy, dx])

    def getOriginForSlice(self, dictionary, slice):
        """Get origin of given slice.

        Args:
            self: DICOMPlugin instance
            dictionary: image dictionary
            slice: slice number (int)
        Returns:
            z,y,x: coordinate for origin of given slice (np.array)
        """

        origin = self.getDicomAttribute(dictionary, tag_for_keyword("ImagePositionPatient"), slice)
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

    def getDicomAttribute(self, dictionary, tag, slice=0):
        """Get DICOM attribute from first image for given slice.

        Args:
            self: DICOMPlugin instance
            dictionary: image dictionary
            tag: DICOM tag of requested attribute.
            slice: which slice to access. Default: slice=0
        """
        # logger.debug("getDicomAttribute: tag", tag, ", slice", slice)
        assert dictionary is not None, "dicomplugin.getDicomAttribute: dictionary is None"
        _, im = dictionary[slice][0]
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
            # logger.debug("Set si[{}]".format(idx))
            if 'RescaleSlope' in im and 'RescaleIntercept' in im:
                _use_float = abs(im.RescaleSlope - 1) > 1e-4 or abs(im.RescaleIntercept) > 1e-4
            if _use_float:
                pixels = float(im.RescaleSlope) * im.pixel_array.astype(float) +\
                         float(im.RescaleIntercept)
            else:
                pixels = im.pixel_array
            if shape != pixels.shape:
                if im.PhotometricInterpretation == 'RGB':
                    # RGB image
                    rgb_dtype = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])
                    si = pixels.copy().view(dtype=rgb_dtype).reshape(pixels.shape[:-1])
                    # si = pixels
                elif 'NumberOfFrames' in im:
                    logger.debug('{}: NumberOfFrames: {}'.format(_name, im.NumberOfFrames))
                    if (im.NumberOfFrames,) + shape == pixels.shape:
                        logger.debug('{}: NumberOfFrames copy pixels'.format(_name, im.NumberOfFrames))
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
                # try:
                #    image.decompress()
                # except NotImplementedError as e:
                #    logger.error("Cannot decompress pixel data: {}".format(e))
                #    raise
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

        sorted_dataset_dict: SortedDatasetDict
        sorted_dataset_dict = {}
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
                # logger.debug(_slice, tg)
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

    def write_3d_numpy(self, si, destination, opts):
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
            level=max(0, si.ndim-2),
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
        if not self.keep_uid:
            si.header.seriesInstanceUID = si.header.new_uid()
        self.serInsUid = si.header.seriesInstanceUID
        logger.debug("{}: {}".format(_name, self.serInsUid))
        self.input_options = opts

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

    def write_4d_numpy(self, si, destination, opts):
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
        assert si.ndim == 4, "write_4d_series: input dimension %d is not 4D." % si.ndim

        steps = si.shape[0]
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
                archive.set_member_naming_scheme(
                    fallback='Image_{:05d}.dcm',
                    level=1,
                    default_extension='.dcm',
                    extensions=self.extensions
                )
            else:  # self.output_dir == 'multi'
                digits = len("{}".format(steps))
                dirn = "{0}{{0:0{1}}}".format(
                    input_order_to_dirname_str(si.input_order),
                    digits)
                archive.set_member_naming_scheme(
                    fallback=os.path.join(dirn, 'Image_{1:05d}.dcm'),
                    level=max(0, si.ndim-2),
                    default_extension='.dcm',
                    extensions=self.extensions
                )
            ifile = 0
            for tag in range(steps):
                for _slice in range(si.slices):
                    if self.keep_uid:
                        sop_ins_uid = si.SOPInstanceUIDs[(tag, _slice)]
                    else:
                        sop_ins_uid = si.header.new_uid()
                    if self.output_dir == 'multi' and _slice == 0:
                        # Restart file number in each subdirectory
                        ifile = 0
                    try:
                        self.write_slice(si.input_order, (tag, _slice), si[tag, _slice],
                                         destination, ifile, sop_ins_uid=sop_ins_uid)
                    except Exception as e:
                        print('DICOMPlugin.write_slice Exception: {}'.format(e))
                        traceback.print_exc(file=sys.stdout)
                        raise
                    ifile += 1
        else:  # self.output_sort == SORT_ON_TAG:
            if self.output_dir == 'single':
                archive.set_member_naming_scheme(
                    fallback=self.input_order + '_{1:05d}.dcm',
                    level=1,
                    default_extension='.dcm',
                    extensions=self.extensions
                )
            else:  # self.output_dir == 'multi'
                digits = len("{}".format(si.slices))
                dirn = "slice{{0:0{0}}}".format(
                    digits)
                archive.set_member_naming_scheme(
                    fallback=os.path.join(dirn, 'Slice_{1:05d}.dcm'),
                    level=max(0, si.ndim-2),
                    default_extension='.dcm',
                    extensions=self.extensions
                )
            ifile = 0
            for _slice in range(si.slices):
                for tag in range(steps):
                    if self.keep_uid:
                        sop_ins_uid = si.SOPInstanceUIDs[(tag, _slice)]
                    else:
                        sop_ins_uid = si.header.new_uid()
                    if self.output_dir == 'multi' and tag == 0:
                        # Restart file number in each subdirectory
                        ifile = 0
                    try:
                        self.write_slice(si.input_order, (tag, _slice), si[tag, _slice],
                                         destination, ifile, sop_ins_uid=sop_ins_uid)
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
        #         ds.save_as(f, write_like_original=False)
        raise ValueError("write_enhanced: to be implemented")

    # noinspection PyPep8Naming,PyArgumentList
    def write_slice(self, input_order, tag, si, destination, ifile,
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
            -   tags (not used)
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
        """

        _name: str = '{}.{}'.format(__name__, self.write_slice.__name__)

        archive: AbstractArchive = destination['archive']
        query = None
        # if destination['files'] is not None and len(destination['files']):
        if destination['files'] and len(destination['files']):
            query = destination['files'][0]
        if self.output_dir == 'single':
            filename = archive.construct_filename(
                tag=(ifile,),
                query=query
            )
        else:
            filename = archive.construct_filename(
                tag=tag,
                query=query
            )
        logger.debug("{}: {} {}".format(_name, filename, self.serInsUid))

        # try:
        #     logger.debug("write_slice slice {}, tag {}".format(slice, tag))
        #     # logger.debug("write_slice {}".format(si.DicomHeaderDict))
        #     tg, member_name, im = si.DicomHeaderDict[0][0]
        #     # tg,member_name,image = si.DicomHeaderDict[slice][tag]
        # except (KeyError, IndexError, TypeError):
        #     print('DICOMPlugin.write_slice: DicomHeaderDict: {}'.format(si.DicomHeaderDict))
        #     raise IndexError("Cannot address dicom_template.DicomHeaderDict[slice=%d][tag=%d]"
        #                      % (slice, tag))
        # except AttributeError:
        # except ValueError:
        #     raise NoDICOMAttributes("Cannot write DICOM object when no DICOM attributes exist.")
        try:
            ds = self.construct_dicom(filename, si.dicomTemplate, si, sop_ins_uid=sop_ins_uid)
        except ValueError:
            ds = self.construct_basic_dicom(si, sop_ins_uid=sop_ins_uid)
            # raise NoDICOMAttributes("Cannot write DICOM object when no DICOM attributes exist.")
        # logger.debug("write_slice member_name {}".format(member_name))
        # self._copy_dicom_group(0x21, im, ds)
        # self._copy_dicom_group(0x29, im, ds)

        # Add header information
        try:
            ds.SliceLocation = pydicom.valuerep.format_number_as_ds(float(si.sliceLocations[0]))
        except (AttributeError, ValueError):
            # Dont know the SliceLocation, so will set this to be the slice index
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
        # logger.debug("write_slice: filename {}".format(filename))

        # Set tag
        # si will always have only the present tag
        self._set_dicom_tag(ds, input_order, si.tags[0][0])

        logger.debug("{}: filename {}".format(_name, filename))
        if archive.transport.name == 'dicom':
            # Store dicom set ds directly
            archive.transport.store(ds)
        else:
            # Store dicom set ds as file
            with archive.open(filename, 'wb') as f:
                ds.save_as(f, write_like_original=False)

    def construct_basic_dicom(self,
                              template: Series = None,
                              filename: str = 'NA',
                              sop_ins_uid:str = None
                              ) -> FileDataset:

        if sop_ins_uid is None:
            raise ValueError('SOPInstanceUID is undefined.')
        # Populate required values for file meta information
        file_meta = FileMetaDataset()
        sop_class_uid = getattr(template, 'SOPClassUID', None)
        if sop_class_uid is None:
            sop_class_uid = '1.2.840.10008.5.1.4.1.1.7'
        file_meta.MediaStorageSOPClassUID = sop_class_uid
        # if template is not None and 'SOPClassUID' in template:
        #     file_meta.MediaStorageSOPClassUID = template.SOPClassUID
        # else:
        #     file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.7'
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
            ds.StudyInstanceUID = template.header.new_uid()
            ds.SeriesInstanceUID = template.header.new_uid()
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
        # if not self.keep_uid:
        #     si.SOPInstanceUID = si.header.new_uid()
        # if 'SOPInstanceUID' in si:
        #     sop_ins_uid = si.SOPInstanceUID
        # else:
        #     sop_ins_uid = si.header.new_uid()
        if sop_ins_uid is None:
            sop_ins_uid = si.header.new_uid()

        # Populate required values for file meta information
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = si.SOPClassUID
        file_meta.MediaStorageSOPInstanceUID = sop_ins_uid
        file_meta.ImplementationClassUID = "%s.1" % self.root
        file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        # file_meta.FileMetaInformationVersion = int(1).to_bytes(2,'big')
        # file_meta.FileMetaInformationGroupLength = 160

        # print("Setting dataset values...")

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
        # ds.SeriesInstanceUID = si.header.seriesInstanceUID
        ds.SeriesInstanceUID = self.serInsUid
        ds.SOPClassUID = si.SOPClassUID
        ds.SOPInstanceUID = sop_ins_uid

        ds.AccessionNumber = si.header.accessionNumber
        ds.PatientName = si.header.patientName
        ds.PatientID = si.header.patientID
        ds.PatientBirthDate = si.header.patientBirthDate

        # Set the transfer syntax
        ds.is_little_endian = True
        ds.is_implicit_VR = True

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

        # logger.debug('DICOMPlugin.insert_pixeldata: arr.dtype %s' % arr.dtype)
        # logger.debug('DICOMPlugin.insert_pixeldata: arr.itemsize  %s' % arr.itemsize)

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
        # ymin = np.nanmin(arr).item()
        # ymax = np.nanmax(arr).item()
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
            # self.smallestPixelValueInSeries = arr.min().astype('int16')
            # self.largestPixelValueInSeries = arr.max().astype('int16')
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

    def _get_tag(self, im: Dataset, input_order: str, opts: dict = None) -> Number:

        if input_order is None:
            return 0
        if input_order == INPUT_ORDER_NONE:
            return 0
        elif input_order == INPUT_ORDER_TIME:
            time_tag = self._choose_tag('time', 'AcquisitionTime')
            # if 'TriggerTime' in opts:
            #    return(float(image.TriggerTime))
            # elif 'InstanceNumber' in opts:
            #    return(float(image.InstanceNumber))
            # else:
            if im.data_element(time_tag).VR == 'TM':
                time_str = im.data_element(time_tag).value
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
                    return float(im.data_element(time_tag).value)
                except ValueError:
                    raise CannotSort("Unable to extract time value from header.")
        elif input_order == INPUT_ORDER_B:
            try:
                return get_ds_b_value(im)
            except IndexError:
                raise CannotSort("Unable to extract b value from header.")
            b_tag = self._choose_tag('b', 'DiffusionBValue')
            try:
                return float(im.data_element(b_tag).value)
            except (KeyError, TypeError):
                pass
            b_tag = self._choose_tag('b', 'csa_header')
            if b_tag == 'csa_header':
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    import nibabel.nicom.csareader as csa
                try:
                    csa_head = csa.get_csa_header(im)
                except csa.CSAReadError:
                    raise CannotSort("Unable to extract b value from header.")
                if csa_head is None:
                    raise CannotSort("Unable to extract b value from header.")
                try:
                    value = csa.get_b_value(csa_head)
                except TypeError:
                    raise CannotSort("Unable to extract b value from header.")
            else:
                try:
                    value = float(im.data_element(b_tag).value)
                except ValueError:
                    raise CannotSort("Unable to extract b value from header.")
            if value is None:
                raise CannotSort("Unable to extract b value from header.")
            return value
        elif input_order == INPUT_ORDER_FA:
            fa_tag = self._choose_tag('fa', 'FlipAngle')
            try:
                return float(im.data_element(fa_tag).value)
            except ValueError:
                raise CannotSort("Unable to extract FA value from header.")
        elif input_order == INPUT_ORDER_TE:
            te_tag = self._choose_tag('te', 'EchoTime')
            try:
                return float(im.data_element(te_tag).value)
            except ValueError:
                raise CannotSort("Unable to extract TE value from header.")
        elif input_order == INPUT_ORDER_AUTO:
            pass
        else:
            # User-defined tag
            if input_order in opts:
                _tag = opts[input_order]
                try:
                    return float(im.data_element(_tag).value)
                except ValueError:
                    raise CannotSort("Unable to extract {} value from header.".format(input_order))
        raise (UnknownTag("Unknown input_order {}.".format(input_order)))

    def _choose_tag(self, tag, default):
        # Example: _tag = choose_tag('b', 'csa_header')
        if tag in self.input_options:
            return self.input_options[tag]
        else:
            return default

    def _set_dicom_tag(self, im, input_order, value):
        if input_order is None:
            pass
        elif input_order == INPUT_ORDER_NONE:
            pass
        elif input_order == INPUT_ORDER_TIME:
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
                # elem = pydicom.dataelem.DataElement(time_tag, 'TM', 0)
                # im.add(elem)
            if im.data_element(time_tag).VR == 'TM':
                time_str = datetime.fromtimestamp(float(value), timezone.utc).strftime("%H%M%S.%f")
                im.data_element(time_tag).value = time_str
            else:
                im.data_element(time_tag).value = float(value)
        elif input_order == INPUT_ORDER_B:
            set_ds_b_value(im, value)
        elif input_order == INPUT_ORDER_FA:
            fa_tag = self._choose_tag('fa', 'FlipAngle')
            if fa_tag not in im:
                VR = pydicom.datadict.dictionary_VR(fa_tag)
                im.add_new(fa_tag, VR, float(value))
            else:
                im.data_element(fa_tag).value = float(value)
        elif input_order == INPUT_ORDER_TE:
            te_tag = self._choose_tag('te', 'EchoTime')
            if te_tag not in im:
                VR = pydicom.datadict.dictionary_VR(te_tag)
                im.add_new(te_tag, VR, float(value))
            else:
                im.data_element(te_tag).value = float(value)
        else:
            # User-defined tag
            if input_order in self.input_options:
                _tag = self.input_options[input_order]
                if _tag not in im:
                    VR = pydicom.datadict.dictionary_VR(_tag)
                    im.add_new(_tag, VR, float(value))
                else:
                    im.data_element(_tag).value = float(value)
            else:
                raise (UnknownTag("Unknown input_order {}.".format(input_order)))

    def simulateAffine(self):
        # shape = (
        #     self.getDicomAttribute(tag_for_keyword('Rows')),
        #     self.getDicomAttribute(tag_for_keyword('Columns')))
        _name: str = '{}.{}'.format(__name__, self.simulateAffine.__name__)

        iop = self.getDicomAttribute(tag_for_keyword('ImageOrientationPatient'))
        if iop is None:
            return
        iop = np.array(list(map(float, iop)))
        iop = np.array(iop).reshape(2, 3).T
        logger.debug('{}: iop\n{}'.format(_name, iop))
        s_norm = np.cross(iop[:, 1], iop[:, 0])
        # Rotation matrix
        R = np.eye(3)
        R[:, :2] = np.fliplr(iop)
        R[:, 2] = s_norm
        if not np.allclose(np.eye(3), np.dot(R, R.T), atol=5e-5):
            raise ValueErrorWrapperPrecisionError('Rotation matrix not nearly orthogonal')

        pix_space = self.getDicomAttribute(tag_for_keyword('PixelSpacing'))
        zs = self.getDicomAttribute(tag_for_keyword('SpacingBetweenSlices'))
        if zs is None:
            zs = self.getDicomAttribute(tag_for_keyword('SliceThickness'))
            if zs is None:
                zs = 1
        zs = float(zs)
        pix_space = list(map(float, pix_space))
        vox = tuple(pix_space + [zs])
        logger.debug('{}: vox {}'.format(_name, vox))

        ipp = self.getDicomAttribute(tag_for_keyword('ImagePositionPatient'))
        if ipp is None:
            return
        ipp = np.array(list(map(float, ipp)))
        logger.debug('{}: ipp {}'.format(_name, ipp))

        orient = R
        logger.debug('{}: orient\n{}'.format(_name, orient))

        aff = np.eye(4)
        aff[:3, :3] = orient * np.array(vox)
        aff[:3, 3] = ipp
        logger.debug('{}: aff\n{}'.format(_name, aff))

    def create_affine(self, hdr):
        """Function to generate the affine matrix for a dicom series
        This method was based on
        (http://nipy.org/nibabel/dicom/dicom_orientation.html)
        :param hdr: list with sorted dicom files
        """
        _name: str = '{}.{}'.format(__name__, self.create_affine.__name__)

        slices = hdr.slices

        # Create affine matrix
        # (http://nipy.sourceforge.net/nibabel/dicom/dicom_orientation.html#dicom-slice-affine)
        iop = self.getDicomAttribute(tag_for_keyword('ImageOrientationPatient'))
        if iop is None:
            return
        image_orient1 = np.array(iop[0:3])
        image_orient2 = np.array(iop[3:6])

        pix_space = self.getDicomAttribute(tag_for_keyword('PixelSpacing'))
        delta_r = float(pix_space[0])
        delta_c = float(pix_space[1])

        ipp = self.getDicomAttribute(tag_for_keyword('ImagePositionPatient'))
        if ipp is None:
            return
        image_pos = np.array(ipp)

        ippn = self.getDicomAttribute(tag_for_keyword('ImagePositionPatient'),
                                      slice=slices - 1)
        if ippn is None:
            return
        last_image_pos = np.array(ippn)

        if slices == 1:
            # Single slice
            step = [0, 0, -1]
        else:
            step = (image_pos - last_image_pos) / (1 - slices)

        # check if this is actually a volume and not all slices on the same location
        if np.linalg.norm(step) == 0.0:
            raise ValueError("NOT_A_VOLUME")

        affine = np.matrix([
            [-image_orient1[0] * delta_c, -image_orient2[0] * delta_r, -step[0], -image_pos[0]],
            [-image_orient1[1] * delta_c, -image_orient2[1] * delta_r, -step[1], -image_pos[1]],
            [image_orient1[2] * delta_c, image_orient2[2] * delta_r, step[2], image_pos[2]],
            [0, 0, 0, 1]
        ])
        logger.debug('{}: affine\n{}'.format(_name, affine))
        return affine

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
