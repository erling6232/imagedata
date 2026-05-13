"""
Data types
==========

SourceList: A list of source input urls.
    SourceList is list[dict].
            - 'archive'  : archive plugin
            - 'files'    : list of file names or regexp. May be empty list.
    SourceList is provided by a calling party.

class ObjectList: A list of all source objects.
    ObjectList is list[tuple[AbstractArchive, Member]]
    ObjectList is collected by self._get_dicom_files().
    - _get_dicom_files() collects all DICOM objects from input sources.
      The DICOM objects are not sorted.

class DatasetList: A list of all source Instance objects.
    DatasetList is list[Instance]
    DatasetList is collected by self._extract_member.

class DatasetDict: The source Instances are sorted according to SeriesUID.
    DatasetDict is defaultdict[SeriesUID, DatasetList]
    DatasetDict is collected by self._catalog_on_instance_uid(), and
      processed by self._select_imaging_datasets() and
      self._select_non_imaging_datasets().
    - _catalog_on_instance_uid() takes an ObjectList and sort objects on
      SeriesUID.
    - _select_imaging_datasets() takes a DatasetDict and return the imaging
      datasets only.
    - _select_non_imaging_datasets() takes a DatasetDict and return the
      non-imaging datasets only.

class SortedDatasetList: Collection of DatasetLists for each slice location (float).
    Sorted by slice location and tag.
    SortedDatasetList is defaultdict[float, DatasetList]
    SortedDatasetList is constructed by self._sort_dataset_geometry(),
    and collected by self._sort_datasets().
    SortedDatasetList is processed by self._get_headers._extract_all_tags(),
    self._sort_dataset_geometry(), and _verify_spacing().
    SortedHeaderDict is collected by self._get_headers()
      and self._get_non_image_headers().

class SortedData: A tuple of SortedDatasetList and Header.
    SortedDataDict is collected by self._sort_datasets() (see
    SortedDataDict).

class SortedDataDict: Collection of SortedData, key is SeriesUID.
    SortedDataDict is defaultdict[SeriesUID, SortedData]
    SortedDataDict is collected by self._sort_datasets() (see
      SortedDatasetList).
    self._process_image_members() NOT in use.

class SortedHeaderDict: Collection of Headers, key is SeriesUID.
    SortedHeaderDict is dict[SeriesUID, Header]
    SortedHeaderDict is collected by self._get_headers()
      and self._get_non_image_headers().

class PixelDict: Collection of pixel data arrays, key is SeriesUID.
    PixelDict is dict[SeriesUID, np.ndarray]
    PixelDict is collected by self._construct_pixel_arrays().
"""

import logging
from collections import defaultdict, namedtuple
import numpy as np
from ...archives.abstractarchive import AbstractArchive, Member
from .instance import Instance
from ...header import Header


logger = logging.getLogger(__name__)


# Type definitions
SourceList = list[dict]

SeriesUID = namedtuple('SeriesUID', 'patientID, studyInstanceUID, seriesInstanceUID, ' +
                       'acquisitionNumber, echoNumber', defaults=(None, None))


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
    """DatasetList is list[Instance]"""

    def __init__(self):
        super().__init__()

    def __str__(self):
        """Get printable description of series"""
        dataset = self[0]
        try:
            message = '{} ({})'.format(dataset.SeriesDescription, dataset.SeriesNumber)
        except AttributeError:
            try:
                message = '{} ({})'.format('', dataset.SeriesNumber)
            except AttributeError:
                message = '{}'.format(dataset.SeriesInstanceUID)
        return message

    def append(self, *args):
        for arg in args:
            assert isinstance(arg, Instance), self.__doc__
        super().append(*args)

    def getDicomAttribute(self, tag):
        instance: Instance = self[0]
        return instance[tag].value

    def getDicomAttributeValues(self, tag) -> list:
        values = []
        instance: Instance
        for instance in self:
            try:
                values.append(instance[tag].value)
            except KeyError:
                pass
        return values

    def get_image_orientation_patient(self) -> np.ndarray:
        instance: Instance = self[0]
        return instance.get_image_orientation_patient()

    def get_image_position_patient(self) -> np.ndarray:
        instance: Instance = self[0]
        return instance.get_image_position_patient()


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
        self.axes = None
        self.spacing = None
        self.transformationMatrix = None
        self.imagePositions = None

    def __setitem__(self, key, value):
        assert isinstance(key, float), self.__doc__
        assert isinstance(value, DatasetList), self.__doc__
        super().__setitem__(key, value)

    def __str__(self):
        """Get printable description of series"""
        dataset: DatasetList = self[next(iter(self))][0]
        try:
            message = '{} ({})'.format(dataset.SeriesDescription, dataset.SeriesNumber)
        except AttributeError:
            try:
                message = '{} ({})'.format('', dataset.SeriesNumber)
            except AttributeError:
                message = '{}'.format(dataset.SeriesInstanceUID)
        return message

    def get_headers(self) -> Header:
        """Extract DICOM attributes

        Args:
            self: DICOMPlugin instance
            sorted_dataset_list: SortedDatasetList
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
        dataset = self[next(iter(self))][0]
        hdr = Header()
        hdr.input_format = 'dicom'
        dataset.copy_attributes_to_header(hdr)

        # Extract geometry
        hdr.axes = self.axes
        hdr.orientation = dataset.get_image_orientation_patient()
        hdr.imagePositions = self.imagePositions
        hdr.transformationMatrix = self.transformationMatrix
        hdr.spacing = self.spacing

        hdr.color = False
        if 'SamplesPerPixel' in dataset and dataset.SamplesPerPixel == 3:
            hdr.color = True

        # Testing IPP and transformationMatrix
        T0 = hdr.transformationMatrix[:3, 3]
        ipp = np.array(T0)
        warned = False
        for i in range(len(self)):
            if not warned and not np.allclose(ipp, hdr.imagePositions[i], rtol=1e-3):
                logger.warning(f'{self}: DICOM ImagePosition is inconsistent with ImageOrientation')
                warned = True
            ipp += hdr.transformationMatrix[:3, 0]
        return hdr


SortedData = tuple[SortedDatasetList, Header]


class SortedDataDict(defaultdict):
    """SortedDataDict is defaultdict[SeriesUID, SortedData]"""

    def __init__(self):
        super().__init__(lambda: SortedData)

    def __setitem__(self, key, value):
        assert isinstance(key, SeriesUID), self.__doc__
        assert isinstance(value, tuple), self.__doc__
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


