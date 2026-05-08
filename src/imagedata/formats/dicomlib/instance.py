import logging
from numbers import Number
from typing import List
import numpy as np
from datetime import datetime, timedelta, timezone
from pydicom.dataset import FileDataset, Dataset, FileMetaDataset
from pydicom.datadict import dictionary_VR, keyword_for_tag, tag_for_keyword
from typing import Any
from ...header import Header
from ...formats import (CannotSort,
                        INPUT_ORDER_FAULTY,
                        INPUT_ORDER_NONE, INPUT_ORDER_TIME, INPUT_ORDER_B,
                        INPUT_ORDER_FA, INPUT_ORDER_TE, INPUT_ORDER_BVECTOR
                        )
from ...apps.diffusion import get_ds_b_vectors, get_ds_b_value, set_ds_b_value, set_ds_b_vector

logger = logging.getLogger(__name__)


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


class UnknownTag(Exception):
    pass


class Instance(FileDataset):
    slice_index: int
    tags: tuple[Number]
    tag_index: tuple[int]

    def __init__(self, *args: Dataset, **kwargs: Any) -> None:
        if len(args) == 1:
            dataset = args[0]
        elif len(args) == 6:
            dataset = args[1]
        else:
            raise ValueError('Unexpected number of arguments ({}) for Instance.__init__'.format(len(args)))
        try:
            file_meta = dataset.file_meta
        except AttributeError:
            file_meta = FileMetaDataset()
        if 'file_meta' in kwargs:
            file_meta = kwargs['file_meta']
        preamble = b"\0" * 128
        if 'preamble' in kwargs:
            preamble = kwargs['preamble']
        super().__init__("",
                         dataset=dataset,
                         file_meta=file_meta,
                         preamble=preamble
                         )
        self.slice_index = None
        self.tags = None
        self.tag_index = None

    def get_tag(self, input_order: str, input_options: dict = None) -> Number:

        if callable(input_options[input_order]):
            return input_options[input_order](self)
        try:
            _ = getattr(self, input_options[input_order])
            return _()
            # return input_options[input_order](self)
        except (KeyError, TypeError) as e:
            try:
                return self.get_float(input_options[input_order])
            except (AttributeError, KeyError, TypeError):
                raise CannotSort('Tag {} not found in data'.format(input_order))
            except (IndexError, ValueError):
                raise CannotSort('Tag {} cannot be extracted from data'.format(input_order))
        except IndexError:
            return None

    def get_tag_tuple(self, faulty: int, input_order: str, input_options: dict) -> tuple:
        tag_list = []
        for order in input_order.split(sep=','):
            try:
                tag = self.get_tag(order, input_options)
            except KeyError:
                if order == INPUT_ORDER_FAULTY:
                    tag = faulty
                else:
                    raise CannotSort(f'Tag {order} not found in dataset')
            if tag is None:
                raise CannotSort(f'Tag {order} not found in data')
            tag_list.append(tag)
        return tuple(tag_list)

    def replace_tag(self, tag_list: list,
                    wanted_time_values: list,
                    tag: tuple, time_tag: int) -> list:
        _name: str = '{}.{}'.format(__name__, self.replace_tag.__name__)
        _times, _mask = _get_fixed_tags(tag_list, tag, time_tag)
        try:
            what_time = _times.index(tag[time_tag])
            self.timestamp = wanted_time_values[what_time]
        except ValueError:
            raise CannotSort(f'{_name}: Internal sorting error')
        vr = dictionary_VR('AcquisitionTime')
        self.add_new('AcquisitionTime', vr, datetime.fromtimestamp(
            wanted_time_values[what_time], timezone.utc
        ).strftime("%H%M%S.%f"))
        try:
            where = tag_list.index((self, tag))
            new_tag = ()
            for i, v in enumerate(tag):
                if i == time_tag:
                    new_tag += (self.timestamp,)
                else:
                    new_tag += (v,)
            tag_list[where] = self, new_tag
        except ValueError:
            raise CannotSort(f'{_name}: Internal sorting error')
        return tag_list

    def set_slice_index(self, slice_index):
        self.slice_index = slice_index

    def set_tags(self, tags):
        self.tags = tags

    def set_tag_index(self, idx):
        self.tag_index = idx

    def get_float(self, tag: str) -> float:
        if self.data_element(tag).VR == 'TM':
            time_str = self.data_element(tag).value
            try:
                if '.' in time_str:
                    tm = datetime.strptime(time_str, "%H%M%S.%f")
                else:
                    tm = datetime.strptime(time_str, "%H%M%S")
            except ValueError:
                raise IndexError("Unable to extract time value from header.")
            td = timedelta(hours=tm.hour,
                           minutes=tm.minute,
                           seconds=tm.second,
                           microseconds=tm.microsecond)
            return td.total_seconds()
        else:
            try:
                return float(self.data_element(tag).value)
            except ValueError:
                raise IndexError("Unable to extract value from header.")

    def get_no_value(self) -> Number:
        return 0

    def get_acquisition_time(self) -> Number:
        return self.get_float('AcquisitionTime')

    def get_trigger_time(self) -> Number:
        return self.get_float('TriggerTime') / 1000.

    def get_b_value(self) -> Number:
        try:
            return get_ds_b_value(self)
        except IndexError:
            raise

    def get_b_vector(self) -> np.ndarray:
        try:
            bvec = get_ds_b_vectors(self)
            if bvec.ndim == 0:
                bvec = np.array([])
            return bvec
        except IndexError:
            return np.array([])

    def get_dti(self) -> tuple[Number, np.ndarray]:
        return self.get_b_value(), self.get_b_vector()

    def get_echo_time(self) -> Number:
        return self.get_float('EchoTime')

    def get_flip_angle(self) -> Number:
        return self.get_float('FlipAngle')

    def get_image_orientation_patient(self) -> np.ndarray:
        try:
            iop = self.data_element('ImageOrientationPatient').value
        except (KeyError, ValueError):
            iop = [0, 0, 1, 0, 1, 0]
        # Reverse orientation vectors from (x,y,z) to (z,y,x)
        return np.array((iop[2], iop[1], iop[0],
                         iop[5], iop[4], iop[3]))

    def get_image_position_patient(self) -> np.ndarray:
        origin = self.data_element('ImagePositionPatient').value
        if origin is not None:
            x = float(origin[0])
            y = float(origin[1])
            z = float(origin[2])
            return np.array([z, y, x])
        raise ValueError('ImagePositionPatient not found')

    def calculate_slice_location(self) -> float:
        """Function to calculate slice location from image position and orientation.

        Args:
            self: image (pydicom dicom object)
        Returns:
            float: calculated slice location for this slice
        Raises:
            ValueError: when sliceLocation cannot be calculated
        """

        def get_attribute(im, tag):
            if tag in im:
                return im[tag].value
            else:
                raise ValueError('Tag {:08x} ({}) not found'.format(
                    tag, keyword_for_tag(tag)
                ))

        def get_normal(im):
            iop = np.array(get_attribute(im, tag_for_keyword('ImageOrientationPatient')))
            normal = np.zeros(3)
            normal[0] = iop[1] * iop[5] - iop[2] * iop[4]
            normal[1] = iop[2] * iop[3] - iop[0] * iop[5]
            normal[2] = iop[0] * iop[4] - iop[1] * iop[3]
            return normal

        _name: str = '{}.{}'.format(__name__, self.calculate_slice_location.__name__)
        try:
            ipp = np.array(get_attribute(self, tag_for_keyword('ImagePositionPatient')),
                           dtype=float)
            _normal = get_normal(self)
            return np.inner(_normal, ipp)
        except ValueError as e:
            raise ValueError(f'{_name}: Cannot calculate slice location: {e}')

    def get_slice_location(self) -> float:
        _name: str = '{}.{}'.format(__name__, self.get_slice_location.__name__)
        try:
            return float(self.SliceLocation)
        except AttributeError:
            logger.debug(f'{_name}: Calculate SliceLocation')
            try:
                return self.calculate_slice_location()
            except ValueError:
                pass
        return 0.0

    def get_pixels_with_shape(self, shape):
        """Get pixels from image object. Reshape image to given shape

        Args:
            self: dicom instance
            shape: requested image shape
        Returns:
            si: numpy array of given shape
        """

        _name: str = '{}.{}'.format(__name__, self.get_pixels_with_shape.__name__)
        _use_float = False
        try:
            if 'RescaleSlope' in self and 'RescaleIntercept' in self:
                _use_float = abs(self.RescaleSlope - 1) > 1e-4 or abs(self.RescaleIntercept) > 1e-4
            pixels: np.ndarray
            if _use_float:
                pixels = (float(self.RescaleSlope) * self.pixel_array.astype(float) +
                          float(self.RescaleIntercept))
            else:
                pixels = self.pixel_array
            if shape != pixels.shape:
                if self.PhotometricInterpretation == 'RGB':
                    # RGB image
                    rgb_dtype = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])
                    si = pixels.copy().view(dtype=rgb_dtype).reshape(pixels.shape[:-1])
                elif 'NumberOfFrames' in self:
                    logger.debug('{}: NumberOfFrames: {}'.format(_name, self.NumberOfFrames))
                    if (self.NumberOfFrames,) + shape == pixels.shape:
                        logger.debug('{}: NumberOfFrames {} copy pixels'.format(_name, self.NumberOfFrames))
                        si = pixels
                    else:
                        logger.debug('{}: NumberOfFrames pixels differ {} {}'.format(
                            _name, (self.NumberOfFrames,) + shape, pixels.shape))
                        raise IndexError(
                            'NumberOfFrames pixels differ {} {}'.format(
                                (self.NumberOfFrames,) + shape, pixels.shape)
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
            if self.BitsAllocated == 1:
                logger.debug(
                    "{}: Binary image, image.shape={}, image shape=({},{},{})".format(
                        _name, self.shape, self.NumberOfFrames, self.Rows, self.Columns))
                _myarr = np.frombuffer(self.PixelData, dtype=np.uint8)
                # Reverse bit order, and copy the array to get a
                # contiguous array
                bits = np.unpackbits(_myarr).reshape(-1, 8)[:, ::-1].copy()
                si = np.fliplr(
                    bits.reshape(
                        1, self.NumberOfFrames, self.Rows, self.Columns))
                if _use_float:
                    si = float(self.RescaleSlope) * si + float(self.RescaleIntercept)
            else:
                raise
        return si

    def copy_attributes_to_header(self, hdr: Header):
        for attribute in attributes:
            dicom_attribute = attribute[0].upper() + attribute[1:]
            try:
                setattr(hdr, attribute,
                        self.data_element(dicom_attribute).value
                        )
            except (AttributeError, KeyError):
                pass
        if 'ReferencedSeriesSequence' in self:
            hdr.referencedSeriesUID = self.ReferencedSeriesSequence[0].SeriesInstanceUID


def _get_fixed_tags(tag_list: list, fixed: list, lookup: int) -> list:
    tags = len(fixed)
    values = []
    fixed_mask = [False for _ in range(tags)]
    for _, tag in tag_list:
        found = True
        for i in range(tags):
            if i != lookup:
                fixed_mask[i] = True
                if tag[i] != fixed[i]:
                    found = False
        if found:
            values.append(tag[lookup])
    return sorted(values), fixed_mask

