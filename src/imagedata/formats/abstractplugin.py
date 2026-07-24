"""Abstract class for image formats.

Defines generic functions.
"""

# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
# Copyright (c) 2017-2026 Erling Andersen, Haukeland University Hospital, Bergen, Norway

from abc import ABCMeta, abstractmethod  # , abstractproperty
import math
import logging
import numpy as np
from collections import namedtuple
from . import BadShapeGiven, NotImageError, shape_to_str, INPUT_ORDER_TIME, SORT_ON_TAG
from ..header import Header
from ..archives.abstractarchive import AbstractArchive

logger = logging.getLogger(__name__)


class NoOtherInstance(Exception):
    pass


class AbstractPlugin(object, metaclass=ABCMeta):
    """Abstract base class definition for imagedata format plugins.
    Plugins must be a subclass of AbstractPlugin and
    must define the attributes set in __init__() and
    the following methods:

    read() method
    write_3d_numpy() method
    write_4d_numpy() method

    Attributes:
        input_order
        tags
        transformationMatrix
    """

    plugin_type = 'format'

    def __init__(self, name, description, authors, version, url):
        object.__init__(self)
        self.__name = name
        self.__description = description
        self.__authors = authors
        self.__version = version
        self.__url = url
        self.input_order = None
        self.tags = None
        self.transformationMatrix = None

    @property
    def name(self):
        """Plugin name

        Single word string describing the image format.
        Typical names: dicom, nifti, itk.
        """
        return self.__name

    @property
    def description(self):
        """Plugin description

        Single line string describing the image format.
        """
        return self.__description

    @property
    def authors(self):
        """Plugin authors

        Multi-line string naming the author(s) of the plugin.
        """
        return self.__authors

    @property
    def version(self):
        """Plugin version

        String giving the plugin version.
        Version scheme: 1.0.0
        """
        return self.__version

    @property
    def url(self):
        """Plugin URL

        URL string to the site of the plugin or the author(s).
        """
        return self.__url

    def read(self, sources, pre_hdr, input_order, opts):
        """Read image data

        Generic version for images which will be sorted on their filenames.

        Args:
            sources: list of sources to image data
            pre_hdr: DICOM template
            input_order: sort order
            opts: Input options (dict)
        Returns:
            Tuple of
                - hdr: Header dict
                - si[tag,slice,rows,columns]: numpy array
        """

        _name: str = '{}.{}'.format(__name__, self.read.__name__)
        hdr = Header()
        hdr.input_format = self.name
        hdr.input_order = input_order

        # image_list: list of tuples (hdr,si)
        logger.debug("{}: sources {}".format(_name, sources))
        image_list = list()
        for source in sources:
            logger.debug("{}: source: {} {}".format(_name, type(source), source))
            archive: AbstractArchive = source['archive']
            scan_files = source['files']
            if scan_files is None or len(scan_files) == 0:
                if archive.base is not None:
                    scan_files = [archive.base]
                else:
                    scan_files = ['*']
            elif archive.base is not None:
                raise ValueError('When is archive.base with source[files]')
            for file_handle in archive.getmembers(scan_files):
                logger.debug("{}: file_handle {}".format(_name, file_handle.filename))
                if self._need_local_file():
                    logger.debug("{}: need local file {}".format(
                        _name, file_handle.filename))
                    f = archive.to_localfile(file_handle)
                    logger.debug("{}: local file {}".format(_name, f))
                    try:
                        info, si = self._read_image(f, opts, hdr)
                    except NotImageError:
                        continue
                else:
                    f = archive.open(file_handle, mode='rb')
                    logger.debug("{}: file {}".format(_name, f))
                    try:
                        info, si = self._read_image(f, opts, hdr)
                    except NotImageError:
                        continue
                    finally:
                        f.close()
                # info is None when no image was read
                if info is not None:
                    image_list.append((info, si))
        if len(image_list) < 1:
            raise ValueError('No image data read')
        info, si = image_list[0]
        if si is not None:
            si = self._reduce_shape(si)
            logger.debug('{}: reduced si {}'.format(_name, si.shape))
            shape = (len(image_list),) + si.shape
            dtype = si.dtype
            logger.debug('{}: shape {}'.format(_name, shape))
            si = np.zeros(shape, dtype)
            i = 0
            for info, img in image_list:
                if 'input_sort' in opts and opts['input_sort'] == SORT_ON_TAG:
                    si[:, i] = img
                else:
                    si[i] = img
                i += 1
            logger.debug('{}: si {}'.format(_name, si.shape))

            # Simplify shape
            si = self._reduce_shape(si)
            logger.debug('{}: reduced si {}'.format(_name, si.shape))

            _shape = si.shape
            logger.debug('{}: _shape {}'.format(_name, _shape))
            _ndim = len(_shape)
            nz = 1
            if _ndim > 2:
                nz = _shape[-3]
            logger.debug('{}: slices {}'.format(_name, nz))

            if 'input_shape' in opts and opts['input_shape']:
                try:
                    input_shape = tuple(int(_) for _ in opts['input_shape'].split('x'))
                except ValueError as e:
                    raise BadShapeGiven(f'Illegal input_shape "{opts['input_shape']}": {e}')
                except AttributeError:
                    input_shape = tuple(opts['input_shape'])
                # Guess the meaning of input_shape
                new_shape = None
                for attempt_shape in [input_shape,
                                      input_shape + si.shape[-3:],
                                      input_shape + si.shape[-2:]]:
                    if math.prod(attempt_shape) == si.size:
                        new_shape = attempt_shape
                        break
                if new_shape is None:
                    raise BadShapeGiven(f'input_shape {input_shape} does not match {si.shape}')
                if input_order == 'none' and len(new_shape) > 3:
                    raise BadShapeGiven(f'input_shape {input_shape} not acceptable for input_order none')
                if len(new_shape) != len(input_order.split(',')) + 3:
                    raise BadShapeGiven(f'input_shape {input_shape} does not match input_order {input_order}')
                logging.warning(f'Modifying input shape from {si.shape} to {new_shape}. Take care!')
                si = si.reshape(new_shape)

        logger.debug('{}: calling _set_tags'.format(_name))
        self._set_tags(image_list, hdr, si)

        if si is not None:
            logger.info("{}: Data shape read: {}".format(_name, shape_to_str(si.shape)))

        # Add any DICOM template
        if pre_hdr is not None:
            hdr.update(pre_hdr)

        logger.debug('{}: hdr {}'.format(_name, hdr))
        return {0: hdr}, {0: si}

    def _need_local_file(self):
        """Does the plugin need access to local files?

        Returns:
            Boolean
                - True: The plugin need access to local filenames
                - False: The plugin can access files given by an open file handle
        """

        return False

    @abstractmethod
    def _read_image(self, f, opts, hdr):
        """Read image data from given file handle

        Args:
            self: format plugin instance
            f: file handle or filename (depending on self._need_local_file)
            opts: Input options (dict)
            hdr: Header dict
        Returns:
            Tuple of
                hdr: Header dict
                    Return values:
                        - info: Internal data for the plugin
                          None if the given file should not be included (e.g. raw file)
                si: numpy array (multi-dimensional)
        """
        pass

    @abstractmethod
    def _set_tags(self, image_list, hdr, si):
        """Set header tags.

        Args:
            self: format plugin instance
            image_list: list with (info,img) tuples
            hdr: Header dict
            si: numpy array (multi-dimensional)
        Returns:
            hdr: Header dict
        """
        pass

    @abstractmethod
    def write_3d_numpy(self, si, destination, opts):
        """Write 3D Series image

        Args:
            self: format plugin instance
            si[slice,rows,columns]: Series array
            destination: dict of archive and filenames
            opts: Output options (dict)
        """
        pass

    @abstractmethod
    def write_4d_numpy(self, si, destination, opts):
        """Write 4D Series image

        Args:
            self: format plugin instance
            si[tag,slice,rows,columns]: Series array
            destination: dict of archive and filenames
            opts: Output options (dict)
        """
        pass

    def getTimeline(self):
        """Get timeline

        Returns:
            Timeline in seconds, as numpy array of floats
                Delta time is given as seconds. First image is t=0.
                Length of array is number of tags.
        Raises:
            ValueError: tags for dataset is not time tags
        """
        if self.input_order == INPUT_ORDER_TIME:
            timeline = [0.0]
            for t in range(1, len(self.tags[0])):
                timeline.append(self.tags[0][t] - self.tags[0][0])
            return np.array(timeline)
        else:
            raise ValueError("No timeline tags are available. Input order: {}".format(
                self.input_order))

    def getPositionForVoxel(self, r, transformation=None):
        """Get patient position for center of given voxel r

        Use Patient Position and Image Orientation to calculate world
        coordinates for given voxel

        Args:
            r: (z,y,x) of voxel in voxel coordinates as numpy.array
            transformation: transformation matrix when different from self.transformationMatrix
        Returns:
            (z,y,x) of voxel in world coordinates (mm) as numpy.array
        """

        if transformation is None:
            transformation = self.transformationMatrix

        newpos = np.dot(transformation, np.hstack((r, [1])))

        return newpos[:3]

    def getVoxelForPosition(self, p, transformation=None):
        """ Get voxel for given patient position p

        Use Patient Position and Image Orientation to calculate world
        coordinates for given voxel

        Args:
            p: (z,y,x) of voxel in world coordinates (mm) as numpy.array
            transformation: transformation matrix when different from self.transformationMatrix
        Returns:
            (z,y,x) of voxel in voxel coordinates as numpy.array
        """

        if transformation is None:
            transformation = self.transformationMatrix

        qinv = np.linalg.inv(transformation)
        r = np.dot(qinv, np.hstack((p, [1])))

        # z,y,x
        return (r + 0.5).astype(int)[:3]

    @staticmethod
    def replace_geometry_attributes(im, gim):
        """Replace geometry attributes in image with values from gim
        """

        im.SliceLocation = gim.SliceLocation
        im.ImagePositionPatient = gim.ImagePositionPatient
        im.ImageOrientationPatient = gim.ImageOrientationPatient
        im.FrameOfReferenceUID = gim.FrameOfReferenceUID
        im.PositionReferenceIndicator = gim.PositionReferenceIndicator
        im.SliceThickness = gim.SliceThickness
        try:
            im.SpacingBetweenSlices = gim.SpacingBetweenSlices
        except AttributeError:
            pass
        im.AcquisitionMatrix = gim.AcquisitionMatrix
        im.PixelSpacing = gim.PixelSpacing

    @staticmethod
    def _reduce_shape(si, axes: namedtuple = None):
        """Reduce shape when leading shape(s) are 1.

        Will not reduce to less than 2-dimensional image.
        Also reduce axes when reducing shape.

        Args:
            self: format plugin instance
            si[...]: Series array
        Raises:
            ValueError: tags for dataset is not time tags
        """

        if si is None:
            return si
        mindim = 2
        while si.ndim > mindim:
            if si.shape[0] == 1:
                si = si.reshape(si.shape[1:])
                if axes is not None:
                    axes = axes[1:]
            else:
                break
        return si

    def _reorder_to_dicom(self, data, flip=False, flipud=False):
        """Reorder data to internal DICOM format.

        Swap axes, except for rows and columns.

        5D:
        data  order: data[rows,columns,slices,tags,d5]
        DICOM order: si  [d5,tags,slices,rows,columns]

        4D:
        data  order: data[rows,columns,slices,tags]
        DICOM order: si  [tags,slices,rows,columns]

        3D:
        data  order: data[rows,columns,slices]
        DICOM order: si  [slices,rows,columns]

        2D:
        data  order: data[rows,columns]
        DICOM order: si  [rows,columns]

        flip: Whether rows and columns are swapped.
        flipud: Whether matrix is transposed
        """

        _name: str = '{}.{}'.format(__name__, self._reorder_to_dicom.__name__)
        logger.debug('{}: shape in {}'.format(_name, data.shape))
        if data.ndim == 5:
            rows, columns, slices, tags, d5 = data.shape
            if flipud:
                rows, columns = columns, rows
            si = np.zeros((d5, tags, slices, rows, columns), data.dtype)
            for d in range(d5):
                for tag in range(tags):
                    for slice in range(slices):
                        si[d, tag, slice, :, :] = self._reorder_slice(data[:, :, slice, tag, d],
                                                                      flip=flip, flipud=flipud)
        elif data.ndim == 4:
            rows, columns, slices, tags = data.shape
            if flipud:
                rows, columns = columns, rows
            si = np.zeros((tags, slices, rows, columns), data.dtype)
            for tag in range(tags):
                for slice in range(slices):
                    si[tag, slice, :, :] = self._reorder_slice(data[:, :, slice, tag],
                                                               flip=flip, flipud=flipud)
        elif data.ndim == 3:
            rows, columns, slices = data.shape
            if flipud:
                rows, columns = columns, rows
            si = np.zeros((slices, rows, columns), data.dtype)
            for slice in range(slices):
                si[slice, :, :] = self._reorder_slice(data[:, :, slice], flip=flip, flipud=flipud)
        elif data.ndim == 2:
            rows, columns = data.shape
            if flipud:
                rows, columns = columns, rows
            si = np.zeros((rows, columns), data.dtype)
            si[:] = self._reorder_slice(data[:], flip=flip, flipud=flipud)
        else:
            raise ValueError('Dimension %d is not implemented' % data.ndim)
        logger.debug('{}: shape out {}'.format(_name, si.shape))
        return si

    def _reorder_from_dicom(self, data, flip=False, flipud=False):
        """Reorder data from internal DICOM format.

        Swap axes, except for rows and columns.

        5D:
        DICOM  order: data[d5,tags,slices,rows,columns]
        return order: si  [rows,columns,slices,tags,d5]

        4D:
        DICOM  order: data[tags,slices,rows,columns]
        return order: si  [rows,columns,slices,tags]

        3D:
        DICOM  order: data[slices,rows,columns]
        return order: si  [rows,columns,slices]

        2D:
        DICOM  order: data[rows,columns]
        return order: si [rows,columns]

        flip: Whether rows and columns are swapped.
        flipud: Whether matrix is transposed
        """

        _name: str = '{}.{}'.format(__name__, self._reorder_from_dicom.__name__)
        logger.debug('{}: shape in {}'.format(_name, data.shape))
        if data.ndim == 5:
            d5, tags, slices, rows, columns = data.shape
            if flipud:
                rows, columns = columns, rows
            si = np.zeros((rows, columns, slices, tags, d5), data.dtype)
            for d in range(d5):
                for tag in range(tags):
                    for slice in range(slices):
                        si[:, :, slice, tag, d] = self._reorder_slice(data[d, tag, slice, :, :],
                                                                      flip=flip, flipud=flipud)
        elif data.ndim == 4:
            tags, slices, rows, columns = data.shape
            if flipud:
                rows, columns = columns, rows
            si = np.zeros((rows, columns, slices, tags), data.dtype)
            for tag in range(tags):
                for slice in range(slices):
                    si[:, :, slice, tag] = self._reorder_slice(data[tag, slice, :, :],
                                                               flip=flip, flipud=flipud)
        elif data.ndim == 3:
            slices, rows, columns = data.shape
            if flipud:
                rows, columns = columns, rows
            si = np.zeros((rows, columns, slices), data.dtype)
            for slice in range(slices):
                si[:, :, slice] = self._reorder_slice(data[slice, :, :], flip=flip, flipud=flipud)
        elif data.ndim == 2:
            rows, columns = data.shape
            if flipud:
                rows, columns = columns, rows
            si = np.zeros((rows, columns), data.dtype)
            si[:, :] = self._reorder_slice(data[:, :], flip=flip, flipud=flipud)
        else:
            raise ValueError('Dimension %d is not implemented' % data.ndim)
        logger.debug('{}: shape out {}'.format(_name, si.shape))
        return si

    @staticmethod
    def _reorder_slice(data: np.ndarray, flip: bool, flipud: bool) -> np.ndarray:
        if flip and flipud:
            return np.fliplr(data).T
        elif flip:
            return np.fliplr(data)
        elif flipud:
            return data.T
        else:
            return data
