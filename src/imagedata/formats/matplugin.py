"""Read/Write Matlab-compatible MAT files
"""

# Copyright (c) 2013-2025 Erling Andersen, Haukeland University Hospital, Bergen, Norway

import logging
import numpy as np
import mimetypes
from collections import namedtuple
import scipy
import scipy.io
from . import NotImageError, WriteNotImplemented, \
    shape_to_str, sort_on_to_str, SORT_ON_SLICE
from ..axis import UniformLengthAxis
from .abstractplugin import AbstractPlugin
from ..archives.abstractarchive import AbstractArchive

logger = logging.getLogger(__name__)

mimetypes.add_type('application/x-matlab-data', '.mat')


class ImageTypeError(Exception):
    """
    Thrown when trying to load or save an image of unknown type.
    """
    pass


class DependencyError(Exception):
    """
    Thrown when a required module could not be loaded.
    """
    pass


class MultipleVariablesInMatlabFile(Exception):
    """
    Reading multiple variables from a MAT file is not implemented.
    """


class MatrixDimensionNotImplemented(Exception):
    """
    Matrix dimension is not implemented.
    """


class MatPlugin(AbstractPlugin):
    """Read/write MAT files.
    """

    name = "mat"
    description = "Read and write MAT files."
    authors = "Erling Andersen"
    version = "1.1.0"
    url = "www.helse-bergen.no"
    extensions = [".mat"]

    def __init__(self):
        super(MatPlugin, self).__init__(self.name, self.description,
                                        self.authors, self.version, self.url)
        self.slices = None
        self.spacing = None
        self.tags = None
        self.output_sort = None

    # noinspection PyTypeChecker,PyUnresolvedReferences
    def _read_image(self, f, opts, hdr):
        """Read image data from given file handle

        Args:
            self: format plugin instance
            f: file handle or filename (depending on self._need_local_file)
            opts: Input options (dict)
            hdr: Header
        Returns:
            Tuple of
                hdr: Header
                    Return values:
                        - info: Internal data for the plugin
                            None if the given file should not be included (e.g. raw file)
                si: numpy array (multi-dimensional)
        """

        _name: str = '{}.{}'.format(__name__, self._read_image.__name__)

        info = {}

        if hdr.input_order == 'auto':
            hdr.input_order = 'none'

        hdr.color = False
        try:
            logger.debug('{}: scipy.io.loadmat({})'.format(_name, f))
            mdictlist = scipy.io.whosmat(f)
            if len(mdictlist) != 1:
                names = []
                for name, shape, dtype in mdictlist:
                    names.append(name)
                logger.debug('{}: scipy.io.loadmat len(mdict) {}'.format(
                    _name, len(mdictlist)))
                logger.debug('{}: Multiple variables in MAT file {}'.format(_name, f))
                # raise MultipleVariablesInMatlabFile('Multiple variables in MAT file {}: '
                #                                     '{}'.format(f, names))
                raise MultipleVariablesInMatlabFile('Multiple variables in MAT file {}'.format(f))
            name, shape, dtype = mdictlist[0]
            logger.debug('{}: name {} shape {} dtype {}'.format(
                _name, name, shape, dtype))
            mdict = scipy.io.loadmat(f, variable_names=(name,))
            logger.debug('{}: variable {}'.format(_name, name))
            si = self._reorder_to_dicom(mdict[name])
            logger.info("{}: Data shape _read_image MAT: {} {}".format(_name, si.shape, si.dtype))
        except MultipleVariablesInMatlabFile:
            raise
        except NotImageError:
            raise NotImageError('{} does not look like a MAT file'.format(f))
        return info, si

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

        _name: str = '{}.{}'.format(__name__, self._set_tags.__name__)

        # Set spacing
        hdr.spacing = (1.0, 1.0, 1.0)

        row_axis = UniformLengthAxis(
            'row',
            0,
            si.shape[-2]
        )
        column_axis = UniformLengthAxis(
            'column',
            0,
            si.shape[-1]
        )
        # Set tags
        nt = nz = 1
        if si.ndim > 2:
            nz = si.shape[-3]
            slice_axis = UniformLengthAxis(
                'slice',
                0,
                nz
            )
            if si.ndim > 3:
                nt = si.shape[-4]
                tag_axis = UniformLengthAxis(
                    hdr.input_order,
                    0,
                    nt
                )
                Axes = namedtuple('Axes', [
                    hdr.input_order, 'slice', 'row', 'column'
                ])
                axes = Axes(tag_axis, slice_axis, row_axis, column_axis)
            else:
                Axes = namedtuple('Axes', [
                    'slice', 'row', 'column'
                ])
                axes = Axes(slice_axis, row_axis, column_axis)
        else:
            Axes = namedtuple('Axes', ['row', 'column'])
            axes = Axes(row_axis, column_axis)
        hdr.axes = axes
        logger.debug('{}: nt {}, nz {}'.format(_name, nt, nz))
        dt = 1
        times = [(_,) for _ in np.arange(0, nt * dt, dt)]
        hdr.tags = {}
        for slice in range(nz):
            hdr.tags[slice] = np.array(times, dtype=tuple)

        hdr.photometricInterpretation = 'MONOCHROME2'
        hdr.color = False

    def write_3d_numpy(self, si, destination, opts):
        """Write 3D numpy image as MAT file

        Args:
            self: MATPlugin instance
            si: Series array (3D or 4D), including these attributes:
                slices,
                spacing,
                tags

            destination: dict of archive and filenames
            opts: Output options (dict)
        """

        _name: str = '{}.{}'.format(__name__, self.write_3d_numpy.__name__)

        if si.color:
            raise WriteNotImplemented(
                "Writing color MAT images not implemented.")

        logger.debug('{}: destination {}'.format(_name, destination))

        self.slices = si.slices
        self.spacing = si.spacing
        self.tags = si.tags

        logger.info("{}: Data shape write: {}".format(_name, shape_to_str(si.shape)))
        if si.ndim == 4 and si.shape[0] == 1:
            si.shape = si.shape[1:]
        assert si.ndim == 2 or si.ndim == 3, \
            "write_3d_series: input dimension %d is not 2D/3D." % si.ndim

        img = self._reorder_from_dicom(si)
        self._write_numpy_to_mat(img, destination, opts)

    def write_4d_numpy(self, si, destination, opts):
        """Write 4D numpy image as MAT files

        Args:
            self: MATPlugin instance
            si[tag,slice,rows,columns]: Series array, including these attributes:
                slices,
                spacing,
                tags

            destination: dict of archive and filenames
            opts: Output options (dict)
        """

        _name: str = '{}.{}'.format(__name__, self.write_4d_numpy.__name__)

        if si.color:
            raise WriteNotImplemented(
                "Writing color MAT images not implemented.")

        logger.debug('{}: destination {}'.format(_name, destination))

        self.slices = si.slices
        self.spacing = si.spacing
        self.tags = si.tags

        # Defaults
        self.output_sort = SORT_ON_SLICE
        if 'output_sort' in opts:
            self.output_sort = opts['output_sort']

        if si.ndim != 4:
            raise ValueError("write_4d_numpy: input dimension {} is not 4D.".format(si.ndim))

        logger.debug("{}: si dtype {}, shape {}, sort {}".format(
            _name, si.dtype, si.shape,
            sort_on_to_str(self.output_sort)))

        steps = si.shape[0]
        slices = si.shape[1]
        if steps != len(si.tags[0]):
            raise ValueError(
                "write_4d_series: tags of dicom template ({}) differ "
                "from input array ({}).".format(len(si.tags[0]), steps))
        if slices != si.slices:
            raise ValueError(
                "write_4d_series: slices of dicom template ({}) differ "
                "from input array ({}).".format(si.slices, slices))

        img = self._reorder_from_dicom(si)
        self._write_numpy_to_mat(img, destination, opts)

    def _write_numpy_to_mat(self, img, destination, opts):
        _name: str = '{}.{}'.format(__name__, self._write_numpy_to_mat.__name__)

        archive: AbstractArchive = destination['archive']
        archive.set_member_naming_scheme(
            fallback='Image.mat',
            level=0,
            default_extension='.mat',
            extensions=self.extensions
        )
        query = None
        if destination['files'] is not None and len(destination['files']):
            query = destination['files'][0]
        filename = archive.construct_filename(
            tag=None,
            query=query
        )

        with archive.open(filename, 'wb') as f:
            logger.debug("{}: Calling savemat".format(_name))
            scipy.io.savemat(f, {'A': img})
