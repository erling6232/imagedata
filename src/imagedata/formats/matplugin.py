"""Read/Write Matlab-compatible MAT files
"""

# Copyright (c) 2013-2024 Erling Andersen, Haukeland University Hospital, Bergen, Norway

import logging
import numpy as np
import mimetypes
import scipy
import scipy.io
from . import NotImageError, input_order_to_dirname_str, WriteNotImplemented, \
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
    version = "1.0.0"
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

        info = {}

        if hdr.input_order == 'auto':
            hdr.input_order = 'none'

        hdr.color = False
        try:
            logger.debug('matplugin._read_image: scipy.io.loadmat({})'.format(f))
            mdictlist = scipy.io.whosmat(f)
            if len(mdictlist) != 1:
                names = []
                for name, shape, dtype in mdictlist:
                    names.append(name)
                logger.debug('matplugin._read_image: scipy.io.loadmat len(mdict) {}'.format(
                    len(mdictlist)))
                logger.debug('matplugin._read_image: Multiple variables in MAT file {}'.format(f))
                # raise MultipleVariablesInMatlabFile('Multiple variables in MAT file {}: '
                #                                     '{}'.format(f, names))
                raise MultipleVariablesInMatlabFile('Multiple variables in MAT file {}'.format(f))
            name, shape, dtype = mdictlist[0]
            logger.debug('matplugin._read_image: name {} shape {} dtype {}'.format(
                name, shape, dtype))
            mdict = scipy.io.loadmat(f, variable_names=(name,))
            logger.debug('matplugin._read_image variable {}'.format(name))
            si = self._reorder_to_dicom(mdict[name])
            logger.info("Data shape _read_image MAT: {} {}".format(si.shape, si.dtype))
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

        # Set spacing
        hdr.spacing = (1.0, 1.0, 1.0)

        axes = list()
        axes.append(UniformLengthAxis(
            'row',
            0,
            si.shape[-2])
        )
        axes.append(UniformLengthAxis(
            'column',
            0,
            si.shape[-1])
        )
        # Set tags
        nt = nz = 1
        if si.ndim > 2:
            nz = si.shape[-3]
            axes.insert(0, UniformLengthAxis(
                'slice',
                0,
                nz)
            )
        if si.ndim > 3:
            nt = si.shape[-4]
            axes.insert(0, UniformLengthAxis(
                input_order_to_dirname_str(hdr.input_order),
                0,
                nt)
            )
        hdr.axes = axes
        logger.debug('matplugin._set_tags nt {}, nz {}'.format(
            nt, nz))
        dt = 1
        times = np.arange(0, nt * dt, dt)
        tags = {}
        for slice in range(nz):
            tags[slice] = np.array(times)
        hdr.tags = tags
        # logger.debug('matplugin._set_tags tags {}'.format(tags))

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

        if si.color:
            raise WriteNotImplemented(
                "Writing color MAT images not implemented.")

        logger.debug('MatPlugin.write_3d_numpy: destination {}'.format(destination))

        self.slices = si.slices
        self.spacing = si.spacing
        self.tags = si.tags

        logger.info("Data shape write: {}".format(shape_to_str(si.shape)))
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

        if si.color:
            raise WriteNotImplemented(
                "Writing color MAT images not implemented.")

        logger.debug('MatPlugin.write_4d_numpy: destination {}'.format(destination))

        self.slices = si.slices
        self.spacing = si.spacing
        self.tags = si.tags

        # Defaults
        self.output_sort = SORT_ON_SLICE
        if 'output_sort' in opts:
            self.output_sort = opts['output_sort']

        if si.ndim != 4:
            raise ValueError("write_4d_numpy: input dimension {} is not 4D.".format(si.ndim))

        logger.debug("write_4d_numpy: si dtype {}, shape {}, sort {}".format(
            si.dtype, si.shape,
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
            logger.debug("_write_numpy_to_mat: Calling savemat")
            scipy.io.savemat(f, {'A': img})
