"""Read/Write Matlab-compatible MAT files
"""

# Copyright (c) 2013-2022 Erling Andersen, Haukeland University Hospital, Bergen, Norway

import os.path
import logging
import numpy as np
import scipy
import scipy.io
from . import NotImageError, input_order_to_dirname_str, WriteNotImplemented,\
    shape_to_str, sort_on_to_str, SORT_ON_SLICE
from ..axis import UniformLengthAxis
from .abstractplugin import AbstractPlugin

logger = logging.getLogger(__name__)


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
    """Read/write MAT files."""

    name = "mat"
    description = "Read and write MAT files."
    authors = "Erling Andersen"
    version = "1.0.0"
    url = "www.helse-bergen.no"

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
                raise MultipleVariablesInMatlabFile('Multiple variables in MAT file {}: '
                                                    '{}'.format(f, names))
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
            -   slices,
            -   spacing,
            -   tags

            destination: dict of archive and filenames
            opts: Output options (dict)
        """

        if si.color:
            raise WriteNotImplemented(
                "Writing color MAT images not implemented.")

        logger.debug('MatPlugin.write_3d_numpy: destination {}'.format(destination))
        archive = destination['archive']
        filename_template = 'Image_%05d.mat'
        if len(destination['files']) > 0 and len(destination['files'][0]) > 0:
            filename_template = destination['files'][0]

        self.slices = si.slices
        self.spacing = si.spacing
        self.tags = si.tags

        logger.info("Data shape write: {}".format(shape_to_str(si.shape)))
        if si.ndim == 4 and si.shape[0] == 1:
            si.shape = si.shape[1:]
        # if si.ndim == 2:
        #    si.shape = (1,) + si.shape
        assert si.ndim == 2 or si.ndim == 3,\
            "write_3d_series: input dimension %d is not 2D/3D." % si.ndim
        # slices = si.shape[0]
        # if slices != si.slices:
        #    raise ValueError("write_3d_series: slices of dicom template ({}) differ "
        #        "from input array ({}).".format(si.slices, slices))

        # newshape = tuple(reversed(si.shape))
        # logger.info("Data shape matlab write: {}".format(shape_to_str(newshape)))
        # logger.debug('matplugin.write_3d_numpy newshape {}'.format(newshape))
        # img = si.reshape(newshape, order='C')
        # img = si.reshape(newshape, order='F')
        # img = np.asfortranarray(si)
        # img = self._dicom_to_mat(si)
        # img = self._reorder_from_dicom(si)
        # logger.debug('matplugin.write_3d_numpy si  {} {}'.format(si.shape, si.dtype))
        # logger.debug('matplugin.write_3d_numpy img {} {}'.format(img.shape, img.dtype))

        # if not os.path.isdir(directory_name):
        #    os.makedirs(directory_name)
        try:
            filename = filename_template % 0
        except TypeError:
            filename = filename_template
        # filename = os.path.join(directory_name, filename)
        if len(os.path.splitext(filename)[1]) == 0:
            filename = filename + '.mat'
        with archive.open(filename, 'wb') as f:
            # scipy.io.savemat(f, {'A': img})
            scipy.io.savemat(f, {'A': self._reorder_from_dicom(si)})

    def write_4d_numpy(self, si, destination, opts):
        """Write 4D numpy image as MAT files

        Args:
            self: MATPlugin instance
            si[tag,slice,rows,columns]: Series array, including these attributes:
            -   slices,
            -   spacing,
            -   tags

            destination: dict of archive and filenames
            opts: Output options (dict)
        """

        if si.color:
            raise WriteNotImplemented(
                "Writing color MAT images not implemented.")

        logger.debug('MatPlugin.write_4d_numpy: destination {}'.format(destination))
        archive = destination['archive']
        filename_template = 'Image_%05d.mat'
        if len(destination['files']) > 0 and len(destination['files'][0]) > 0:
            filename_template = destination['files'][0]

        self.slices = si.slices
        self.spacing = si.spacing
        self.tags = si.tags

        # Defaults
        self.output_sort = SORT_ON_SLICE
        if 'output_sort' in opts:
            self.output_sort = opts['output_sort']

        # Should we allow to write 3D volume?
        # if si.ndim == 3:
        #    si.shape = (1,) + si.shape
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

        # if not os.path.isdir(directory_name):
        #    os.makedirs(directory_name)

        filename = filename_template % 0
        # filename = os.path.join(directory_name, filename)
        if len(os.path.splitext(filename)[1]) == 0:
            filename = filename + '.mat'
        with archive.open(filename, 'wb') as f:
            # scipy.io.savemat(f, {'A': self._reorder_from_dicom(si)})
            img = self._reorder_from_dicom(si)
            logger.debug("write_4d_numpy: Calling savemat")
            scipy.io.savemat(f, {'A': img})
