"""Read/Write image files supported by Pillow
"""

# Copyright (c) 2026 Erling Andersen, Haukeland University Hospital, Bergen, Norway

import os.path
import sys
import logging
import numpy as np
import traceback
from collections import namedtuple
from PIL import Image, UnidentifiedImageError

from .abstractplugin import AbstractPlugin
from ..archives.abstractarchive import AbstractArchive
from ..axis import UniformLengthAxis
from .. import formats

logger = logging.getLogger(__name__)


class ImageTypeError(Exception):
    """
    Thrown when trying to load or save an image of unknown type.
    """
    pass


class PillowPlugin(AbstractPlugin):
    name = 'pillow'
    description = 'Pillow plugin for reading and writing image files'
    authors = 'Erling Andersen'
    version = "0.0.1"
    url = "https://github.com/erling6232/imagedata"
    # extensions = [".png", ".jpeg", ".jpg", ".tiff", ".gif"]
    extensions = Image.registered_extensions().keys()

    def __init__(self, *args, **kwargs):
        super(PillowPlugin, self).__init__(self.name, self.description,
                                           self.authors, self.version, self.url)

    def _read_image(self, f, opts, hdr):
        """Read image data from given file handle

        Args:
            f: file handle
            opts: options
            hdr: image header
        Returns:
            Tuple of
                hdr: image header
                si: numpy array
        """

        _name: str = '{}.{}'.format(__name__, self._read_image.__name__)

        info = {}

        if hdr.input_order == 'auto':
            hdr.input_order = 'none'

        hdr.color = False
        try:
            img = Image.open(f)
            info = img

            logger.debug('{}: name {} shape {} dtype {}'.format(
                _name, img.filename, img.format, img.size, img.mode))
            match info.mode.split(';'):
                case ['RGB']:
                    hdr.photometricInterpretation = 'RGB'
                    hdr.color = True
                    rgb_dtype = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])
                    _ = np.asarray(img)
                    si = np.asarray(_).view(dtype=rgb_dtype).reshape(_.shape[:-1])
                    si = self._reorder_to_dicom(si)
                case ['1'] | ['L'] | ['I'] | ['I', _] | ['F']:
                    hdr.photometricInterpretation = 'MONOCHROME2'
                    hdr.color = False
                    si = self._reorder_to_dicom(np.asarray(img))
                case _:
                    raise formats.NotImageError(f'Image mode {info.mode} not supported')
            logger.info("{}: Data shape _read_image PIL: {} {}".format(_name, si.shape, si.dtype))
        except UnidentifiedImageError:
            raise formats.NotImageError('{} is not supported by PIL'.format(f))
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

        info, img = image_list[0]
        hdr.input_format = info.format.lower()

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
        frames = getattr(info, "n_frames", 1)
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
        hdr.tags = {}
        for slice in range(nz):
            hdr.tags[slice] = np.empty(nt, dtype=tuple)
            for i, _ in enumerate(np.arange(0, nt * dt, dt)):
                hdr.tags[slice][i] = (_,)

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

        logger.debug(f'{_name}: destination {destination}')
        archive: AbstractArchive = destination['archive']
        ext = opts['output_formats'][0] if len(opts['output_formats']) > 0 else 'png'
        archive.set_member_naming_scheme(
            fallback='Image_{:05d}.'+ext,
            level=max(0, si.ndim - 2),
            default_extension='.'+ext,
            extensions=self.extensions
        )

        self.slices = si.slices
        try:
            self.tags = si.tags
        except ValueError:
            self.tags = None
        self.output_dir = 'single'
        if 'output_dir' in opts:
            self.output_dir = opts['output_dir']

        logging.info(f'{_name}: Data shape write: {formats.shape_to_str(si.shape)}')
        assert si.ndim == 2 or si.ndim == 3, \
            f'{_name}: input dimension {si.ndim} is not 2D/3D.'

        try:
            if 'serdes' in opts and opts['serdes'] is not None:
                self.descr = opts['serdes']
            else:
                self.descr = ''
        except ValueError:  # Unsure about correct exception
            self.descr = ''

        if si.ndim < 3:
            self.write_slice(None, si, destination, 0)
        else:
            for _slice in range(si.slices):
                self.write_slice((_slice,), si[_slice], destination, _slice)


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

        logger.debug(f'{_name}: destination {destination}')
        archive: AbstractArchive = destination['archive']

        self.slices = si.slices
        try:
            self.tags = si.tags
        except ValueError:
            self.tags = None

        # Defaults
        self.output_dir = 'single'
        if 'output_dir' in opts:
            self.output_dir = opts['output_dir']

        if si.ndim != 4:
            raise ValueError(f'{_name}: input dimension {si.ndim} is not 4D.')

        logging.debug('{_name}: si.tags {si.tags[0]} si.slices {si.slices}')
        steps, slices, ny, nx = si.shape[:]
        if steps != len(si.tags[0]):
            raise ValueError(
                f'{_name}: tags of dicom template ({si.tags[0]}) differ '
                'from input array ({steps}).')
        if slices != si.slices:
            raise ValueError(
                f'{_name}: slices of dicom template ({si.slices}) differ '
                'from input array ({slices}).')

        self.output_sort = formats.SORT_ON_SLICE
        if 'output_sort' not in opts or opts['output_sort'] is None:
            self.output_sort = si.input_sort
        elif 'output_sort' in opts:
            self.output_sort = opts['output_sort']

        self.descr = ''
        try:
            if 'serdes' in opts and opts['serdes'] is not None:
                self.descr = opts['serdes']
        except ValueError:  # Unsure about correct exception
            pass

        match self.output_sort:
            case formats.SORT_ON_SLICE:
                match self.output_dir:
                    case 'single':
                        # Filenames: Image_00000.dcm, sort slice fastest
                        archive.set_member_naming_scheme(
                            fallback='Image_{0:05d}.png',
                            level=1,
                            default_extension='.png',
                            extensions=self.extensions)
                    case 'multi':
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
                                'Image_{' + '{}'.format(len(dirn)) + ':05d}.png'),
                            level=max(0, si.ndim - 2),
                            default_extension='.png',
                            extensions=self.extensions
                        )
                ifile = 0
                for tag in np.ndindex(steps):
                    for _slice in range(si.slices):
                        if si.tags[_slice][tag] is None:
                            continue
                        _tag = tag + (_slice,)
                        match self.output_dir:
                            case 'multi':
                                if _slice == 0:
                                    ifile = 0  # Restart file number in each subdirectory
                                _file_tag = _tag
                            case 'single':
                                _file_tag = (ifile,)
                        try:
                            _t = si.header.tags[_slice][tag]
                            if _t is None:
                                continue
                            self.write_slice(_file_tag, si[_tag], destination, ifile)
                        except Exception:
                            traceback.print_exc(file=sys.stdout)
                            raise
                        ifile += 1
            case formats.SORT_ON_TAG:
                match self.output_dir:
                    case 'single':
                        # Filenames: Image_00000.png, sort tags fastest
                        archive.set_member_naming_scheme(
                            fallback='Image_{:05d}.png',
                            level=1,
                            default_extension='.png',
                            extensions=self.extensions
                        )
                    case 'multi':
                        # Filenames: slice/tag0/../tagN/Image_00000.png, sort tags fastest
                        digits = len("{}".format(si.slices))
                        dirn = ["slice{{0:0{0}}}".format(digits)]
                        for i, order in enumerate(si.input_order.split(',')[:-1]):
                            digits = len("{}".format(steps[i]))
                            dirn.append(
                                "{0}{{{1}:0{2}}}".format(
                                    order,
                                    i + 1,
                                    digits
                                )
                            )
                        order = si.input_order.split(',')[-1]
                        digits = len("{}".format(steps[-1]))
                        archive.set_member_naming_scheme(
                            fallback=os.path.join(
                                *dirn,
                                order + '{' + '{}'.format(len(dirn)) + ':0{}'.format(digits) + '}.png'),
                            level=max(0, si.ndim - 2),
                            default_extension='.png',
                            extensions=self.extensions
                        )
                ifile = 0
                for _slice in range(si.slices):
                    for tag in np.ndindex(steps):
                        _tag = (_slice,) + tag
                        if si.tags[_slice][tag] is None:
                            continue
                        match self.output_dir:
                            case 'multi':
                                if tag == 0:
                                    ifile = 0  # Restart file number in each subdirectory
                                _file_tag = _tag
                            case 'single':
                                _file_tag = (ifile,)
                        try:
                            _t = si.header.tags[_slice][tag]
                            if _t is None:
                                continue
                            self.write_slice(_file_tag, si[tag + (_slice,)], destination, ifile)
                        except Exception:
                            traceback.print_exc(file=sys.stdout)
                            raise
                        ifile += 1

    def write_3d_pillow(self, si, archive, destination):
        _name: str = '{}.{}'.format(__name__, self.write_3d_pillow.__name__)


        query = None
        if destination['files'] is not None and len(destination['files']):
            query = destination['files'][0]

        if si.ndim < 3:
            self.write_slice(None, si, destination, 0)
        else:
            for s in range(si.slices):
                self.write_slice((s,), si[s], destination, s)


    def write_slice(self, tag, si, destination, ifile):
        _name: str = '{}.{}'.format(__name__, self.write_slice.__name__)

        archive: AbstractArchive = destination['archive']
        query = None
        if destination['files'] and len(destination['files']):
            query = destination['files'][0]
        filename = archive.construct_filename(
            tag=tag,
            query=query
        )

        with archive.open(filename, 'wb') as f:
            logger.debug(f'{_name}: Saving savemat')
            try:
                if si.color:
                    img = Image.fromarray(
                        si.view(dtype=np.uint8).reshape(si.shape + (3,))
                    )
                else:
                    img = Image.fromarray(si)
            except TypeError as e:
                raise ImageTypeError(f'File type does not support present data type:\n  {e}')
            img.save(f)

