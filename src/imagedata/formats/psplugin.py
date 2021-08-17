"""Read/Write PostScript files
"""

# Copyright (c) 2013-2019 Erling Andersen, Haukeland University Hospital,
# Bergen, Norway

import os.path
import locale
import logging
import magic
import tempfile
import numpy as np
from ghostscript import _gsprint as gs
from ghostscript import GhostscriptError
import imageio

import imagedata.formats
from imagedata.formats.abstractplugin import AbstractPlugin

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


class PSPlugin(AbstractPlugin):
    """Read PostScript files.
    Writing PostScript files is not implemented."""

    name = "ps"
    description = "Read PostScript files."
    authors = "Erling Andersen"
    version = "1.0.0"
    url = "www.helse-bergen.no"

    def __init__(self):
        super(PSPlugin, self).__init__(self.name, self.description,
                                       self.authors, self.version, self.url)

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

        self.dpi = 150  # dpi
        self.driver = 'png16m'
        self.rotate = 0
        legal_attributes = {'dpi', 'driver', 'rotate'}
        if 'psopt' in opts and opts['psopt']:
            for expr in opts['psopt'].split(','):
                attr,value = expr.split('=')
                if attr in legal_attributes:
                    setattr(self, attr, value)
                else:
                    raise ValueError('Unknown attribute {} set in psopt'.format(attr))
        self.dpi = int(self.dpi)
        self.rotate = int(self.rotate)
        if self.rotate not in {0, 90}:
            raise ValueError('psopt rotate value {} is not implemented'.format(self.rotate))
        with tempfile.TemporaryDirectory() as tempdir:
            logger.debug("PSPlugin.read: tempdir {}".format(tempdir))
            try:
                # Convert filename to PNG
                self._convert_to_png(f, tempdir, "fname%02d.png")
                # self._pdf_to_png(f, os.path.join(tempdir.name, "fname.png"))
            except imagedata.formats.NotImageError:
                raise imagedata.formats.NotImageError('{} does not look like a PostScript file'.format(f))
            # Read the PNG file(s)
            image_list = list()
            for fname in sorted(os.listdir(tempdir)):
                filename = os.path.join(tempdir, fname)
                logger.debug("PSPlugin.read: call ITKPlugin {}".format(filename))
                img = imageio.imread(filename)
                image_list.append(img)
        if len(image_list) < 1:
            raise ValueError('No image data read')
        img = image_list[0]
        shape = (len(image_list),) + img.shape
        dtype = img.dtype
        si = np.zeros(shape, dtype)
        for i, img in enumerate(image_list):
            logger.debug('read: img {} si {}'.format(img.shape, si.shape))
            si[i] = img
        hdr['spacing'] = np.array([1,1,1])
        # Color space: RGB
        hdr['photometricInterpretation'] = 'MONOCHROME2'
        hdr['color'] = False
        if self.driver == 'png16m' and si.shape[-1] == 3:
            # Photometric interpretation = 'RGB'
            hdr['photometricInterpretation'] = 'RGB'
            hdr['color'] = True
        if self.rotate == 90:
            si = np.rot90(si, axes=(1,2))
        # Let a single page be a 2D image
        if si.ndim == 3 and si.shape[0] == 1:
            si.shape = si.shape[1:]
        logger.debug('read: si {}'.format(si.shape))
        return True, si

    def _need_local_file(self):
        """Do the plugin need access to local files?

        Returns:
            Boolean
                - True: The plugin need access to local filenames
                - False: The plugin can access files given by an open file handle
        """

        return True

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

        #super(PSPlugin, self)._set_tags(image_list, hdr, si)

        # Default spacing and orientation
        hdr['spacing'] = np.array([1,1,1])
        hdr['imagePositions'] = {}
        hdr['imagePositions'][0] = np.array([0,0,0])
        hdr['orientation'] = np.array([0,1,0,-1,0,0])

        # Set tags
        axes = list()
        _actual_shape = si.shape
        _color = False
        if 'color' in hdr and hdr['color']:
            _actual_shape = si.shape[:-1]
            _color = True
        _actual_ndim = len(_actual_shape)
        nz = 1
        axes.append(imagedata.axis.UniformLengthAxis(
            'row',
            hdr['imagePositions'][0][1],
            _actual_shape[-2],
            hdr['spacing'][1])
        )
        axes.append(imagedata.axis.UniformLengthAxis(
            'column',
            hdr['imagePositions'][0][2],
            _actual_shape[-1],
            hdr['spacing'][2])
        )
        if _actual_ndim > 2:
            nz = _actual_shape[-3]
            axes.insert(0, imagedata.axis.UniformLengthAxis(
                'slice',
                hdr['imagePositions'][0][0],
                nz,
                hdr['spacing'][0])
            )
        if _color:
            axes.append(imagedata.axis.VariableAxis(
                'rgb',
                ['r', 'g', 'b'])
            )
        hdr['axes'] = axes

        tags = {}
        for slice in range(nz):
            tags[slice] = np.array([0])
        hdr['tags'] = tags
        return

    def _convert_to_png(self, filename, tempdir, fname):
        """Convert file from PostScript to PNG

        Args:
            filename: PostScript file
            tempdir:  Output directory
            fname:    Output filename
                    Multi-page PostScript files will be converted to fname-N.png
        """

        # Verify that the input file is a PostScript file
        if magic.from_file(filename, mime=True) != 'application/postscript' and \
           magic.from_file(filename, mime=True) != 'application/pdf':
            raise imagedata.formats.NotImageError('{} does not look like a PostScript or PDF file'.format(filename))

        args = [
            "gs",  # actual value doesn't matter
            "-dNOPAUSE", "-dBATCH", "-dSAFER", "-dQUIET",
            # "-sDEVICE=pnggray",
            # "-sDEVICE=png16m",
            "-r{}".format(self.dpi),
            "-sDEVICE={}".format(self.driver),
            "-sOutputFile=" + os.path.join(tempdir, fname),
            # "-c", ".setpdfwrite",
            "-f", filename
        ]

        # arguments have to be bytes, encode them
        encoding = locale.getpreferredencoding()
        args = [a.encode(encoding) for a in args]
        logger.debug('_convert_to_png: args {}'.format(args))

        try:
            instance = gs.new_instance()
            code = gs.init_with_args(instance, args)
            code1 = gs.exit(instance)
            if code == 0 or code == gs.e_Quit:
                code = code1
            gs.delete_instance(instance)
            if not (code == 0 or code == gs.e_Quit):
                raise DependencyError("Cannot run Ghostscript: {}".format(code))
        except GhostscriptError as e:
            logger.error('_convert_to_png: error: {}'.format(e))
            raise DependencyError("Cannot run Ghostscript: {}".format(e))

    # @staticmethod
    # def _pdf_to_png(inputPath, outputPath):
    #     """Convert from pdf to png by using python gfx
    #
    #     The resolution of the output png can be adjusted in the config file
    #     under General -> zoom, typical value 150
    #     The quality of the output png can be adjusted in the config file under
    #     General -> antiAlise, typical value 5
    #
    #     :param inputPath: path to a pdf file
    #     :param outputPath: path to the location where the output png will be
    #         saved
    #     """
    #     print("converting pdf {} {}".format(inputPath, outputPath))
    #     gfx.setparameter("zoom", config.readConfig("zoom"))  # Gives the image higher resolution
    #     doc = gfx.open("pdf", inputPath)
    #     img = gfx.ImageList()
    #
    #     img.setparameter("antialise", config.readConfig("antiAliasing"))  # turn on antialising
    #     page1 = doc.getPage(1)
    #     img.startpage(page1.width, page1.height)
    #     page1.render(img)
    #     img.endpage()
    #     img.save(outputPath)

    def write_3d_numpy(self, si, destination, opts):
        """Write 3D numpy image as PostScript file

        Args:
            self: ITKPlugin instance
            si: Series array (3D or 4D), including these attributes:
            -   slices,
            -   spacing,
            -   imagePositions,
            -   transformationMatrix,
            -   orientation,
            -   tags

            destination: dict of archive and filenames
            opts: Output options (dict)
        Raises:
            imagedata.formats.WriteNotImplemented: Always, writing is not implemented.
        """
        raise imagedata.formats.WriteNotImplemented(
            'Writing PostScript files is not implemented.')

    def write_4d_numpy(self, si, destination, opts):
        """Write 4D numpy image as PostScript files

        Args:
            self: ITKPlugin instance
            si[tag,slice,rows,columns]: Series array, including these attributes:
            -   slices,
            -   spacing,
            -   imagePositions,
            -   transformationMatrix,
            -   orientation,
            -   tags

            destination: dict of archive and filenames
            opts: Output options (dict)
        Raises:
            imagedata.formats.WriteNotImplemented: Always, writing is not implemented.
        """
        raise imagedata.formats.WriteNotImplemented(
            'Writing PostScript files is not implemented.')
