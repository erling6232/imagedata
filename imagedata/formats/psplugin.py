"""Read/Write PostScript files
"""

# Copyright (c) 2013-2018 Erling Andersen, Haukeland University Hospital,
# Bergen, Norway

import os.path
import sys
import locale
import logging
import fs
import tempfile
import numpy as np
#import gfx
import ghostscript

import imagedata.formats
#from imagedata.formats.abstractplugin import AbstractPlugin
from imagedata.formats.itkplugin import ITKPlugin

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

class PSPlugin(ITKPlugin):
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

        Input:
        - self: format plugin instance
        - f: file handle or filename (depending on self._need_local_file)
        - opts: Input options (dict)
        - hdr: Header dict
        Output:
        - hdr: Header dict
        Return values:
        - info: Internal data for the plugin
          None if the given file should not be included (e.g. raw file)
        - si: numpy array (multi-dimensional)
        """

        info = None
        #with tempfile.TemporaryDirectory() as tempdir:
        tempdir = tempfile.TemporaryDirectory()
        logging.debug("PSPlugin.read: tempdir {}".format(tempdir))
        try:
            # Convert filename to PNG
            self._convert_to_png(f, tempdir.name, "fname%02d.png")
            #self._pdf_to_png(f, os.path.join(tempdir.name, "fname.png"))
        except imagedata.formats.NotImageError:
            raise imagedata.formats.NotImageError('{} does not look like a PostScript file'.format(f))
        # Call ITKPlugin to read the PNG file(s)
        image_list = list()
        for fname in sorted(os.listdir(tempdir.name)):
            filename = os.path.join(tempdir.name, fname)
            logging.debug("PSPlugin.read: call ITKPlugin {}".format(filename))
            info, img = super(PSPlugin, self)._read_image(filename, opts, hdr)
            image_list.append((info, img))
            logging.debug("PSPlugin.read: returned from ITKPlugin")
        if len(image_list) < 1:
            raise ValueError('No image data read')
        info, img = image_list[0]
        shape = (len(image_list),) + img.shape
        dtype = img.dtype
        si = np.zeros(shape, dtype)
        i = 0
        for info, img in image_list:
            logging.debug('read: img {} si {}'.format(img.shape, si.shape))
            si[i] = img
            i += 1
        # Let a single page be a 2D image
        if si.ndim == 3 and si.shape[0] == 1:
            si.shape = si.shape[1:]
        logging.debug('read: si {}'.format(si.shape))
        return(info, si)

    def _need_local_file(self):
        """Do the plugin need access to local files?

        Return values:
        - True: The plugin need access to local filenames
        - False: The plugin can access files given by an open file handle
        """

        return(True)

    def _set_tags(self, image_list, hdr, si):
        """Set header tags.

        Input:
        - self: format plugin instance
        - image_list: list with (info,img) tuples
        - hdr: Header dict
        - si: numpy array (multi-dimensional)
        Output:
        - hdr: Header dict
        """

        super(PSPlugin, self)._set_tags(image_list, hdr, si)

    def _convert_to_png(self, filename, tempdir, fname):
        """Convert file from PostScript to PNG

        Input:
        - filename: PostScript file
        - tempdir:  Output directory
        - fname:    Output filename
                    Multi-page PostScript files will be converted to fname-N.png
        """

        args = [
            "gs", # actual value doesn't matter
            "-dNOPAUSE", "-dBATCH", "-dSAFER", "-dQUIET",
            "-sDEVICE=pnggray",
            "-sOutputFile=" + os.path.join(tempdir, fname),
            #"-c", ".setpdfwrite",
            "-f",  filename
            ]

        # arguments have to be bytes, encode them
        encoding = locale.getpreferredencoding()
        args = [a.encode(encoding) for a in args]
        logging.debug('_convert_to_png: args {}'.format(args))

        ghostscript.Ghostscript(*args)

    def _pdf_to_png(self, inputPath, outputPath):
        """Convert from pdf to png by using python gfx
        
        The resolution of the output png can be adjusted in the config file 
        under General -> zoom, typical value 150 
        The quality of the output png can be adjusted in the config file under 
        General -> antiAlise, typical value 5
        
        :param inputPath: path to a pdf file
        :param outputPath: path to the location where the output png will be 
            saved
        """
        print("converting pdf {} {}".format(inputPath, outputPath))
        gfx.setparameter("zoom", config.readConfig("zoom")) #Gives the image higher resolution
        doc = gfx.open("pdf", inputPath)
        img = gfx.ImageList()
        
        img.setparameter("antialise", config.readConfig("antiAliasing")) # turn on antialising
        page1 = doc.getPage(1)
        img.startpage(page1.width,page1.height)
        page1.render(img)
        img.endpage()
        img.save(outputPath)

    def write_3d_numpy(self, si, destination, opts):
        """Write 3D numpy image as PostScript file

        Input:
        - self: ITKPlugin instance
        - si: Series array (3D or 4D), including these attributes:
            slices
            spacing
            imagePositions
            transformationMatrix
            orientation
            tags
        - destination: dict of archive and filenames
        - opts: Output options (dict)
        """
        raise imagedata.formats.WriteNotImplemented(
                'Writing PostScript files is not implemented.')

    def write_4d_numpy(self, si, destination, opts):
        """Write 4D numpy image as PostScript files

        Input:
        - self: ITKPlugin instance
        - si[tag,slice,rows,columns]: Series array, including these attributes:
            slices
            spacing
            imagePositions
            transformationMatrix
            orientation
            tags
        - destination: dict of archive and filenames
        - opts: Output options (dict)
        """
        raise imagedata.formats.WriteNotImplemented(
                'Writing PostScript files is not implemented.')
