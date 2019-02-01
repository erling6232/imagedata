#!/usr/bin/env python3

"""Read/Write PostScript files
"""

# Copyright (c) 2013-2018 Erling Andersen, Haukeland University Hospital,
# Bergen, Norway

import os.path
import sys
import logging
import fs
import tempfile

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
    """Read/write PostScript files."""

    name = "ps"
    description = "Read and write PostScript files."
    authors = "Erling Andersen"
    version = "1.0.0"
    url = "www.helse-bergen.no"

    def __init__(self):
        super(PSPlugin, self).__init__(self.name, self.description,
            self.authors, self.version, self.url)

    def read(self, urls, files, pre_hdr, input_order, opts):
        """Read image data

        Input:
                    - urls: list of urls to image data
        - files: list of files inside a single url.
            = None: No files given
        - pre_hdr: Pre-filled header dict. Might be None
        - input_order
        - opts: Input options (dict)
        Output:
        - hdr: Header dict
            input_format
            input_order
            slices
            spacing
            imagePositions
            transformationMatrix
            orientation
            tags
        - si[tag,slice,rows,columns]: numpy array
        """

        hdr = {}
        hdr['input_format'] = self.name
        hdr['input_order'] = input_order

        if len(urls) > 1 and files is not None:
            raise FilesGivenForMultipleURLs("Files shall not be given when there are multiple URLs")

        nfiles = 0
        with tempfile.TemporaryDirectory() as tempdir:
            logging.debug("PSPlugin.read: Created tempdir {}".format(tempdir))
            # Scan filelist to determine data size
            for url in urls:
                logging.debug("PSPlugin.read: peek url: {} {}".format(type(url), url))
                with fs.open_fs(url) as archive:
                    scan_files = files
                    if scan_files is None:
                        scan_files = archive.walk.files()
                    for path in sorted(scan_files):
                        logging.debug("PSPlugin.read peek filehandle {}".format(path))
                        if archive.hassyspath(path):
                            filename = archive.getsyspath(path)
                            tmp_fs = None
                        else:
                            # Copy file to a local instance
                            tmp_fs = fs.tempfs.TempFS()
                            fs.copy.copy_fs(archive,path, tmp_fs,os.path.basename(path))
                            filename = tmp_fs.getsyspath(os.path.basename(path))
                        logging.debug("PSPlugin.read peek filename {}".format(filename))
                        try:
                            # Convert filename to PNG
                            self.convert_to_png(filename, tempdir, "%05d" % nfiles)
                            nfiles += 1
                        except imagedata.formats.NotImageError:
                            raise imagedata.formats.NotImageError('{} does not look like a PostScript file'.format(path))

            # Call ITKPlugin to read the PNG file(s)
            logging.debug("PSPlugin.read: call ITKPlugin")
            hdr,si = super(PSPlugin, self).read(tempdir, None, pre_hdr, input_order, opts)
            logging.debug("PSPlugin.read: returned from ITKPlugin")
        return hdr,si

    def convert_to_png(self, filename, tempdir, fname):
        """Convert file from PostScript to PNG

        Input:
        - filename: PostScript file
        - tempdir:  Output directory
        - fname:    Output filename
                    Multi-page PostScript files will be converted to fname-N.png
        """
        pass
