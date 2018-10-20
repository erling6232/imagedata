#!/usr/bin/env python3

"""Read/Write Matlab-compatible MAT files
"""

# Copyright (c) 2013-2018 Erling Andersen, Haukeland University Hospital, Bergen, Norway

import os.path
import sys
import logging
import fs
import itk
import numpy as np
import scipy, scipy.io

import imagedata.formats
from imagedata.formats.abstractplugin import AbstractPlugin

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
        super(MATPlugin, self).__init__(self.name, self.description,
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
            tags
        - si[tag,slice,rows,columns]: numpy array
        """

        hdr = {}
        hdr['input_format'] = self.name
        hdr['input_order'] = input_order

        # if len(filelist) > 1: raise ValueError("What to do with multiple input files?")
        #if len(filelist) > 1: raise imagedata.formats.NotImageError('%s does not look like a ITK file.' % filelist[0])
        if len(urls) > 1 and files is not None:
            raise FilesGivenForMultipleURLs("Files shall not be given when there are multiple URLs")
        # Scan filelist to determine data size
        url = urls[0]
        logging.debug("matplugin.read: url: {} {}".format(type(url), url))
        with fs.open_fs(url) as archive:
            if files is None:
                files = archive.walk.files()
            for path in files:
                logging.debug("matplugin.read filehandle {}".format(path))
                if archive.hassyspath(path):
                    filename = archive.getsyspath(path)
                    tmp_fs = None
                else:
                    # Copy file to a local instance
                    tmp_fs = fs.tempfs.TempFS()
                    fs.copy.copy_fs(archive,path, tmp_fs,os.path.basename(path))
                    filename = tmp_fs.getsyspath(os.path.basename(path))
                logging.debug("matplugin.read load filename {}".format(filename))
                try:
                    mdict = scipy.io.loadmat(filename)
                    if len(mdict) > 1:
                        raise MultipleVariablesInMatlabFile('Multiple variables in Matlab file {}: {}'(filename, mdict.keys()))
                    key = mdict.keys()[0]
                    logging.debug('matplugin.read variable {}'.format(key))
                    si = mdict[key]
                    logging.info("Data shape read MAT: {}".format(si.shape))
                    if si.ndim == 2:
                        nt, nz, ny, nx = (1, 1,) + si.shape
                    elif si.ndim == 3:
                        nt, nz, ny, nx = (1,) + si.shape
                    elif si.ndim == 4:
                        nt, nz, ny, nx = si.shape
                    else:
                        raise MatrixDimensionNotImplemented('Matrix dimension {} is not implemented'.format(si.shape))

                except imagedata.formats.NotImageError:
                    raise imagedata.formats.NotImageError('{} does not look like a MAT file'.format(path))
        logging.debug('MATPlugin.read: nt,nz,ny,nx: {} {} {} {}'.format(nt,nz,ny,nx))
        hdr['slices'] = nz

        # Set spacing
        hdr['spacing'] = (1.0, 1.0, 1.0)

        # Set tags
        dt = 1
        times = np.arange(0, nt*dt, dt)
        tags = {}
        for slice in range(nz):
            tags[slice] = np.array(times)
        hdr['tags'] = tags

        logging.info("Data shape read MAT: {}".format(imagedata.formats.shape_to_str(si.shape)))

        # Add any DICOM template
        if pre_hdr is not None:
            hdr.update(pre_hdr)

        return hdr,si

    def write_3d_numpy(self, si, dirname, filename_template, opts):
        """Write 3D numpy image as MAT file

        Input:
        - self: MATPlugin instance
        - si: Series array (3D or 4D), including these attributes:
            slices
            spacing
            tags
        - dirname: directory name
        - filename_template: template including %d for image number
        - opts: Output options (dict)
        """

        self.slices               = si.slices
        self.spacing              = si.spacing
        self.tags                 = si.tags

        logging.info("Data shape write: {}".format(imagedata.formats.shape_to_str(si.shape)))
        if si.ndim == 3:
            si.shape = (1,) + si.shape
        assert si.ndim == 4, "write_3d_series: input dimension %d is not 3D." % (si.ndim-1)
        if si.shape[0] != 1:
            raise ValueError("Attempt to write 4D image ({}) using write_3d_numpy".format(si.shape))
        slices = si.shape[1]
        if slices != si.slices:
            raise ValueError("write_3d_series: slices of dicom template ({}) differ from input array ({}).".format(si.slices, slices))

        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        try:
            filename = filename_template % (0)
        except TypeError:
            filename = filename_template
        filename = os.path.join(dirname, filename)
        scipy.io.savemat(filename, {'A': si[0,...]})

    def write_4d_numpy(self, si, dirname, filename_template, opts):
        """Write 4D numpy image as MAT files

        Input:
        - self: MATPlugin instance
        - si[tag,slice,rows,columns]: Series array, including these attributes:
            slices
            spacing
            tags
        - dirname: directory name
        - filename_template: template including %d for image number
        - opts: Output options (dict)
        """

        self.slices               = si.slices
        self.spacing              = si.spacing
        self.tags                 = si.tags

        # Should we allow to write 3D volume?
        if si.ndim == 3:
            si.shape = (1,) + si.shape
        if si.ndim != 4:
            raise ValueError("write_4d_numpy: input dimension %d is not 4D.".format(si.ndim))

        logging.debug("write_4d_numpy: si dtype {}, shape {}, sort {}".format(
            si.dtype, si.shape,
            imagedata.formats.sort_on_to_str(opts.output_sort)))

        steps  = si.shape[0]
        slices = si.shape[1]
        if steps != len(si.tags[0]):
            raise ValueError("write_4d_series: tags of dicom template ({}) differ from input array ({}).".format(len(si.tags[0]), steps))
        if slices != si.slices:
            raise ValueError("write_4d_series: slices of dicom template ({}) differ from input array ({}).".format(si.slices, slices))

        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        filename = filename_template % (0)
        filename = os.path.join(dirname, filename)
        scipy.io.savemat(filename, {'A': si})

    def copy(self, other=None):
        logging.debug("MATPlugin::copy")
        if other is None: other = MATPlugin()
        return AbstractPlugin.copy(self, other=other)
