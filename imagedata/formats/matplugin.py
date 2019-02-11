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
        super(MatPlugin, self).__init__(self.name, self.description,
            self.authors, self.version, self.url)

    def read(self, sources, pre_hdr, input_order, opts):
        """Read image data

        Input:
        - sources: list of sources to image data
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
        #if len(filelist) > 1: raise imagedata.formats.NotImageError('%s does not look like a MAT file.' % filelist[0])
        #if len(urls) > 1 and files is not None:
        #    raise FilesGivenForMultipleURLs("Files shall not be given when there are multiple URLs")
        # Scan filelist to determine data size
        image_list = list()
        for source in sources:
            logging.debug("matplugin.read: source: {} {}".format(type(source), source))
            archive = source['archive']
            scan_files = source['files']
            if scan_files is None or len(scan_files) == 0:
                scan_files = archive.getnames()
            for path in archive.getmembers(scan_files):
                logging.debug("matplugin.read filehandle {}".format(path))
                with archive.getmember(path, mode='rb') as f:
                    image_list.append(
                        self._read_image(f, opts, hdr)
                    )
        if len(image_list) < 1:
            raise ValueError('No image data read')
        shape = (len(image_list),) + image_list[0].shape
        dtype = image_list[0].dtype
        logging.debug('matplugin.read: shape {}'.format(shape))
        si = np.zeros(shape, dtype)
        i = 0
        for img in image_list:
            logging.debug('matplugin.read: img {} si {}'.format(img.shape,
                si.shape))
            si[i] = img
            i += 1

        # Simplify shape
        self._reduce_shape(si)

        nt = nz = 1
        ny, nx = si.shape[-2:]
        if si.ndim > 2:
            nz = si.shape[-3]
        if si.ndim > 3:
            nt = si.shape[-4]

        logging.debug('MatPlugin.read: nt,nz,ny,nx: {} {} {} {}'.format(nt,nz,ny,nx))
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

        logging.info("Data shape read Mat: {}".format(imagedata.formats.shape_to_str(si.shape)))

        # Add any DICOM template
        if pre_hdr is not None:
            hdr.update(pre_hdr)

        return hdr,si

    def _read_image(self, f, opts, hdr):
        try:
            logging.debug('matplugin._read_image: scipy.io.loadmat({})'.format(f))
            mdictlist = scipy.io.whosmat(f)
            if len(mdictlist) != 1:
                names=[]
                for name,shape,dtype in mdictlist: names.append(name)
                logging.debug('matplugin._read_image: scipy.io.loadmat len(mdict) {}'.format(len(mdictlist)))
                logging.debug('matplugin._read_image: Multiple variables in MAT file {}: {}'.format(f, names))
                raise MultipleVariablesInMatlabFile('Multiple variables in MAT file {}: {}'.format(f, names))
            name,shape,dtype = mdictlist[0]
            logging.debug('matplugin._read_image: name {} shape {} dtype {}'.format(name,shape,dtype))
            mdict = scipy.io.loadmat(f, variable_names=(name,))
            logging.debug('matplugin._read_image variable {}'.format(name))
            newshape = tuple(reversed(shape))
            logging.debug('matplugin._read_image newshape {}'.format(newshape))
            #si = mdict[name].reshape(newshape, order='F')
            #si = self._mat_to_dicom(mdict[name])
            si = self._reorder_data(mdict[name])
            logging.info("Data shape _read_image MAT: {} {}".format(si.shape, si.dtype))
            if si.ndim == 2:
                nt, nz, ny, nx = (1, 1,) + si.shape
            elif si.ndim == 3:
                #si = self.reorder_3d_data(si)
                nt, nz, ny, nx = (1,) + si.shape
            elif si.ndim == 4:
                #si = self.reorder_4d_data(si)
                nt, nz, ny, nx = si.shape
            else:
                raise MatrixDimensionNotImplemented('Matrix dimension {} is not implemented'.format(si.shape))

        except imagedata.formats.NotImageError:
            raise imagedata.formats.NotImageError('{} does not look like a MAT file'.format(path))
        return(si)

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
        save_shape = si.shape
        if si.ndim == 4 and si.shape[0] == 1: si.shape = si.shape[1:]
        assert si.ndim == 3, "write_3d_series: input dimension %d is not 3D." % (si.ndim)
        slices = si.shape[0]
        if slices != si.slices:
            raise ValueError("write_3d_series: slices of dicom template ({}) differ from input array ({}).".format(si.slices, slices))

        newshape = tuple(reversed(si.shape))
        logging.info("Data shape matlab write: {}".format(imagedata.formats.shape_to_str(newshape)))
        logging.debug('matplugin.write_3d_numpy newshape {}'.format(newshape))
        #img = si.reshape(newshape, order='C')
        #img = si.reshape(newshape, order='F')
        #img = np.asfortranarray(si)
        img = self._dicom_to_mat(si)
        logging.debug('matplugin.write_3d_numpy si  {} {}'.format(si.shape, si.dtype))
        logging.debug('matplugin.write_3d_numpy img {} {}'.format(img.shape, img.dtype))

        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        try:
            filename = filename_template % (0)
        except TypeError:
            filename = filename_template
        filename = os.path.join(dirname, filename)
        scipy.io.savemat(filename, {'A': img})
        si.shape = save_shape

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

        save_shape = si.shape
        # Should we allow to write 3D volume?
        if si.ndim == 3:
            si.shape = (1,) + si.shape
        if si.ndim != 4:
            raise ValueError("write_4d_numpy: input dimension %d is not 4D.".format(si.ndim))

        logging.debug("write_4d_numpy: si dtype {}, shape {}, sort {}".format(
            si.dtype, si.shape,
            imagedata.formats.sort_on_to_str(opts['output_sort'])))

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
        scipy.io.savemat(filename, {'A': self._dicom_to_mat(si)})
        si.shape = save_shape

    def copy(self, other=None):
        logging.debug("MatPlugin::copy")
        if other is None: other = MatPlugin()
        return AbstractPlugin.copy(self, other=other)
