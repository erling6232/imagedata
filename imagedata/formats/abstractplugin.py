"""Abstract class for image formats.

Defines generic functions.
"""

# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
# Copyright (c) 2017-2018 Erling Andersen, Haukeland University Hospital, Bergen, Norway

from abc import ABCMeta, abstractmethod, abstractproperty
import copy
import logging
from datetime import date, datetime, time, timedelta
import math
import numpy as np
import pydicom.dataset
import imagedata.formats

class NoOtherInstance(Exception): pass

class AbstractPlugin(object, metaclass=ABCMeta):
    """Abstract base class definition for imagedata format plugins.
    Plugins must be a subclass of AbstractPlugin and
    must define the atttributes set in __init__() and
    the following methods:

    read() method
    write_3d_numpy() method
    write_4d_numpy() method
    """

    def __init__(self, name, description, authors, version, url):
        object.__init__(self)
        self.__name              = name
        self.__description       = description
        self.__authors           = authors
        self.__version           = version
        self.__url               = url

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

        Input:
        - sources: list of sources to image data
        - pre_hdr: DICOM template
        - input_order: sort order
        - opts: Input options (dict)
        Output:
        - hdr: Header dict
        - si[tag,slice,rows,columns]: numpy array
        """

        hdr = {}
        hdr['input_format'] = self.name
        hdr['input_order'] = input_order

        # image_list: list of tuples (hdr,si)
        logging.debug("AbstractPlugin.read: sources {}".format(sources))
        image_list = list()
        for source in sources:
            logging.debug("AbstractPlugin.read: source: {} {}".format(type(source), source))
            archive = source['archive']
            scan_files = source['files']
            if scan_files is None or len(scan_files) == 0:
                scan_files = archive.getnames()
            logging.debug("AbstractPlugin.read: scan_files {}".format(scan_files))
            for file_handle in archive.getmembers(scan_files):
                logging.debug("AbstractPlugin.read: file_handle {}".format(file_handle))
                if self._need_local_file():
                    logging.debug("AbstractPlugin.read: need local file {}".format(file_handle))
                    f = archive.to_localfile(file_handle)
                else:
                    f = archive.open(file_handle, mode='rb')
                logging.debug("AbstractPlugin.read: file {}".format(f))
                info, si = self._read_image(f, opts, hdr)
                # info is None when no image was read
                if info is not None:
                    image_list.append((info, si))
        if len(image_list) < 1:
            raise ValueError('No image data read')
        info, si = image_list[0]
        self._reduce_shape(si)
        logging.debug('AbstractPlugin.read: reduced si {}'.format(si.shape))
        shape = (len(image_list),) + si.shape
        dtype = si.dtype
        logging.debug('AbstractPlugin.read: shape {}'.format(shape))
        si = np.zeros(shape, dtype)
        i = 0
        for info, img in image_list:
            logging.debug('AbstractPlugin.read: img {} si {} {}'.format(img.shape, si.shape, si.dtype))
            si[i] = img
            i += 1
        logging.debug('AbstractPlugin.read: si {}'.format(si.shape))

        # Simplify shape
        self._reduce_shape(si)
        logging.debug('AbstractPlugin.read: reduced si {}'.format(si.shape))

        _shape = si.shape
        if 'color' in hdr and hdr['color']:
            _shape = si.shape[:-1]
            logging.debug('AbstractPlugin.read: color')
        logging.debug('AbstractPlugin.read: _shape {}'.format(_shape))
        _ndim = len(_shape)
        nt = nz = 1
        ny, nx = _shape[-2:]
        if _ndim > 2:
            nz = _shape[-3]
        if _ndim > 3:
            nt = _shape[-4]
        hdr['slices'] = nz
        logging.debug('AbstractPlugin.read: slices {}'.format(nz))

        # hdr['spacing'], hdr['tags']
        logging.debug('AbstractPlugin.read: calling _set_tags')
        self._set_tags(image_list, hdr, si)
        #logging.debug('AbstractPlugin.read: return  _set_tags: {}'.format(hdr))

        logging.info("Data shape read: {}".format(imagedata.formats.shape_to_str(si.shape)))

        # Add any DICOM template
        if pre_hdr is not None:
            hdr.update(pre_hdr)

        logging.debug('AbstractPlugin.read: hdr {}'.format(
            hdr.keys()))
        #logging.debug('AbstractPlugin.read: hdr {}'.format(hdr))
        return hdr,si

    def _need_local_file(self):
        """Do the plugin need access to local files?

        Return values:
        - True: The plugin need access to local filenames
        - False: The plugin can access files given by an open file handle
        """

        return(False)

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def write_3d_numpy(self, si, destination, opts):
        #def write_3d_numpy(self, si, destination, filename_template, opts):
        """Write 3D Series image

        Input:
        - self: format plugin instance
        - si[slice,rows,columns]: Series array
        - destination: dict of archive and filenames
        #- filename_template: template including %d for image number
        - opts: Output options (dict)
        """
        pass

    @abstractmethod
    def write_4d_numpy(self, si, destination, opts):
        #def write_4d_numpy(self, si, destination, filename_template, opts):
        """Write 4D Series image

        Input:
        - self: format plugin instance
        - si[tag,slice,rows,columns]: Series array
        - destination: dict of archive and filenames
        #- filename_template: template including %d for image number
        - opts: Output options (dict)
        """
        pass

    def getTimeline(self):
        """Get timeline
        
        Returns:
        - timeline in seconds, as numpy array of floats
            Delta time is given as seconds. First image is t=0.
            Length of array is number of tags.
        Exceptions:
        - ValueError: tags for dataset is not time tags
        """
        if self.input_order == imagedata.formats.INPUT_ORDER_TIME:
            timeline = []
            timeline.append(0.0)
            for t in range(1, len(self.tags[0])):
                timeline.append(self.tags[0][t] - self.tags[0][0])
            return np.array(timeline)
        else:
            raise ValueError("No timeline tags are available. Input order: {}".format(self.input_order))

    '''
    def getQform(self):
        """Get Nifti version of the transformation matrix, aka qform

        Input:
        - self.spacing
        - self.imagePositions
        - self.orientation
        Returns:
        - transformation matrix as numpy array
        """

        def normalize_column(x,row):
            val = np.vdot(x, x)

            if val > 0:
                x = x / math.sqrt(val)
            else:
                shape = x.shape
                x = np.array([0., 0., 0.])
                x[row]=1
                x.shape = shape
            return x

        #debug = None
        debug = True

        ds,dr,dc    = self.spacing
        z,y,x       = self.imagePositions[0]
        slices      = len(self.imagePositions)
        T0          = self.imagePositions[0]
        Tn          = self.imagePositions[slices-1]
        orient      = self.orientation
        #print("ds,dr,dc={},{},{}".format(ds,dr,dc))
        #print("z ,y ,x ={},{},{}".format(z,y,x))

        Q = np.eye(4)
        # Set column 3 and row 3 to zeros, except [3,3]
        colr=np.array([[orient[3]], [orient[4]], [orient[5]]])
        colc=np.array([[orient[0]], [orient[1]], [orient[2]]])
        colr = normalize_column(colr,0)
        colc = normalize_column(colc,1)
        k=np.cross(colr, colc, axis=0)

        Q[:3, :3] = np.hstack((colr, colc, k))
        if debug:
            logging.debug("Q")
            logging.debug( Q)

        if debug:
            logging.debug("determinant(Q) {}".format(np.linalg.det(Q)))
        if np.linalg.det(Q) < 0:
            Q[:3,2] = -Q[:3,2]

        # Scale matrix
        diagVox = np.eye(3)
        diagVox[0,0] = dc
        diagVox[1,1] = dr
        diagVox[2,2] = ds
        if debug:
            logging.debug("diagVox")
            logging.debug( diagVox)
            logging.debug("Q without scaling {}".format(Q.dtype))
            logging.debug( Q)
        Q[:3,:3] = np.dot(Q[:3,:3],diagVox)
        if debug:
            logging.debug("Q with scaling {}".format(Q.dtype))
            logging.debug( Q)

        # Add translations
        Q[0,3] = x; Q[1,3] = y; Q[2,3] = z       # pos x,y,z
        if debug:
            logging.debug("Q with translations")
            logging.debug( Q)
        # Q now equals dicom_to_patient in spm_dicom_convert

        return Q
    '''

    '''
    def setQform(self, A):
        """Set transformationMatrix from Nifti affine A."""
        #print("setQform:  input\n{}".format(A))
        M=np.eye(4)
        M[:3,0]=A[2::-1,2]
        M[:3,1]=A[2::-1,0]
        M[:3,2]=A[2::-1,1]
        M[:3,3]=A[2::-1,3]
        #print("setQform: output\n{}".format(M))
        self.transformationMatrix=M
        return
    '''

    def getPositionForVoxel(self, r, transformation=None):
        """Get patient position for center of given voxel r

        Use Patient Position and Image Orientation to calculate world
        coordinates for given voxel
        Input:
        - r: (z,y,x) of voxel in voxel coordinates as numpy.array
        - transformation: transformation matrix when different from
                          self.transformationMatrix
        Output:
        - (z,y,x) of voxel in world coordinates (mm) as numpy.array
        """

        if transformation is None:
            transformation = self.transformationMatrix
        #Q = self.getTransformationMatrix()

        #V = np.array([[r[2]], [r[1]], [r[0]], [1]])  # V is [x,y,z,1]

        newpos = np.dot(transformation, np.hstack((r, [1])))

        #return np.array([newpos[2,0],newpos[1,0],newpos[0,0]])   # z,y,x
        return newpos[:3]

    def getVoxelForPosition(self, p, transformation=None):
        """ Get voxel for given patient position p

        Use Patient Position and Image Orientation to calculate world
        coordinates for given voxel
        Input:
        - p: (z,y,x) of voxel in world coordinates (mm) as numpy.array
        - transformation: transformation matrix when different from
                          self.transformationMatrix
        Output:
        - (z,y,x) of voxel in voxel coordinates as numpy.array
        """

        if transformation is None:
            transformation = self.transformationMatrix
        #Q = self.getTransformationMatrix()

        #V = np.array([[p[2]], [p[1]], [p[0]], [1]])    # V is [x,y,z,1]

        Qinv = np.linalg.inv(transformation)
        r = np.dot(Qinv, np.hstack((p, [1])))

        # z,y,x
        #return np.array([int(r[2,0]+0.5),int(r[1,0]+0.5),int(r[0,0]+0.5)], dtype=int)
        #return int(r+0.5)[:3]
        return (r+0.5).astype(int)[:3]

    def replace_geometry_attributes(im, gim):
        """Replace geometry attributes in im with values from gim
        """

        im.SliceLocation              = gim.SliceLocation
        im.ImagePositionPatient       = gim.ImagePositionPatient
        im.ImageOrientationPatient    = gim.ImageOrientationPatient
        im.FrameOfReferenceUID        = gim.FrameOfReferenceUID
        im.PositionReferenceIndicator = gim.PositionReferenceIndicator
        im.SliceThickness             = gim.SliceThickness
        try:
            im.SpacingBetweenSlices = gim.SpacingBetweenSlices
        except:
            pass
        im.AcquisitionMatrix          = gim.AcquisitionMatrix
        im.PixelSpacing               = gim.PixelSpacing

    def _reduce_shape(self, si):
        """Reduce shape when leading shape(s) are 1.
        Will not reduce to less than 2-dimensional image.
        
        Input:
        - self: format plugin instance
        - si[...]: Series array
        Returns:
        Exceptions:
        - ValueError: tags for dataset is not time tags
        """

        # Color image?
        mindim = 2
        if si.shape[-1] == 3:
            mindim += 1

        while si.ndim > mindim:
            if si.shape[0] == 1:
                si.shape = si.shape[1:]
            else:
                break

    def _reorder_to_dicom(self, data, flip=False):
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
        """

        logging.debug('AbstractPlugin._reorder_to_dicom: shape in {}'.format(
            data.shape))
        if data.ndim == 5:
            rows,columns,slices,tags,d5 = data.shape
            if flip:
                rows, columns = columns, rows
            si = np.zeros((d5,tags,slices,rows,columns), data.dtype)
            for d in range(d5):
                for tag in range(tags):
                    for slice in range(slices):
                        if flip:
                            si[d,tag,slice,:,:] = \
                            np.fliplr(data[:,:,slice,tag,d]).T
                        else:
                            si[d,tag,slice,:,:] = data[:,:,slice,tag,d]
        elif data.ndim == 4:
            rows,columns,slices,tags = data.shape
            if flip:
                rows, columns = columns, rows
            si = np.zeros((tags,slices,rows,columns), data.dtype)
            for tag in range(tags):
                for slice in range(slices):
                    if flip:
                        si[tag,slice,:,:] = np.fliplr(data[:,:,slice,tag]).T
                    else:
                        si[tag,slice,:,:] = data[:,:,slice,tag]
        elif data.ndim == 3:
            rows,columns,slices = data.shape
            if flip:
                rows, columns = columns, rows
            si = np.zeros((slices,rows,columns), data.dtype)
            for slice in range(slices):
                if flip:
                    si[slice,:,:] = np.fliplr(data[:,:,slice]).T
                else:
                    si[slice,:,:] = data[:,:,slice]
        elif data.ndim == 2:
            rows,columns = data.shape
            if flip:
                rows, columns = columns, rows
            si = np.zeros((rows,columns), data.dtype)
            if flip:
                si[:] = np.fliplr(data[:]).T
            else:
                si[:] = data[:]
        else:
            raise ValueError('Dimension %d is not implemented' % data.ndim)
        logging.debug('AbstractPlugin._reorder_to_dicom: shape out {}'.format(
            si.shape))
        return(si)

    def _reorder_from_dicom(self, data, flip=False):
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
        """

        logging.debug('AbstractPlugin._reorder_from_dicom: shape in {}'.format(data.shape))
        if data.ndim == 5:
            d5,tags,slices,rows,columns = data.shape
            if flip:
                rows, columns = columns, rows
            si = np.zeros((rows,columns,slices,tags,d5), data.dtype)
            for d in range(d5):
                for tag in range(tags):
                    for slice in range(slices):
                        if flip:
                            si[:,:,slice,tag,d] = \
                            np.fliplr(data[d,tag,slice,:,:]).T
                        else:
                            si[:,:,slice,tag,d] = data[d,tag,slice,:,:]
        elif data.ndim == 4:
            tags,slices,rows,columns = data.shape
            if flip:
                rows, columns = columns, rows
            si = np.zeros((rows,columns,slices,tags), data.dtype)
            for tag in range(tags):
                for slice in range(slices):
                    if flip:
                        si[:,:,slice,tag] = np.fliplr(data[tag,slice,:,:]).T
                    else:
                        si[:,:,slice,tag] = data[tag,slice,:,:]
        elif data.ndim == 3:
            slices,rows,columns = data.shape
            if flip:
                rows, columns = columns, rows
            si = np.zeros((rows,columns,slices), data.dtype)
            for slice in range(slices):
                if flip:
                    si[:,:,slice] = np.fliplr(data[slice,:,:]).T
                else:
                    si[:,:,slice] = data[slice,:,:]
        elif data.ndim == 2:
            rows,columns = data.shape
            if flip:
                rows, columns = columns, rows
            si = np.zeros((rows,columns), data.dtype)
            if flip:
                si[:] = np.fliplr(data[:]).T
            else:
                si[:] = data[:]
        else:
            raise ValueError('Dimension %d is not implemented' % data.ndim)
        logging.debug('AbstractPlugin._reorder_from_dicom: shape out {}'.format(
            si.shape))
        return(si)

    def copy(self, other=None):
        """Make a copy of the instance

        Returns:
        - new instance of imagedata format plugin
        """

        if other is None:
            raise NoOtherInstance("No other instance to copy to.")

        other.__input_order       = self.__input_order
        #for attr in self.__dict__:
        #    logging.debug("AbstractPlugin::copy {}".format(attr))
        if self.__sort_on is not None:
            other.__sort_on          = self.__sort_on
        if self.__sliceLocations is not None:
            other.__sliceLocations   = self.__sliceLocations.copy()
        if self.__DicomHeaderDict is not None:
            other.__DicomHeaderDict  = self.__DicomHeaderDict.copy()
            for slice in self.__DicomHeaderDict:
                other.__DicomHeaderDict[slice] = []
                for tg,fname,im in self.__DicomHeaderDict[slice]:
                    # Create new dataset by making a deep copy of im
                    info = pydicom.dataset.Dataset()
                    for key in im.keys():
                        if key != (0x7fe0, 0x0010):
                            el = im[key]
                            info.add_new(el.tag, el.VR, el.value)
                    other.__DicomHeaderDict[slice].append((tg,fname,info))
        if self.__tags is not None:
            other.__tags = {}
            for slice in self.__tags.keys():
                other.__tags[slice] = self.__tags[slice].copy()
        if self.__spacing is not None:
            other.__spacing          = self.__spacing.copy()
        if self.__imagePositions is not None:
            other.__imagePositions = {}
            for slice in self.__imagePositions.keys():
                other.__imagePositions[slice] = self.__imagePositions[slice].copy()
        if self.__orientation is not None:
            other.__orientation      = self.__orientation.copy()
        if self.__seriesNumber is not None:
            other.__seriesNumber     = self.__seriesNumber
        if self.__seriesDescription is not None:
            other.__seriesDescription= self.__seriesDescription
        if self.__imageType is not None:
            other.__imageType        = self.__imageType
        return other

