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

    @abstractmethod
    def read(self, urls, files, pre_hdr, input_order, opts):
        """Read image data

        Input:
        - urls: list of urls to image data
        - files: list of files inside a single url.
            = None: No files given
        - pre_hdr: DICOM template
        - input_order: sort order
        - opts: Input options (dict)
        Output:
        - hdr: Header dict
        - si[tag,slice,rows,columns]: numpy array
        """
        pass

    @abstractmethod
    def write_3d_numpy(self, si, dirname, filename_template, opts):
        """Write 3D Series image

        Input:
        - self: format plugin instance
        - si[slice,rows,columns]: Series array
        - dirname: directory name
        - filename_template: template including %d for image number
        - opts: Output options (dict)
        """
        pass

    @abstractmethod
    def write_4d_numpy(self, si, dirname, filename_template, opts):
        """Write 4D Series image

        Input:
        - self: format plugin instance
        - si[tag,slice,rows,columns]: Series array
        - dirname: directory name
        - filename_template: template including %d for image number
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

