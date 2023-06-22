"""Abstract class for image formats.

Defines generic functions.
"""

# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
# Copyright (c) 2017-2022 Erling Andersen, Haukeland University Hospital, Bergen, Norway

from abc import ABCMeta, abstractmethod  # , abstractproperty
import logging
import numpy as np
from . import NotImageError, shape_to_str, INPUT_ORDER_TIME
from ..header import Header

logger = logging.getLogger(__name__)


class NoOtherInstance(Exception):
    pass


class AbstractPlugin(object, metaclass=ABCMeta):
    """Abstract base class definition for imagedata format plugins.
    Plugins must be a subclass of AbstractPlugin and
    must define the atttributes set in __init__() and
    the following methods:

    read() method
    write_3d_numpy() method
    write_4d_numpy() method

    Attributes:
        input_order
        tags
        transformationMatrix
    """

    plugin_type = 'format'

    def __init__(self, name, description, authors, version, url):
        object.__init__(self)
        self.__name = name
        self.__description = description
        self.__authors = authors
        self.__version = version
        self.__url = url
        self.input_order = None
        self.tags = None
        self.transformationMatrix = None

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

        Args:
            sources: list of sources to image data
            pre_hdr: DICOM template
            input_order: sort order
            opts: Input options (dict)
        Returns:
            Tuple of
                - hdr: Header dict
                - si[tag,slice,rows,columns]: numpy array
        """

        hdr = Header()
        hdr.input_format = self.name
        hdr.input_order = input_order

        # image_list: list of tuples (hdr,si)
        logger.debug("AbstractPlugin.read: sources {}".format(sources))
        image_list = list()
        for source in sources:
            logger.debug("AbstractPlugin.read: source: {} {}".format(type(source), source))
            archive = source['archive']
            scan_files = source['files']
            # if scan_files is None or len(scan_files) == 0:
            #     scan_files = archive.getnames()
            # logger.debug("AbstractPlugin.read: scan_files {}".format(scan_files))
            for file_handle in archive.getmembers(scan_files):
                logger.debug("AbstractPlugin.read: file_handle {}".format(file_handle.filename))
                if self._need_local_file():
                    logger.debug("AbstractPlugin.read: need local file {}".format(
                        file_handle.filename))
                    f = archive.to_localfile(file_handle)
                    logger.debug("AbstractPlugin.read: local file {}".format(f))
                    info, si = self._read_image(f, opts, hdr)
                else:
                    f = archive.open(file_handle, mode='rb')
                    logger.debug("AbstractPlugin.read: file {}".format(f))
                    try:
                        info, si = self._read_image(f, opts, hdr)
                    except NotImageError:
                        raise
                    finally:
                        f.close()
                # info is None when no image was read
                if info is not None:
                    image_list.append((info, si))
        if len(image_list) < 1:
            raise ValueError('No image data read')
        info, si = image_list[0]
        self._reduce_shape(si)
        logger.debug('AbstractPlugin.read: reduced si {}'.format(si.shape))
        shape = (len(image_list),) + si.shape
        dtype = si.dtype
        logger.debug('AbstractPlugin.read: shape {}'.format(shape))
        si = np.zeros(shape, dtype)
        i = 0
        for info, img in image_list:
            # logger.debug('AbstractPlugin.read: img {} si {} {}'.format(
            #     img.shape, si.shape, si.dtype))
            si[i] = img
            i += 1
        logger.debug('AbstractPlugin.read: si {}'.format(si.shape))

        # Simplify shape
        self._reduce_shape(si)
        logger.debug('AbstractPlugin.read: reduced si {}'.format(si.shape))

        _shape = si.shape
        if hdr.color:
            _shape = si.shape[:-1]
            logger.debug('AbstractPlugin.read: color')
        logger.debug('AbstractPlugin.read: _shape {}'.format(_shape))
        _ndim = len(_shape)
        nz = 1
        if _ndim > 2:
            nz = _shape[-3]
        logger.debug('AbstractPlugin.read: slices {}'.format(nz))

        logger.debug('AbstractPlugin.read: calling _set_tags')
        self._set_tags(image_list, hdr, si)
        # logger.debug('AbstractPlugin.read: return  _set_tags: {}'.format(hdr))

        logger.info("Data shape read: {}".format(shape_to_str(si.shape)))

        # Add any DICOM template
        if pre_hdr is not None:
            hdr.update(pre_hdr)

        logger.debug('AbstractPlugin.read: hdr {}'.format(hdr))
        return hdr, si

    def _need_local_file(self):
        """Do the plugin need access to local files?

        Returns:
            Boolean
                - True: The plugin need access to local filenames
                - False: The plugin can access files given by an open file handle
        """

        return False

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def write_3d_numpy(self, si, destination, opts):
        """Write 3D Series image

        Args:
            self: format plugin instance
            si[slice,rows,columns]: Series array
            destination: dict of archive and filenames
            opts: Output options (dict)
        """
        pass

    @abstractmethod
    def write_4d_numpy(self, si, destination, opts):
        """Write 4D Series image

        Args:
            self: format plugin instance
            si[tag,slice,rows,columns]: Series array
            destination: dict of archive and filenames
            opts: Output options (dict)
        """
        pass

    def getTimeline(self):
        """Get timeline

        Returns:
            Timeline in seconds, as numpy array of floats
                Delta time is given as seconds. First image is t=0.
                Length of array is number of tags.
        Raises:
            ValueError: tags for dataset is not time tags
        """
        if self.input_order == INPUT_ORDER_TIME:
            timeline = [0.0]
            for t in range(1, len(self.tags[0])):
                timeline.append(self.tags[0][t] - self.tags[0][0])
            return np.array(timeline)
        else:
            raise ValueError("No timeline tags are available. Input order: {}".format(
                self.input_order))

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

        q = np.eye(4)
        # Set column 3 and row 3 to zeros, except [3,3]
        colr=np.array([[orient[3]], [orient[4]], [orient[5]]])
        colc=np.array([[orient[0]], [orient[1]], [orient[2]]])
        colr = normalize_column(colr,0)
        colc = normalize_column(colc,1)
        k=np.cross(colr, colc, axis=0)

        q[:3, :3] = np.hstack((colr, colc, k))
        if debug:
            logger.debug("q")
            logger.debug( q)

        if debug:
            logger.debug("determinant(q) {}".format(np.linalg.det(q)))
        if np.linalg.det(q) < 0:
            q[:3,2] = -q[:3,2]

        # Scale matrix
        diagVox = np.eye(3)
        diagVox[0,0] = dc
        diagVox[1,1] = dr
        diagVox[2,2] = ds
        if debug:
            logger.debug("diagVox")
            logger.debug( diagVox)
            logger.debug("q without scaling {}".format(q.dtype))
            logger.debug( q)
        q[:3,:3] = np.dot(q[:3,:3],diagVox)
        if debug:
            logger.debug("q with scaling {}".format(q.dtype))
            logger.debug( q)

        # Add translations
        q[0,3] = x; q[1,3] = y; q[2,3] = z       # pos x,y,z
        if debug:
            logger.debug("q with translations")
            logger.debug( q)
        # q now equals dicom_to_patient in spm_dicom_convert

        return q
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

        Args:
            r: (z,y,x) of voxel in voxel coordinates as numpy.array
            transformation: transformation matrix when different from self.transformationMatrix
        Returns:
            (z,y,x) of voxel in world coordinates (mm) as numpy.array
        """

        if transformation is None:
            transformation = self.transformationMatrix
        # q = self.getTransformationMatrix()

        # V = np.array([[r[2]], [r[1]], [r[0]], [1]])  # V is [x,y,z,1]

        newpos = np.dot(transformation, np.hstack((r, [1])))

        # return np.array([newpos[2,0],newpos[1,0],newpos[0,0]])   # z,y,x
        return newpos[:3]

    def getVoxelForPosition(self, p, transformation=None):
        """ Get voxel for given patient position p

        Use Patient Position and Image Orientation to calculate world
        coordinates for given voxel

        Args:
            p: (z,y,x) of voxel in world coordinates (mm) as numpy.array
            transformation: transformation matrix when different from self.transformationMatrix
        Returns:
            (z,y,x) of voxel in voxel coordinates as numpy.array
        """

        if transformation is None:
            transformation = self.transformationMatrix
        # q = self.getTransformationMatrix()

        # V = np.array([[p[2]], [p[1]], [p[0]], [1]])    # V is [x,y,z,1]

        qinv = np.linalg.inv(transformation)
        r = np.dot(qinv, np.hstack((p, [1])))

        # z,y,x
        # return np.array([int(r[2,0]+0.5),int(r[1,0]+0.5),int(r[0,0]+0.5)], dtype=int)
        # return int(r+0.5)[:3]
        return (r + 0.5).astype(int)[:3]

    @staticmethod
    def replace_geometry_attributes(im, gim):
        """Replace geometry attributes in image with values from gim
        """

        im.SliceLocation = gim.SliceLocation
        im.ImagePositionPatient = gim.ImagePositionPatient
        im.ImageOrientationPatient = gim.ImageOrientationPatient
        im.FrameOfReferenceUID = gim.FrameOfReferenceUID
        im.PositionReferenceIndicator = gim.PositionReferenceIndicator
        im.SliceThickness = gim.SliceThickness
        try:
            im.SpacingBetweenSlices = gim.SpacingBetweenSlices
        except AttributeError:
            pass
        im.AcquisitionMatrix = gim.AcquisitionMatrix
        im.PixelSpacing = gim.PixelSpacing

    @staticmethod
    def _reduce_shape(si, axes=None):
        """Reduce shape when leading shape(s) are 1.

        Will not reduce to less than 2-dimensional image.
        Also reduce axes when reducing shape.

        Args:
            self: format plugin instance
            si[...]: Series array
        Raises:
            ValueError: tags for dataset is not time tags
        """

        # Color image?
        mindim = 2
        if si.shape[-1] == 3:
            mindim += 1

        while si.ndim > mindim:
            if si.shape[0] == 1:
                si.shape = si.shape[1:]
                if axes is not None:
                    del axes[0]
            else:
                break

    def _reorder_to_dicom(self, data, flip=False, flipud=False):
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
        flipud: Whether matrix is transposed
        """

        logger.debug('AbstractPlugin._reorder_to_dicom: shape in {}'.format(
            data.shape))
        if data.ndim == 5:
            rows, columns, slices, tags, d5 = data.shape
            if flipud:
                rows, columns = columns, rows
            si = np.zeros((d5, tags, slices, rows, columns), data.dtype)
            for d in range(d5):
                for tag in range(tags):
                    for slice in range(slices):
                        si[d, tag, slice, :, :] = self._reorder_slice(data[:, :, slice, tag, d],
                                                                      flip=flip, flipud=flipud)
                        # if flip:
                        #    si[d,tag,slice,:,:] = \
                        #    (data[:,:,slice,tag,d]).T
                        #    #np.fliplr(data[:,:,slice,tag,d]).T
                        # else:
                        #    si[d,tag,slice,:,:] = data[:,:,slice,tag,d]
        elif data.ndim == 4:
            rows, columns, slices, tags = data.shape
            if flipud:
                rows, columns = columns, rows
            si = np.zeros((tags, slices, rows, columns), data.dtype)
            for tag in range(tags):
                for slice in range(slices):
                    si[tag, slice, :, :] = self._reorder_slice(data[:, :, slice, tag],
                                                               flip=flip, flipud=flipud)
                    # if flip:
                    #    si[tag,slice,:,:] = (data[:,:,slice,tag]).T
                    #    #si[tag,slice,:,:] = np.fliplr(data[:,:,slice,tag]).T
                    # else:
                    #    si[tag,slice,:,:] = data[:,:,slice,tag]
        elif data.ndim == 3:
            rows, columns, slices = data.shape
            if flipud:
                rows, columns = columns, rows
            si = np.zeros((slices, rows, columns), data.dtype)
            for slice in range(slices):
                si[slice, :, :] = self._reorder_slice(data[:, :, slice], flip=flip, flipud=flipud)
                # if flip:
                #    si[slice,:,:] = (data[:,:,slice]).T
                #    #si[slice,:,:] = np.fliplr(data[:,:,slice]).T
                # else:
                #    si[slice,:,:] = data[:,:,slice]
        elif data.ndim == 2:
            rows, columns = data.shape
            if flipud:
                rows, columns = columns, rows
            si = np.zeros((rows, columns), data.dtype)
            si[:] = self._reorder_slice(data[:], flip=flip, flipud=flipud)
            # if flip:
            #    si[:] = (data[:]).T
            #    #si[:] = np.fliplr(data[:]).T
            # else:
            #    si[:] = data[:]
        else:
            raise ValueError('Dimension %d is not implemented' % data.ndim)
        logger.debug('AbstractPlugin._reorder_to_dicom: shape out {}'.format(
            si.shape))
        return si

    def _reorder_from_dicom(self, data, flip=False, flipud=False):
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
        flipud: Whether matrix is transposed
        """

        logger.debug('AbstractPlugin._reorder_from_dicom: shape in {}'.format(data.shape))
        if data.ndim == 5:
            d5, tags, slices, rows, columns = data.shape
            if flipud:
                rows, columns = columns, rows
            si = np.zeros((rows, columns, slices, tags, d5), data.dtype)
            for d in range(d5):
                for tag in range(tags):
                    for slice in range(slices):
                        si[:, :, slice, tag, d] = self._reorder_slice(data[d, tag, slice, :, :],
                                                                      flip=flip, flipud=flipud)
                        # if flip:
                        #    si[:,:,slice,tag,d] = \
                        #    np.fliplr(data[d,tag,slice,:,:]).T
                        # else:
                        #    si[:,:,slice,tag,d] = data[d,tag,slice,:,:]
        elif data.ndim == 4:
            tags, slices, rows, columns = data.shape
            if flipud:
                rows, columns = columns, rows
            si = np.zeros((rows, columns, slices, tags), data.dtype)
            for tag in range(tags):
                for slice in range(slices):
                    si[:, :, slice, tag] = self._reorder_slice(data[tag, slice, :, :],
                                                               flip=flip, flipud=flipud)
                    # if flip:
                    #    si[:,:,slice,tag] = np.fliplr(data[tag,slice,:,:]).T
                    # else:
                    #    si[:,:,slice,tag] = data[tag,slice,:,:]
        elif data.ndim == 3:
            slices, rows, columns = data.shape
            if flipud:
                rows, columns = columns, rows
            si = np.zeros((rows, columns, slices), data.dtype)
            for slice in range(slices):
                si[:, :, slice] = self._reorder_slice(data[slice, :, :], flip=flip, flipud=flipud)
                # if flip:
                #    si[:,:,slice] = np.fliplr(data[slice,:,:]).T
                # else:
                #    si[:,:,slice] = data[slice,:,:]
        elif data.ndim == 2:
            rows, columns = data.shape
            if flipud:
                rows, columns = columns, rows
            si = np.zeros((rows, columns), data.dtype)
            si[:, :] = self._reorder_slice(data[:, :], flip=flip, flipud=flipud)
            # if flip:
            #    si[:] = np.fliplr(data[:]).T
            # else:
            #    si[:] = data[:]
        else:
            raise ValueError('Dimension %d is not implemented' % data.ndim)
        logger.debug('AbstractPlugin._reorder_from_dicom: shape out {}'.format(
            si.shape))
        return si

    @staticmethod
    def _reorder_slice(data, flip, flipud):
        if flip and flipud:
            return np.fliplr(data).T
        elif flip:
            return np.fliplr(data)
        elif flipud:
            return data.T
        else:
            return data
