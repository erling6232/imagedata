"""Read/Write image files using ITK
"""

# Copyright (c) 2013-2018 Erling Andersen, Haukeland University Hospital, Bergen, Norway

import os.path
import sys
import logging
import mimetypes
import tempfile
import fs
import fs.tempfs
import itk
import numpy as np
import scipy

import imagedata.formats
import imagedata.axis
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

class ITKPlugin(AbstractPlugin):
    """Read/write ITK files."""

    name = "itk"
    description = "Read and write ITK files."
    authors = "Erling Andersen"
    version = "1.0.0"
    url = "www.helse-bergen.no"

    def __init__(self, name=None, description=None,
                 authors=None, version=None, url=None):
        if name        is not None: self.name        = name
        if description is not None: self.description = description
        if authors     is not None: self.authors     = authors
        if version     is not None: self.version     = version
        if url         is not None: self.url         = url
        super(ITKPlugin, self).__init__(self.name, self.description,
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

        o = None
        logging.debug("itkplugin._read_image")
        logging.debug("itkplugin._read_image filehandle {}".format(f))
        if f.endswith('.raw'):
            return(None, None)
        try:
            # https://blog.kitware.com/itk-python-image-pixel-types/
            reader = itk.imread(f)
            #imagetype = itk.Image[itk.F, 3]
            #reader = itk.ImageFileReader[imagetype].New()
            #reader.SetFileName(f)
            #reader.Update()

            #img = itk.GetArrayFromImage(reader.GetOutput())
            img = itk.GetArrayFromImage(reader)
            self._reduce_shape(img)
            logging.info("Data shape read ITK: {}".format(img.shape))

            o = reader
        except imagedata.formats.NotImageError as e:
            logging.error('itkplugin._read_image: inner exception {}'.format(e))
            raise imagedata.formats.NotImageError('{} does not look like a ITK file'.format(f))

        # Color image?
        hdr['photometricInterpretation'] = 'MONOCHROME2'
        hdr['color'] = False
        if o.GetNumberOfComponentsPerPixel() == 3:
            logging.debug('ITKPlugin._read_image: RGB color')
            hdr['photometricInterpretation'] = 'RGB'
            hdr['color'] = True

        return(o, img)

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

        def transformMatrix(direction, origin):
            matrix = itk.GetArrayFromMatrix(direction)
            A = np.array([[matrix[2,2], matrix[1,2], matrix[0,2], origin[2]],
                          [matrix[2,1], matrix[1,1], matrix[0,1], origin[1]],
                          [matrix[2,0], matrix[1,0], matrix[0,0], origin[0]],
                          [          0,           0,           0,         1]])
            return A

        """
        orientation = self.orientation
        rotation = np.zeros([3,3])
        # X axis
        rotation[0,0] = orientation[0]
        rotation[0,1] = orientation[1]
        rotation[0,2] = orientation[2]
        # Y axis
        rotation[1,0] = orientation[3]
        rotation[1,1] = orientation[4]
        rotation[1,2] = orientation[5]
        # Z axis = X cross Y
        rotation[2,0] = orientation[1]*orientation[5]-orientation[2]*orientation[4]
        rotation[2,1] = orientation[2]*orientation[3]-orientation[0]*orientation[5]
        rotation[2,2] = orientation[0]*orientation[4]-orientation[1]*orientation[3]
        logging.debug(rotation)

        # Set direction by modifying default orientation in place
        d=image.GetDirection()
        dv=d.GetVnlMatrix()
        for col in range(3):
            v=itk.vnl_vector.D()
            v.set_size(3)
            v.put(0, rotation[col,0])
            v.put(1, rotation[col,1])
            v.put(2, rotation[col,2])
            dv.set_column(col,v)
        """

        o, img = image_list[0]
        spacing   = o.GetSpacing()
        origin    = o.GetOrigin()
        direction = o.GetDirection()

        # Set spacing
        v=spacing.GetVnlVector()
        logging.debug('ITKPlugin._set_tags: hdr {}'.format(hdr))
        logging.debug('ITKPlugin._set_tags: spacing {} {} {}'.format(v.get(2), v.get(1), v.get(0)))
        hdr['spacing'] = (v.get(2), v.get(1), v.get(0))
        if v.size() > 3:
            dt = v.get(3)
        else:
            dt = 1

        # Set imagePositions for first slice
        v=origin.GetVnlVector()
        hdr['imagePositions'] = {}
        hdr['imagePositions'][0] = np.array([v.get(2), v.get(1), v.get(0)])

        # Do not calculate transformationMatrix here. Will be calculated by Series() when needed.
        #self.transformationMatrix = transformMatrix(direction, hdr['imagePositions'][0])
        #hdr['transformationMatrix'] = self.transformationMatrix
        #logging.debug('ITKPlugin._set_tags: transformationMatrix=\n{}'.format(self.transformationMatrix))

        # Set image orientation
        iop = self.orientationFromVnlMatrix(direction)
        logging.debug('ITKPlugin._set_tags: iop=\n{}'.format(iop))
        hdr['orientation'] = np.array((iop[2],iop[1],iop[0],
                                       iop[5],iop[4],iop[3]))

        # Set tags
        axes = list()
        _actual_shape = si.shape
        _color = False
        if 'color' in hdr and hdr['color']:
            _actual_shape = si.shape[:-1]
            _color = True
            logging.debug('ITKPlugin.read: color')
        _actual_ndim = len(_actual_shape)
        nt = nz = 1
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
        if _actual_ndim > 3:
            nt = _actual_shape[-4]
            axes.insert(0, imagedata.axis.UniformLengthAxis(
                imagedata.formats.input_order_to_dirname_str(hdr['input_order']),
                0,
                nt,
                dt)
            )
        times = np.arange(0, nt*dt, dt)
        tags = {}
        for slice in range(nz):
            tags[slice] = np.array(times)
        if _color:
            axes.append(
                imagedata.axis.VariableAxis(
                    'rgb',
                    ['r', 'g', 'b']
                )
            )
        hdr['axes'] = axes
        hdr['tags'] = tags

        logging.info("Data shape read DCM: {}".format(imagedata.formats.shape_to_str(si.shape)))

    def write_3d_numpy(self, si, destination, opts):
        """Write 3D numpy image as ITK file

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

        if si.color:
            raise imagedata.formats.WriteNotImplemented(
                    "Writing color ITK images not implemented.")

        logging.debug('ITKPlugin.write_3d_numpy: destination {}'.format(destination))
        archive = destination['archive']
        filename_template = 'Image_%05d.mha'
        if len(destination['files']) > 0 and len(destination['files'][0]) > 0:
            filename_template = destination['files'][0]

        self.shape                = si.shape
        self.slices               = si.slices
        self.spacing              = si.spacing
        self.imagePositions       = si.imagePositions
        self.transformationMatrix = si.transformationMatrix
        #self.orientation          = si.orientation
        self.tags                 = si.tags
        self.origin, self.orientation, self.normal = si.get_transformation_components_xyz()

        logging.info("Data shape write: {}".format(imagedata.formats.shape_to_str(si.shape)))
        #if si.ndim == 2:
        #    si.shape = (1,) + si.shape
        #if si.ndim == 3:
        #    si.shape = (1,) + si.shape
        #assert si.ndim == 4, "write_3d_series: input dimension %d is not 3D." % (si.ndim-1)
        #if si.shape[0] != 1:
        #    raise ValueError("Attempt to write 4D image ({}) using write_3d_numpy".format(si.shape))
        assert si.ndim == 2 or si.ndim == 3, "write_3d_series: input dimension %d is not 2D/3D." % (si.ndim)
        #slices = si.shape[1]
        #if slices != si.slices:
        #    raise ValueError("write_3d_series: slices of dicom template ({}) differ from input array ({}).".format(si.slices, slices))

        #if not os.path.isdir(dirname):
        #    os.makedirs(dirname)
        try:
            filename = filename_template % (0)
        except TypeError:
            filename = filename_template
        #filename = os.path.join(dirname, filename)
        self.write_numpy_itk(si, archive, filename)

    def write_4d_numpy(self, si, destination, opts):
        """Write 4D numpy image as ITK files

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

        if si.color:
            raise imagedata.formats.WriteNotImplemented(
                    "Writing color ITK images not implemented.")

        logging.debug('ITKPlugin.write_4d_numpy: destination {}'.format(destination))
        archive = destination['archive']
        filename_template = 'Image_%05d.mha'
        if len(destination['files']) > 0 and len(destination['files'][0]) > 0:
            filename_template = destination['files'][0]

        self.shape                = si.shape
        self.slices               = si.slices
        self.spacing              = si.spacing
        self.imagePositions       = si.imagePositions
        self.transformationMatrix = si.transformationMatrix
        #self.orientation          = si.orientation
        self.tags                 = si.tags
        self.origin, self.orientation, self.normal = si.get_transformation_components_xyz()

        # Defaults
        self.output_sort = imagedata.formats.SORT_ON_SLICE
        if 'output_sort' in opts:
            self.output_sort = opts['output_sort']

        # Should we allow to write 3D volume?
        #if si.ndim == 3:
        #    si.shape = (1,) + si.shape
        if si.ndim != 4:
            raise ValueError("write_4d_numpy: input dimension %d is not 4D.".format(si.ndim))

        logging.debug("write_4d_numpy: si dtype {}, shape {}, sort {}".format(
            si.dtype, si.shape,
            imagedata.formats.sort_on_to_str(self.output_sort)))

        steps  = si.shape[0]
        slices = si.shape[1]
        if steps != len(si.tags[0]):
            raise ValueError("write_4d_series: tags of dicom template ({}) differ from input array ({}).".format(len(si.tags[0]), steps))
        if slices != si.slices:
            raise ValueError("write_4d_series: slices of dicom template ({}) differ from input array ({}).".format(si.slices, slices))

        #if not os.path.isdir(dirname):
        #    os.makedirs(dirname)

        logging.debug('write_4d_numpy: si[0,0,0,0]={}'.format(
            si[0,0,0,0]))
        if self.output_sort == imagedata.formats.SORT_ON_TAG:
            for slice in range(slices):
                filename = filename_template % (slice)
                #filename = os.path.join(dirname, filename)
                self.write_numpy_itk(si[:,slice,...], archive, filename)
        else: # default: imagedata.formats.SORT_ON_SLICE:
            for tag in range(steps):
                filename = filename_template % (tag)
                #filename = os.path.join(dirname, filename)
                self.write_numpy_itk(si[tag,...], archive, filename)

    def write_numpy_itk(self, si, archive, filename):
        """Write single volume to file

        Input:
        - self: ITKPlugin instance, including these attributes:
            slices (not used)
            spacing
            imagePositions
            transformationMatrix
            orientation (not used)
            tags (not used)
        - si: numpy 3D array [slice,row,column]
        - archive: archive object
        - filename: file name, possibly without extentsion
        """

        if si.ndim != 2 and si.ndim != 3:
            raise ValueError("write_numpy_itk: input dimension %d is not 2D/3D." % si.ndim)
        if np.issubdtype(si.dtype, np.floating):
            arr=np.float32(np.nan_to_num(si))
            #arr=np.nan_to_num(si)
        else:
            arr=si.copy()
        if arr.dtype == np.int32:
            logging.debug("write_numpy_itk: arr {}".format(arr.dtype))
            arr=arr.astype(np.float32)
            #arr=arr.astype(np.uint16)
        if arr.dtype == np.complex64 or arr.dtype == np.complex128:
            arr = np.absolute(arr)
        if arr.dtype == np.double:
            arr = arr.astype(np.float32)
        logging.debug("write_numpy_itk: arr {}".format(arr.dtype))

        # Write it
        logging.debug("write_numpy_itk: arr {} {}".format(arr.shape, arr.dtype))
        image = itk.GetImageFromArray(arr)
        #img = itk.GetArrayFromImage(image)
        #logging.debug("write_numpy_itk: shape {} {}".format(arr.shape, img.shape))
        fromImageType = self.get_image_type(image)
        image = self.get_image_from_numpy(image)

        logging.debug("write_numpy_itk: imagetype {} filename {}".format(fromImageType,filename))
        """
        import itk

        PixelType = itk.ctype(“unsigned char”)
        image = itk.imread("input_filename.png", PixelType)
        dimension = image.GetImageDimension()
        InputImageType = type(image)
        OutputPixelType = itk.ctype(“float”)
        OutputImageType = itk.Image[OutputPixelType, dimension]
        cast = itk.CastImageFilter[InputImageType, OutputImageType].New(image)
        itk.imwrite(cast, "output_filename.png")
        """
        #try:
        #    writer = itk.ImageFileWriter[fromImageType].New()
        #except KeyError:
        #    # Does not support the output format
        #    raise
        ##writer.SetInput(image.GetPointer())
        #writer.SetInput(image)
        ##writer.SetInput(image.GetOutput())

        if len(os.path.splitext(filename)[1]) == 0:
            filename = filename + '.mha'
        ext = os.path.splitext(filename)[1]
        logging.debug('write_numpy_itk: ext %s' % ext)
        #writer.SetFileName(filename)

        ##writer.SetRAWFileName(filename+".raw")
        ##writer.SetInput(flipper.GetOutput())
        ## writer.SetFileTypeToBinary()
        ##writer.Write()
        #writer.Update()
        f = tempfile.NamedTemporaryFile(
                suffix=ext, delete=False)
        logging.debug('write_numpy_itk: write local file %s' % f.name)
        itk.imwrite(image, f.name)
        f.close()
        logging.debug('write_numpy_itk: written local file %s' % f)
        logging.debug('write_numpy_itk: copy to archive %s as %s' % (
            archive, filename))
        archive.add_localfile(f.name, filename)
        os.unlink(f.name)

    def orientationFromVnlMatrix(self, direction):
        tr=direction.GetVnlMatrix()
        arr = []
        for c in range(2):
            for r in range(3):
                arr.append(float(tr.get(r,c)))
        return arr

    # def numpyMatrixFromVnlMatrix(self, direction):
    #     tr=direction.GetVnlMatrix()
    #     arr = np.zeros([3,3])
    #     for c in range(3):
    #         for r in range(3):
    #             arr[r,c] = float(tr.get(r,c))
    #     return arr

    def set_direction_from_dicom_header(self, image):
        orientation = self.orientation
        rotation = np.zeros([3,3])
        # X axis
        rotation[0,0] = orientation[2]
        rotation[0,1] = orientation[1]
        rotation[0,2] = orientation[0]
        # Y axis
        rotation[1,0] = orientation[5]
        rotation[1,1] = orientation[4]
        rotation[1,2] = orientation[3]
        # Z axis = X cross Y
        rotation[2,0] = orientation[1]*orientation[3]-orientation[0]*orientation[4]
        rotation[2,1] = orientation[0]*orientation[5]-orientation[2]*orientation[3]
        rotation[2,2] = orientation[2]*orientation[4]-orientation[1]*orientation[5]
        logging.debug('set_direction_from_dicom_header: rotation:\n{}'.format(rotation))

        # Set direction by modifying default orientation in place
        d=image.GetDirection()
        dv=d.GetVnlMatrix()
        for col in range(3):
            v=itk.vnl_vector.D()
            v.set_size(3)
            v.put(0, rotation[col,0])
            v.put(1, rotation[col,1])
            v.put(2, rotation[col,2])
            dv.set_column(col,v)

    def set_direction_from_transformation_matrix(self, image):
        M = self.transformationMatrix

        # Set direction by modifying default orientation in place
        d=image.GetDirection()
        dv=d.GetVnlMatrix()
        for col in range(3):
            v=itk.vnl_vector.D()
            v.set_size(3)
            v.put(0, M[2-col,2])
            v.put(1, M[2-col,1])
            v.put(2, M[2-col,0])
            dv.set_column(col,v)

    def get_image_from_numpy(self, image, image_type = False):
        """Returns an itk Image created from the supplied scipy ndarray.
        If the image_type is supported, will be automatically transformed to that type,
        otherwise the most suitable is selected.
        
        @note always use this instead of directly the itk.PyBuffer, as that
                object transposes the image axes.
        
        @param arr an array
        @type arr scipy.ndarray
        @param image_type an itk image type
        @type image_type itk.Image (template)

        @return an instance of itk.Image holding the array's data
        @rtype itk.Image (instance)
        """

        def itkMatrix_from_orientation(orientation, normal):
            oT = orientation.reshape((2,3)).T
            colr = oT[:,0].reshape((3,1))
            colc = oT[:,1].reshape((3,1))
            coln = normal.reshape((3,1))
            if len(self.shape) < 3:
                M = np.hstack((colr[:2], colc[:2])).reshape((2,2))
            else:
                M = np.hstack((colr, colc, coln)).reshape((3,3))
            itkM = itk.GetMatrixFromArray(M)
            return itkM

        # The itk_py_converter transposes the image dimensions. This has to be countered.
        ##arr = si
        #arr = scipy.transpose(si)

        # determine image type if not supplied
        #if not image_type:
        #    image_type = self.get_image_type_from_array(arr)

        # convert
        #itk_py_converter = itk.PyBuffer[image_type]
        #image = itk_py_converter.GetImageFromArray(arr)
        #img2 = itk.GetArrayFromImage(image)
        #logging.debug("get_image_from_numpy: from shape {} to shape {}".format(
        #    arr.shape, img2.shape))

        #self.set_direction_from_dicom_header(image)
        #self.set_direction_from_transformation_matrix(image)
        image.SetDirection(
            itkMatrix_from_orientation(
                self.orientation, self.normal))

        z,y,x = self.imagePositions[0]
        logging.debug("get_image_from_numpy: (z,y,x)=({},{},{}) ({})".format(z,y,x,type(z)))
        if isinstance(z, np.int64):
            logging.debug("get_image_from_numpy: SetOrigin int")
            if len(self.shape) < 3:
                image.SetOrigin([int(x),int(y)])
            else:
                image.SetOrigin([int(x),int(y),int(z)])
        else:
            logging.debug("get_image_from_numpy: SetOrigin float")
            if len(self.shape) < 3:
                image.SetOrigin((float(x),float(y)))
            else:
                image.SetOrigin((float(x),float(y),float(z)))

        dz, dy, dx = self.spacing
        dx=float(dx); dy=float(dy); dz=float(dz)
        if len(self.shape) < 3:
            image.SetSpacing([dx, dy])
        else:
            image.SetSpacing([dx, dy, dz])

        return image

    # def get_image_type_from_array(self, arr): # tested
    #     """
    #     Returns the image type of the supplied array as itk.Image template.
    #     @param arr: an scipy.ndarray array
    #
    #     @return a template of itk.Image
    #     @rtype itk.Image
    #
    #     @raise DependencyError if the itk wrapper do not support the target image type
    #     @raise ImageTypeError if the array dtype is unsupported
    #     """
    #     # mapping from scipy to the possible itk types, in order from most to least suitable
    #     # ! this assumes char=8bit, short=16bit and long=32bit (minimal values)
    #     scipy_to_itk_types = {scipy.bool_: [itk.SS, itk.UC, itk.US, itk.SS, itk.UL, itk.SL],
    #         scipy.uint8: [itk.UC, itk.US, itk.SS, itk.UL, itk.SL],
    #         scipy.uint16: [itk.US, itk.UL, itk.SL],
    #         scipy.uint32: [itk.UL],
    #         scipy.uint64: [],
    #         scipy.int8: [itk.SC, itk.SS, itk.SL],
    #         scipy.int16: [itk.SS, itk.SL],
    #         scipy.int32: [itk.SL],
    #         scipy.int64: [],
    #         scipy.float32: [itk.F, itk.D],
    #         scipy.float64: [itk.D],
    #         scipy.float128: []}
    #
    #     if arr.ndim <= 1:
    #         raise(DependencyError('Itk does not support images with less than 2 dimensions.'))
    #
    #     # chek for unknown array data type
    #     if not arr.dtype.type in scipy_to_itk_types:
    #         raise(ImageTypeError('The array dtype {} could not be mapped to any itk image type.'.format(arr.dtype)))
    #
    #     # check if any valid conversion exists
    #     if 0 == len(scipy_to_itk_types[arr.dtype.type]):
    #         raise(ImageTypeError('No valid itk type for the pixel data dtype {}.'.format(arr.dtype)))
    #
    #     # iterate and try out candidate templates
    #     ex = None
    #     for itk_type in scipy_to_itk_types[arr.dtype.type]:
    #         try:
    #             return itk.Image[itk_type, arr.ndim]
    #         except Exception as e: # pass raised exception, as a list of ordered possible itk pixel types is processed and some of them might not be compiled with the current itk wrapper module
    #             ex = e
    #             pass
    #     # if none fitted, examine error and eventually translate, otherwise rethrow
    #     if type(ex) == KeyError:
    #         raise(DependencyError('The itk python wrappers were compiled without support the combination of {} dimensions and at least one of the following pixel data types (which are compatible with dtype {}): {}.'.format(arr.ndim, arr.dtype, scipy_to_itk_types[arr.dtype.type])))
    #     else:
    #         raise

    def get_image_type(self, image):
        """
        Returns the image type of the supplied image as itk.Image template.
        @param image: an instance of itk.Image

        @return a template of itk.Image
        @rtype itk.Image
        """
        try:
            return itk.Image[itk.template(image)[1][0],
                itk.template(image)[1][1]]
        except IndexError as e:
            raise(NotImplementedError, 'The python wrappers of ITK define no template class for this data type.')

    # def reverse_3d_shape(self, shape):
    #     #if len(shape) == 4:
    #     #    t,slices,rows,columns = shape
    #     #else:
    #     #    slices,rows,columns = shape
    #     #return((columns,rows,slices))
    #     return tuple(reversed(shape))
    #
    # def reverse_4d_shape(self, shape):
    #     #if len(shape) == 4:
    #     #    t,slices,rows,columns = shape
    #     #else:
    #     #    slices,rows,columns = shape
    #     #    t = 1
    #     #return((columns,rows,slices,t))
    #     return tuple(reversed(shape))

    # def reorder_3d_data(self, data):
    #     # Reorder data
    #     # ITK order:   sitk[columns,rows,slices]
    #     # DICOM order: data[t,slices,rows,columns]
    #     itk_shape = self.reverse_3d_shape(data.shape)
    #
    #     rows,columns,slices = itk_shape
    #     if len(data.shape) == 3:
    #         logging.info("From DCM shape: {}x{}x{}".format(data.shape[0],data.shape[1],data.shape[2]))
    #         si = np.zeros([slices,columns,rows], data.dtype)
    #         for z in range(slices):
    #             #si[:,:,z] = data[z,:,:].T
    #             #si[z,:,:] = data[z,:,:].T
    #             si[z,:,:] = data[z,:,:]
    #     elif len(data.shape) == 4:
    #         logging.info("From DCM shape: {}tx{}x{}x{}".format(data.shape[0],data.shape[1],data.shape[2],data.shape[3]))
    #         si = np.zeros([slices,columns,rows], data.dtype)
    #         for z in range(slices):
    #             #si[:,:,z] = data[0,z,:,:].T
    #             #si[z,:,:] = data[0,z,:,:].T
    #             si[z,:,:] = data[0,z,:,:]
    #     else:
    #         raise(ValueError("Unknown shape {}.".format(len(data.shape))))
    #     logging.info("To   ITK shape: cols:{} rows:{} slices:{}".format(columns,rows,slices))
    #     return si
    #
    # def reorder_4d_data(self, data):
    #     # Reorder data
    #     # ITK order:   sitk[columns,rows,slices,t]
    #     # DICOM order: data[t,slices,rows,columns]
    #     itk_shape = self.reverse_4d_shape(data.shape)
    #
    #     rows,columns,slices,t = itk_shape
    #     if len(data.shape) != 4:
    #         raise(ValueError("Not 4D data set."))
    #     logging.info("From DCM shape: %dtx%dx%dx%d" % (data.shape[0],data.shape[1],data.shape[2],data.shape[3]))
    #     si = np.zeros([slices,columns,rows,t], data.dtype)
    #     for i in range(t):
    #         for z in range(slices):
    #             #si[z,:,:,i] = data[i,z,:,:].T
    #             si[z,:,:,i] = data[i,z,:,:]
    #     logging.debug("To   ITK shape: %dx%dx%dx%dt" % (columns,rows,slices,t))
    #     return si

    def copy(self, other=None):
        logging.debug("ITKPlugin::copy")
        if other is None: other = ITKPlugin()
        return AbstractPlugin.copy(self, other=other)
