"""Read/Write image files using ITK
"""

# Copyright (c) 2013-2022 Erling Andersen, Haukeland University Hospital, Bergen, Norway

import os.path
import logging
import tempfile
import itk
import numpy as np
from . import NotImageError, input_order_to_dirname_str, shape_to_str, WriteNotImplemented,\
    SORT_ON_SLICE, SORT_ON_TAG, sort_on_to_str
from ..axis import UniformLengthAxis, VariableAxis
from .abstractplugin import AbstractPlugin

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


# noinspection PyUnresolvedReferences
class ITKPlugin(AbstractPlugin):
    """Read/write ITK files."""

    name = "itk"
    description = "Read and write ITK files."
    authors = "Erling Andersen"
    version = "1.1.0"
    url = "www.helse-bergen.no"

    def __init__(self, name=None, description=None,
                 authors=None, version=None, url=None):
        if name is not None:
            self.name = name
        if description is not None:
            self.description = description
        if authors is not None:
            self.authors = authors
        if version is not None:
            self.version = version
        if url is not None:
            self.url = url
        super(ITKPlugin, self).__init__(self.name, self.description,
                                        self.authors, self.version, self.url)
        self.shape = None
        self.slices = None
        self.spacing = None
        self.imagePositions = None
        self.transformationMatrix = None
        # self.orientation          = si.orientation
        self.tags = None
        self.origin = None
        self.orientation = None
        self.normal = None
        self.output_sort = None

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

        logger.debug("itkplugin._read_image")
        logger.debug("itkplugin._read_image filehandle {}".format(f))
        if f.endswith('.raw'):
            return None, None
        try:
            # https://blog.kitware.com/itk-python-image-pixel-types/
            reader = itk.imread(f)
            img = itk.GetArrayFromImage(reader)
            self._reduce_shape(img)
            logger.info("Data shape read ITK: {}".format(img.shape))

            o = reader
        except NotImageError as e:
            logger.error('itkplugin._read_image: inner exception {}'.format(e))
            raise NotImageError('{} does not look like a ITK file'.format(f))

        # Color image?
        hdr.photometricInterpretation = 'MONOCHROME2'
        hdr.color = False
        if o.GetNumberOfComponentsPerPixel() == 3:
            logger.debug('ITKPlugin._read_image: RGB color')
            hdr.photometricInterpretation = 'RGB'
            hdr.color = True

        return o, img

    def _need_local_file(self):
        """Do the plugin need access to local files?

        Returns:
            Boolean. True: The plugin need access to local filenames.
                False: The plugin can access files given by an open file handle
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

        # def transformMatrix(direction, origin):
        #     matrix = itk.GetArrayFromMatrix(direction)
        #     A = np.array([[matrix[2,2], matrix[1,2], matrix[0,2], origin[2]],
        #                   [matrix[2,1], matrix[1,1], matrix[0,1], origin[1]],
        #                   [matrix[2,0], matrix[1,0], matrix[0,0], origin[0]],
        #                   [          0,           0,           0,         1]])
        #     return A

        # orientation = self.orientation
        # rotation = np.zeros([3,3])
        # # X axis
        # rotation[0,0] = orientation[0]
        # rotation[0,1] = orientation[1]
        # rotation[0,2] = orientation[2]
        # # Y axis
        # rotation[1,0] = orientation[3]
        # rotation[1,1] = orientation[4]
        # rotation[1,2] = orientation[5]
        # # Z axis = X cross Y
        # rotation[2,0] = orientation[1]*orientation[5]-orientation[2]*orientation[4]
        # rotation[2,1] = orientation[2]*orientation[3]-orientation[0]*orientation[5]
        # rotation[2,2] = orientation[0]*orientation[4]-orientation[1]*orientation[3]
        # logger.debug(rotation)
        #
        # # Set direction by modifying default orientation in place
        # d=image.GetDirection()
        # dv=d.GetVnlMatrix()
        # for col in range(3):
        #     v=itk.vnl_vector.D()
        #     v.set_size(3)
        #     v.put(0, rotation[col,0])
        #     v.put(1, rotation[col,1])
        #     v.put(2, rotation[col,2])
        #     dv.set_column(col,v)

        o, img = image_list[0]
        spacing = o.GetSpacing()
        origin = o.GetOrigin()
        direction = o.GetDirection()

        # Set spacing
        v = spacing.GetVnlVector()
        logger.debug('ITKPlugin._set_tags: hdr {}'.format(hdr))
        logger.debug('ITKPlugin._set_tags: spacing {} {} {}'.format(v.get(2), v.get(1), v.get(0)))
        hdr.spacing = (float(v.get(2)), float(v.get(1)), float(v.get(0)))
        if v.size() > 3:
            dt = float(v.get(3))
        else:
            dt = 1.0

        # Set imagePositions for first slice
        v = origin.GetVnlVector()
        hdr.imagePositions = {0: np.array([v.get(2), v.get(1), v.get(0)])}

        # Do not calculate transformationMatrix here. Will be calculated by Series() when needed.
        # self.transformationMatrix = transformMatrix(direction, hdr['imagePositions'][0])
        # hdr['transformationMatrix'] = self.transformationMatrix
        # logger.debug('ITKPlugin._set_tags: transformationMatrix=\n{}'.format(
        #     self.transformationMatrix))

        # Set image orientation
        iop = self._orientation_from_vnl_matrix(direction)
        logger.debug('ITKPlugin._set_tags: iop=\n{}'.format(iop))
        hdr.orientation = np.array((iop[2], iop[1], iop[0],
                                    iop[5], iop[4], iop[3]))

        # Set tags
        axes = list()
        _actual_shape = si.shape
        _color = False
        if hdr.color:
            _actual_shape = si.shape[:-1]
            _color = True
            logger.debug('ITKPlugin.read: color')
        _actual_ndim = len(_actual_shape)
        nt = nz = 1
        axes.append(UniformLengthAxis(
            'row',
            hdr.imagePositions[0][1],
            _actual_shape[-2],
            hdr.spacing[1])
        )
        axes.append(UniformLengthAxis(
            'column',
            hdr.imagePositions[0][2],
            _actual_shape[-1],
            hdr.spacing[2])
        )
        if _actual_ndim > 2:
            nz = _actual_shape[-3]
            axes.insert(0, UniformLengthAxis(
                'slice',
                hdr.imagePositions[0][0],
                nz,
                hdr.spacing[0])
                        )
        if _actual_ndim > 3:
            nt = _actual_shape[-4]
            axes.insert(0, UniformLengthAxis(
                input_order_to_dirname_str(hdr.input_order),
                0,
                nt,
                dt)
                        )
        times = np.arange(0, nt * dt, dt)
        tags = {}
        for _slice in range(nz):
            tags[_slice] = np.array(times)
        if _color:
            axes.append(
                VariableAxis(
                    'rgb',
                    ['r', 'g', 'b']
                )
            )
        hdr.axes = axes
        hdr.tags = tags

        logger.info("Data shape read DCM: {}".format(shape_to_str(si.shape)))

    def write_3d_numpy(self, si, destination, opts):
        """Write 3D numpy image as ITK file

        Args:
            self: ITKPlugin instance
            si: Series array (3D or 4D), including these attributes:
            - slices
            - spacing
            - imagePositions
            - transformationMatrix
            - orientation
            - tags

            destination: dict of archive and filenames
            opts: Output options (dict)
        """

        if si.color:
            raise WriteNotImplemented(
                "Writing color ITK images not implemented.")

        logger.debug('ITKPlugin.write_3d_numpy: destination {}'.format(destination))
        archive = destination['archive']
        filename_template = 'Image_%05d.mha'
        if len(destination['files']) > 0 and len(destination['files'][0]) > 0:
            filename_template = destination['files'][0]

        self.shape = si.shape
        self.slices = si.slices
        self.spacing = si.spacing
        self.imagePositions = si.imagePositions
        self.transformationMatrix = si.transformationMatrix
        # self.orientation          = si.orientation
        self.tags = si.tags
        self.origin, self.orientation, self.normal = si.get_transformation_components_xyz()

        logger.info("Data shape write: {}".format(shape_to_str(si.shape)))
        # if si.ndim == 2:
        #    si.shape = (1,) + si.shape
        # if si.ndim == 3:
        #    si.shape = (1,) + si.shape
        # assert si.ndim == 4, "write_3d_series: input dimension %d is not 3D." % (si.ndim-1)
        # if si.shape[0] != 1:
        #    raise ValueError("Attempt to write 4D image ({}) using write_3d_numpy".format(
        #        si.shape))
        assert si.ndim == 2 or si.ndim == 3,\
            "write_3d_series: input dimension %d is not 2D/3D." % si.ndim
        # slices = si.shape[1]
        # if slices != si.slices:
        #    raise ValueError("write_3d_series: slices of dicom template ({}) differ "
        #        "from input array ({}).".format(si.slices, slices))

        # if not os.path.isdir(directory_name):
        #    os.makedirs(directory_name)
        try:
            filename = filename_template % 0
        except TypeError:
            filename = filename_template
        # filename = os.path.join(directory_name, filename)
        self.write_numpy_itk(si, archive, filename)

    def write_4d_numpy(self, si, destination, opts):
        """Write 4D numpy image as ITK files

        Args:
            self: ITKPlugin instance
            si: [tag,slice,rows,columns]: Series array, including these attributes:
            - slices
            - spacing
            - imagePositions
            - transformationMatrix
            - orientation
            - tags

            destination: dict of archive and filenames
            opts: Output options (dict)
        """

        if si.color:
            raise WriteNotImplemented(
                "Writing color ITK images not implemented.")

        logger.debug('ITKPlugin.write_4d_numpy: destination {}'.format(destination))
        archive = destination['archive']
        filename_template = 'Image_%05d.mha'
        if len(destination['files']) > 0 and len(destination['files'][0]) > 0:
            filename_template = destination['files'][0]

        self.shape = si.shape
        self.slices = si.slices
        self.spacing = si.spacing
        self.imagePositions = si.imagePositions
        self.transformationMatrix = si.transformationMatrix
        # self.orientation          = si.orientation
        self.tags = si.tags
        self.origin, self.orientation, self.normal = si.get_transformation_components_xyz()

        # Defaults
        self.output_sort = SORT_ON_SLICE
        if 'output_sort' in opts:
            self.output_sort = opts['output_sort']

        # Should we allow to write 3D volume?
        # if si.ndim == 3:
        #    si.shape = (1,) + si.shape
        if si.ndim != 4:
            raise ValueError("write_4d_numpy: input dimension {} is not 4D.".format(si.ndim))

        logger.debug("write_4d_numpy: si dtype {}, shape {}, sort {}".format(
            si.dtype, si.shape,
            sort_on_to_str(self.output_sort)))

        steps = si.shape[0]
        slices = si.shape[1]
        if steps != len(si.tags[0]):
            raise ValueError(
                "write_4d_series: tags of dicom template ({}) differ "
                "from input array ({}).".format(len(si.tags[0]), steps))
        if slices != si.slices:
            raise ValueError(
                "write_4d_series: slices of dicom template ({}) differ "
                "from input array ({}).".format(si.slices, slices))

        # if not os.path.isdir(directory_name):
        #    os.makedirs(directory_name)

        logger.debug('write_4d_numpy: si[0,0,0,0]={}'.format(
            si[0, 0, 0, 0]))
        if self.output_sort == SORT_ON_TAG:
            for _slice in range(slices):
                filename = filename_template % _slice
                # filename = os.path.join(directory_name, filename)
                self.write_numpy_itk(si[:, _slice, ...], archive, filename)
        else:  # default: SORT_ON_SLICE:
            for tag in range(steps):
                filename = filename_template % tag
                # filename = os.path.join(directory_name, filename)
                self.write_numpy_itk(si[tag, ...], archive, filename)

    def write_numpy_itk(self, si, archive, filename):
        """Write single volume to file

        Args:
            self: ITKPlugin instance, including these attributes:
                - slices (not used)
                - spacing
                - imagePositions
                - transformationMatrix
                - orientation (not used)
                - tags (not used)

            si: numpy 3D array [slice,row,column]
            archive: archive object
            filename: file name, possibly without extentsion
        """

        if si.ndim != 2 and si.ndim != 3:
            raise ValueError("write_numpy_itk: input dimension %d is not 2D/3D." % si.ndim)
        if np.issubdtype(si.dtype, np.floating):
            arr = np.float32(np.nan_to_num(si))
        else:
            arr = si.copy()
        if arr.dtype == np.int32:
            logger.debug("write_numpy_itk: arr {}".format(arr.dtype))
            arr = arr.astype(np.float32)
            # arr=arr.astype(np.uint16)
        if arr.dtype == np.complex64 or arr.dtype == np.complex128:
            arr = np.absolute(arr)
        if arr.dtype == np.double:
            arr = arr.astype(np.float32)
        logger.debug("write_numpy_itk: arr {}".format(arr.dtype))

        # Write it
        logger.debug("write_numpy_itk: arr {} {}".format(arr.shape, arr.dtype))
        image = itk.GetImageFromArray(arr)
        from_image_type = self._get_image_type(image)
        image = self.get_image_from_numpy(image)

        logger.debug("write_numpy_itk: imagetype {} filename {}".format(from_image_type, filename))

        if len(os.path.splitext(filename)[1]) == 0:
            filename = filename + '.mha'
        ext = os.path.splitext(filename)[1]
        logger.debug('write_numpy_itk: ext %s' % ext)

        f = tempfile.NamedTemporaryFile(
            suffix=ext, delete=False)
        logger.debug('write_numpy_itk: write local file %s' % f.name)
        itk.imwrite(image, f.name)
        f.close()
        logger.debug('write_numpy_itk: written local file %s' % f)
        logger.debug('write_numpy_itk: copy to archive %s as %s' % (
            archive, filename))
        archive.add_localfile(f.name, filename)
        os.unlink(f.name)

    @staticmethod
    def _orientation_from_vnl_matrix(direction):
        tr = direction.GetVnlMatrix()
        arr = []
        for c in range(2):
            for r in range(3):
                arr.append(float(tr.get(r, c)))
        return arr

    def set_direction_from_dicom_header(self, image):
        orientation = self.orientation
        rotation = np.zeros([3, 3])
        # X axis
        rotation[0, 0] = orientation[2]
        rotation[0, 1] = orientation[1]
        rotation[0, 2] = orientation[0]
        # Y axis
        rotation[1, 0] = orientation[5]
        rotation[1, 1] = orientation[4]
        rotation[1, 2] = orientation[3]
        # Z axis = X cross Y
        rotation[2, 0] = orientation[1] * orientation[3] - orientation[0] * orientation[4]
        rotation[2, 1] = orientation[0] * orientation[5] - orientation[2] * orientation[3]
        rotation[2, 2] = orientation[2] * orientation[4] - orientation[1] * orientation[5]
        logger.debug('set_direction_from_dicom_header: rotation:\n{}'.format(rotation))

        # Set direction by modifying default orientation in place
        d = image.GetDirection()
        dv = d.GetVnlMatrix()
        for col in range(3):
            v = itk.vnl_vector.D()
            v.set_size(3)
            v.put(0, rotation[col, 0])
            v.put(1, rotation[col, 1])
            v.put(2, rotation[col, 2])
            dv.set_column(col, v)

    def set_direction_from_transformation_matrix(self, image):
        m = self.transformationMatrix

        # Set direction by modifying default orientation in place
        d = image.GetDirection()
        dv = d.GetVnlMatrix()
        for col in range(3):
            v = itk.vnl_vector.D()
            v.set_size(3)
            v.put(0, m[2 - col, 2])
            v.put(1, m[2 - col, 1])
            v.put(2, m[2 - col, 0])
            dv.set_column(col, v)

    def get_image_from_numpy(self, image):
        """Returns an itk Image created from the supplied scipy ndarray.

        If the image_type is supported, will be automatically transformed to that type,
        otherwise the most suitable is selected.

        Note: always use this instead of directly the itk.PyBuffer, as that
                object transposes the image axes.

        Args:
            image an array, type image np.ndarray

        Returns:
            an instance of itk.Image holding the array's data, type itk.Image (instance)
        """

        def itkMatrix_from_orientation(orientation, normal):
            o_t = orientation.reshape((2, 3)).T
            colr = o_t[:, 0].reshape((3, 1))
            colc = o_t[:, 1].reshape((3, 1))
            coln = normal.reshape((3, 1))
            if len(self.shape) < 3:
                m = np.hstack((colr[:2], colc[:2])).reshape((2, 2))
            else:
                m = np.hstack((colr, colc, coln)).reshape((3, 3))
            return m

        image.SetDirection(
            itkMatrix_from_orientation(
                self.orientation, self.normal))

        z, y, x = self.imagePositions[0]
        logger.debug("get_image_from_numpy: (z,y,x)=({},{},{}) ({})".format(z, y, x, type(z)))
        if isinstance(z, np.int64):
            logger.debug("get_image_from_numpy: SetOrigin int")
            if len(self.shape) < 3:
                image.SetOrigin([int(x), int(y)])
            else:
                image.SetOrigin([int(x), int(y), int(z)])
        else:
            logger.debug("get_image_from_numpy: SetOrigin float")
            if len(self.shape) < 3:
                image.SetOrigin([float(x), float(y)])
            else:
                image.SetOrigin([float(x), float(y), float(z)])

        logger.debug("get_image_from_numpy: SetSpacing float")
        dz, dy, dx = self.spacing
        dx = float(dx)
        dy = float(dy)
        dz = float(dz)
        if len(self.shape) < 3:
            image.SetSpacing([dx, dy])
        else:
            image.SetSpacing([dx, dy, dz])

        return image

    @staticmethod
    def _get_image_type(image):
        """Returns the image type of the supplied image as itk.Image template.

        Args:
            image: an instance of itk.Image

        Returns:
            a template of itk.Image, type itk.Image
        """
        try:
            return itk.Image[itk.template(image)[1][0],
                             itk.template(image)[1][1]]
        except IndexError:
            raise (NotImplementedError,
                   'The python wrappers of ITK define no template class for this data type.')
