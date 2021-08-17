"""Read/Write Nifti-1 files
"""

# Copyright (c) 2013-2018 Erling Andersen, Haukeland University Hospital, Bergen, Norway

import os.path
import tempfile
import logging
import numpy as np
import imagedata.formats
import imagedata.axis
from imagedata.formats.abstractplugin import AbstractPlugin
import nibabel
import nibabel.spatialimages

logger = logging.getLogger(__name__)


class NoInputFile(Exception):
    pass


class FilesGivenForMultipleURLs(Exception):
    pass


# noinspection PyUnresolvedReferences
class NiftiPlugin(AbstractPlugin):
    """Read/write Nifti-1 files."""

    name = "nifti"
    description = "Read and write Nifti-1 files."
    authors = "Erling Andersen"
    version = "1.0.0"
    url = "www.helse-bergen.no"

    """
    data - getter and setter - NumPy array
    read() method
    write() method
    """

    def __init__(self):
        super(NiftiPlugin, self).__init__(self.name, self.description,
                                          self.authors, self.version, self.url)
        self.shape = None
        self.slices = None
        self.spacing = None
        self.transformationMatrix = None
        self.imagePositions = None
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

        logger.debug("niftiplugin::read filehandle {}".format(f))
        # TODO: Read nifti directly from open file object
        #      Should be able to do something like:
        #
        # with archive.open(member_name) as member:
        #    # Create a nibabel image using
        #    # the existing file handle.
        #    fmap = nibabel.nifti1.Nifti1Image.make_file_map()
        #    #nibabel.nifti1.Nifti1Header
        #    fmap['image'].fileobj = member
        #    img = nibabel.Nifti1Image.from_file_map(fmap)
        #
        logger.debug("niftiplugin::read load f {}".format(f))
        try:
            img = nibabel.load(f)
        except nibabel.spatialimages.ImageFileError:
            raise imagedata.formats.NotImageError(
                '{} does not look like a nifti file.'.format(f))
        except Exception:
            raise
        info = img.header
        si = self._reorder_to_dicom(img.get_data(), flip=False, flipud=True)
        return info, si

    def _need_local_file(self):
        """Do the plugin need access to local files?

        Returns:
            Boolean:
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

        info, si = image_list[0]
        _data_shape = info.get_data_shape()
        nt = nz = 1
        nx, ny = _data_shape[:2]
        if len(_data_shape) > 2:
            nz = _data_shape[2]
        if len(_data_shape) > 3:
            nt = _data_shape[3]
        logger.debug("_set_tags: ny {}, nx {}, nz {}, nt {}".format(ny, nx, nz, nt))
        logger.debug('NiftiPlugin.read: get_qform\n{}'.format(info.get_qform()))
        logger.debug('NiftiPlugin.read: info.get_zooms() {}'.format(info.get_zooms()))
        _xyzt_units = info.get_xyzt_units()
        _data_zooms = info.get_zooms()
        _dim_info = info.get_dim_info()
        logger.debug("_set_tags: get_dim_info(): {}".format(info.get_dim_info()))
        logger.debug("_set_tags: get_xyzt_units(): {}".format(info.get_xyzt_units()))
        dt = dz = 1
        dx, dy = _data_zooms[:2]
        if len(_data_zooms) > 2:
            dz = _data_zooms[2]
        if len(_data_zooms) > 3:
            dt = _data_zooms[3]
        if _xyzt_units[0] == 'meter':
            dx, dy, dz = dx * 1000., dy * 1000., dz * 1000.
        elif _xyzt_units[0] == 'micron':
            dx, dy, dz = dx / 1000., dy / 1000., dz / 1000.
        if _xyzt_units[1] == 'msec':
            dt = dt / 1000.
        elif _xyzt_units[1] == 'usec':
            dt = dt / 1000000.
        self.spacing = (float(dz), float(dy), float(dx))
        hdr['spacing'] = (float(dz), float(dy), float(dx))

        # Simplify shape
        self._reduce_shape(si)

        sform, scode = info.get_sform(coded=True)
        qform, qcode = info.get_qform(coded=True)
        qfac = info['pixdim'][0]
        if qfac not in (-1, 1):
            raise ValueError('qfac (pixdim[0]) should be 1 or -1')

        # Image orientation and positions
        hdr['imagePositions'] = {}
        if sform is not None and scode != 0:
            logger.debug("Method 3 - sform: orientation")

            # Note: rz, ry, rx, cz, cy, cx
            iop = np.array([
                sform[2, 0] / dx,
                - sform[1, 0] / dx,  # NIfTI is RAS+, DICOM is LPS+
                - sform[0, 0] / dx,  # NIfTI is RAS+, DICOM is LPS+

                sform[2, 1] / dy,
                - sform[1, 1] / dy,  # NIfTI is RAS+, DICOM is LPS+
                - sform[0, 1] / dy  # NIfTI is RAS+, DICOM is LPS+
                # - sform[2,1] / dy,
                #  sform[1,1] / dy,     # NIfTI is RAS+, DICOM is LPS+
                #  sform[0,1] / dy      # NIfTI is RAS+, DICOM is LPS+
            ])
            for _slice in range(nz):
                _p = np.array([
                    - (sform[0, 2] * _slice + sform[0, 3]),  # NIfTI is RAS+, DICOM is LPS+
                    - (sform[1, 2] * _slice + sform[1, 3]),  # NIfTI is RAS+, DICOM is LPS+
                    (sform[2, 2] * _slice + sform[2, 3])
                ])
                hdr['imagePositions'][_slice] = _p[::-1]  # Reverse x,y,z

        elif qform is not None and qcode != 0:
            logger.debug("Method 2 - qform: orientation")
            qoffset_x, qoffset_y, qoffset_z = qform[0:3, 3]
            a, b, c, d = info.get_qform_quaternion()

            rx = - (a * a + b * b - c * c - d * d)
            ry = - (2 * b * c + 2 * a * d)
            rz = (2 * b * d - 2 * a * c)

            cx = - (2 * b * c - 2 * a * d)
            cy = - (a * a + c * c - b * b - d * d)
            cz = (2 * c * d + 2 * a * b)

            # normal from quaternion derived once and saved for position calculation ... do not handle qfac here ... do it later
            tx = - (2 * b * d + 2 * a * c)  # NIfTI is RAS+, DICOM is LPS+
            ty = - (2 * c * d - 2 * a * b)  # NIfTI is RAS+, DICOM is LPS+
            tz = (a * a + d * d - c * c - b * b)

            iop = np.array([rz, ry, rx, cz, cy, cx])
            for _slice in range(nz):
                _p = np.array([
                    tx * qfac * dz * _slice - qoffset_x,  # NIfTI is RAS+, DICOM is LPS+
                    ty * qfac * dz * _slice - qoffset_y,  # NIfTI is RAS+, DICOM is LPS+
                    tz * qfac * dz * _slice + qoffset_z
                ])
                hdr['imagePositions'][_slice] = _p[::-1]  # Reverse x,y,z
        else:
            logger.debug("Method 1 - assume axial: orientation")
            iop = np.array([0, 0, 1, 0, 1, 0])
            for _slice in range(nz):
                _p = np.array([
                    0,  # NIfTI is RAS+, DICOM is LPS+
                    0,  # NIfTI is RAS+, DICOM is LPS+
                    dz * _slice
                ])
                hdr['imagePositions'][_slice] = _p[::-1]  # Reverse x,y,z
        hdr['orientation'] = iop

        self.shape = si.shape

        times = [0]
        if nt > 1:
            times = np.arange(0, nt * dt, dt)
        assert len(times) == nt, "Wrong timeline calculated (times={}) (nt={})".format(len(times), nt)
        logger.debug("_set_tags: times {}".format(times))
        tags = {}
        for z in range(nz):
            tags[z] = np.array(times)
        hdr['tags'] = tags

        axes = list()
        if si.ndim > 3:
            axes.append(imagedata.axis.UniformLengthAxis(
                imagedata.formats.input_order_to_dirname_str(hdr['input_order']),
                0,
                nt,
                dt)
            )
        if si.ndim > 2:
            axes.append(imagedata.axis.UniformLengthAxis(
                'slice',
                0,
                nz,
                dz)
            )
        axes.append(imagedata.axis.UniformLengthAxis(
            'row',
            0,
            ny,
            dy)
        )
        axes.append(imagedata.axis.UniformLengthAxis(
            'column',
            0,
            nx,
            dx)
        )
        hdr['axes'] = axes

        hdr['photometricInterpretation'] = 'MONOCHROME2'
        hdr['color'] = False

    # def nifti_to_affine(self, affine, shape):
    #
    #     if len(shape) != 4:
    #         raise ValueError("4D only (was: %dD)" % len(shape))
    #
    #     q = affine.copy()
    #
    #     logger.debug("q from nifti_to_affine():\n{}".format(q))
    #     # Swap row 0 (z) and 2 (x)
    #     q[[0, 2],:] = q[[2, 0],:]
    #     # Swap column 0 (z) and 2 (x)
    #     q[:,[0, 2]] = q[:,[2, 0]]
    #     logger.debug("q swap nifti_to_affine():\n{}".format(q))
    #
    #     analyze_to_dicom = np.eye(4)
    #     analyze_to_dicom[0,3] = 1
    #     analyze_to_dicom[1,3] = 1
    #     analyze_to_dicom[2,3] = 1
    #     dicom_to_analyze = np.linalg.inv(analyze_to_dicom)
    #     q = np.dot(q,dicom_to_analyze)
    #     logger.debug("q after dicom_to_analyze:\n{}".format(q))
    #
    #     analyze_to_dicom = np.eye(4)
    #     analyze_to_dicom[0,3] = -1
    #     analyze_to_dicom[1,1] = -1
    #     rows = shape[2]
    #     analyze_to_dicom[1,3] = rows
    #     analyze_to_dicom[2,3] = -1
    #     dicom_to_analyze = np.linalg.inv(analyze_to_dicom)
    #     q = np.dot(q,dicom_to_analyze)
    #     logger.debug("q after rows dicom_to_analyze:\n{}".format(q))
    #
    #     patient_to_tal = np.eye(4)
    #     patient_to_tal[0,0] = -1
    #     patient_to_tal[1,1] = -1
    #     tal_to_patient = np.linalg.inv(patient_to_tal)
    #     q = np.dot(tal_to_patient,q)
    #     logger.debug("q after tal_to_patient:\n{}".format(q))
    #
    #     return q

    # def affine_to_nifti(self, shape):
    #     q = self.transformationMatrix.copy()
    #     logger.debug("Affine from self.transformationMatrix:\n{}".format(q))
    #     # Swap row 0 (z) and 2 (x)
    #     q[[0, 2],:] = q[[2, 0],:]
    #     # Swap column 0 (z) and 2 (x)
    #     q[:,[0, 2]] = q[:,[2, 0]]
    #     logger.debug("Affine swap self.transformationMatrix:\n{}".format(q))
    #
    #     # q now equals dicom_to_patient in spm_dicom_convert
    #
    #     # Convert space
    #     analyze_to_dicom = np.eye(4)
    #     analyze_to_dicom[0,3] = -1
    #     analyze_to_dicom[1,1] = -1
    #     #if len(shape) == 3:
    #     #    rows = shape[1]
    #     #else:
    #     #    rows = shape[2]
    #     rows = shape[-2]
    #     analyze_to_dicom[1,3] = rows
    #     analyze_to_dicom[2,3] = -1
    #     logger.debug("analyze_to_dicom:\n{}".format(analyze_to_dicom))
    #
    #     patient_to_tal = np.eye(4)
    #     patient_to_tal[0,0] = -1
    #     patient_to_tal[1,1] = -1
    #     logger.debug("patient_to_tal:\n{}".format(patient_to_tal))
    #
    #     q = np.dot(patient_to_tal,q)
    #     logger.debug("q with patient_to_tal:\n{}".format(q))
    #     q = np.dot(q,analyze_to_dicom)
    #     # q now equals mat in spm_dicom_convert
    #
    #     analyze_to_dicom = np.eye(4)
    #     analyze_to_dicom[0,3] = 1
    #     analyze_to_dicom[1,3] = 1
    #     analyze_to_dicom[2,3] = 1
    #     logger.debug("analyze_to_dicom:\n{}".format(analyze_to_dicom))
    #     q = np.dot(q,analyze_to_dicom)
    #
    #     logger.debug("q nifti:\n{}".format(q))
    #     return q

    @staticmethod
    def _get_geometry_from_affine(hdr, q):
        """Extract geometry attributes from Nifti header

        Args:
            self: NiftiPlugin instance
            q: nifti Qform
            hdr['spacing']
        Returns:
            hdr: header dict
                - hdr['imagePositions'][0]
                - hdr['orientation']
                - hdr['transformationMatrix']
        """

        # Swap back from nifti patient space, flip x and y directions
        affine = np.dot(np.diag([-1, -1, 1, 1]), q)
        # Set imagePositions for first slice
        x, y, z = affine[0:3, 3]
        hdr['imagePositions'] = {0: np.array([z, y, x])}
        logger.debug("getGeometryFromAffine: hdr imagePositions={}".format(hdr['imagePositions']))
        # Set slice orientation
        ds, dr, dc = hdr['spacing']
        logger.debug("getGeometryFromAffine: spacing ds {}, dr {}, dc {}".format(ds, dr, dc))

        colr = affine[:3, 0][::-1] / dr
        colc = affine[:3, 1][::-1] / dc
        # T0 = affine[:3,3][::-1]
        orient = []
        logger.debug("getGeometryFromAffine: affine\n{}".format(affine))
        for i in range(3):
            orient.append(colc[i])
        for i in range(3):
            orient.append(colr[i])
        logger.debug("getGeometryFromAffine: orient {}".format(orient))
        hdr['orientation'] = orient
        return

    # noinspection PyPep8Naming
    def create_affine_xyz(self):
        """Create affine in xyz.
        """

        def normalize(v):
            """Normalize a vector

            https://stackoverflow.com/questions/21030391/how-to-normalize-an-array-in-numpy

            Args:
                v: 3D vector
            Returns:
                normalized 3D vector
            """
            norm = np.linalg.norm(v, ord=1)
            if norm == 0:
                norm = np.finfo(v.dtype).eps
            return v / norm

        ds, dr, dc = self.spacing
        colr = normalize(np.array(self.orientation[3:6])).reshape((3,)) * [-1, -1, 1]
        colc = normalize(np.array(self.orientation[0:3])).reshape((3,)) * [-1, -1, 1]
        # T0 = self.imagePositions[0][::-1].reshape(3, )  # x,y,z
        if self.slices > 1:
            # Tn = self.imagePositions[self.slices - 1][::-1].reshape(3, )  # x,y,z
            # k = Tn
            k = np.cross(colc, colr, axis=0)
            k = k * ds
        else:
            k = np.cross(colc, colr, axis=0)
            k = k * ds

        L = np.zeros((4, 4))
        L[:3, 1] = colr * dr
        L[:3, 0] = colc * dc
        L[:3, 2] = k
        L[:3, 3] = self.origin * [-1, -1, 1]
        L[3, 3] = 1
        return L

    # def getQformFromTransformationMatrix(self):
    #     # def matrix_from_orientation(orientation, normal):
    #     #    oT = orientation.reshape((2,3)).T
    #     #    colr = oT[:,0].reshape((3,1))
    #     #    colc = oT[:,1].reshape((3,1))
    #     #    coln = normal.reshape((3,1))
    #     #    if len(self.shape) < 3:
    #     #        M = np.hstack((colr[:2], colc[:2])).reshape((2,2))
    #     #    else:
    #     #        M = np.hstack((colr, colc, coln)).reshape((3,3))
    #     #    return M
    #
    #     def normalize(v):
    #         """Normalize a vector
    #
    #         https://stackoverflow.com/questions/21030391/how-to-normalize-an-array-in-numpy
    #
    #         :param v: 3D vector
    #         :return: normalized 3D vector
    #         """
    #         norm = np.linalg.norm(v, ord=1)
    #         if norm == 0:
    #             norm = np.finfo(v.dtype).eps
    #         return v / norm
    #
    #     def L_from_orientation(orientation, normal, spacing):
    #         """
    #         orientation: row, then column index direction cosines
    #         """
    #         _ds, _dr, _dc = spacing
    #         _colr = normalize(np.array(orientation[3:6])).reshape((3,))
    #         _colc = normalize(np.array(orientation[0:3])).reshape((3,))
    #         _t0 = self.imagePositions[0][::-1].reshape(3, )  # x,y,z
    #         if self.slices > 1:
    #             _tn = self.imagePositions[self.slices - 1][::-1].reshape(3, )  # x,y,z
    #             # k = _tn
    #             _k = np.cross(_colr, _colc, axis=0)
    #             _k = _k * _ds
    #         else:
    #             _k = np.cross(_colr, _colc, axis=0)
    #             _k = _k * _ds
    #
    #         _L = np.zeros((4, 4))
    #         _L[:3, 0] = _t0[:]
    #         _L[3, 0] = 1
    #         _L[:3, 1] = _k
    #         _L[3, 1] = 1 if self.slices > 1 else 0
    #         _L[:3, 2] = _colr * [-1, -1, 1] * _dr
    #         _L[:3, 3] = _colc * [-1, -1, 1] * _dc
    #         return _L
    #
    #     # M = self.transformationMatrix
    #     # M = matrix_from_orientation(self.orientation, self.normal)
    #     # ipp = self.origin
    #     # q = np.array([[M[2,2], M[2,1], M[2,0], ipp[0]],
    #     #              [M[1,2], M[1,1], M[1,0], ipp[1]],
    #     #              [M[0,2], M[0,1], M[0,0], ipp[2]],
    #     #              [     0,      0,      0,     1 ]]
    #     #        )
    #
    #     if self.slices > 1:
    #         r = np.array([[1, 1, 1, 0], [1, 1, 0, 1], [1, self.slices, 0, 0], [1, 1, 0, 0]])
    #     else:
    #         r = np.array([[1, 0, 1, 0], [1, 0, 0, 1], [1, self.slices, 0, 0], [1, 0, 0, 0]])
    #     l = L_from_orientation(self.orientation, self.normal, self.spacing)
    #
    #     # Linv = np.linalg.inv(L)
    #     # Aspm = np.dot(r, np.linalg.inv(l))
    #     to_ones = np.eye(4)
    #     to_ones[:, 3] = 1
    #     # A = np.dot(Aspm, to_ones)
    #
    #     ds, dr, dc = self.spacing
    #     colr = normalize(np.array(self.orientation[3:6])).reshape((3,))
    #     colc = normalize(np.array(self.orientation[0:3])).reshape((3,))
    #     coln = normalize(np.cross(colc, colr, axis=0))
    #     t_0 = self.imagePositions[0][::-1].reshape(3, )  # x,y,z
    #     if self.slices > 1:
    #         t_n = self.imagePositions[self.slices - 1][::-1].reshape((3,))  # x,y,z
    #         abcd = np.array([1, 1, self.slices, 1]).reshape((4,))
    #         one = np.ones((1,))
    #         efgh = np.concatenate((t_n, one))
    #     else:
    #         abcd = np.array([0, 0, 1, 0]).reshape((4,))
    #         # zero = np.zeros((1,))
    #         efgh = np.concatenate((n * ds, zeros))
    #
    #     # From derivations/spm_dicom_orient.py
    #
    #     # premultiplication matrix to go from 0 to 1 based indexing
    #     one_based = np.eye(4)
    #     one_based[:3, 3] = (1, 1, 1)
    #     # premult for swapping row and column indices
    #     row_col_swap = np.eye(4)
    #     row_col_swap[:, 0] = np.eye(4)[:, 1]
    #     row_col_swap[:, 1] = np.eye(4)[:, 0]
    #
    #     # various worming matrices
    #     orient_pat = np.hstack([colr.reshape(3, 1), colc.reshape(3, 1)])
    #     orient_cross = coln
    #     pos_pat_0 = t_0
    #     if self.slices > 1:
    #         missing_r_col = (t_0 - t_n) / (1 - self.slices)
    #         pos_pat_N = t_n
    #     pixel_spacing = [dr, dc]
    #     NZ = self.slices
    #     slice_thickness = ds
    #
    #     R3 = np.dot(orient_pat, np.diag(pixel_spacing))
    #     # R3 = orient_pat * np.diag(pixel_spacing)
    #     r = np.zeros((4, 2))
    #     r[:3, :] = R3
    #
    #     # The following is specific to the SPM algorithm.
    #     x1 = np.ones(4)
    #     y1 = np.ones(4)
    #     y1[:3] = pos_pat_0
    #
    #     to_inv = np.zeros((4, 4))
    #     to_inv[:, 0] = x1
    #     to_inv[:, 1] = abcd
    #     to_inv[0, 2] = 1
    #     to_inv[1, 3] = 1
    #     inv_lhs = np.zeros((4, 4))
    #     inv_lhs[:, 0] = y1
    #     inv_lhs[:, 1] = efgh
    #     inv_lhs[:, 2:] = r
    #
    #     def spm_full_matrix(x2, y2):
    #         rhs = to_inv[:, :]
    #         rhs[:, 1] = x2
    #         lhs = inv_lhs[:, :]
    #         lhs[:, 1] = y2
    #         return np.dot(lhs, np.linalg.inv(rhs))
    #
    #     if self.slices > 1:
    #         x2_ms = np.array([1, 1, NZ, 1])
    #         y2_ms = np.ones((4,))
    #         y2_ms[:3] = pos_pat_N
    #         A_ms = spm_full_matrix(x2_ms, y2_ms)
    #         A = A_ms
    #     else:
    #         orient = np.zeros((3, 3))
    #         orient[:3, :2] = orient_pat
    #         orient[:, 2] = orient_cross
    #         x2_ss = np.array([0, 0, 1, 0])
    #         y2_ss = np.zeros((4,))
    #         # y2_ss[:3] = orient * np.array([0, 0, slice_thickness])
    #         y2_ss[:3] = np.dot(orient, np.array([0, 0, slice_thickness]))
    #         A_ss = spm_full_matrix(x2_ss, y2_ss)
    #         A = A_ss
    #
    #     A = np.dot(A, row_col_swap)
    #
    #     multi_aff = np.eye(4)
    #     multi_aff[:3, :2] = R3
    #     trans_z_N = np.array([0, 0, self.slices - 1, 1])
    #     multi_aff[:3, 2] = missing_r_col
    #     multi_aff[:3, 3] = pos_pat_0
    #     # est_pos_pat_N = np.dot(multi_aff, trans_z_N)
    #
    #     # Flip voxels in y
    #     analyze_to_dicom = np.eye(4)
    #     analyze_to_dicom[1, 1] = -1
    #     # analyze_to_dicom[1,3] = shape[1]+1
    #     analyze_to_dicom[1, 3] = self.slices
    #     logger.debug("getQformFromTransformationMatrix: analyze_to_dicom\n{}".format(analyze_to_dicom))
    #     # dicom_to_analyze = np.linalg.inv(analyze_to_dicom)
    #     # q = np.dot(q,dicom_to_analyze)
    #     q = np.dot(A, analyze_to_dicom)
    #     # ## 2019.07.03 # q = np.dot(q,analyze_to_dicom)
    #     # ## 2019.07.03 # logger.debug("q after rows dicom_to_analyze:\n{}".format(q))
    #     # Flip mm coords in x and y directions
    #     patient_to_tal = np.diag([1, -1, -1, 1])
    #     # patient_to_tal = np.eye(4)
    #     # patient_to_tal[0,0] = -1
    #     # patient_to_tal[1,1] = -1
    #     # tal_to_patient = np.linalg.inv(patient_to_tal)
    #     # q = np.dot(tal_to_patient,q)
    #     logger.debug("getQformFromTransformationMatrix: patient_to_tal\n{}".format(patient_to_tal))
    #     q = np.dot(patient_to_tal, q)
    #     logger.debug("getQformFromTransformationMatrix: q after\n{}".format(q))
    #
    #     return q

    # def create_affine(self, sorted_dicoms):
    #     """
    #     Function to generate the affine matrix for a dicom series
    #     From dicom2nifti:common.py: https://github.com/icometrix/dicom2nifti/blob/master/dicom2nifti/common.py
    #     This method was based on (http://nipy.org/nibabel/dicom/dicom_orientation.html)
    #     :param sorted_dicoms: list with sorted dicom files
    #     """
    #
    #     # Create affine matrix (http://nipy.sourceforge.net/nibabel/dicom/dicom_orientation.html#dicom-slice-affine)
    #     image_orient1 = np.array(sorted_dicoms[0].ImageOrientationPatient)[0:3]
    #     image_orient2 = np.array(sorted_dicoms[0].ImageOrientationPatient)[3:6]
    #
    #     delta_r = float(sorted_dicoms[0].PixelSpacing[0])
    #     delta_c = float(sorted_dicoms[0].PixelSpacing[1])
    #
    #     image_pos = np.array(sorted_dicoms[0].ImagePositionPatient)
    #
    #     last_image_pos = np.array(sorted_dicoms[-1].ImagePositionPatient)
    #
    #     if len(sorted_dicoms) == 1:
    #         # Single slice
    #         step = [0, 0, -1]
    #     else:
    #         step = (image_pos - last_image_pos) / (1 - len(sorted_dicoms))
    #
    #     # check if this is actually a volume and not all slices on the same location
    #     if np.linalg.norm(step) == 0.0:
    #         raise imagedata.formats.NotImageError("Not a volume")
    #
    #     affine = np.array(
    #         [[-image_orient1[0] * delta_c, -image_orient2[0] * delta_r, -step[0], -image_pos[0]],
    #          [-image_orient1[1] * delta_c, -image_orient2[1] * delta_r, -step[1], -image_pos[1]],
    #          [image_orient1[2] * delta_c, image_orient2[2] * delta_r, step[2], image_pos[2]],
    #          [0, 0, 0, 1]]
    #     )
    #     return affine, np.linalg.norm(step)

    def write_3d_numpy(self, si, destination, opts):
        """Write 3D numpy image as Nifti file

        Args:
            self: NiftiPlugin instance
            si: Series array (3D or 4D), including these attributes:
            -   slices,
            -   spacing,
            -   imagePositions,
            -   transformationMatrix,
            -   orientation,
            -   tags

            destination: dict of archive and filenames
            opts: Output options (dict)
        """

        if si.color:
            raise imagedata.formats.WriteNotImplemented(
                "Writing color Nifti images not implemented.")

        logger.debug('NiftiPlugin.write_3d_numpy: destination {}'.format(destination))
        archive = destination['archive']
        filename_template = 'Image.nii.gz'
        if len(destination['files']) > 0 and len(destination['files'][0]) > 0:
            filename_template = destination['files'][0]

        self.shape = si.shape
        self.slices = si.slices
        self.spacing = si.spacing
        self.transformationMatrix = si.transformationMatrix
        self.imagePositions = si.imagePositions
        self.tags = si.tags
        self.origin, self.orientation, self.normal = si.get_transformation_components_xyz()

        logger.info("Data shape write: {}".format(imagedata.formats.shape_to_str(si.shape)))
        assert si.ndim == 2 or si.ndim == 3, "write_3d_series: input dimension %d is not 3D." % si.ndim

        fsi = self._reorder_from_dicom(si, flip=False, flipud=True)
        shape = fsi.shape

        affine_xyz = self.create_affine_xyz()
        nifti_header = nibabel.Nifti1Header()
        nifti_header.set_dim_info(freq=0, phase=1, slice=2)
        nifti_header.set_data_shape(shape)
        dz, dy, dx = self.spacing
        if si.ndim < 3:
            nifti_header.set_zooms((dx, dy))
        else:
            nifti_header.set_zooms((dx, dy, dz))
        nifti_header.set_data_dtype(fsi.dtype)
        nifti_header.set_sform(affine_xyz, code=1)
        nifti_header.set_xyzt_units(xyz='mm')
        img = nibabel.Nifti1Image(fsi, None, nifti_header)
        try:
            filename = filename_template % 0
        except TypeError:
            filename = filename_template
        self.write_numpy_nifti(img, archive, filename)

    def write_4d_numpy(self, si, destination, opts):
        """Write 4D numpy image as Nifti file

        Args:
            self: NiftiPlugin instance
            si[tag,slice,rows,columns]: Series array, including these attributes:
            -   slices,
            -   spacing,
            -   imagePositions,
            -   transformationMatrix,
            -   orientation,
            -   tags

            destination: dict of archive and filenames
            opts: Output options (dict)
        """

        if si.color:
            raise imagedata.formats.WriteNotImplemented(
                "Writing color Nifti images not implemented.")

        logger.debug('ITKPlugin.write_4d_numpy: destination {}'.format(destination))
        archive = destination['archive']
        filename_template = 'Image.nii.gz'
        if len(destination['files']) > 0 and len(destination['files'][0]) > 0:
            filename_template = destination['files'][0]

        self.shape = si.shape
        self.slices = si.slices
        self.spacing = si.spacing
        self.transformationMatrix = si.transformationMatrix
        self.imagePositions = si.imagePositions
        self.tags = si.tags
        self.origin, self.orientation, self.normal = si.get_transformation_components_xyz()

        # Defaults
        self.output_sort = imagedata.formats.SORT_ON_SLICE
        if 'output_sort' in opts:
            self.output_sort = opts['output_sort']

        # Should we allow to write 3D volume?
        if si.ndim == 2:
            si.shape = (1, 1,) + si.shape
        elif si.ndim == 3:
            si.shape = (1,) + si.shape
        if si.ndim != 4:
            raise ValueError("write_4d_numpy: input dimension {} is not 4D.".format(si.ndim))

        logger.debug("write_4d_numpy: si dtype {}, shape {}, sort {}".format(
            si.dtype, si.shape,
            imagedata.formats.sort_on_to_str(self.output_sort)))

        steps = si.shape[0]
        slices = si.shape[1]
        if steps != len(si.tags[0]):
            raise ValueError(
                "write_4d_series: tags of dicom template ({}) differ from input array ({}).".format(len(si.tags[0]),
                                                                                                    steps))
        if slices != si.slices:
            raise ValueError(
                "write_4d_series: slices of dicom template ({}) differ from input array ({}).".format(si.slices,
                                                                                                      slices))

        fsi = self._reorder_from_dicom(si, flip=False, flipud=True)
        shape = fsi.shape

        affine_xyz = self.create_affine_xyz()
        nifti_header = nibabel.Nifti1Header()
        nifti_header.set_dim_info(freq=0, phase=1, slice=2)
        nifti_header.set_data_shape(shape)
        dz, dy, dx = self.spacing
        nifti_header.set_zooms((dx, dy, dz, 1))
        nifti_header.set_data_dtype(fsi.dtype)
        nifti_header.set_sform(affine_xyz, code=1)
        # NiftiHeader.set_slice_duration()
        # NiftiHeader.set_slice_times(times)
        nifti_header.set_xyzt_units(xyz='mm', t='sec')
        img = nibabel.Nifti1Image(fsi, None, nifti_header)
        try:
            filename = filename_template % 0
        except TypeError:
            filename = filename_template
        self.write_numpy_nifti(img, archive, filename)

    @staticmethod
    def write_numpy_nifti(img, archive, filename):
        """Write nifti data to file

        Args:
            self: ITKPlugin instance, including these attributes:
            - slices (not used)
            - spacing
            - imagePositions
            - transformationMatrix
            - orientation (not used)
            - tags (not used)

            img: Nifti1Image
            archive: archive object
            filename: file name, possibly without extentsion
        """

        if len(os.path.splitext(filename)[1]) == 0:
            filename = filename + '.nii.gz'
        ext = os.path.splitext(filename)[1]
        if filename.endswith('.nii.gz'):
            ext = '.nii.gz'
        logger.debug('write_numpy_nifti: ext %s' % ext)

        f = tempfile.NamedTemporaryFile(
            suffix=ext, delete=False)
        logger.debug('write_numpy_nifti: write local file %s' % f.name)
        img.to_filename(f.name)
        f.close()
        logger.debug('write_numpy_nifti: copy to file %s' % filename)
        _ = archive.add_localfile(f.name, filename)
        os.unlink(f.name)
