"""Read/Write Nifti-1 files
"""

# Copyright (c) 2013-2022 Erling Andersen, Haukeland University Hospital, Bergen, Norway

import os.path
import tempfile
import logging
import math
import nibabel
import nibabel.spatialimages
import numpy as np
from . import NotImageError, WriteNotImplemented, input_order_to_dirname_str,\
    shape_to_str, sort_on_to_str,\
    SORT_ON_SLICE
from ..axis import UniformLengthAxis
from .abstractplugin import AbstractPlugin

# import nitransforms

logger = logging.getLogger(__name__)

NIFTI_XFORM_UNKNOWN = 0
NIFTI_XFORM_SCANNER_ANAT = 1
NIFTI_XFORM_ALIGNED_ANAT = 2
NIFTI_XFORM_TALAIRACH = 3
NIFTI_XFORM_MNI_152 = 4


class NoInputFile(Exception):
    pass


class FilesGivenForMultipleURLs(Exception):
    pass


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
            hdr: Header
        Returns:
            Tuple of
                hdr: Header
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
            raise NotImageError(
                '{} does not look like a nifti file.'.format(f))
        except Exception:
            raise
        info = img
        si = self._reorder_to_dicom(
            np.asanyarray(img.dataobj),
            flip=False,
            flipud=True)
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
            image_list: list with (img,si) tuples
            hdr: Header
            si: numpy array (multi-dimensional)
        Returns:
            hdr: Header
        """

        img, si = image_list[0]
        info = img.header
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
        # _dim_info = info.get_dim_info()
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
        hdr.spacing = (float(dz), float(dy), float(dx))

        # Simplify shape
        self._reduce_shape(si)

        sform, scode = info.get_sform(coded=True)
        qform, qcode = info.get_qform(coded=True)
        qfac = info['pixdim'][0]
        if qfac not in (-1, 1):
            raise ValueError('qfac (pixdim[0]) should be 1 or -1')

        # Image orientation and positions
        hdr.imagePositions = {}
        if sform is not None and scode != 0:
            logger.debug("Method 3 - sform: orientation")

            for c in range(4):  # NIfTI is RAS+, DICOM is LPS+
                for r in range(2):
                    sform[r, c] = - sform[r, c]
            Q = sform[:3, :3]
            # p = sform[:3, 3]
            p = nibabel.affines.apply_affine(sform, (0, ny - 1, 0))
            if np.linalg.det(Q) < 0:
                Q[:3, 1] = - Q[:3, 1]
            # Note: rz, ry, rx, cz, cy, cx
            iop = np.array([
                Q[2, 0] / dx, Q[1, 0] / dx, Q[0, 0] / dx,
                Q[2, 1] / dy, Q[1, 1] / dy, Q[0, 1] / dy
            ])

            for _slice in range(nz):
                _p = np.array([
                    (Q[0, 2] * _slice + p[0]),  # NIfTI is RAS+, DICOM is LPS+
                    (Q[1, 2] * _slice + p[1]),
                    (Q[2, 2] * _slice + p[2])
                ])
                hdr.imagePositions[_slice] = _p[::-1]

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

            # normal from quaternion derived once and saved for position calculation ...
            # ... do not handle qfac here ... do it later
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
                hdr.imagePositions[_slice] = _p[::-1]  # Reverse x,y,z
        else:
            logger.debug("Method 1 - assume axial: orientation")
            iop = np.array([0, 0, 1, 0, 1, 0])
            for _slice in range(nz):
                _p = np.array([
                    0,  # NIfTI is RAS+, DICOM is LPS+
                    0,  # NIfTI is RAS+, DICOM is LPS+
                    dz * _slice
                ])
                hdr.imagePositions[_slice] = _p[::-1]  # Reverse x,y,z
        hdr.orientation = iop

        self.shape = si.shape

        times = [0]
        if nt > 1:
            times = np.arange(0, nt * dt, dt)
        assert len(times) == nt,\
            "Wrong timeline calculated (times={}) (nt={})".format(len(times), nt)
        logger.debug("_set_tags: times {}".format(times))
        tags = {}
        for z in range(nz):
            tags[z] = np.array(times)
        hdr.tags = tags

        axes = list()
        if si.ndim > 3:
            axes.append(UniformLengthAxis(
                input_order_to_dirname_str(hdr.input_order),
                0,
                nt,
                dt)
            )
        if si.ndim > 2:
            axes.append(UniformLengthAxis(
                'slice',
                0,
                nz,
                dz)
            )
        axes.append(UniformLengthAxis(
            'row',
            0,
            ny,
            dy)
        )
        axes.append(UniformLengthAxis(
            'column',
            0,
            nx,
            dx)
        )
        hdr.axes = axes

        hdr.photometricInterpretation = 'MONOCHROME2'
        hdr.color = False

        # Set dummy DicomHeaderDict
        hdr.DicomHeaderDict = {}
        for _slice in range(nz):
            hdr.DicomHeaderDict[_slice] = []
            for tag in range(nt):
                hdr.DicomHeaderDict[_slice].append(
                    (times[tag], None, hdr.empty_ds())
                )

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
            hdr.spacing
        Returns:
            hdr: header
                - hdr.imagePositions[0]
                - hdr.orientation
                - hdr.transformationMatrix
        """

        # Swap back from nifti patient space, flip x and y directions
        affine = np.dot(np.diag([-1, -1, 1, 1]), q)
        # Set imagePositions for first slice
        x, y, z = affine[0:3, 3]
        hdr.imagePositions = {0: np.array([z, y, x])}
        logger.debug("getGeometryFromAffine: hdr imagePositions={}".format(hdr.imagePositions))
        # Set slice orientation
        ds, dr, dc = hdr.spacing
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
        hdr.orientation = orient
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

        # NIfTI is RAS+, DICOM is LPS+
        colr = normalize(np.array(self.orientation[3:6])).reshape((3,)) * [1, 1, -1]
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
        L[:3, 2] = -k
        ny = self.shape[-2]
        p = self.getPositionForVoxel((0, ny - 1, 0))[::-1]
        # L[:3, 3] = self.origin * [-1, -1, 1]
        L[:3, 3] = p * [-1, -1, 1]
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
    #     logger.debug("getQformFromTransformationMatrix: analyze_to_dicom\n{}".format(
    #         analyze_to_dicom))
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
    #     logger.debug("getQformFromTransformationMatrix: patient_to_tal\n{}".format(
    #         patient_to_tal))
    #     q = np.dot(patient_to_tal, q)
    #     logger.debug("getQformFromTransformationMatrix: q after\n{}".format(q))
    #
    #     return q

    # def create_affine(self, sorted_dicoms):
    #     """
    #     Function to generate the affine matrix for a dicom series
    #     From dicom2nifti:common.py:
    #     https://github.com/icometrix/dicom2nifti/blob/master/dicom2nifti/common.py
    #     This method was based on (http://nipy.org/nibabel/dicom/dicom_orientation.html)
    #     :param sorted_dicoms: list with sorted dicom files
    #     """
    #
    #     # Create affine matrix
    #     (http://nipy.sourceforge.net/nibabel/dicom/dicom_orientation.html#dicom-slice-affine)
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
    #         raise NotImageError("Not a volume")
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
            raise WriteNotImplemented(
                "Writing color Nifti images not implemented.")

        logger.debug('NiftiPlugin.write_3d_numpy: destination {}'.format(destination))
        archive = destination['archive']
        filename_template = 'Image.nii.gz'
        if len(destination['files']) > 0 and len(destination['files'][0]) > 0:
            filename_template = destination['files'][0]

        # TODO # self._save_dicom_to_nifti(si)
        self.shape = si.shape
        self.slices = si.slices
        self.spacing = si.spacing
        self.transformationMatrix = si.transformationMatrix
        self.imagePositions = si.imagePositions
        self.tags = si.tags
        self.origin, self.orientation, self.normal = si.get_transformation_components_xyz()
        # slice_direction = _find_slice_direction(si, self.transformationMatrix, self.normal)

        logger.info("Data shape write: {}".format(shape_to_str(si.shape)))
        assert si.ndim == 2 or si.ndim == 3,\
            "write_3d_series: input dimension %d is not 3D." % si.ndim

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
            raise WriteNotImplemented(
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
        self.output_sort = SORT_ON_SLICE
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

    def _save_dicom_to_nifti(self, si):
        """Convert DICOM to Nifti"""
        hdr = nibabel.Nifti1Header()
        img = si
        if si.slices > 1:
            hdr, slice_direction = self._header_dicom_to_nifti(hdr, si)
            if slice_direction < 0:
                hdr, img = self._nii_flip_z(hdr, si)
                slice_direction = abs(slice_direction)
            img = self._nii_set_ortho(hdr, img)
        self._nii_save_attributes(si, hdr)

    def _header_dicom_to_nifti(self, hdr, si):
        # COL/ROW
        inPlanePhaseEncodingDirection = si.getDicomAttribute('InPlanePhaseEncodingDirection')
        if inPlanePhaseEncodingDirection == 'ROW':
            hdr.set_dim_info(freq=1, phase=0, slice=2)
        elif inPlanePhaseEncodingDirection == 'COL':
            hdr.set_dim_info(freq=0, phase=1, slice=2)
        slice_direction = 0
        if si.slices < 2:
            q44, slice_direction = self._nifti_dicom_mat(si)
            hdr.set_sform(q44, code=NIFTI_XFORM_UNKNOWN)
            hdr.set_qform(q44, code=NIFTI_XFORM_UNKNOWN)
        else:
            q44, slice_direction = self._nifti_dicom_mat(si)
            hdr.set_sform(q44, NIFTI_XFORM_SCANNER_ANAT)
            hdr.set_qform(q44, NIFTI_XFORM_SCANNER_ANAT)
        return hdr, slice_direction

    def _nifti_dicom_mat(self, si):
        """Create NIfTI header based on values from DICOM header"""

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

        origin, orientation, normal = si.get_transformation_components_xyz()
        spacing = si.spacing[::-1]  # x,y,z

        q = np.zeros((3, 3))
        q[0] = normalize(orientation[:3])
        q[1] = normalize(orientation[3:])
        q[2] = np.cross(q[0], q[1], axis=0)
        q = np.transpose(q)
        if np.linalg.det(q) < 0:
            q[:2, 2] = - q[:2, 2]

        diagVox = np.diag(spacing)

        q = np.matmul(q, diagVox)

        q44 = np.zeros((4, 4))
        q44[:3, :3] = q
        q44[:3, 3] = origin
        q44[3, 3] = 1

        slice_direction = self._find_slice_direction(si, q44, normal)
        for c in range(4):  # LPS to nifti RAS
            for r in range(2):  # Swap rows 0 and 1
                q44[r, c] = - q44[r, c]
        return q44, slice_direction

    def _nii_flip_z(self, hdr, si):
        """Flip slice order"""

        if si.slices < 2:
            return si
        # LOAD_MAT33(s,h->srow_x[0],h->srow_x[1],h->srow_x[2],
        #            h->srow_y[0],h->srow_y[1], h->srow_y[2],
        #            h->srow_z[0],h->srow_z[1],h->srow_z[2]);
        sform = hdr.get_sform()[:3, :3]
        # LOAD_MAT44(Q44,h->srow_x[0],h->srow_x[1],h->srow_x[2],h->srow_x[3],
        #            h->srow_y[0],h->srow_y[1],h->srow_y[2],h->srow_y[3],
        #            h->srow_z[0],h->srow_z[1],h->srow_z[2],h->srow_z[3]);
        # q44 = np.eye(4)
        # q44[:3, :3] = sform
        q44 = hdr.get_sform()
        # vec4 v= setVec4(0.0f,0.0f,(float) h->dim[3]-1.0f);
        v = np.array([0, 0, si.slices - 1, 1], dtype=float)
        # v = nifti_vect44mat44_mul(v, Q44); //after flip this voxel will be the origin
        v = np.matmul(v, q44)  # after flip this voxel will be the origin
        # mat33 mFlipZ;
        # LOAD_MAT33(mFlipZ,1.0f, 0.0f, 0.0f, 0.0f,1.0f,0.0f, 0.0f,0.0f,-1.0f);
        mFlipZ = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=float)
        # s= nifti_mat33_mul( s , mFlipZ );
        sform = np.matmul(sform, mFlipZ)
        # LOAD_MAT44(Q44, s.m[0][0],s.m[0][1],s.m[0][2],v.v[0],
        #        s.m[1][0],s.m[1][1],s.m[1][2],v.v[1],
        #        s.m[2][0],s.m[2][1],s.m[2][2],v.v[2]);
        q44[:3, :3] = sform
        q44[:, 3] = v
        # setQSForm(h,Q44, true);
        hdr.set_sform(q44, NIFTI_XFORM_SCANNER_ANAT)
        hdr.set_qform(q44, NIFTI_XFORM_SCANNER_ANAT)
        # printMessage("nii_flipImgY dims %dx%dx%d %d \n",h->dim[1],h->dim[2],
        #     dim3to7,h->bitpix/8);
        # return self._nii_flip_image_z(hdr, si)
        return hdr, self._reorder_from_dicom(si, flipud=True)

    def _nii_set_ortho(self, hdr, img):

        def isMat44Canonical(R):
            # returns true if diagonals >0 and all others =0
            #  no rotation is necessary - already in perfect orthogonal alignment
            for i in range(3):
                for j in range(3):
                    if (i == j) and (R[i, j] <= 0):
                        return False
                    if (i != j) and (R[i, j] != 0):
                        return False
            return True

        def xyz2mm(R, v):
            ret = np.zeros(3)
            for i in range(3):
                ret[i] = R[i, 0]*v[0] + R[i, 1]*v[1] + R[i, 2]*v[2] + R[i, 3]
            return ret

        def getDistance(v, _min):
            # Scalar distance between two 3D points - Pythagorean theorem
            return math.sqrt(math.pow((v[0] - _min[0]), 2) +
                             math.pow((v[1] - _min[1]), 2) +
                             math.pow((v[2] - _min[2]), 2))

        def minCornerFlip(h):
            # Orthogonal rotations and reflections applied as 3x3 matrices will cause the origin
            # to shift. A simple solution is to first compute the most left, posterior, inferior
            # voxel in the source image. This voxel will be at location i,j,k = 0,0,0, so we can
            # simply use this as the offset for the final 4x4 matrix...
            # vec3i flipVecs[8]
            # vec3 corner[8], min
            flipVecs = {}
            corner = {}
            # mat44 s = sFormMat(h);
            s = h.get_sform()
            for i in range(8):
                flipVecs[i] = np.zeros(3)
                flipVecs[i][0] = -1 if (i & 1) == 1 else 1
                flipVecs[i][1] = -1 if (i & 2) == 1 else 1
                flipVecs[i][2] = -1 if (i & 4) == 1 else 1
                corner[i] = np.array([0., 0., 0.])  # assume no reflections
                if (flipVecs[i][0]) < 1:
                    corner[i][0] = h.dim[1]-1  # reflect X
                if (flipVecs[i][1]) < 1:
                    corner[i][1] = h.dim[2]-1  # reflect Y
                if (flipVecs[i][2]) < 1:
                    corner[i][2] = h.dim[3]-1  # reflect Z
                corner[i] = xyz2mm(s, corner[i])
            # find extreme edge from ALL corners....
            _min = corner[0]
            for i in range(8):
                for j in range(3):
                    if corner[i][j] < _min[j]:
                        _min[j] = corner[i][j]
            # dx: observed distance from corner
            min_dx = getDistance(corner[0], _min)
            min_index = 0  # index of corner closest to _min
            # see if any corner is closer to absmin than the first one...
            for i in range(8):
                dx = getDistance(corner[i], _min)
                if dx < min_dx:
                    min_dx = dx
                    min_index = i
            # _min = corner[minIndex]  # this is the single corner closest to _min from all
            return corner[min_index], flipVecs[min_index]

        def getOrthoResidual(orig, transform):
            # mat33 mat = matDotMul33(orig, transform);
            mat = orig @ transform
            return np.sum(mat)

        def getBestOrient(R, flipVec):
            # flipVec reports flip: [1 1 1]=no flips, [-1 1 1] flip X dimension
            # LOAD_MAT33(orig,R.m[0][0],R.m[0][1],R.m[0][2],
            #            R.m[1][0],R.m[1][1],R.m[1][2],
            #            R.m[2][0],R.m[2][1],R.m[2][2]);
            ret = np.eye(3) * flipVec
            orig = R[:3, :3]
            best = 0.0
            for rot in range(6):  # 6 rotations
                if rot == 0:
                    # LOAD_MAT33(newmat,flipVec.v[0],0,0, 0,flipVec.v[1],0, 0,0,flipVec.v[2])
                    newmat = np.eye(3) * flipVec
                elif rot == 1:
                    # LOAD_MAT33(newmat,flipVec.v[0],0,0, 0,0,flipVec.v[1], 0,flipVec.v[2],0)
                    newmat = np.array([[flipVec[0], 0, 0], [0, 0, flipVec[1]], [0, flipVec[2], 0]])
                elif rot == 2:
                    # LOAD_MAT33(newmat,0,flipVec.v[0],0, flipVec.v[1],0,0, 0,0,flipVec.v[2])
                    newmat = np.array([[0, flipVec[0], 0], [flipVec[1], 0, 0], [0, 0, flipVec[2]]])
                elif rot == 3:
                    # LOAD_MAT33(newmat,0,flipVec.v[0],0, 0,0,flipVec.v[1], flipVec.v[2],0,0)
                    newmat = np.array([[0, flipVec[0], 0], [0, 0, flipVec[1]], [flipVec[2], 0, 0]])
                elif rot == 4:
                    # LOAD_MAT33(newmat,0,0,flipVec.v[0], flipVec.v[1],0,0, 0,flipVec.v[2],0)
                    newmat = np.array([[0, 0, flipVec[0]], [flipVec[1], 0, 0], [0, flipVec[2], 0]])
                elif rot == 5:
                    # LOAD_MAT33(newmat,0,0,flipVec.v[0], 0,flipVec.v[1],0, flipVec.v[2],0,0)
                    newmat = np.array([[0, 0, flipVec[0]], [0, flipVec[1], 0], [flipVec[2], 0, 0]])
                newval = getOrthoResidual(orig, newmat)
                if newval > best:
                    best = newval
                    ret = newmat
            return ret

        def setOrientVec(m):
            # Assumes isOrthoMat NOT computed on INVERSE, hence return INVERSE of solution...
            # e.g. [-1,2,3] means reflect x axis, [2,1,3] means swap x and y dimensions
            ret = np.array([0, 0, 0])
            for i in range(3):
                for j in range(3):
                    if m[i, j] > 0:
                        ret[j] = i+1
                    elif m[i, j] < 0:
                        ret[j] = - (i + 1)
            return ret

        def orthoOffsetArray(dim, stepBytesPerVox):
            # return lookup table of length dim with values incremented by stepBytesPerVox
            #  e.g. if Dim=10 and stepBytes=2: 0,2,4..18, is stepBytes=-2 18,16,14...0
            # size_t *lut= (size_t *)malloc(dim*sizeof(size_t));
            lut = np.zeros(dim)
            if stepBytesPerVox > 0:
                lut[0] = 0
            else:
                lut[0] = -stepBytesPerVox * (dim - 1)
            if dim > 1:
                for i in range(1, dim):
                    lut[i] = lut[i-1] + stepBytesPerVox
            return lut

        def reOrientImg(img, outDim, outInc, bytePerVox, nvol):
            # Reslice data to new orientation
            # Generate look up tables
            xLUT = orthoOffsetArray(outDim[0], bytePerVox*outInc[0])
            yLUT = orthoOffsetArray(outDim[1], bytePerVox*outInc[1])
            zLUT = orthoOffsetArray(outDim[2], bytePerVox*outInc[2])
            # Convert data
            # number of voxels in spatial dimensions [1,2,3]
            # bytePerVol = bytePerVox*outDim[0]*outDim[1]*outDim[2]
            # o = 0  # output address
            # inbuf = (uint8_t *) malloc(bytePerVol)  # we convert 1 volume at a time
            # outbuf = (uint8_t *) img  # source image
            for vol in range(nvol):  # for each volume
                # memcpy(&inbuf[0], &outbuf[vol*bytePerVol], bytePerVol)  # copy source volume
                inbuf = np.copy(img[vol])
                for z in range(outDim[2]):
                    for y in range(outDim[1]):
                        for x in range(outDim[0]):
                            logger.error('Has not verified adressing')
                            # memcpy(&outbuf[o], &inbuf[xLUT[x]+yLUT[y]+zLUT[z]], bytePerVox)
                            img[vol, z, y, x] = inbuf[xLUT[x], yLUT[y], zLUT[z]]
                            # o += bytePerVox

        def reOrient(img, h, orientVec, orient, minMM):
            # e.g. [-1,2,3] means reflect x axis, [2,1,3] means swap x and y dimensions

            nvox = img.columns * img.rows * img.slices
            if nvox < 1:
                return img
            outDim = np.zeros(3)
            outInc = np.zeros(3)
            for i in range(3):  # set dimensions, pixdim
                outDim[i] = h.dim[abs(orientVec[i])]
                if abs(orientVec[i]) == 1:
                    outInc[i] = 1
                elif abs(orientVec[i]) == 2:
                    outInc[i] = h.dim[1]
                elif abs(orientVec[i]) == 3:
                    outInc[i] = h.dim[1]*h.dim[2]
                if orientVec[i] < 0:
                    outInc[i] = -outInc[i]  # flip
            nvol = 1  # convert all non-spatial volumes from source to destination
            for vol in range(4, 8):
                if h.dim[vol] > 1:
                    nvol = nvol * h.dim[vol]
            reOrientImg(img, outDim, outInc, h.bitpix / 8, nvol)
            # now change the header....
            outPix = np.array([h.pixdim[abs(orientVec[0])],
                               h.pixdim[abs(orientVec[1])],
                               h.pixdim[abs(orientVec[2])]])
            for i in range(3):
                h.dim[i+1] = outDim[i]
                h.pixdim[i+1] = outPix[i]
            # mat44 s = sFormMat(h);
            s = h.get_sform()
            # mat33 mat; //computer transform
            # LOAD_MAT33(mat, s.m[0][0],s.m[0][1],s.m[0][2],
            #                      s.m[1][0],s.m[1][1],s.m[1][2],
            #                      s.m[2][0],s.m[2][1],s.m[2][2]);
            mat = s[:3, :3]  # Computer transform
            # mat = matMul33(  mat, orient);
            mat = mat @ orient
            # s = setMat44Vec(mat, minMM); //add offset
            s = np.eye(4)
            s[:3, :3] = mat
            s[:3, 3] = minMM  # Add offset
            # mat2sForm(h,s);
            h.set_sform(s)
            # h->qform_code = h->sform_code; //apply to the quaternion as well
            _, sform_code = h.get_sform(coded=True)
            # float dumdx, dumdy, dumdz;
            # nifti_mat44_to_quatern(s, &h->quatern_b, &h->quatern_c, &h->quatern_d,
            #     &h->qoffset_x, &h->qoffset_y, &h->qoffset_z,
            #     &dumdx, &dumdy, &dumdz,&h->pixdim[0]) ;
            h.set_qform(s, code=sform_code)
            return img

        # mat44 s = sFormMat(h);
        s = hdr.get_sform()
        h = hdr  # TODO
        if isMat44Canonical(s):
            logger.debug("Image in perfect alignment: no need to reorient")
            return img
        # vec3i  flipV;
        flipV = np.zeros(3)
        minMM, flipV = minCornerFlip(hdr)
        orient = getBestOrient(s, flipV)
        orientVec = setOrientVec(orient)
        if orientVec[0] == 1 and orientVec[1] == 2 and orientVec[2] == 3:
            logger.debug("Image already near best orthogonal alignment: no need to reorient")
            return img
        is24 = False
        if h.bitpix == 24:  # RGB stored as planar data. Treat as 3 8-bit slices
            return img
            is24 = True
            h.bitpix = 8
            h.dim[3] = h.dim[3] * 3
        img = reOrient(img, h, orientVec, orient, minMM)
        if is24:
            h.bitpix = 24
            h.dim[3] = h.dim[3] / 3
        logger.debug("NewRotation= %d %d %d\n", orientVec.v[0], orientVec.v[1], orientVec.v[2])
        logger.debug("MinCorner= %.2f %.2f %.2f\n", minMM.v[0], minMM.v[1], minMM.v[2])
        return img

    def _nii_save_attributes(self, si, hdr):
        pass

    def _find_slice_direction(self, si, affine, normal):
        """Return slice direction

        Returns
         None : unknown
         1 : sag,
         2 : cor
         3 : axial
         - : flipped
         """
        if si.ndim < 3:
            return None
        slice_direction = 1
        if abs(normal[1]) >= abs(normal[0]) and abs(normal[1]) >= abs(normal[2]):
            slice_direction = 2
        if abs(normal[2]) >= abs(normal[0]) and abs(normal[2]) >= abs(normal[1]):
            slice_direction = 3
        # pos = si.patientPosition(slice_direction)
        pos = si.imagePositions[0][::-1][slice_direction-1]
        x = np.array([0, 0, si.ndim - 1, 1], dtype=float).reshape((1, 4))
        # pos1v = nifti_vect44mat44_mul(x, affine)
        pos1v = x @ affine
        pos1 = pos1v[0, slice_direction - 1]
        # Same direction? Note Python indices from 0
        flip = (pos > affine[slice_direction-1, 3]) != (pos1 > affine[slice_direction-1, 3])
        if flip:
            slice_direction = - slice_direction
        return slice_direction
