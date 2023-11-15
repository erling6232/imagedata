"""Read/Write Nifti-1 files
"""

# Copyright (c) 2013-2023 Erling Andersen, Haukeland University Hospital, Bergen, Norway

import os.path
import tempfile
import logging
import math
import nibabel
import nibabel.data
import nibabel.spatialimages
from ..series import Series
import numpy as np
from . import NotImageError, WriteNotImplemented, input_order_to_dirname_str
from ..axis import UniformLengthAxis
from .abstractplugin import AbstractPlugin

# import nitransforms
NIFTI_INTENT_NONE = 0
NIFTI_XFORM_UNKNOWN = 0
NIFTI_XFORM_SCANNER_ANAT = 1
NIFTI_XFORM_ALIGNED_ANAT = 2
NIFTI_XFORM_TALAIRACH = 3
NIFTI_XFORM_MNI_152 = 4
NIFTI_XFORM_TEMPLATE_OTHER = 5

logger = logging.getLogger(__name__)


class NiftiPlugin(AbstractPlugin):
    """Read/write Nifti-1 files.
    """

    name = "nifti"
    description = "Read and write Nifti-1 files."
    authors = "Erling Andersen"
    version = "2.0.0"
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
        except nibabel.spatialimages.ImageDataError:
            raise NotImageError(
                '{} does not look like a nifti file.'.format(f))
        except Exception:
            raise

        if hdr.input_order == 'auto':
            hdr.input_order = 'none'

        hdr.color = False

        flip_y = True  # Always
        if flip_y:
            img = self._nii_flip_y(img)

        slice_direction = self._verify_nifti_slice_direction(img)
        if slice_direction < 0:
            img = self._nii_flip_slices(img)
            # slice_direction = abs(slice_direction)

        si = self._reorder_to_dicom(
            np.asanyarray(img.dataobj),
            flip=False,
            flipud=True)
        return img, si

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

            # for c in range(4):  # NIfTI is RAS+, DICOM is LPS+
            #     for r in range(2):
            #         sform[r, c] = - sform[r, c]

            hdr.transformationMatrix = np.eye(4, dtype=np.float64)
            # NIfTI is RAS+, DICOM is LPS+
            hdr.transformationMatrix[:3, 0] = sform[:3, 2][::-1]
            hdr.transformationMatrix[:3, 1] = sform[:3, 1][::-1]
            hdr.transformationMatrix[:3, 2] = sform[:3, 0][::-1]
            hdr.transformationMatrix[:3, 3] = sform[:3, 3][::-1]

            q = sform[:3, :3]
            # # p = sform[:3, 3]
            # p = nibabel.affines.apply_affine(sform, (0, ny - 1, 0))
            # if np.linalg.det(q) < 0:
            #     q[:3, 1] = - q[:3, 1]
            # # Note: rz, ry, rx, cz, cy, cx
            iop = np.array([
                q[2, 0] / dx, q[1, 0] / dx, q[0, 0] / dx,
                q[2, 1] / dy, q[1, 1] / dy, q[0, 1] / dy
            ])
            #
            p = sform[:3, 3]
            for _slice in range(nz):
                _p = np.array([
                    (q[0, 2] * _slice + p[0]),  # NIfTI is RAS+, DICOM is LPS+
                    (q[1, 2] * _slice + p[1]),
                    (q[2, 2] * _slice + p[2])
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
        assert len(times) == nt, \
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

    # noinspection PyPep8Naming
    def write_3d_numpy(self, si, destination, opts):
        """Write 3D numpy image as Nifti file

        Args:
            self: NiftiPlugin instance
            si: Series array (3D or 4D), including these attributes:
                slices,
                spacing,
                imagePositions,
                transformationMatrix,
                orientation,
                tags

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

        img = self._save_dicom_to_nifti(si)
        try:
            filename = filename_template % 0
        except TypeError:
            filename = filename_template
        self.write_numpy_nifti(img, archive, filename)

    def write_4d_numpy(self, si, destination, opts):
        """Write 4D numpy image as Nifti file

        si[tag,slice,rows,columns]: Series array, including these attributes:
                slices, spacing, imagePositions, transformationMatrix,
                orientation, tags

        Args:
            si (imagedata.Series): Series array
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

        img = self._save_dicom_to_nifti(si)
        try:
            filename = filename_template % 0
        except TypeError:
            filename = filename_template
        self.write_numpy_nifti(img, archive, filename)

    @staticmethod
    def write_numpy_nifti(img, archive, filename):
        """Write nifti data to file

        Args:
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

    def _save_dicom_to_nifti(self, si: Series) -> nibabel.Nifti1Image:
        """Convert DICOM to Nifti
        dcm2niix.saveDcm2Nii

        Args:
            si (Series): input Series instance
        Returns:
            (nibabel.Nifti1Image): nifti instance
        """

        img = self._nii_load_image(si)
        slice_direction = 0
        if si.slices > 1:
            slice_direction = self._header_dicom_to_nifti_2(img.header, si)
        if slice_direction < 0:
            img = self._nii_flip_slices(img)
            # slice_direction = abs(slice_direction)
        # img = self._nii_set_ortho(img)
        flip_y = True  # Always
        if flip_y:
            img = self._nii_flip_y(img)
        return img

    def _nii_load_image(self, si):
        """Create Nifti1Image from Series
        dcm2niix.nii_loadImgXL

        Args:
            si (Series): input Series instance
        Returns:
            (nibabel.Nifti1Image): nifti instance
        """
        hdr = self._header_dicom_to_nifti(si, compute_sform=True)
        img = nibabel.Nifti1Image(self._reorder_from_dicom(si, flipud=True), None, hdr)
        # img = self._nii_load_image_core(dcm, hdr)
        # if img is None:
        #     return img
        # if hdr.get_data_dtype() == 128:  # DT_RGB24
        raw = hdr.structarr
        if raw['datatype'] == 128:  # DT_RGB24
            # Do this before Y-flip, or RGB order can be flipped
            img = self._nii_rgb_to_planar(img, hdr, si.isPlanarRGB)
        # if dcm.CSA.mosaicSlices > 1:
        #     img = self._nii_de_mosaic(img, hdr, dcm.CSA.mosaicSlices
        # n_acq = si.slices
        # dim = hdr.get_data_shape()
        # if n_acq > 1 and (dim[2] % n_acq and dim[2] > n_acq):
        #     # dim[3] = dim[2] // n_acq
        #     # dim[2] = n_acq
        #     dim = (dim[0], dim[1], n_acq, dim[2] // n_acq)
        # if hdr.dim[0] > 3 and dcm.patientPositionSequentialRepeats > 1:
        #     # Swizzle 3rd and 4th dimension (Philips stores time as 3rd dimension)
        #     img = self._nii_xytz_xyzt(img, hdr, dcm.patientPositionsSequentialRepeats)
        self._header_dicom_to_nifti_sform(hdr, si)
        return img

    # def _nii_load_image_core(self, dcm, hdr):
    #     """dcm2niix.nii_loadImgCore
    #     """
    #     raise Exception('Not implemented')
    #     return img

    def _nii_rgb_to_planar(self, img, hdr, is_planar_rgb):
        """dcm2niix.nii_rgb2planar
        """
        raise Exception('Not implemented')

    def _nii_xytz_xyzt(self, img, hdr, patient_positions_sequential_repeats):
        """dcm2niix.nii_XYTZ_XYZT
        """
        raise Exception('Not implemented')

    def _header_dicom_to_nifti(self, dcm, compute_sform=False):
        """dcm2niix.headerDcm2Nii
        """
        hdr = nibabel.Nifti1Header()
        if dcm.itemsize == 1 and dcm.axes[0].name == 'rgb':
            hdr.set_intent('estimate')
        hdr.set_data_dtype(dcm.dtype)
        hdr.set_data_shape(dcm.shape[::-1])
        # hdr.set_slope_inter(slope, inter)
        ds, dr, dc = dcm.spacing
        if (dcm.ndim - dcm.color * 1) < 3:
            hdr.set_zooms((dc, dr))
        elif (dcm.ndim - dcm.color * 1) < 4:
            hdr.set_zooms((dc, dr, ds))
        else:
            if dcm.input_order == 'time':
                dt = dcm.timeline[1] - dcm.timeline[0]
                hdr.set_zooms((dc, dr, ds, dt))
            else:
                hdr.set_zooms((dc, dr, ds, 1))
        hdr.set_xyzt_units(xyz='mm', t='sec')
        affine = np.zeros((4, 4))
        affine[0, 0] = -1
        affine[1, 2] = 1
        affine[2, 1] = -1
        affine[0, 3] = dcm.shape[-1 - dcm.color*1] / 2  # C
        affine[1, 3] = dcm.shape[-2 - dcm.color*1] / 2  # R
        try:
            affine[2, 3] = dcm.shape[-3 - dcm.color*1] / 2  # S
        except IndexError:
            # Probably a 2D image
            pass
        hdr.set_qform(affine, NIFTI_XFORM_UNKNOWN)
        hdr.set_sform(affine, NIFTI_XFORM_SCANNER_ANAT)
        hdr.set_intent(NIFTI_INTENT_NONE)
        if compute_sform:
            self._header_dicom_to_nifti_2(hdr, dcm)
        return hdr

    def _header_dicom_to_nifti_2(self, hdr: nibabel.Nifti1Header, dcm: Series):
        """Set Nifti1 header from Series instance
        dcm2niix.headerDcm2Nii2

        Args:
            hdr (nibabel.Nifti1Header): nifti header
            dcm (Series): Series instance
        Returns:
            sliceDir (int): 0=unknown,1=sag,2=coro,3=axial,-=reversed slices
        """
        # """dcm2niix.headerDcm2Nii2"""
        # if hdr.slice_code == nibabel.NIFTI_SLICE_UNKNOWN:
        #     hdr.set_slice_code(d.CSA.sliceOrder)
        # if hdr.slice_code == nibabel.NIFTI_SLICE_UNKNOWN:
        #     hdr.set_slice_code(d2.CSA.sliceOrder)
        # txt = "TE=%.2g;TIME=%.3f".format(d.TE, d.acquisitionTime)
        # if d.CSA.phaseEncodingDirectionPositive >= 0:
        #     txt += ";phase=%d".format(d.CSA.phaseEncodingDirectionPositive)
        inPlanePhaseEncodingDirection = dcm.getDicomAttribute('InPlanePhaseEncodingDirection')
        if inPlanePhaseEncodingDirection == 'ROW':
            hdr.set_dim_info(freq=1, phase=0, slice=2)
        elif inPlanePhaseEncodingDirection == 'COL':
            hdr.set_dim_info(freq=0, phase=1, slice=2)
        # if d.CSA.multiBandFactor > 1):
        #     txt += ";mb=%d".format(d.CSA.multiBandFactor)
        # hdr.set_description(txt)
        return self._header_dicom_to_nifti_sform(hdr, dcm)

    def _header_dicom_to_nifti_sform(self, hdr: nibabel.Nifti1Header, dcm: Series):
        """dcm2niix.headerDcm2NiiSForm

        Args:
            hdr (nibabel.Nifti1Header): nifti header
            dcm (Series): Series instance
        Returns:
            sliceDir (int): 0=unknown,1=sag,2=coro,3=axial,-=reversed slices
        """
        if dcm.slices < 2:
            # Do not care direction for single slice
            q44, slice_direction = self._set_nii_header_x(dcm)
            hdr.set_qform(q44, NIFTI_XFORM_SCANNER_ANAT)
            hdr.set_sform(q44, NIFTI_XFORM_SCANNER_ANAT)
            return slice_direction
        is_ok = False
        for i in range(6):
            if dcm.orientation[i] != 0.0:
                is_ok = True
        if not is_ok:
            # We will have to guess,
            # assume axial acquisition saved in standard Siemens style?
            dcm.orientation = [0, 1, 0, 0, 0, 1]
        q44, slice_direction = self._set_nii_header_x(dcm)
        hdr.set_qform(q44, NIFTI_XFORM_SCANNER_ANAT)
        hdr.set_sform(q44, NIFTI_XFORM_SCANNER_ANAT)
        return slice_direction

    def _set_nii_header_x(self, dcm: Series):
        """dcm2niix.set_nii_header_x

        Args:
            dcm (Series): Series instance
        Returns:
            q44 (numpy.ndarray): affine matrix
            sliceDir (int): 0=unknown,1=sag,2=coro,3=axial,-=reversed slices
        """
        q44 = self._nifti_dicom_to_mat(dcm.orientation, dcm.imagePositions[0], dcm.spacing)
        # if dcm.isSegamiOasis:
        #     q44 = np.array([[-h.pixdim[1], 0, 0, 0],
        #                     [0, -h.pixdim[2], 0, 0],
        #                     [0, 0, h.pixdim[3], 0],
        #                     [0, 0, 0, 0]])
        #     originVx = np.array([(h.dim[1]+1.0)/2, (h.dim[2]+1.0])/2, (h.dim[3]+1.0)/2])
        #     originMm = np.matmul(originVx, q44)
        #     for i in range(4):
        #         q44[i, 3] = -originMm[i]  # Set origin to center voxel
        #     return q44, slice_direction
        # if d.CSA.mosaicSlices > 1:
        #     pass
        slice_direction = self._verify_slice_direction(dcm, q44)
        # LPS to nifti RAS, xform matrix before reorient
        q44[:2, :4] = -q44[:2, :4]
        return q44, slice_direction

    def _nifti_dicom_to_mat(self, orient, patient_position, spacing):
        """dcm2niix.nifti_dicom2mat

        Args:
            orient (np.ndarray): zyx
            patient_position (np.ndarray): image position of first slice, zyx
            spacing (np.ndarray):, ds, dr, dc
        """
        q = np.array([[orient[2], orient[1], orient[0]],
                      [orient[5], orient[4], orient[3]],
                      [0, 0, 0]], dtype=np.float64)
        # Normalize row 0
        val = q[0, 0] * q[0, 0] + q[0, 1] * q[0, 1] + q[0, 2] * q[0, 2]
        if val > 0.0:
            val = 1.0 / math.sqrt(val)
            q[0] = val * q[0]
        else:
            q[0, 0] = 1.0
            q[0, 1] = 0.0
            q[0, 2] = 0.0
        # Normalize row 1
        val = q[1, 0] * q[1, 0] + q[1, 1] * q[1, 1] + q[1, 2] * q[1, 2]
        if val > 0.0:
            val = 1.0 / math.sqrt(val)
            q[1] = val * q[1]
        else:
            q[1, 0] = 0.0
            q[1, 1] = 1.0
            q[1, 2] = 0.0
        # Row 3 is the cross product of rows 1 and 2
        q[2] = np.cross(q[0], q[1])
        q = q.T
        if np.linalg.det(q) < 0:
            q[0, 2] = -q[0, 2]
            q[1, 2] = -q[1, 2]
            q[2, 2] = -q[2, 2]
        # Next scale matrix
        diag_vox = np.array([[spacing[2], 0.0, 0.0],
                             [0.0, spacing[1], 0.0],
                             [0.0, 0.0, spacing[0]]])
        q = np.matmul(q, diag_vox)
        q44 = np.eye(4)
        q44[0:3, 0:3] = q
        q44[0:3, 3] = patient_position[::-1]
        return q44

    def _verify_slice_direction(self, dcm: Series, r: np.ndarray):
        """dcm2niix.verify_slice_dir

        Args:
            dcm (Series): Series instance
            r (numpy.ndarray): affine matrix in nifti orientation
        Returns:
            sliceDir (int): 0=unknown,1=sag,2=coro,3=axial,-=reversed slices
        """
        slice_direction = 0
        if dcm.slices < 2:
            return slice_direction
        # find Z-slice direction: row with highest magnitude of 1st column
        slice_direction = 1
        if (abs(r[1, 2]) >= abs(r[0, 2])) and (abs(r[1, 2]) >= abs(r[2, 2])):
            slice_direction = 2
        if (abs(r[2, 2]) >= abs(r[0, 2])) and (abs(r[2, 2]) >= abs(r[1, 2])):
            slice_direction = 3
        pos_dicom = None
        try:
            # Last slice in stack
            pos_dicom = dcm.imagePositions[dcm.slices-1][::-1][slice_direction-1]  # zyx to xyz
        except ValueError:
            pass
        x = np.array([0.0, 0.0, dcm.slices-1.0, 1.0])
        # pos1v = np.matmul(x, r)
        pos1v = _nifti_vect44mat44_mul(x, r)
        pos_nifti = pos1v[slice_direction-1]  # -1 as C index from 0
        if pos_dicom is None:
            # Do some guess work
            orient = dcm.orientation  # in zyx
            read_v = np.array([orient[2], orient[1], orient[0]])
            phase_v = np.array([orient[5], orient[4], orient[3]])
            slice_v = np.cross(read_v, phase_v)
            flip = np.sum(slice_v) < 0
        else:
            # same direction? note C indices from 0
            flip = (pos_dicom > r[slice_direction-1, 3]) != (pos_nifti > r[slice_direction-1, 3])
        if flip:
            r[:, 2] = -r[:, 2]
            slice_direction = -slice_direction
        return slice_direction

    def _verify_nifti_slice_direction(self, img: nibabel.Nifti1Image) -> int:
        """Calculate slice direction.

        Args:
            img (nibabel.Nifti1Image): nifti image instance
        Returns:
            sliceDir (int): 0=unknown,1=sag,2=coro,3=axial,-=reversed slices
        """
        h = img.header
        slice_direction = 0
        dim = h.get_data_shape()
        if dim[2] < 2:
            return slice_direction

        # find Z-slice direction: row with highest magnitude of 1st column
        q44 = h.get_sform()
        slice_direction = -1  # By convention all sagital series are reversed
        if (abs(q44[1, 2]) >= abs(q44[0, 2])) and (abs(q44[1, 2]) >= abs(q44[2, 2])):
            slice_direction = 2
        if (abs(q44[2, 2]) >= abs(q44[0, 2])) and (abs(q44[2, 2]) >= abs(q44[1, 2])):
            slice_direction = 3

        if slice_direction < 0:
            sform = h.get_sform()[:3, :3]
            q44 = h.get_sform()
            v = np.array([0, 0, dim[2] - 1, 1], dtype=float)
            v = _nifti_vect44mat44_mul(v, q44)  # after flip this voxel will be the origin
            mFlipZ = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64)
            sform = np.matmul(sform, mFlipZ)
            q44[:3, :3] = sform
            q44[:, 3] = v

        q44[:2, :4] = -q44[:2, :4]  # Swap rows 0 and 1 (Nifti RAS to LPS)
        img.set_qform(q44, NIFTI_XFORM_SCANNER_ANAT)
        img.set_sform(q44, NIFTI_XFORM_SCANNER_ANAT)
        return slice_direction

    def _nii_flip_slices(self, img: nibabel.Nifti1Image) -> nibabel.Nifti1Image:
        """Flip slice order in img
        dcm2niix.nii_flipZ

        Args:
            img (nibabel.Nifti1Image): nifti instance
        """

        hdr = img.header
        dim = hdr.get_data_shape()
        if dim[2] < 2:
            return img
        sform = hdr.get_sform()[:3, :3]
        q44 = hdr.get_sform()
        v = np.array([0, 0, dim[2] - 1, 1], dtype=float)
        v = _nifti_vect44mat44_mul(v, q44)  # after flip this voxel will be the origin
        mFlipZ = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64)
        sform = np.matmul(sform, mFlipZ)
        q44[:3, :3] = sform
        q44[:, 3] = v
        hdr.set_qform(q44, NIFTI_XFORM_SCANNER_ANAT)
        hdr.set_sform(q44, NIFTI_XFORM_SCANNER_ANAT)
        return self._nii_flip_image_slices(img)

    def _nii_flip_image_slices(self, img: nibabel.Nifti1Image) -> nibabel.Nifti1Image:
        """Flip slice order of actual image.
        DICOM slice order opposite of NIfTI.
        dcm2niix.nii_flipImgZ

        Args:
            img (nibabel.Nifti1Image): nifti instance
        """
        hdr = img.header
        dim = hdr.get_data_shape()
        slices = dim[2]
        # note truncated toward zero, so half_volume=2 regardless of 4 or 5 slices
        half_volume = slices // 2
        if half_volume < 1:
            return img
        data = np.asarray(img.dataobj)
        for z in range(half_volume):
            # swap order of slices
            tmp = np.array(data[:, :, z, ...])
            data[:, :, z, ...] = data[:, :, slices - z - 1, ...]
            data[:, :, slices - z - 1, ...] = tmp
        return nibabel.Nifti1Image(data, img.affine, img.header)

    def _nii_set_ortho(self, img: nibabel.Nifti1Image):
        """
        Set ortho

        Args:
            img (nibabel.Nifti1Image): nifti image
        """

        def isMat44Canonical(R: np.ndarray) -> bool:
            # returns true if diagonals >0 and all others =0
            #  no rotation is necessary - already in perfect orthogonal alignment
            for i in range(3):
                for j in range(3):
                    if (i == j) and (R[i, j] <= 0):
                        return False
                    if (i != j) and (R[i, j] != 0):
                        return False
            return True

        def xyz2mm(R: np.ndarray, v: np.ndarray) -> np.ndarray:
            ret = np.zeros(3)
            for i in range(3):
                ret[i] = R[i, 0] * v[0] + R[i, 1] * v[1] + R[i, 2] * v[2] + R[i, 3]
            return ret

        def getDistance(v: np.ndarray, _min: np.ndarray) -> float:
            # Scalar distance between two 3D points - Pythagorean theorem
            return math.sqrt(math.pow((v[0] - _min[0]), 2) +
                             math.pow((v[1] - _min[1]), 2) +
                             math.pow((v[2] - _min[2]), 2))

        def minCornerFlip(h: nibabel.Nifti1Header) -> tuple:
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
            dim = h.get_data_shape()
            for i in range(8):
                flipVecs[i] = np.zeros(3, dtype=int)
                flipVecs[i][0] = -1 if (i & 1) == 1 else 1
                flipVecs[i][1] = -1 if (i & 2) == 1 else 1
                flipVecs[i][2] = -1 if (i & 4) == 1 else 1
                corner[i] = np.array([0., 0., 0.])  # assume no reflections
                if (flipVecs[i][0]) < 1:
                    corner[i][0] = dim[0] - 1  # reflect X
                if (flipVecs[i][1]) < 1:
                    corner[i][1] = dim[1] - 1  # reflect Y
                if (flipVecs[i][2]) < 1:
                    corner[i][2] = dim[2] - 1  # reflect Z
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

        def getOrthoResidual(orig: np.ndarray, transform: np.ndarray) -> float:
            # mat33 mat = matDotMul33(orig, transform);
            mat = orig @ transform
            return np.sum(mat)

        def getBestOrient(R: np.ndarray, flipVec: np.ndarray) -> np.ndarray:
            # flipVec reports flip: [1 1 1]=no flips, [-1 1 1] flip X dimension
            # LOAD_MAT33(orig,R.m[0][0],R.m[0][1],R.m[0][2],
            #            R.m[1][0],R.m[1][1],R.m[1][2],
            #            R.m[2][0],R.m[2][1],R.m[2][2]);
            ret = np.eye(3) * flipVec
            orig = R[:3, :3]
            best = 0.0
            for rot in range(6):  # 6 rotations
                newmat = np.eye(3)
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

        def setOrientVec(m: np.ndarray) -> np.ndarray:
            # Assumes isOrthoMat NOT computed on INVERSE, hence return INVERSE of solution...
            # e.g. [-1,2,3] means reflect x axis, [2,1,3] means swap x and y dimensions
            ret = np.array([0, 0, 0], dtype=int)
            for i in range(3):
                for j in range(3):
                    if m[i, j] > 0:
                        ret[j] = i + 1
                    elif m[i, j] < 0:
                        ret[j] = - (i + 1)
            return ret

        def orthoOffsetArray(dim: int, stepBytesPerVox: int) -> np.ndarray:
            # return lookup table of length dim with values incremented by stepBytesPerVox
            #  e.g. if Dim=10 and stepBytes=2: 0,2,4..18, is stepBytes=-2 18,16,14...0
            # size_t *lut= (size_t *)malloc(dim*sizeof(size_t));
            lut = np.zeros(dim, dtype=int)
            if stepBytesPerVox > 0:
                lut[0] = 0
            else:
                lut[0] = -stepBytesPerVox * (dim - 1)
            if dim > 1:
                for i in range(1, dim):
                    lut[i] = lut[i - 1] + stepBytesPerVox
            return lut

        def reOrientImg(img: nibabel.Nifti1Image, outDim: np.ndarray, outInc: np.ndarray,
                        nvol: int) -> nibabel.Nifti1Image:
            # Reslice data to new orientation
            # Generate look up tables
            xLUT = orthoOffsetArray(outDim[0], outInc[0])
            yLUT = orthoOffsetArray(outDim[1], outInc[1])
            zLUT = orthoOffsetArray(outDim[2], outInc[2])
            # Convert data
            # number of voxels in spatial dimensions [1,2,3]
            perVol = outDim[0]*outDim[1]*outDim[2]
            # o = 0  # output address
            # inbuf = (uint8_t *) malloc(bytePerVol)  # we convert 1 volume at a time
            # outbuf = (uint8_t *) img  # source image
            inbuf = np.asarray(img.dataobj).flatten()  # copy source volume
            outbuf = np.empty_like(inbuf)
            o = 0
            for vol in range(nvol):  # for each volume
                # memcpy(&inbuf[0], &outbuf[vol*bytePerVol], bytePerVol)  # copy source volume
                for z in range(outDim[2]):
                    for y in range(outDim[1]):
                        for x in range(outDim[0]):
                            # memcpy(&outbuf[o], &inbuf[xLUT[x]+yLUT[y]+zLUT[z]], bytePerVox)
                            outbuf[o] = inbuf[vol * perVol + xLUT[x] + yLUT[y] + zLUT[z]]
                            o += 1
            outbuf = np.reshape(outbuf, tuple(outDim))
            return nibabel.Nifti1Image(outbuf, img.affine, img.header)

        def reOrient(img, h, orientVec, orient, minMM):
            # e.g. [-1,2,3] means reflect x axis, [2,1,3] means swap x and y dimensions

            columns, rows, slices = h.get_data_shape()
            nvox = columns * rows * slices
            if nvox < 1:
                return img
            outDim = np.zeros(3, dtype=int)
            outInc = np.zeros(3, dtype=int)
            for i in range(3):  # set dimensions, pixdim
                outDim[i] = h['dim'][abs(orientVec[i])]
                if abs(orientVec[i]) == 1:
                    outInc[i] = 1
                elif abs(orientVec[i]) == 2:
                    outInc[i] = h['dim'][1]
                elif abs(orientVec[i]) == 3:
                    outInc[i] = int(h['dim'][1]) * int(h['dim'][2])
                if orientVec[i] < 0:
                    outInc[i] = -outInc[i]  # flip
            nvol = 1  # convert all non-spatial volumes from source to destination
            for vol in range(4, 8):
                if h['dim'][vol] > 1:
                    nvol = nvol * h['dim'][vol]
            img = reOrientImg(img, outDim, outInc, nvol)
            # now change the header....
            outPix = np.array([h['pixdim'][abs(orientVec[0])],
                               h['pixdim'][abs(orientVec[1])],
                               h['pixdim'][abs(orientVec[2])]])
            for i in range(3):
                h['dim'][i + 1] = outDim[i]
                h['pixdim'][i + 1] = outPix[i]
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

        h = img.header
        s = h.get_sform()
        if isMat44Canonical(s):
            logger.debug("Image in perfect alignment: no need to reorient")
            return img
        flipV = np.zeros(3)
        minMM, flipV = minCornerFlip(h)
        orient = getBestOrient(s, flipV)
        orientVec = setOrientVec(orient)
        if orientVec[0] == 1 and orientVec[1] == 2 and orientVec[2] == 3:
            logger.debug("Image already near best orthogonal alignment: no need to reorient")
            return img
        img = reOrient(img, h, orientVec, orient, minMM)
        logger.debug("NewRotation= %d %d %d\n", orientVec[0], orientVec[1], orientVec[2])
        logger.debug("MinCorner= %.2f %.2f %.2f\n", minMM[0], minMM[1], minMM[2])
        return img

    def _nii_flip_y(self, img: nibabel.Nifti1Image) -> nibabel.Nifti1Image:
        """Flip image along Y direction.
        dcm2niix.nii_flipY

        Args:
            img (nibabel.Nifti1Image): input instance
        Returns:
            (nibabel.Nifti1Image): flipped nifti image
        """
        hdr = img.header
        dim = hdr.get_data_shape()
        s = hdr.get_sform()[:3, :3]
        q44 = hdr.get_sform()
        v = np.array([0, dim[1] - 1, 0, 1], dtype=float)
        v = _nifti_vect44mat44_mul(v, q44)
        m_flip_y = np.eye(3, dtype=float)
        m_flip_y[1, 1] = -1
        s = np.matmul(s, m_flip_y)
        q44[:3, :3] = s
        q44[:3, 3] = v[:3]
        img.set_qform(q44, NIFTI_XFORM_SCANNER_ANAT)
        img.set_sform(q44, NIFTI_XFORM_SCANNER_ANAT)
        return self._nii_flip_image_y(img)

    def _nii_flip_image_y(self, img: nibabel.Nifti1Image) -> nibabel.Nifti1Image:
        """Flip image data along Y direction.
        dcm2niix.nii_flipImgY

        Args:
            img (nibabel.Nifti1Image): input instance
        Returns:
            (nibabel.Nifti1Image): flipped nifti image
        """
        hdr = img.header
        dim = hdr.get_data_shape()
        y_size = dim[1]
        half_y = y_size // 2
        data = np.asarray(img.dataobj)
        # Swap order of Y lines
        for y in range(half_y):
            tmp = np.array(data[:, y, ...])
            data[:, y, ...] = data[:, y_size - y - 1, ...]
            data[:, y_size - y - 1, ...] = tmp
        return nibabel.Nifti1Image(data, img.affine, img.header)


def _nifti_vect44mat44_mul(v, m):
    """multiply vector * 4x4matrix
    """
    vO = np.zeros(4)
    for i in range(4):  # multiply Pcrs * m
        for j in range(4):
            vO[i] += m[i, j] * v[j]
    return vO
