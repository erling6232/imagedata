"""Read/Write Nifti-1 files
"""

# Copyright (c) 2013-2018 Erling Andersen, Haukeland University Hospital, Bergen, Norway

import os.path
import sys
import tempfile
import logging
import fs
import math
import numpy as np

import imagedata.formats
import imagedata.axis
from imagedata.formats.abstractplugin import AbstractPlugin
import nibabel, nibabel.nicom.dicomwrappers

class NoInputFile(Exception): pass
class FilesGivenForMultipleURLs(Exception): pass

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

        info = None
        logging.debug("niftiplugin::read filehandle {}".format(f))
        #TODO: Read nifti directly from open file object
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
        logging.debug("niftiplugin::read load f {}".format(f))
        try:
            img = nibabel.load(f)
        except nibabel.spatialimages.ImageFileError:
            raise imagedata.formats.NotImageError(
                    '{} does not look like a nifti file.'.format(f))
        except Exception:
            raise
        info = img.header
        si = self._reorder_to_dicom(img.get_data(), flip=True)
        return (info, si)

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

        info, si = image_list[0]
        _data_shape = info.get_data_shape()
        nt = nz = 1
        nx, ny = _data_shape[:2]
        if len(_data_shape) > 2:
            nz = _data_shape[2]
        if len(_data_shape) > 3:
            nt = _data_shape[3]
        logging.debug("_set_tags: ny {}, nx {}, nz {}, nt {}".format(ny,nx,nz,nt))
        #nifti_affine = image_list[0].get_affine()
        nifti_affine = info.get_qform()
        logging.debug('NiftiPlugin.read: get_qform\n{}'.format(info.get_qform()))
        logging.debug('NiftiPlugin.read: info.get_zooms() {}'.format(info.get_zooms()))
        _data_zooms = info.get_zooms()
        dt = dz = 1
        dx, dy = _data_zooms[:2]
        if len(_data_zooms) > 2:
            dz = _data_zooms[2]
        if len(_data_zooms) > 3:
            dt = _data_zooms[3]
        self.spacing = (dz, dy, dx)
        hdr['spacing'] = (dz, dy, dx)

        # Simplify shape
        self._reduce_shape(si)

        dim_info = info.get_dim_info()
        #qform = info.get_qform()
        #sform = info.get_sform()
        xyzt_units = info.get_xyzt_units()

        #self.transformationMatrix = self.nifti_to_affine(nifti_affine, si.shape)
        # Prerequisites for setQform: self.spacing and self.slices
        # self.spacing is set already
        #self.setQform(nifti_affine)
        #hdr['transformationMatrix'] = self.transformationMatrix
        self.shape = si.shape
        self.getGeometryFromAffine(hdr, nifti_affine, hdr['slices'])
        logging.debug("NiftiPlugin::read: hdr[orientation] {}".format(hdr['orientation']))
        #logging.debug("NiftiPlugin::read: hdr[transformationMatrix]\n{}".format(hdr['transformationMatrix']))

        logging.debug("_set_tags: get_dim_info(): {}".format(info.get_dim_info()))
        logging.debug("_set_tags: get_xyzt_units(): {}".format(info.get_xyzt_units()))
        times = [0]
        if nt > 1:
            #try:
            #    times = info.get_slice_times()
            #    print("_set_tags: times", times)
            #except nibabel.spatialimages.HeaderDataError:
            times = np.arange(0, nt*dt, dt)
        assert len(times) == nt, "Wrong timeline calculated (times={}) (nt={})".format(len(times), nt)
        logging.debug("_set_tags: times {}".format(times))
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

    '''
    def nifti_to_affine(self, affine, shape):

        if len(shape) != 4:
            raise ValueError("4D only (was: %dD)" % len(shape))

        Q = affine.copy()

        logging.debug("Q from nifti_to_affine():\n{}".format(Q))
        # Swap row 0 (z) and 2 (x)
        Q[[0, 2],:] = Q[[2, 0],:]
        # Swap column 0 (z) and 2 (x)
        Q[:,[0, 2]] = Q[:,[2, 0]]
        logging.debug("Q swap nifti_to_affine():\n{}".format(Q))

        analyze_to_dicom = np.eye(4)
        analyze_to_dicom[0,3] = 1
        analyze_to_dicom[1,3] = 1
        analyze_to_dicom[2,3] = 1
        dicom_to_analyze = np.linalg.inv(analyze_to_dicom)
        Q = np.dot(Q,dicom_to_analyze)
        logging.debug("Q after dicom_to_analyze:\n{}".format(Q))

        analyze_to_dicom = np.eye(4)
        analyze_to_dicom[0,3] = -1
        analyze_to_dicom[1,1] = -1
        rows = shape[2]
        analyze_to_dicom[1,3] = rows
        analyze_to_dicom[2,3] = -1
        dicom_to_analyze = np.linalg.inv(analyze_to_dicom)
        Q = np.dot(Q,dicom_to_analyze)
        logging.debug("Q after rows dicom_to_analyze:\n{}".format(Q))

        patient_to_tal = np.eye(4)
        patient_to_tal[0,0] = -1
        patient_to_tal[1,1] = -1
        tal_to_patient = np.linalg.inv(patient_to_tal)
        Q = np.dot(tal_to_patient,Q)
        logging.debug("Q after tal_to_patient:\n{}".format(Q))

        return Q
    '''

    '''
    def affine_to_nifti(self, shape):
        Q = self.transformationMatrix.copy()
        logging.debug("Affine from self.transformationMatrix:\n{}".format(Q))
        # Swap row 0 (z) and 2 (x)
        Q[[0, 2],:] = Q[[2, 0],:]
        # Swap column 0 (z) and 2 (x)
        Q[:,[0, 2]] = Q[:,[2, 0]]
        logging.debug("Affine swap self.transformationMatrix:\n{}".format(Q))

        # Q now equals dicom_to_patient in spm_dicom_convert

        # Convert space
        analyze_to_dicom = np.eye(4)
        analyze_to_dicom[0,3] = -1
        analyze_to_dicom[1,1] = -1
        #if len(shape) == 3:
        #    rows = shape[1]
        #else:
        #    rows = shape[2]
        rows = shape[-2]
        analyze_to_dicom[1,3] = rows
        analyze_to_dicom[2,3] = -1
        logging.debug("analyze_to_dicom:\n{}".format(analyze_to_dicom))

        patient_to_tal = np.eye(4)
        patient_to_tal[0,0] = -1
        patient_to_tal[1,1] = -1
        logging.debug("patient_to_tal:\n{}".format(patient_to_tal))

        Q = np.dot(patient_to_tal,Q)
        logging.debug("Q with patient_to_tal:\n{}".format(Q))
        Q = np.dot(Q,analyze_to_dicom)
        # Q now equals mat in spm_dicom_convert

        analyze_to_dicom = np.eye(4)
        analyze_to_dicom[0,3] = 1
        analyze_to_dicom[1,3] = 1
        analyze_to_dicom[2,3] = 1
        logging.debug("analyze_to_dicom:\n{}".format(analyze_to_dicom))
        Q = np.dot(Q,analyze_to_dicom)

        logging.debug("Q nifti:\n{}".format(Q))
        return Q
    '''

    def getGeometryFromAffine(self, hdr, Q, slices):
        """Extract geometry attributes from Nifti header

        Input:
        - self: NiftiPlugin instance
        - Q: nifti Qform
        - slices: number of slices
        - hdr['spacing']
        Output:
        - hdr: header dict
            hdr['imagePositions'][0]
            hdr['orientation']
            hdr['transformationMatrix']
        """

        # Swap back from nifti patient space, flip x and y directions
        affine = np.dot(np.diag([-1,-1,1,1]), Q)
        # Set imagePositions for first slice
        x,y,z = affine[0:3,3]
        hdr['imagePositions'] = {0: np.array([z,y,x])}
        logging.debug("getGeometryFromAffine: hdr imagePositions={}".format(hdr['imagePositions']))
        # Set slice orientation
        ds,dr,dc = hdr['spacing']
        logging.debug("getGeometryFromAffine: spacing ds {}, dr {}, dc {}".format(ds,dr,dc))

        colr = affine[:3,0][::-1] / dr
        colc = affine[:3,1][::-1] / dc
        #T0 = affine[:3,3][::-1]
        orient = []
        logging.debug("getGeometryFromAffine: affine\n{}".format(affine))
        for i in range(3):
            orient.append(colc[i])
        for i in range(3):
            orient.append(colr[i])
        logging.debug("getGeometryFromAffine: orient {}".format(orient))
        hdr['orientation'] = orient

        return

        # Transformation matrix
        #Flip voxels in y
        analyze_to_dicom = np.eye(4)
        analyze_to_dicom[1,1] = -1
        try:
            analyze_to_dicom[1,3] = self.shape[2]+1
        except IndexError: # 2D
            analyze_to_dicom[1,3] = 1
        logging.debug("getGeometryFromAffine: analyze_to_dicom\n{}".format(analyze_to_dicom))
        #dicom_to_analyze = np.linalg.inv(analyze_to_dicom)
        #Q = np.dot(Q,dicom_to_analyze)
        Q = np.dot(Q,analyze_to_dicom)
        logging.debug("Q after rows dicom_to_analyze:\n{}".format(Q))
        # Flip mm coords in x and y directions
        patient_to_tal = np.diag([-1,-1,1,1])
        #patient_to_tal = np.eye(4)
        #patient_to_tal[0,0] = -1
        #patient_to_tal[1,1] = -1
        #tal_to_patient = np.linalg.inv(patient_to_tal)
        #Q = np.dot(tal_to_patient,Q)
        logging.debug("getGeometryFromAffine: patient_to_tal\n{}".format(patient_to_tal))
        Q = np.dot(patient_to_tal,Q)
        logging.debug("getGeometryFromAffine: Q after\n{}".format(Q))
        
        M = np.array([[Q[2,2], Q[2,1], Q[2,0], Q[2,3]],
                      [Q[1,2], Q[1,1], Q[1,0], Q[1,3]],
                      [Q[0,2], Q[0,1], Q[0,0], Q[0,3]],
                      [     0,      0,      0,     1 ]]
                )
        hdr['transformationMatrix'] = M

    def create_affine_xyz(self):
        """Create affine in xyz.
        """

        def normalize(v):
            """Normalize a vector

            https://stackoverflow.com/questions/21030391/how-to-normalize-an-array-in-numpy

            :param v: 3D vector
            :return: normalized 3D vector
            """
            norm=np.linalg.norm(v, ord=1)
            if norm==0:
                norm=np.finfo(v.dtype).eps
            return v/norm

        ds, dr, dc = self.spacing
        colr = normalize(np.array(self.orientation[3:6])).reshape((3,))
        colc = normalize(np.array(self.orientation[0:3])).reshape((3,))
        T0 = self.imagePositions[0][::-1].reshape(3,)  # x,y,z
        if self.slices > 1:
            # Stack of multiple slices
            Tn = self.imagePositions[self.slices-1][::-1].reshape(3,)  # x,y,z
            k = -(T0-Tn)/(1-self.slices)
        else:
            # Single slice
            k = np.cross(colr, colc, axis=0)
            k = k * ds

        L = np.zeros((4,4))
        L[:3, 0] = colr * dr
        L[:3, 1] = colc * dc
        L[:3, 2] = k
        L[:3, 3] = T0[:]
        L[ 3, 3] = 1
        return L

    def getQformFromTransformationMatrix(self, shape):
        #def matrix_from_orientation(orientation, normal):
        #    oT = orientation.reshape((2,3)).T
        #    colr = oT[:,0].reshape((3,1))
        #    colc = oT[:,1].reshape((3,1))
        #    coln = normal.reshape((3,1))
        #    if len(self.shape) < 3:
        #        M = np.hstack((colr[:2], colc[:2])).reshape((2,2))
        #    else:
        #        M = np.hstack((colr, colc, coln)).reshape((3,3))
        #    return M

        def normalize(v):
            """Normalize a vector

            https://stackoverflow.com/questions/21030391/how-to-normalize-an-array-in-numpy

            :param v: 3D vector
            :return: normalized 3D vector
            """
            norm=np.linalg.norm(v, ord=1)
            if norm==0:
                norm=np.finfo(v.dtype).eps
            return v/norm

        def L_from_orientation(orientation, normal, spacing):
            """
            orientation: row, then column index direction cosines
            """
            ds, dr, dc = spacing
            colr = normalize(np.array(orientation[3:6])).reshape((3,))
            colc = normalize(np.array(orientation[0:3])).reshape((3,))
            T0 = self.imagePositions[0][::-1].reshape(3,)  # x,y,z
            if self.slices > 1:
                Tn = self.imagePositions[self.slices-1][::-1].reshape(3,)  # x,y,z
                k = Tn
                k = np.cross(colr, colc, axis=0)
                k = k * ds
            else:
                k = np.cross(colr, colc, axis=0)
                k = k * ds

            L = np.zeros((4,4))
            L[:3, 0] = T0[:]
            L[ 3, 0] = 1
            L[:3, 1] = k
            L[ 3, 1] = 1 if self.slices > 1 else 0
            L[:3, 2] = colr * dr
            L[:3, 3] = colc * dc
            return L

        #M = self.transformationMatrix
        #M = matrix_from_orientation(self.orientation, self.normal)
        #ipp = self.origin
        #Q = np.array([[M[2,2], M[2,1], M[2,0], ipp[0]],
        #              [M[1,2], M[1,1], M[1,0], ipp[1]],
        #              [M[0,2], M[0,1], M[0,0], ipp[2]],
        #              [     0,      0,      0,     1 ]]
        #        )

        if self.slices > 1:
            R=np.array([[1, 1, 1, 0], [1, 1, 0, 1], [1, self.slices, 0, 0], [1, 1, 0, 0]])
        else:
            R=np.array([[1, 0, 1, 0], [1, 0, 0, 1], [1, self.slices, 0, 0], [1, 0, 0, 0]])
        L = L_from_orientation(self.orientation, self.normal, self.spacing)

        Linv = np.linalg.inv(L)
        Aspm = np.dot(R, np.linalg.inv(L))
        to_ones = np.eye(4)
        to_ones[:,3] = 1
        A = np.dot(Aspm, to_ones)

        ds, dr, dc = self.spacing
        colr = normalize(np.array(self.orientation[3:6])).reshape((3,))
        colc = normalize(np.array(self.orientation[0:3])).reshape((3,))
        coln = normalize(np.cross(colc, colr, axis=0))
        T0 = self.imagePositions[0][::-1].reshape(3,)  # x,y,z
        if self.slices > 1:
            Tn = self.imagePositions[self.slices-1][::-1].reshape((3,))  # x,y,z
            abcd = np.array([1, 1, self.slices, 1]).reshape((4,))
            one = np.ones((1,))
            efgh = np.concatenate((Tn, one))
        else:
            abcd = np.array([0, 0, 1, 0]).reshape((4,))
            zero = np.zeros((1,))
            efgh = np.concatenate((n * ds, zeros))

        # From derivations/spm_dicom_orient.py

        # premultiplication matrix to go from 0 to 1 based indexing
        one_based = np.eye(4)
        one_based[:3,3] = (1,1,1)
        # premult for swapping row and column indices
        row_col_swap = np.eye(4)
        row_col_swap[:,0] = np.eye(4)[:,1]
        row_col_swap[:,1] = np.eye(4)[:,0]

        # various worming matrices
        orient_pat = np.hstack([colr.reshape(3,1), colc.reshape(3,1)])
        orient_cross = coln
        pos_pat_0 = T0
        if self.slices > 1:
            missing_r_col = (T0-Tn)/(1-self.slices)
            pos_pat_N = Tn
        pixel_spacing = [dr, dc]
        NZ = self.slices
        slice_thickness = ds

        R3 = np.dot(orient_pat, np.diag(pixel_spacing))
        #R3 = orient_pat * np.diag(pixel_spacing)
        R = np.zeros((4,2))
        R[:3,:] = R3

        # The following is specific to the SPM algorithm.
        x1 = np.ones((4))
        y1 = np.ones((4))
        y1[:3] = pos_pat_0

        to_inv = np.zeros((4,4))
        to_inv[:,0] = x1
        to_inv[:,1] = abcd
        to_inv[0,2] = 1
        to_inv[1,3] = 1
        inv_lhs = np.zeros((4,4))
        inv_lhs[:,0] = y1
        inv_lhs[:,1] = efgh
        inv_lhs[:,2:] = R

        def spm_full_matrix(x2, y2):
            rhs = to_inv[:,:]
            rhs[:,1] = x2
            lhs = inv_lhs[:,:]
            lhs[:,1] = y2
            return np.dot(lhs, np.linalg.inv(rhs))

        if self.slices > 1:
            x2_ms = np.array([1, 1, NZ, 1])
            y2_ms = np.ones((4,))
            y2_ms[:3] = pos_pat_N
            A_ms = spm_full_matrix(x2_ms, y2_ms)
            A = A_ms
        else:
            orient = np.zeros((3,3))
            orient[:3,:2] = orient_pat
            orient[:,2] = orient_cross
            x2_ss = np.array([0, 0, 1, 0])
            y2_ss = np.zeros((4,))
            #y2_ss[:3] = orient * np.array([0, 0, slice_thickness])
            y2_ss[:3] = np.dot(orient, np.array([0, 0, slice_thickness]))
            A_ss = spm_full_matrix(x2_ss, y2_ss)
            A = A_ss

        A = np.dot(A, row_col_swap)

        multi_aff = np.eye(4)
        multi_aff[:3,:2] = R3
        trans_z_N = np.array([0, 0, self.slices-1, 1])
        multi_aff[:3,2] = missing_r_col
        multi_aff[:3,3] = pos_pat_0
        est_pos_pat_N = np.dot(multi_aff, trans_z_N)

        #Flip voxels in y
        analyze_to_dicom = np.eye(4)
        analyze_to_dicom[1,1] = -1
        #analyze_to_dicom[1,3] = shape[1]+1
        analyze_to_dicom[1,3] = self.slices
        logging.debug("getQformFromTransformationMatrix: analyze_to_dicom\n{}".format(analyze_to_dicom))
        #dicom_to_analyze = np.linalg.inv(analyze_to_dicom)
        #Q = np.dot(Q,dicom_to_analyze)
        Q = np.dot(A,analyze_to_dicom)
        ### 2019.07.03 # Q = np.dot(Q,analyze_to_dicom)
        ### 2019.07.03 # logging.debug("Q after rows dicom_to_analyze:\n{}".format(Q))
        # Flip mm coords in x and y directions
        patient_to_tal = np.diag([1,-1,-1,1])
        #patient_to_tal = np.eye(4)
        #patient_to_tal[0,0] = -1
        #patient_to_tal[1,1] = -1
        #tal_to_patient = np.linalg.inv(patient_to_tal)
        #Q = np.dot(tal_to_patient,Q)
        logging.debug("getQformFromTransformationMatrix: patient_to_tal\n{}".format(patient_to_tal))
        Q = np.dot(patient_to_tal,Q)
        logging.debug("getQformFromTransformationMatrix: Q after\n{}".format(Q))

        return Q

    def create_affine(sorted_dicoms):
        """
        Function to generate the affine matrix for a dicom series
        From dicom2nifti:common.py: https://github.com/icometrix/dicom2nifti/blob/master/dicom2nifti/common.py
        This method was based on (http://nipy.org/nibabel/dicom/dicom_orientation.html)
        :param sorted_dicoms: list with sorted dicom files
        """

        # Create affine matrix (http://nipy.sourceforge.net/nibabel/dicom/dicom_orientation.html#dicom-slice-affine)
        image_orient1 = numpy.array(sorted_dicoms[0].ImageOrientationPatient)[0:3]
        image_orient2 = numpy.array(sorted_dicoms[0].ImageOrientationPatient)[3:6]

        delta_r = float(sorted_dicoms[0].PixelSpacing[0])
        delta_c = float(sorted_dicoms[0].PixelSpacing[1])

        image_pos = numpy.array(sorted_dicoms[0].ImagePositionPatient)

        last_image_pos = numpy.array(sorted_dicoms[-1].ImagePositionPatient)

        if len(sorted_dicoms) == 1:
            # Single slice
            step = [0, 0, -1]
        else:
            step = (image_pos - last_image_pos) / (1 - len(sorted_dicoms))

        # check if this is actually a volume and not all slices on the same location
        if numpy.linalg.norm(step) == 0.0:
            raise ConversionError("NOT_A_VOLUME")

        affine = numpy.array(
            [[-image_orient1[0] * delta_c, -image_orient2[0] * delta_r, -step[0], -image_pos[0]],
             [-image_orient1[1] * delta_c, -image_orient2[1] * delta_r, -step[1], -image_pos[1]],
             [image_orient1[2] * delta_c, image_orient2[2] * delta_r, step[2], image_pos[2]],
             [0, 0, 0, 1]]
        )
        return affine, numpy.linalg.norm(step)

    '''
    def reverse_3d_shape(self, shape):
        if len(shape) == 4:
            t,slices,rows,columns = shape
        else:
            slices,rows,columns = shape
        return((columns,rows,slices))

    def reverse_4d_shape(self, shape):
        if len(shape) == 4:
            t,slices,rows,columns = shape
        else:
            slices,rows,columns = shape
            t = 1
        return((columns,rows,slices,t))

    def reorder_data_in_3d(self, data):
        # Reorder data
        # NIFTI order: snii[columns,rows,slices]
        # DICOM order: data[t,slices,rows,columns]
        nifti_shape = self.reverse_3d_shape(data.shape)

        columns,rows,slices = nifti_shape
        logging.info("NIFTI shape: %dx%dx%d, dtype %s" % (columns,rows,slices, data.dtype))
        if len(data.shape) == 2:
            logging.info("DCM shape: %dx%d" % (data.shape[0],data.shape[1]))
            si = np.fliplr(data.T)
        elif len(data.shape) == 3:
            logging.info("DCM shape: %dx%dx%d" % (data.shape[0],data.shape[1],data.shape[2]))
            si = np.zeros([columns,rows,slices], data.dtype)
            for z in range(slices):
                si[:,:,z] = np.fliplr(data[z,:,:].T)
        elif len(data.shape) == 4:
            logging.info("DCM shape: %dtx%dx%dx%d" % (data.shape[0],data.shape[1],data.shape[2],data.shape[3]))
            si = np.zeros([columns,rows,slices], data.dtype)
            for z in range(slices):
                si[:,:,z] = np.fliplr(data[0,z,:,:].T)
        else:
            raise ValueError("Unknown shape %d." % len(data.shape))
        return si

    def reorder_data_in_4d(self, data):
        # Reorder data
        # NIFTI order: snii[columns,rows,slices,t]
        # DICOM order: data[t,slices,rows,columns]
        nifti_shape = self.reverse_4d_shape(data.shape)

        columns,rows,slices,t = nifti_shape
        logging.info("NIFTI shape: %dx%dx%dx%dt, dtype %s" % (columns,rows,slices,t, data.dtype))

        logging.info("DCM shape: %dtx%dx%dx%d" % (data.shape[0],data.shape[1],data.shape[2],data.shape[3]))
        si = np.zeros([columns,rows,slices,t], data.dtype)
        #logging.debug('reorder_data_in_4d: got data {}'.format(data.shape))
        #logging.debug('reorder_data_in_4d: got si {}'.format(si.shape))
        #logging.debug('reorder_data_in_4d: slices {}'.format(slices))
        for i in range(t):
            for z in range(slices):
                #logging.debug('reorder_data_in_4d: .T {} {}'.format(z,i))
                si[:,:,z,i] = np.fliplr(data[i,z,:,:].T)
        #logging.debug('reorder_data_in_4d: return')
        return si
    '''

    def write_3d_numpy(self, si, destination, opts):
        """Write 3D numpy image as Nifti file

        Input:
        - self: NiftiPlugin instance
        - si: Series array (3D or 4D), including these attributes:
            slices*
            spacing*
            imagePositions*
            transformationMatrix*
            orientation*
            tags*
        - destination: dict of archive and filenames
        - opts: Output options (dict)
        """

        if si.color:
            raise imagedata.formats.WriteNotImplemented(
                    "Writing color Nifti images not implemented.")

        logging.debug('NiftiPlugin.write_3d_numpy: destination {}'.format(destination))
        archive = destination['archive']
        filename_template = 'Image.nii.gz'
        if len(destination['files']) > 0 and len(destination['files'][0]) > 0:
            filename_template = destination['files'][0]

        self.shape                = si.shape
        self.slices               = si.slices
        self.spacing              = si.spacing
        self.transformationMatrix = si.transformationMatrix
        self.imagePositions       = si.imagePositions
        #self.orientation          = si.orientation
        self.tags                 = si.tags
        self.origin, self.orientation, self.normal = si.get_transformation_components_xyz()

        logging.info("Data shape write: {}".format(imagedata.formats.shape_to_str(si.shape)))
        #if si.ndim == 2:
        #    si.shape = (1,) + si.shape
        #elif si.ndim == 3:
        #    si.shape = (1,) + si.shape
        #assert si.ndim == 4, "write_3d_series: input dimension %d is not 3D." % (si.ndim-1)
        assert si.ndim == 2 or si.ndim == 3, "write_3d_series: input dimension %d is not 3D." % (si.ndim)
        #if si.shape[0] != 1:
        #    raise ValueError("Attempt to write 4D image ({}) using write_3d_numpy".format(si.shape))
        #slices = si.shape[1]
        #slices = si.shape[0]
        #if slices != si.slices:
        #    raise ValueError("write_3d_series: slices of dicom template ({}) differ from input array ({}).".format(si.slices, slices))

        #fsi=self.reorder_data_in_3d(si)
        fsi = self._reorder_to_dicom(si, flip=True)
        shape = fsi.shape

        affine_xyz = self.create_affine_xyz()
        # The Nifti patient space flips the x and y directions
        qform = np.dot(np.diag([-1,-1,1,1]), affine_xyz)
        NiftiHeader = nibabel.Nifti1Header()
        NiftiHeader.set_dim_info(freq=0, phase=1, slice=2)
        NiftiHeader.set_data_shape(shape)
        dz,dy,dx = self.spacing
        if si.ndim < 3:
            NiftiHeader.set_zooms((dx,dy))
        else:
            NiftiHeader.set_zooms((dx,dy,dz))
        NiftiHeader.set_data_dtype(fsi.dtype)
        NiftiHeader.set_qform(qform, code=1)
        #NiftiHeader.set_slice_duration()
        #NiftiHeader.set_slice_times(times)
        #NiftiHeader.set_xyzt_units(xyz='mm', t='sec')
        NiftiHeader.set_xyzt_units(xyz='mm')
        img = nibabel.Nifti1Image(fsi, None, NiftiHeader)
        #if not os.path.isdir(dirname):
        #    os.makedirs(dirname)
        try:
            filename = filename_template % (0)
        except TypeError:
            filename = filename_template
        self.write_numpy_nifti(img, archive, filename)

    def write_4d_numpy(self, si, destination, opts):
        """Write 4D numpy image as Nifti file

        Input:
        - self: NiftiPlugin instance
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
                    "Writing color Nifti images not implemented.")

        logging.debug('ITKPlugin.write_4d_numpy: destination {}'.format(destination))
        archive = destination['archive']
        filename_template = 'Image.nii.gz'
        if len(destination['files']) > 0 and len(destination['files'][0]) > 0:
            filename_template = destination['files'][0]

        #fsi=self.reorder_data_in_4d(si)
        #shape = fsi.shape

        self.shape                = si.shape
        self.slices               = si.slices
        self.spacing              = si.spacing
        self.transformationMatrix = si.transformationMatrix
        self.imagePositions       = si.imagePositions
        #self.orientation          = si.orientation
        self.tags                 = si.tags
        self.origin, self.orientation, self.normal = si.get_transformation_components_xyz()

        # Defaults
        self.output_sort = imagedata.formats.SORT_ON_SLICE
        if 'output_sort' in opts:
            self.output_sort = opts['output_sort']

        # Should we allow to write 3D volume?
        if si.ndim == 2:
            si.shape = (1,1,) + si.shape
        elif si.ndim == 3:
            si.shape = (1,) + si.shape
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

        #fsi=self.reorder_data_in_4d(si)
        fsi = self._reorder_to_dicom(si, flip=True)
        shape = fsi.shape

        affine_xyz = self.create_affine_xyz()
        # The Nifti patient space flips the x and y directions
        qform = np.dot(np.diag([-1,-1,1,1]), affine_xyz)
        logging.debug("write_4d_numpy: get qform")
        logging.debug("write_4d_numpy: got qform")
        NiftiHeader = nibabel.Nifti1Header()
        NiftiHeader.set_dim_info(freq=0, phase=1, slice=2)
        NiftiHeader.set_data_shape(shape)
        dz,dy,dx = self.spacing
        NiftiHeader.set_zooms((dx, dy, dz, 1))
        NiftiHeader.set_data_dtype(fsi.dtype)
        NiftiHeader.set_qform(qform, code=1)
        #NiftiHeader.set_slice_duration()
        #NiftiHeader.set_slice_times(times)
        NiftiHeader.set_xyzt_units(xyz='mm', t='sec')
        img = nibabel.Nifti1Image(fsi, None, NiftiHeader)
        try:
            filename = filename_template % (0)
        except TypeError:
            filename = filename_template
        self.write_numpy_nifti(img, archive, filename)

    def write_numpy_nifti(self, img, archive, filename):
        """Write nifti data to file

        Input:
        - self: ITKPlugin instance, including these attributes:
            slices (not used)
            spacing
            imagePositions
            transformationMatrix
            orientation (not used)
            tags (not used)
        - img: Nifti1Image
        - archive: archive object
        - filename: file name, possibly without extentsion
        """

        if len(os.path.splitext(filename)[1]) == 0:
            filename = filename + '.nii.gz'
        ext = os.path.splitext(filename)[1]
        if filename.endswith('.nii.gz'):
            ext = '.nii.gz'
        logging.debug('write_numpy_nifti: ext %s' % ext)

        f = tempfile.NamedTemporaryFile(
                suffix=ext, delete=False)
        logging.debug('write_numpy_nifti: write local file %s' % f.name)
        img.to_filename(f.name)
        f.close()
        logging.debug('write_numpy_nifti: copy to file %s' % filename)
        fh = archive.add_localfile(f.name, filename)
        os.unlink(f.name)

    def copy(self, other=None):
        if other is None: other = NiftiPlugin()
        return AbstractPlugin.copy(self, other=other)
