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
from imagedata.formats.abstractplugin import AbstractPlugin
import nibabel

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
        info = img.get_header()
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
        try:
            ny,nx,nz,nt = info.get_data_shape()
            logging.debug("_set_tags: ny {}, nx {}, nz {}, nt {}".format(ny,nx,nz,nt))
        except ValueError:
            ny,nx,nz = info.get_data_shape()
            nt = 1
        #nifti_affine = image_list[0].get_affine()
        nifti_affine = info.get_sform()
        logging.debug('NiftiPlugin.read: get_sform\n{}'.format(info.get_sform()))
        logging.debug('NiftiPlugin.read: info.get_zooms() {}'.format(info.get_zooms()))
        try:
            dx, dy, dz = info.get_zooms()
        except ValueError:
            dx, dy, dz, dt = info.get_zooms()
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
        nz = 1
        if si.ndim > 2:
            nz = si.shape[-3]
        #self.setQform(nifti_affine)
        #hdr['transformationMatrix'] = self.transformationMatrix
        self.shape = si.shape
        self.getGeometryFromAffine(hdr, nifti_affine, hdr['slices'])
        logging.debug("NiftiPlugin::read: hdr[orientation] {}".format(hdr['orientation']))
        logging.debug("NiftiPlugin::read: hdr[transformationMatrix]\n{}".format(hdr['transformationMatrix']))

        logging.debug("_set_tags: get_dim_info(): {}".format(info.get_dim_info()))
        logging.debug("_set_tags: get_xyzt_units(): {}".format(info.get_xyzt_units()))
        times = [0]
        if nt > 1:
            #try:
            #    times = info.get_slice_times()
            #    print("_set_tags: times", times)
            #except nibabel.spatialimages.HeaderDataError:
                dx, dy, dz, dt = info.get_zooms()
                times = np.arange(0, nt*dt, dt)
        assert len(times) == nt, "Wrong timeline calculated (times={}) (nt={})".format(len(times), nt)
        logging.debug("_set_tags: times {}".format(times))
        tags = {}
        for z in range(nz):
            tags[z] = np.array(times)
        hdr['tags'] = tags

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
        # Set imagePositions for first slice
        x,y,z = Q[0:3,3]
        hdr['imagePositions'] = {0: np.array([z,y,x])}
        logging.debug("getGeometryFromAffine: hdr imagePositions={}".format(hdr['imagePositions']))
        # Set slice orientation
        ds,dr,dc = hdr['spacing']
        logging.debug("getGeometryFromAffine: spacing ds {}, dr {}, dc {}".format(ds,dr,dc))
        orient = []
        logging.debug("getGeometryFromAffine: Q\n{}".format(Q))
        orient.append(Q[2,2]/dc)
        orient.append(Q[1,2]/dc)
        orient.append(Q[0,2]/dc)
        orient.append(Q[2,1]/dr)
        orient.append(Q[1,1]/dr)
        orient.append(Q[0,1]/dr)
        logging.debug("getGeometryFromAffine: orient {}".format(orient))
        hdr['orientation'] = orient
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

    def getQformFromTransformationMatrix(self, shape):
        M = self.transformationMatrix
        Q = np.array([[M[2,2], M[2,1], M[2,0], M[2,3]],
                      [M[1,2], M[1,1], M[1,0], M[1,3]],
                      [M[0,2], M[0,1], M[0,0], M[0,3]],
                      [     0,      0,      0,     1 ]]
                )

        #Flip voxels in y
        analyze_to_dicom = np.eye(4)
        analyze_to_dicom[1,1] = -1
        analyze_to_dicom[1,3] = shape[1]+1
        logging.debug("getQformFromTransformationMatrix: analyze_to_dicom\n{}".format(analyze_to_dicom))
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
        logging.debug("getQformFromTransformationMatrix: patient_to_tal\n{}".format(patient_to_tal))
        Q = np.dot(patient_to_tal,Q)
        logging.debug("getQformFromTransformationMatrix: Q after\n{}".format(Q))

        return Q

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
        if len(destination['files'][0]) > 0:
            filename_template = destination['files'][0]

        self.slices               = si.slices
        self.spacing              = si.spacing
        self.transformationMatrix = si.transformationMatrix
        self.orientation          = si.orientation
        self.tags                 = si.tags

        logging.info("Data shape write: {}".format(imagedata.formats.shape_to_str(si.shape)))
        save_shape = si.shape
        if si.ndim == 2:
            si.shape = (1,) + si.shape
        #elif si.ndim == 3:
        #    si.shape = (1,) + si.shape
        #assert si.ndim == 4, "write_3d_series: input dimension %d is not 3D." % (si.ndim-1)
        assert si.ndim == 3, "write_3d_series: input dimension %d is not 3D." % (si.ndim)
        #if si.shape[0] != 1:
        #    raise ValueError("Attempt to write 4D image ({}) using write_3d_numpy".format(si.shape))
        #slices = si.shape[1]
        slices = si.shape[0]
        if slices != si.slices:
            raise ValueError("write_3d_series: slices of dicom template ({}) differ from input array ({}).".format(si.slices, slices))

        #fsi=self.reorder_data_in_3d(si)
        fsi = self._reorder_to_dicom(si, flip=True)
        shape = fsi.shape

        #qform = self.affine_to_nifti(shape)
        #qform = self.getQform()
        qform = self.getQformFromTransformationMatrix(shape)
        NiftiHeader = nibabel.Nifti1Header()
        NiftiHeader.set_dim_info(freq=0, phase=1, slice=2)
        NiftiHeader.set_data_shape(shape)
        dz,dy,dx = self.spacing
        NiftiHeader.set_zooms((dx,dy,dz))
        NiftiHeader.set_data_dtype(fsi.dtype)
        #NiftiHeader.set_qform(qform, code=2)
        NiftiHeader.set_sform(qform, code=2)
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
        #if len(os.path.splitext(filename)[1]) == 0:
        #    filename = filename + '.nii.gz'
        #img.to_filename(os.path.join(dirname, filename))
        self.write_numpy_nifti(img, archive, filename)
        si.shape = save_shape

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
        if len(destination['files'][0]) > 0:
            filename_template = destination['files'][0]

        #fsi=self.reorder_data_in_4d(si)
        #shape = fsi.shape

        self.slices               = si.slices
        self.spacing              = si.spacing
        self.transformationMatrix = si.transformationMatrix
        self.orientation          = si.orientation
        self.tags                 = si.tags

        # Defaults
        self.output_sort = imagedata.formats.SORT_ON_SLICE
        if 'output_sort' in opts:
            self.output_sort = opts['output_sort']

        save_shape = si.shape
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

        fsi=self.reorder_data_in_4d(si)
        shape = fsi.shape

        #qform = self.affine_to_nifti(shape)
        #qform = self.getQform()
        logging.debug("write_4d_numpy: get qform")
        qform = self.getQformFromTransformationMatrix(shape)
        logging.debug("write_4d_numpy: got qform")
        NiftiHeader = nibabel.Nifti1Header()
        NiftiHeader.set_dim_info(freq=0, phase=1, slice=2)
        NiftiHeader.set_data_shape(shape)
        dz,dy,dx = self.spacing
        NiftiHeader.set_zooms((dx, dy, dz, 1))
        NiftiHeader.set_data_dtype(fsi.dtype)
        NiftiHeader.set_qform(qform, code=1)
        NiftiHeader.set_sform(qform, code=1)
        #NiftiHeader.set_slice_duration()
        #NiftiHeader.set_slice_times(times)
        NiftiHeader.set_xyzt_units(xyz='mm', t='sec')
        img = nibabel.Nifti1Image(fsi, None, NiftiHeader)
        #if not os.path.isdir(dirname):
        #    os.makedirs(dirname)
        try:
            filename = filename_template % (0)
        except TypeError:
            filename = filename_template
        #if len(os.path.splitext(filename)[1]) == 0:
        #    filename = filename + '.nii.gz'
        #img.to_filename(os.path.join(dirname, filename))
        self.write_numpy_nifti(img, archive, filename)
        si.shape = save_shape

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
