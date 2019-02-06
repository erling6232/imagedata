"""Read/Write DICOM files
"""

# Copyright (c) 2013-2018 Erling Andersen, Haukeland University Hospital, Bergen, Norway

import os
import logging
import math
from datetime import date, datetime, time, timedelta
import numpy as np
import fs
import pydicom, pydicom.uid
from pydicom.datadict import tag_for_name

from imagedata.series import Series
import imagedata.formats
from imagedata.formats.dicomlib.uid import get_uid, uid_append_instance
from imagedata.formats.dicomlib.copy_general_dicom_attributes import copy_general_dicom_attributes
from imagedata.formats.abstractplugin import AbstractPlugin

class FilesGivenForMultipleURLs(Exception): pass
class NoDICOMAttributes(Exception): pass
class UnevenSlicesError(Exception): pass

class DICOMPlugin(AbstractPlugin):
    """Read/write DICOM files."""

    name = "dicom"
    description = "Read and write DICOM files."
    authors = "Erling Andersen"
    version = "1.1.1"
    url = "www.helse-bergen.no"

    root="2.16.578.1.37.1.1.4"
    smallint = ('bool8', 'byte', 'ubyte', 'ushort', 'uint16', 'int8', 'uint8')

    def __init__(self):
        super(DICOMPlugin, self).__init__(self.name, self.description,
                self.authors, self.version, self.url)

    def getOriginForSlice(self, slice):
        """Get origin of given slice.

        Input:
        - self: DICOMPlugin instance
        - slice: slice number (int)
        Output:
        - z,y,x: coordinate for origin of given slice (np.array)
        """

        origin = self.getDicomAttribute(tag_for_name("ImagePositionPatient"), slice)
        x=float(origin[0]); y=float(origin[1]); z=float(origin[2])
        return np.array([z, y, x])

    def extractDicomAttributes(self, hdr):
        """Extract DICOM attributes

        Input:
        - self: DICOMPlugin instance
        - hdr: header dict
        Output:
        - hdr: header dict
            seriesNumber
            seriesDescription
            imageType
            spacing
            orientation
            imagePositions
        """
        hdr['seriesNumber'] = self.getDicomAttribute(tag_for_name("SeriesNumber"))
        hdr['seriesDescription'] = self.getDicomAttribute(tag_for_name("SeriesDescription"))
        it = self.getDicomAttribute(tag_for_name("ImageType"))
        #logging.debug("ImageType {} {} {}".format(len(it), type(it), it))
        #for it1 in it:
            #logging.debug("ImageType {} {} {}".format(len(it1), type(it1), it1))
        it2 = "\\".join(it)
        #logging.debug("ImageType {} {} {}".format(len(it2), type(it2), it2))
        hdr['imageType'] = self.getDicomAttribute(tag_for_name("ImageType"))

        # Spacing
        pixel_spacing = self.getDicomAttribute(tag_for_name("PixelSpacing"))
        dy = 1.0; dx = 1.0
        if pixel_spacing is not None:
            # Notice that DICOM row spacing comes first, column spacing second!
            dy=float(pixel_spacing[0]); dx=float(pixel_spacing[1])
        try:
            dz = float(self.getDicomAttribute(tag_for_name("SpacingBetweenSlices")))
        except:
            try:
                dz = float(self.getDicomAttribute(tag_for_name("SliceThickness")))
            except TypeError:
                dz = 1.0
        hdr['spacing'] = np.array([dz, dy, dx])

        ## Image position (patient)
        # Reverse orientation vectors from (x,y,z) to (z,y,x)
        iop = self.getDicomAttribute(tag_for_name("ImageOrientationPatient"))
        if iop is not None:
            hdr['orientation'] = np.array((iop[2],iop[1],iop[0],
                                           iop[5],iop[4],iop[3]))

        # Extract imagePositions, convert from xyz to zyx
        hdr['imagePositions'] = {}
        for slice in hdr['DicomHeaderDict']:
            for tag,fname,im in hdr['DicomHeaderDict'][slice]:
                try:
                    # getOriginForSlice gives position as zyx
                    hdr['imagePositions'][slice] = self.getOriginForSlice(slice)
                except TypeError:
                    pass

    def setDicomAttribute(self, tag, value):
        # Ignore if no real dicom header exists
        if self.DicomHeaderDict is not None:
            for slice in self.DicomHeaderDict:
                for tg,fname,im in self.DicomHeaderDict[slice]:
                    if tag not in im:
                        VR = pydicom.datadict.dictionary_VR(tag)
                        im.add_new(tag, VR, value)
                    else:
                        im[tag].value = value

    def getDicomAttribute(self, tag, slice=0):
        # Get DICOM attribute from first image for given slice
        #logging.debug("getDicomAttribute: tag", tag, ", slice", slice)
        tg, fname, im = self.DicomHeaderDict[slice][0]
        if tag in im:
            return im[tag].value
        else:
            return None

    def removePrivateTags(self):
        # Ignore if no real dicom header exists
        if self.DicomHeaderDict is not None:
            for slice in self.DicomHeaderDict:
                for tg,fname,im in self.DicomHeaderDict[slice]:
                    im.remove_private_tags()

    def read(self, sources, pre_hdr, input_order, opts):
        """Read image data

        Input:
        - self: DICOMPlugin instance
        - sources: list of sources to image data
        - pre_hdr: Pre-filled header dict. Can be None
        - input_order: sort order
        - opts: input options (dict)
        Output:
        - hdr: Header dict
            input_format
            input_order
            slices
            sliceLocations
            DicomHeaderDict
            tags
            seriesNumber
            seriesDescription
            imageType
            spacing
            orientation
            imagePositions
        - si[tag,slice,rows,columns]: multi-dimensional numpy array
        """

        self.input_order = input_order

        # Read DICOM headers
        logging.debug('DICOMPlugin.read: sources %s' % sources)
        try:
            hdr,shape = self.read_headers(sources, input_order, opts)
        except:
            raise

        # Look-up first image to determine pixel type
        tag, member_name, im = hdr['DicomHeaderDict'][0][0]
        if 'RescaleSlope' in im:
            matrix_dtype = float
        else:
            matrix_dtype = np.uint16
        logging.debug("DICOMPlugin.read: matrix_dtype %s" % matrix_dtype)

        logging.debug("SOPClassUID: {}".format(self.getDicomAttribute(tag_for_name("SOPClassUID"))))
        logging.debug("TransferSyntaxUID: {}".format(self.getDicomAttribute(tag_for_name("TransferSyntaxUID"))))
        if 'headers_only' in opts and opts['headers_only']: return hdr,None

        # Load DICOM image data
        si = np.zeros(shape, matrix_dtype)
        for slice in hdr['DicomHeaderDict']:
            for tag,member_name,im in hdr['DicomHeaderDict'][slice]:
                archive,fname = member_name
                tgs = np.array(hdr['tags'][slice])
                #idx = np.where(hdr.tags[slice] == tag)[0][0] # tags is not numpy array
                idx = np.where(tgs == tag)[0][0]
                if 'NumberOfFrames' in im:
                    if im.NumberOfFrames == 1: idx = (idx, slice)
                else:
                    idx = (idx, slice)
                # Simplify index when image is 3D, remove tag index
                if si.ndim == 3:
                    idx = idx[1:]
                with archive.getmember(fname, mode='rb') as f:
                    im=pydicom.filereader.dcmread(f)
                #try:
                #    im.decompress()
                #except NotImplementedError as e:
                #    logging.error("Cannot decompress pixel data: {}".format(e))
                #    raise
                try:
                    #logging.debug("Set si[{}]".format(idx))
                    if 'RescaleSlope' in im:
                        si[idx] = float(im.RescaleSlope) * im.pixel_array.astype(float) + float(im.RescaleIntercept)
                    else:
                        si[idx] = im.pixel_array
                except UnboundLocalError:
                    # A bug in pydicom appears when reading binary images
                    if im.BitsAllocated == 1:
                        logging.debug("Binary image, si.shape={}, image shape=({},{},{})".format(si.shape, im.NumberOfFrames, im.Rows, im.Columns))
                        #try:
                        #    im.decompress()
                        #except NotImplementedError as e:
                        #    logging.error("Cannot decompress pixel data: {}".format(e))
                        #    raise
                        myarr = np.frombuffer(im.PixelData, dtype=np.uint8)
                        # Reverse bit order, and copy the array to get a
                        # contiguous array
                        bits = np.unpackbits(myarr).reshape(-1,8)[:,::-1].copy()
                        si[idx] = np.fliplr(
                                bits.reshape(
                                    1, im.NumberOfFrames, im.Rows, im.Columns))
                        if 'RescaleSlope' in im:
                            si[idx] = float(im.RescaleSlope) * si[idx] + float(im.RescaleIntercept)
                    else:
                        raise
                except Exception as e:
                    logging.warning("Cannot read pixel data: {}".format(e))
                    raise

        if 'correct_acq' in opts and opts['correct_acq']:
            si = self.correct_acqtimes_for_dynamic_series(hdr,si)

        # Add any DICOM template
        if pre_hdr is not None:
            hdr.update(pre_hdr)

        #logging.debug("DICOMPlugin::read: hdr: {}".format(hdr.keys()))
        #logging.debug("DICOMPlugin::read: hdr.imageType: {}".format(hdr['imageType']))

        self.simulateAffine()
        self.create_affine(hdr)
        return hdr,si

    def read_headers(self, sources, input_order, opts):
        """Read DICOM headers only

        Input:
        - self: DICOMPlugin instance
        - sources: list of sources to image data
        - input_order: sort order
        - opts: input options (dict)
        Output:
        - hdr: header dict
        - shape: required shape of image data
        """

        import traceback
        logging.debug('DICOMPlugin.read_headers: sources %s' % sources)
        try:
            hdr,shape = self.get_dicom_headers(sources, input_order, opts)
        except UnevenSlicesError:
            raise
        except ValueError:
            #traceback.print_exc()
            #logging.info("process_member: Could not read {}".format(member_name))
            raise imagedata.formats.NotImageError(
                'Does not look like a DICOM file: {}'.format(urls))

        self.extractDicomAttributes(hdr)

        return hdr,shape

    def get_dicom_headers(self, sources, input_order, opts=None):
        """Get DICOM headers.

        Input:
         - self: DICOMPlugin instance
         - sources: list of sources to image data
         - input_order: Determine how to sort the input images
         - opts: options (dict)
        Output:
         - hdr dict
         - shape
        """

        logging.debug("DICOMPlugin.get_dicom_headers: sources: {} {}".format(type(sources), sources))
        self._default_opts(opts)

        #if len(sources) > 1 and files is not None:
        #    raise FilesGivenForMultipleURLs("Files shall not be given when there are multiple URLs")
        imageDict = {}
        for source in sources:
            archive = source['archive']
            scan_files = source['files']
            logging.debug("DICOMPlugin.get_dicom_headers: archive: {}".format(archive))
            if scan_files is None or len(scan_files) == 0:
                scan_files = archive.getnames()
            logging.debug("get_dicom_headers: source: {} {}".format(type(source), source))
            for path in archive.getmembers(scan_files):
                if os.path.basename(path) == "DICOMDIR": continue
                with archive.getmember(path, mode='rb') as f:
                    #logging.debug("get_dicom_headers: calling self.process_member({})".format(path))
                    self.process_member(imageDict, archive, path, f, opts)
        return self.sort_images(imageDict, input_order, opts)

    def sort_images(self, headerDict, input_order, opts):
        """Sort DICOM images.

        Input:
        - self: DICOMPlugin instance
        - headerDict: dict where sliceLocations are keys
        - input_order: determine how to sort the input images
        - opts: options (dict)
        Output:
        - hdr dict
            input_format
            input_order
            slices
            sliceLocations
            DicomHeaderDict
            tags
        - shape
        """
        hdr = {}
        hdr['input_format'] = self.name
        hdr['input_order'] = input_order
        sliceLocations = sorted(headerDict)
        hdr['slices'] = len(sliceLocations)
        hdr['sliceLocations'] = sliceLocations

        # Verify same number of images for each slice
        if len(headerDict) == 0:
            raise ValueError("No DICOM images found.")
        count = np.zeros(len(headerDict), dtype=np.int)
        islice = 0
        for sloc in sorted(headerDict):
            count[islice] += len(headerDict[sloc])
            islice += 1
        logging.debug("sort_images: tags per slice: {}".format(count))
        if min(count) != max(count) and 'AcceptUnevenSlices' not in opts['input_options']:
            logging.error("sort_images: tags per slice: {}".format(count))
            raise UnevenSlicesError("Different number of images in each slice.")

        # Extract all tags and sort them per slice
        tagList = {}
        islice = 0
        for sloc in sorted(headerDict):
            tagList[islice] = []
            i = 0
            for archive,fname,im in sorted(headerDict[sloc]):
                if input_order == imagedata.formats.INPUT_ORDER_FAULTY:
                    tag = i
                else:
                    tag = self.get_tag(im, input_order, opts)
                if tag not in tagList[islice] or 'AcceptDuplicateTag' in opts['input_options']:
                    tagList[islice].append(tag)
                else:
                    print("WARNING: Duplicate tag", tag)
                i += 1
            islice += 1
        for islice in range(len(headerDict)):
            tagList[islice] = sorted(tagList[islice])
        # Sort images based on position in tagList
        sorted_headers = {}
        islice = 0
        frames = None; rows = None; columns = None
        i = 0
        for sloc in sorted(headerDict):
            # Pre-fill sorted_headers
            sorted_headers[islice] = [False for x in range(count[islice])]
            for archive,fname,im in sorted(headerDict[sloc]):
                if input_order == imagedata.formats.INPUT_ORDER_FAULTY:
                    tag = i
                else:
                    tag = self.get_tag(im, input_order, opts)
                idx = tagList[islice].index(tag)
                #sorted_headers[islice].insert(idx, (tag, (archive,fname), im))
                sorted_headers[islice][idx] = (tag, (archive,fname), im)
                rows = im.Rows; columns = im.Columns
                if 'NumberOfFrames' in im: frames = im.NumberOfFrames
                i += 1
            islice += 1
        self.DicomHeaderDict = sorted_headers
        hdr['DicomHeaderDict'] = sorted_headers
        #hdr['tagList'] = tagList[0]
        #import pprint
        #pprint.pprint(tagList)
        hdr['tags'] = tagList
        nz = len(headerDict)
        if frames is not None and frames > 1: nz = frames
        if len(tagList[0]) > 1:
            shape = ((len(tagList[0]), nz, rows, columns))
        else:
            shape = (nz, rows, columns)
        return hdr,shape

    def process_member(self, imageDict, archive, member_name, member, opts):
        import traceback
        try:
            im=pydicom.filereader.dcmread(member, stop_before_pixels=True)
        except:
            #traceback.print_exc()
            #logging.info("process_member: Could not read {}".format(member_name))
            return

        if 'input_serinsuid' in opts and opts['input_serinsuid']:
            if im.SeriesInstanceUID != opts['input_serinsuid']: return
        if 'input_echo' in opts and opts['input_echo']:
            if int(im.EchoNumbers) != int(opts['input_echo']): return

        try:
            sloc=im.SliceLocation
            if sloc not in imageDict:
                imageDict[sloc] = []
            imageDict[sloc].append((archive, member_name, im))
            #logging.debug("process_member: added sloc {} {}".format(sloc, member_name))
        except:
            traceback.print_exc()
            logging.debug("process_member: no SliceLocation in {}".format(member_name))
            if 0 not in imageDict:
                imageDict[0] = []
            imageDict[0].append((archive, member_name, im))
        #logging.debug("process_member: imageDict len: {}".format(len(imageDict)))

    def correct_acqtimes_for_dynamic_series(self, hdr, si):
        #si[t,slice,rows,columns]

        # Extract acqtime for each image
        slices = len(hdr['sliceLocations'])
        timesteps = self.count_timesteps(hdr)
        logging.info("Slices: %d, apparent time steps: %d, actual time steps: %d" % (slices, len(hdr['tags']), timesteps))
        new_shape = (timesteps, slices, si.shape[2], si.shape[3])
        newsi = np.zeros(new_shape, dtype=si.dtype)
        acq = np.zeros([slices, timesteps])
        for slice in hdr['DicomHeaderDict']:
            t = 0
            for tg,fname,im in hdr['DicomHeaderDict'][slice]:
                #logging.debug(slice, tg, fname)
                acq[slice,t] = tg
                t += 1

        # Correct acqtimes by setting acqtime for each slice of a volume to
        # the smallest time
        for t in range(acq.shape[1]):
            min_acq = np.min(acq[:,t])
            for slice in range(acq.shape[0]):
                acq[slice,t] = min_acq

        # Set new acqtime for each image
        for slice in hdr['DicomHeaderDict']:
            t = 0
            for tg,fname,im in hdr['DicomHeaderDict'][slice]:
                im.AcquisitionTime = "%f" % acq[slice,t]
                newsi[t,slice,:,:] = si[t,slice,:,:]
                t += 1

        # Update taglist in hdr
        newTagList = {}
        for slice in hdr['DicomHeaderDict']:
            newTagList[slice] = []
            for t in range(acq.shape[1]):
                newTagList[slice].append(acq[0,t])
        hdr['tags'] = newTagList
        return newsi

    def count_timesteps(self, hdr):
        slices = len(hdr['sliceLocations'])
        timesteps = np.zeros([slices], dtype=int)
        for slice in hdr['DicomHeaderDict']:
            t = 0
            for tg,fname,im in hdr['DicomHeaderDict'][slice]:
                timesteps[slice] += 1
        if timesteps.min() != timesteps.max():
            raise ValueError("Number of time steps ranges from %d to %d." % (timesteps.min(), timesteps.max()))
        return timesteps.max()

    def _default_opts(self, opts):
        """Make sure that required options are set.
        """

        if 'input_options' not in opts:
            opts['input_options'] = {}

    def write_3d_numpy(self, si, dirname, filename_template, opts):
        """Write 3D Series image as DICOM files

        Input:
        - self: DICOMPlugin instance
        - si: Series array (3D or 4D)
        - dirname: given output directory
        - filename_template: template including %d for image number
        - opts: Output options (dict)
        """

        self.instanceNumber = 0

        save_shape = si.shape
        if si.ndim == 3:
            si.shape = (1,) + si.shape
        assert si.ndim == 4, "write_3d_series: input dimension %d is not 3D." % (si.ndim-1)
        if si.shape[0] != 1:
            raise ValueError("Attempt to write 4D image ({}) using write_3d_numpy".format(si.shape))
        slices = si.shape[1]
        if slices != si.slices:
            raise ValueError("write_3d_series: slices of dicom template (%d) differ from input array (%d)." % (si.slices, slices))

        self.calculate_rescale(si)
        logging.info("Smallest pixel value in series: {}".format(self.smallestPixelValueInSeries))
        logging.info("Largest  pixel value in series: {}".format(self.largestPixelValueInSeries))
        self.today = date.today().strftime("%Y%m%d")
        self.now   = datetime.now().strftime("%H%M%S.%f")
        self.serInsUid = get_uid()
        logging.debug("write_3d_series {} {}".format(dirname, self.serInsUid))

        ifile = 0
        for slice in range(slices):
            try:
                filename = filename_template % (slice)
            except TypeError as e:
                filename = filename_template + "_{}".format(slice)
            if dirname and not os.path.isdir(dirname):
                os.makedirs(dirname)
            self.write_slice(0, slice, si, dirname, filename, ifile)
            ifile += 1
        si.shape = save_shape

    def write_4d_numpy(self, si, dirname, filename_template, opts):
        """Write 4D Series image as DICOM files

        Input
        - self: DICOMPlugin instance
        - si[tag,slice,rows,columns]: Series array
        - dirname:  directory name
        - filename_template: template including %d for image number
        - opts: Output options (dict)

        - si.series_number is inserted into each dicom object
        - si.series_description is inserted into each dicom object
        - si.image_type: Dicom image type attribute
        - opts['output_sort']: Which tag will sort the output images (slice or tag)
        - opts['output_dir']: Store all images in a single or multiple directories
        """

        self.DicomHeaderDict = si.DicomHeaderDict

        # Defaults
        self.output_sort = imagedata.formats.SORT_ON_SLICE
        if 'output_sort' in opts:
            self.output_sort = opts['output_sort']
        self.output_dir = 'single'
        if 'output_dir' in opts:
            self.output_dir = opts['output_dir']

        self.instanceNumber = 0
        save_shape = si.shape
        # Should we allow to write 3D volume?
        if si.ndim == 3:
            si.shape = (1,) + si.shape
        if si.ndim != 4:
            raise ValueError("write_4d_numpy: input dimension %d is not 4D." % si.ndim)
        logging.debug("write_4d_numpy: si dtype {}, shape {}".format(si.dtype, si.shape))
        steps  = si.shape[0]
        slices = si.shape[1]
        if steps != len(si.tags[0]):
            raise ValueError("write_4d_series: tags of dicom template ({}) differ from input array ({}).".format(len(si.tags[0]), steps))
        if slices != si.slices:
            raise ValueError("write_4d_series: slices of dicom template ({}) differ from input array ({}).".format(si.slices, slices))

        self.calculate_rescale(si)
        logging.info("Smallest pixel value in series: {}".format(
            self.smallestPixelValueInSeries))
        logging.info("Largest  pixel value in series: {}".format(
            self.largestPixelValueInSeries))
        self.today = date.today().strftime("%Y%m%d")
        self.now   = datetime.now().strftime("%H%M%S.%f")
        self.seriesTime = self.getDicomAttribute(tag_for_name("AcquisitionTime"))
        self.serInsUid = get_uid()

        if self.output_sort == imagedata.formats.SORT_ON_SLICE:
            ifile = 0
            digits = len("{}".format(steps))
            for tag in range(steps):
                for slice in range(slices):
                    if self.output_dir == 'single':
                        dirn = dirname
                        if not os.path.isdir(dirn):
                            os.makedirs(dirn)
                        filename = filename_template % (ifile)
                    else: # self.output_dir == 'multi'
                        dirn = os.path.join(dirname, "{0}{1:0{2}}".format(
                            imagedata.formats.input_order_to_dirname_str(si.input_order),
                            tag, digits))
                        if not os.path.isdir(dirn):
                            os.makedirs(dirn)
                            ifile = 0
                        filename = filename_template % (ifile)
                    self.write_slice(tag, slice, si, dirn, filename, ifile)
                    ifile += 1
        else: # self.output_sort == imagedata.formats.SORT_ON_TAG:
            ifile = 0
            digits = len("{}".format(slices))
            for slice in range(slices):
                for tag in range(steps):
                    if self.output_dir == 'single':
                        dirn = dirname
                        if not os.path.isdir(dirn):
                            os.makedirs(dirn)
                        filename = filename_template % (ifile)
                    else: # self.output_dir == 'multi'
                        dirn = os.path.join(dirname,
                            "slice{0:0{1}}".format(slice, digits))
                        if not os.path.isdir(dirn):
                            os.makedirs(dirn)
                            ifile = 0
                        filename = filename_template % (ifile)
                    self.write_slice(tag, slice, si, dirn, filename, ifile)
                    ifile += 1
        si.shape = save_shape

    def write_slice(self, tag, slice, si, dirname, filename, ifile):
        """Write single slice to DICOM file

        Input:
        - self: DICOMPlugin instance
        - tag: tag index
        - slice: slice index
        - si: Series instance, including these attributes:
            slices
            sliceLocations
            DicomHeaderDict
            tags (not used)
            seriesNumber
            seriesDescription
            imageType
            frame
            spacing
            orientation
            imagePositions
        - dirname:  directory name
        - filename: file name, possible without '.dcm' extension
        - ifile: instance number in series
        """

        logging.debug("write_slice {} {}".format(filename, self.serInsUid))

        if np.issubdtype(si.dtype, np.floating):
            safe_si = np.nan_to_num(si)
        else:
            safe_si = si.copy()

        try:
            tg,fname,im = si.DicomHeaderDict[slice][tag]
        except IndexError:
            raise IndexError("Cannot address dicom_template.DicomHeaderDict[slice=%d][tag=%d]" % (slice, tag))
        except AttributeError:
            raise NoDICOMAttributes("Cannot write DICOM object when no DICOM attributes exist.")
        ds = self.construct_dicom(filename, im, self.serInsUid)
        self.copy_dicom_group(0x21, im, ds)
        self.copy_dicom_group(0x29, im, ds)

        # Add header information
        ds.SliceLocation = si.sliceLocations[slice]
        dz,dy,dx = si.spacing
        ds.PixelSpacing = [str(dy), str(dx)]
        ds.SliceThickness = str(dz)
        ipp=si.imagePositions[slice]
        if ipp.shape == (3,1): ipp.shape = (3,)
        z,y,x=ipp[:]
        ds.ImagePositionPatient = [str(x),str(y),str(z)]
        # Reverse orientation vectors from zyx to xyz
        ds.ImageOrientationPatient = [
                si.orientation[2], si.orientation[1], si.orientation[0],
                si.orientation[5], si.orientation[4], si.orientation[3]]
        ds.SeriesNumber = si.seriesNumber
        ds.SeriesDescription = si.seriesDescription
        ds.ImageType = "\\".join(si.imageType)
        try:
            ds.FrameOfReferenceUID = si.frameOfReferenceUID
        except:
            pass

        ds.SmallestPixelValueInSeries = self.smallestPixelValueInSeries.astype('uint16')
        ds.LargestPixelValueInSeries  = self.largestPixelValueInSeries.astype('uint16')
        ds[0x0028,0x0108].VR='US'
        ds[0x0028,0x0109].VR='US'
        ds.WindowCenter = self.center
        ds.WindowWidth  = self.width
        if np.issubdtype(safe_si.dtype, np.floating):
            ds.SmallestImagePixelValue    = ((safe_si[tag,slice].min()-self.b)/self.a).astype('uint16')
            ds.LargestImagePixelValue     = ((safe_si[tag,slice].max()-self.b)/self.a).astype('uint16')
            try:
                ds.RescaleSlope = "%f" % self.a
            except OverflowError:
                ds.RescaleSlope = "%d" % int(self.a)
            ds.RescaleIntercept = "%f" % self.b
        else:
            #if si.dtype == 'uint16':
            ds.SmallestImagePixelValue    = safe_si[tag,slice].min().astype('uint16')
            ds.LargestImagePixelValue     = safe_si[tag,slice].max().astype('uint16')
            if 'RescaleSlope'     in ds: del ds.RescaleSlope
            if 'RescaleIntercept' in ds: del ds.RescaleIntercept
        ds[0x0028,0x0106].VR='US'
        ds[0x0028,0x0107].VR='US'
        # General Image Module Attributes
        ds.InstanceNumber = ifile+1
        ds.ContentDate = self.today
        ds.ContentTime = self.now
        #ds.AcquisitionTime = self.add_time(self.seriesTime, timeline[tag])
        ds.Rows    = si.shape[2]
        ds.Columns = si.shape[3]
        self.insert_pixeldata(ds, safe_si[tag,slice])
        if os.path.splitext(filename)[1] == '.dcm':
            fn = filename
        else:
            fn = filename+'.dcm'
        ds.save_as(os.path.join(dirname,fn), write_like_original=False)

    def construct_dicom(self, filename, template, serInsUid):

        self.instanceNumber += 1
        SOPInsUID = uid_append_instance(serInsUid, self.instanceNumber)

        # Populate required values for file meta information
        file_meta = pydicom.dataset.Dataset()
        file_meta.MediaStorageSOPClassUID = template.SOPClassUID
        file_meta.MediaStorageSOPInstanceUID = SOPInsUID
        file_meta.ImplementationClassUID = "%s.1" % (self.root)
        file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        #file_meta.FileMetaInformationVersion = int(1).to_bytes(2,'big')
        #file_meta.FileMetaInformationGroupLength = 160

        # print("Setting dataset values...")

        # Create the FileDataset instance
        # (initially no data elements, but file_meta supplied)
        ds = pydicom.dataset.FileDataset(
                filename,
                {},
                file_meta=file_meta,
                preamble=b"\0" * 128)

        # Add the data elements
        # -- not trying to set all required here. Check DICOM standard
        copy_general_dicom_attributes(template, ds)
        ds.SeriesInstanceUID = serInsUid
        ds.SOPClassUID = template.SOPClassUID
        ds.SOPInstanceUID = SOPInsUID

        # Set the transfer syntax
        ds.is_little_endian = True
        ds.is_implicit_VR = True

        return ds

    def copy_dicom_group(self, groupno, ds_in, ds_out):
        sub_dataset = ds_in.group_dataset(groupno)
        for data_element in sub_dataset:
            if data_element.VR != "SQ":
                ds_out[data_element.tag] = ds_in[data_element.tag]

    def insert_pixeldata(self, ds, arr):
        """Insert pixel data into dicom object

        If float array, scale to uint16
        """

        if arr.dtype in self.smallint:
                ds.PixelData = arr.tostring()
        elif np.issubdtype(arr.dtype, np.bool_):
                ds.PixelData = arr.astype('uint16').tostring()
        else:
                # Other high precision data type, like float
                rescaled = (arr-self.b)/self.a
                ds.PixelData = rescaled.astype('uint16').tostring()

        ds[0x7fe0,0x0010].VR='OW'
        ds.SamplesPerPixel = 1
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15

    def calculate_rescale(self, arr):
        """y = ax + b
        x in 0:65535 correspond to y in ymin:ymax
        2^16 = 65536 possible steps in 16 bits dicom
        """
        # Window center/width
        ymin = np.nanmin(arr)
        ymax = np.nanmax(arr)
        self.center = (ymax-ymin)/2
        self.width  = max(1, ymax-ymin)
        # y = ax + b, 
        if arr.dtype in self.smallint or np.issubdtype(arr.dtype, np.bool_):
            # No need to rescale
            self.a = None
            self.b = None
            self.smallestPixelValueInSeries = arr.min()
            self.largestPixelValueInSeries  = arr.max()
        else:
            # Other high precision data type, like float
            # Must rescale data
            self.b = ymin
            if math.fabs(ymax-ymin) >1e-6:
                self.a = (ymax-ymin)/65535.
            else:
                self.a = 1.0
            logging.debug("Rescale slope %f, rescale intercept %s" % (self.a, self.b))
            self.smallestPixelValueInSeries = (ymin-self.b)/self.a
            self.largestPixelValueInSeries  = (ymax-self.b)/self.a
        return self.a, self.b

    def add_time(self, now, add):
        """Add time to present time now
        Input:
        - now: string hhmmss.ms
        - add: float [s]
        Output:
        - newtime: string hhmmss.ms
        """
        tnow = datetime.strptime(now, "%H%M%S.%f")
        s = int(add)
        ms = (add-s)*1000.
        tadd = timedelta(seconds=s, milliseconds=ms)
        tnew = tnow + tadd
        return tnew.strftime("%H%M%S.%f")
        
    #def copy(self, other=None):
    #    if other is None: other = DICOMPlugin()
    #    return AbstractPlugin.copy(self, other=other)

    def get_tag(self, im, input_order, opts):
        if input_order is None:
            return 0
        if input_order == imagedata.formats.INPUT_ORDER_NONE:
            return 0
        elif input_order == imagedata.formats.INPUT_ORDER_TIME:
            #return im.AcquisitionTime
            assert 'input_options' in opts, 'No input_options in opts'
            if 'TriggerTime' in opts['input_options']:
                return(float(im.TriggerTime))
            elif 'InstanceNumber' in opts['input_options']:
                return(float(im.InstanceNumber))
            else:
                if '.' in im.AcquisitionTime:
                    tm = datetime.strptime(im.AcquisitionTime, "%H%M%S.%f")
                else:
                    tm = datetime.strptime(im.AcquisitionTime, "%H%M%S")
                td = timedelta(hours=tm.hour,
                                minutes=tm.minute,
                                seconds=tm.second,
                                microseconds=tm.microsecond)
                return td.total_seconds()
        elif input_order == imagedata.formats.INPUT_ORDER_B:
            csa_head = csa.get_csa_header(im)
            try:
                value = csa.get_b_value(csa_head)
            except:
                raise ValueError("Unable to extract b value from header.")
            return value
        elif input_order == imagedata.formats.INPUT_ORDER_FA:
            return float(im.FlipAngle)
        elif input_order == imagedata.formats.INPUT_ORDER_TE:
            return float(im.EchoTime)
        else:
            raise(UnknownTag("Unknown numerical input_order {}.".format(imagedata.formats.input_order_to_str(input_order))))

    def simulateAffine(self):
        shape = (
            self.getDicomAttribute(tag_for_name('Rows')),
            self.getDicomAttribute(tag_for_name('Columns')))
        iop = self.getDicomAttribute(tag_for_name('ImageOrientationPatient'))
        iop = np.array(list(map(float, iop)))
        iop = np.array(iop).reshape(2, 3).T
        logging.debug('simulateAffine: iop\n{}'.format(iop))
        s_norm = np.cross(iop[:, 1], iop[:, 0])
        # Rotation matrix
        R = np.eye(3)
        R[:, :2] = np.fliplr(iop)
        R[:, 2] = s_norm
        if not np.allclose(np.eye(3), np.dot(R, R.T), atol=5e-5):
            raise ValueErrorWrapperPrecisionError('Rotation matrix not nearly orthogonal')

        pix_space = self.getDicomAttribute(tag_for_name('PixelSpacing'))
        zs = self.getDicomAttribute(tag_for_name('SpacingBetweenSlices'))
        if zs is None:
            zs = self.getDicomAttribute(tag_for_name('SliceThickness'))
            if zs is None:
                zs = 1
        zs = float(zs)
        pix_space = list(map(float, pix_space))
        vox = tuple(pix_space + [zs])
        logging.debug('simulateAffine: vox {}'.format(vox))

        ipp = self.getDicomAttribute(tag_for_name('ImagePositionPatient'))
        ipp = np.array(list(map(float, ipp)))
        logging.debug('simulateAffine: ipp {}'.format(ipp))

        orient = R
        logging.debug('simulateAffine: orient\n{}'.format(orient))

        aff = np.eye(4)
        aff[:3, :3] = orient * np.array(vox)
        aff[:3, 3] = ipp
        logging.debug('simulateAffine: aff\n{}'.format(aff))

    def create_affine(self, hdr):
        """Function to generate the affine matrix for a dicom series
        This method was based on
        (http://nipy.org/nibabel/dicom/dicom_orientation.html)
        :param sorted_dicoms: list with sorted dicom files
        """

        slices = hdr['slices']

        # Create affine matrix
        # (http://nipy.sourceforge.net/nibabel/dicom/dicom_orientation.html#dicom-slice-affine)
        iop = self.getDicomAttribute(tag_for_name('ImageOrientationPatient'))
        image_orient1 = np.array(iop[0:3])
        image_orient2 = np.array(iop[3:6])

        pix_space = self.getDicomAttribute(tag_for_name('PixelSpacing'))
        delta_r = float(pix_space[0])
        delta_c = float(pix_space[1])

        ipp = self.getDicomAttribute(tag_for_name('ImagePositionPatient'))
        image_pos = np.array(ipp)

        ippn = self.getDicomAttribute(tag_for_name('ImagePositionPatient'),
                slice=slices-1)
        last_image_pos = np.array(ippn)

        if slices == 1:
            # Single slice
            step = [0, 0, -1]
        else:
            step = (image_pos - last_image_pos) / (1 - slices)

        # check if this is actually a volume and not all slices on the same
        # location
        if np.linalg.norm(step) == 0.0:
            raise ValueError("NOT_A_VOLUME")

        affine = np.matrix([
            [-image_orient1[0] * delta_c, -image_orient2[0] * delta_r, -step[0], -image_pos[0]],
            [-image_orient1[1] * delta_c, -image_orient2[1] * delta_r, -step[1], -image_pos[1]],
            [ image_orient1[2] * delta_c,  image_orient2[2] * delta_r,  step[2],  image_pos[2]],
            [0, 0, 0, 1]
            ])
        logging.debug('create_affine: affine\n{}'.format(affine))
