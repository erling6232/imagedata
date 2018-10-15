#!/usr/bin/env python3

"""Read/Write image files using Xite (biff format)
"""

# Copyright (c) 2018 Erling Andersen, Haukeland University Hospital,
# Bergen, Norway

import os.path
import sys
import logging
import struct
import fs
import numpy as np

import imagedata.formats
from imagedata.formats.abstractplugin import AbstractPlugin

class ImageTypeError(Exception):
    """Thrown when trying to load or save an image of unknown type.
    """
    pass

class PixelTypeNotSupported(Exception):
    """Thrown when pixel type is not supported.
    """
    pass

class VaryingImageSize(Exception):
    """Thrown when the bands are of varying size.
    """
    pass

class NoBands(Exception):
    """Thrown when no bands are defined.
    """
    pass

class BiffPlugin(AbstractPlugin):
    """Read/write Xite biff files."""

    name = "biff"
    description = "Read and write Xite biff files."
    authors = "Erling Andersen"
    version = "1.0.0"
    url = "www.helse-bergen.no"

    Ilittle_endian_mask = 0x010000
    Ipixtyp_mask = 0x0ffff
    Icolor_mask = 0x0100


    def __init__(self):
        super(BiffPlugin, self).__init__(self.name, self.description,
              self.authors, self.version, self.url)

    def read(self, urls, files, pre_hdr, input_order, opts):
        """Read image data

        Input:
        - urls: list of urls to image data
        - files: list of files inside a single url.
            = None: No files given
        - pre_hdr: Pre-filled header dict.  Might be None
        - input_order
        - opts: Input options (dict)
        Output:
        - hdr: Header dict
            input_format
            input_order
            input_sort
            slices
            tags
        - si[tag,slice,rows,columns]: numpy array
        """
        
        hdr = {}
        hdr['input_format'] = self.name
        hdr['input_order'] = input_order

        if len(urls) > 1 and files is not None:
            raise FilesGivenForMultipleURLs("Files shall not be given when there are multiple URLs")

        # Scan filelist to determine data size
        url = urls[0]
        logging.debug("biffplugin.read: url: {} {}".format(type(url), url))
        with fs.open_fs(url) as archive:
            if files is None:
                files = archive.walk.files()
            for path in files:
                logging.debug("biffplugin.read filehandle {}".format(path))
                if archive.hassyspath(path):
                    filename = archive.getsyspath(path)
                    tmp_fs = None
                else:
                    # Copy file to a local instance
                    tmp_fs = fs.tempfs.TempFS()
                    fs.copy.copy_fs(archive,path, tmp_fs,os.path.basename(path))
                    filename = tmp_fs.getsyspath(os.path.basename(path))
                logging.debug("biffplugin.read load filename {}".format(filename))
                try:
                    self.Iopen_image(filename, 'r')
                except:
                    raise imagedata.formats.NotImageError('{} does not look like a BIFF file'.format(path))
                logging.debug("biffplugin.read img")
                if self.nbands == 0:
                    raise imagedata.formats.EmptyImageError('{} has no image'.format(path))
                ny = self.Iysize()
                nx = self.Ixsize()
                logging.debug("biffplugin.read ny {} nx {}".format(ny,nx))
                if 'input_shape' in opts and opts.input_shape:
                    nt,delim,nz = opts.input_shape.partition('x')
                    if len(nz) == 0:
                        raise BadShapeGiven('input_shape {} is not like (t)x(z)'.format(opts.input_shape))
                    nt,nz = (int(nt), int(nz))
                    if nt*nz != self.nbands:
                        raise BadShapeGiven('input_shape {} does not match {} bands'.format(opts.input_shape, self.nbands))
                else:
                    # Assume the bands are slices
                    nt,nz = (1,self.nbands)
                dtype,dformat = self.dtype_from_biff(self.Ipixtyp())
                if dtype == np.complex64:
                    dfloat = np.float32
                elif dtype == np.complex128:
                    dfloat = np.float64
                si = np.zeros([nt,nz,ny,nx], dtype)
                iband = 0 # Band numbers start from zero
                if 'input_sort' in opts and opts.input_sort == imagedata.formats.SORT_ON_TAG:
                    hdr['input_sort'] = imagedata.formats.SORT_ON_TAG
                    for slice in range(nz):
                        for tag in range(nt):
                            si[tag,slice,:,:] = self.Iread_band(iband)
                            iband += 1
                else: # opts.input_sort == imagedata.formats.SORT_ON_SLICE:
                    hdr['input_sort'] = imagedata.formats.SORT_ON_SLICE
                    for tag in range(nt):
                        for slice in range(nz):
                            si[tag,slice,:,:] = self.Iread_band(iband)
                            if tag==0 and slice==0:
                                logging.debug('BiffPlugin.read: si(0) {}'.format(si[0,0,0,:4]))
                            iband += 1
        # Overcome a problem where the first pixel is set to some arbitrary number
        #si[0,0,0,0] = 0
        logging.debug('BiffPlugin.read: si(0) {}'.format(si[0,0,0,0]))

        logging.debug('BiffPlugin.read: nt,nz,ny,nx: {} {} {} {}'.format(nt,nz,ny,nx))
        hdr['slices'] = nz
        hdr['tags']   = {}
        for slice in range(nz):
            hdr['tags'][slice] = np.array([t for t in range(nt)])

        # Add any DICOM template
        if pre_hdr is not None:
            hdr.update(pre_hdr)

        return hdr,si

    def write_3d_numpy(self, si, dirname, filename_template, opts):
        """Write 3D numpy image as Xite biff file

        Input:
        - self: BiffPlugin instance
        - si: Series array (3D or 4D), including these attributes:
            input_sort
            slices
            tags
        - dirname: directory name
        - filename_template: template including %d for image number
        - opts: Output options (dict)
        """

        logging.debug('BiffPlugin.write_3d_series: si {}'.format(type(si)))
        logging.debug('BiffPlugin.write_3d_series: si {}'.format(si.__dict__))
        self.slices               = si.slices
        self.tags                 = si.tags
        self.output_dir           = opts.output_dir

        logging.info("Data shape write: {}".format(imagedata.formats.shape_to_str(si.shape)))
        if si.ndim == 3:
            si.shape = (1,) + si.shape
        assert si.ndim == 4, "write_3d_series: input dimension %d is not 3D." % (si.ndim-1)
        if si.shape[0] != 1:
            raise ValueError("Attempt to write 4D image ({}) using write_3d_numpy".format(si.shape))
        slices,ny,nx = si.shape[1:]
        if slices != si.slices:
            raise ValueError("write_3d_series: slices of dicom template ({}) differ from input array ({}).".format(si.slices, slices))

        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        try:
            filename = filename_template % (0)
        except TypeError:
            filename = filename_template
        filename = os.path.join(dirname, filename)

        if np.issubdtype(si.dtype, np.floating):
            self.arr=np.nan_to_num(si)
        else:
            #self.arr=si.copy()
            self.arr=np.array(si)
        self.pixtyp = self.pixtyp_from_dtype(self.arr.dtype)

        try:
            if opts.serdes is not None:
                self.descr = opts.serdes
            else:
                self.descr = ''
        except:
            self.descr = ''

        self.Iset_text('')
        self.Iopen_image(filename, 'w')

        iband = 0
        for slice in range(slices):
            self.Iwrite_band(iband, self.arr[0,slice])
            iband += 1
        logging.debug('BiffPlugin.write_3d_series: filename {}'.format(filename))
        self.Iwrite_text()
        self.Iclose_image()

    def write_4d_numpy(self, si, dirname, filename_template, opts):
        """Write 4D numpy image as Xite biff file

        Input:
        - self: BiffPlugin instance
        - si[tag,slice,rows,columns]: Series array, including these attributes:
            input_sort
            slices
            tags
        - dirname: directory name
        - filename_template: template including %d for image number
        - opts: Output options (dict)
        """

        self.slices               = si.slices
        self.tags                 = si.tags
        self.output_dir           = opts.output_dir

        # Should we allow to write 3D volume?
        if si.ndim == 3:
            si.shape = (1,) + si.shape
        if si.ndim != 4:
            raise ValueError("write_4d_numpy: input dimension %d is not 4D.".format(si.ndim))

        logging.debug("write_4d_numpy: si dtype {}, shape {}, sort {}".format(
            si.dtype, si.shape,
            imagedata.formats.sort_on_to_str(opts.output_sort)))

        logging.debug("write_4d_numpy: si.tags {} si.slices {}".format(len(si.tags[0]), si.slices))
        steps,slices,ny,nx = si.shape[:]
        if steps != len(si.tags[0]):
            raise ValueError("write_4d_series: tags of dicom template ({}) differ from input array ({}).".format(len(si.tags[0]), steps))
        if slices != si.slices:
            raise ValueError("write_4d_series: slices of dicom template ({}) differ from input array ({}).".format(si.slices, slices))

        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        
        if 'output_sort' not in opts or opts.output_sort is None:
            self.output_sort = si.input_sort
        else:
            self.output_sort = opts.output_sort

        if np.issubdtype(si.dtype, np.floating):
            self.arr=np.nan_to_num(si)
        else:
            self.arr=si.copy()
        self.pixtyp = self.pixtyp_from_dtype(self.arr.dtype)

        try:
            if opts.serdes is not None:
                self.descr = opts.serdes
            else:
                self.descr = ''
        except:
            self.descr = ''

        if self.output_dir == 'single':
            try:
                filename = filename_template % (0)
            except TypeError:
                filename = filename_template
            filename = os.path.join(dirname, filename)
            self.Iset_text('')
            self.Iopen_image(filename, 'w')
            iband = 0
            if self.output_sort == imagedata.formats.SORT_ON_TAG:
                for slice in range(slices):
                    for tag in range(steps):
                        self.Iwrite_band(iband, self.arr[tag,slice])
                        iband += 1
            else: # default: imagedata.formats.SORT_ON_SLICE:
                for tag in range(steps):
                    for slice in range(slices):
                        self.Iwrite_band(iband, self.arr[tag,slice])
                        iband += 1
            logging.debug('BiffPlugin.write_4d_series: filename {}'.format(filename))
            self.Iwrite_text()
            self.Iclose_image()
        else: # self.output_dir == 'multi'
            if self.output_sort == imagedata.formats.SORT_ON_TAG:
                digits = len("{}".format(slices))
                for slice in range(slices):
                    filename="slice{0:0{1}}".format(slice, digits)
                    filename = os.path.join(dirname, filename)
                    self.Iset_text('')
                    self.Iopen_image(filename, 'w')
                    iband = 0
                    for tag in range(steps):
                        self.Iwrite_band(iband, self.arr[tag,slice])
                        iband += 1
                    logging.debug('BiffPlugin.write_4d_series: filename {}'.format(filename))
                    self.Iwrite_text()
                    self.Iclose_image()
            else: # self.output_sort == imagedata.formats.SORT_ON_SLICE:
                digits = len("{}".format(steps))
                for tag in range(steps):
                    filename="{0}{1:0{2}}".format(imagedata.formats.input_order_to_dirname_str(si.input_order),
                                                  tag, digits)
                    filename = os.path.join(dirname, filename)
                    self.Iset_text('')
                    self.Iopen_image(filename, 'w')
                    iband = 0
                    for slice in range(slices):
                        self.Iwrite_band(iband, self.arr[tag,slice])
                        iband += 1
                    logging.debug('BiffPlugin.write_4d_series: filename {}'.format(filename))
                    self.Iwrite_text()
                    self.Iclose_image()


    def dtype_from_biff(self, pixtyp):
        """Return NumPy dtype for given Xite pixel type

        Input:
        - pixtyp: Xite pixel type
        Output:
        - dtype: Corresponding NumPy dtype
        - dformat: Format code, to be used by the struct module
                   Complex numbers are not supported directly by struct.
                   These data types are coded as 'ff' and 'dd' (length 2)
        """
        pixtyp = pixtyp & self.Ipixtyp_mask
        if pixtyp == -1:
            raise PixelTypeNotSupported("Pixel type 'Unknown' is not supported")
        elif pixtyp == 0:
            raise PixelTypeNotSupported("Pixel type 'bit' is not supported")
        elif pixtyp == 1:
            raise PixelTypeNotSupported("Pixel type 'bit2' is not supported")
        elif pixtyp == 2:
            raise PixelTypeNotSupported("Pixel type 'nibble' is not supported")
        elif pixtyp == 3:
            return np.uint8,'B'
        elif pixtyp == 4:
            return np.int8,'b'
        elif pixtyp == 5:
            return np.uint16,'H'
        elif pixtyp == 6:
            return np.int16,'h'
        elif pixtyp == 7:
            return np.int32,'i'
        elif pixtyp == 8:
            return np.float32,'f'
        elif pixtyp == 9:
            return np.complex64,'ff'
        elif pixtyp == 10:
            return np.float64,'d'
        elif pixtyp == 11:
            return np.complex128,'dd'
        else:
            raise PixelTypeNotSupported("Unknown pixel type.")

    def pixtyp_from_dtype(self, dtype):
        """Return Xite pixel type for given NumPy dtype

        Input:
        - dtype:  NumPy dtype
        Output:
        - pixtyp: Corresponding Xite pixel type
        """
        if dtype == np.uint8:
            return 3
        elif dtype == np.int8:
            return 4
        elif dtype == np.uint16 or dtype == '>i2' or dtype == '<i2':
            return 5
        elif dtype == np.int16:
            return 6
        elif dtype == np.int32:
            return 7
        elif dtype == np.float32:
            return 8
        elif dtype == np.complex64:
            return 9
        elif dtype == np.float64:
            return 10
        elif dtype == np.complex128:
            return 11
        else:
            raise PixelTypeNotSupported("NumPy dtype {} not supported in Xite.".format(dtype))

    def Iopen_image(self, filename, mode):
        """Open the file 'filename' with 'mode' access, and
        connect it to self image

        Input:
        - filename
        - mode: access mode ('r' or 'w')
        Output:
        - self: image struct
        """
        assert mode=='r' or mode=='w', "Wrong access mode {} given".format(mode)

        self.status = None
        if mode == 'r':
            self.f = open(filename, 'rb')
            self.Iread_info()
            self.status = 'read'
        elif mode == 'w':
            if len(os.path.splitext(filename)[1]) == 0:
                filename = filename + '.biff'
            self.f = open(filename, 'wb')
            self.Iset_info()
            self.status = 'write'
        else:
            raise ValueError('Unknown access mode "{}"'.format(mode))

    def Iread_info(self):
        """Initiate image info field with data from file.
        Called by Iopen_image when image is opened
        with readonly or readwrite access.

        Input:
        - self.f : file descriptor
        Output:
        - self : image info
        - self.bands : band info (dict of bands)
        """
        header = self.f.read(96)
        magic = header[:4]
        #logging.debug('Iread_info: magic {}'.format(magic))
        if magic != b'BIFF':
            logging.debug('Iread_info: magic {} giving up'.format(magic))
            raise ImageTypeError('File is not BIFF format')
        self.param = [0,0,0,0,0,0,0,0]
        try:
            (magic, col,
                self.ninfoblks, self.nbandblks, self.ntextblks, self.nblocks,
                title,
                self.param[0], self.param[1], self.param[2], self.param[3], 
                self.param[4], self.param[5], self.param[6], self.param[7], 
                nfreechars, self.nbands) = struct.unpack('>4s4s4i32s10i', header)
        except Exception as e:
            logging.debug('Iread_info: exception\n{}'.format(e))
            raise ImageTypeError('{}'.format(e))
        self.col = col == b'C'
        logging.debug('Iread_info: magic {} colour {}'.format(magic, self.col))
        logging.debug('Iread_info: ninfoblks {} nbandblks {} ntextblks {} nblocks {}'.format(self.ninfoblks, self.nbandblks, self.ntextblks, self.nblocks))
        self.title = title.decode('utf-8')
        logging.debug('Iread_info: self.title {}'.format(self.title))
        logging.debug('Iread_info: self.param {}'.format(self.param))
        self.nchars = self.ntextblks * 512 - nfreechars
        logging.debug('Iread_info: nfreechars {} self.nchars {}'.format(nfreechars, self.nchars))
        # Initiate character position to 0
        self.pos = 0
        self.text = None
        self.textbufblks = 0
        logging.debug('Iread_info: self.nbands {}'.format(self.nbands))

        self.bands = {}
        # Read data concerning each band
        for bandnr in range(self.nbands):
            band_header = self.f.read(32)
            try:
                (self.pt, self.xsz, self.ysz, self.xst, self.yst, self.xmg,
                    self.ymg, _) = struct.unpack('>8i', band_header)
            except Exception as e:
                logging.debug('Iread_info: exception\n{}'.format(e))
                raise ImageTypeError('{}'.format(e))
            pt = self.pt &  self.Ipixtyp_mask
            self.little_endian = (self.pt & self.Ilittle_endian_mask) != 0
            #logging.debug('Band {} pt {} xsz {} ysz {} xst {} yst {} xmg {} ymg {} little {}'.format(bandnr, pt, self.xsz, self.ysz,
            #            self.xst, self.yst, self.xmg, self.ymg, self.little_endian))

            self.bands[bandnr] = self.Init_band(self.pt | (self.col * self.Icolor_mask), self.xsz, self.ysz)
        # Skip file to beginning of next 512 bytes block
        rest = 0 - 96 - self.nbands*32;
        while rest < 0: rest += 512
        #logging.debug('Iread_info: skipping {} bytes'.format(rest))
        _ = self.f.read(rest)

    def Init_band(self, pt, xsize, ysize):
        """Create an "absent" band. Storage for pixels will
        not be allocated, but the band info field will
        be allocated and initialized

        Input:
        - pt : pixel type
        - xsize, ysize: size of band
        """
        binfo = {}
        binfo['status']     = 'absent';
        binfo['xsize']      = xsize;
        binfo['ysize']      = ysize;
        binfo['xstart']     = 1;
        binfo['ystart']     = 1;
        binfo['xmag']       = 1;
        binfo['ymag']       = 1;
        binfo['pixtyp']     = pt; # Also stores info about endian-ness and colortable
        binfo['roi_xsize']  = xsize;
        binfo['roi_ysize']  = ysize;
        binfo['roi_xstart'] = 1;
        binfo['roi_ystart'] = 1;
        return binfo
        
    def Iread_band(self, bandnr):
        """read BIFF band from file

        Iread_band reads band number 'bandnr' from file into self image.
        The byte order of the stored band is compared to the byte
        order of the host computer. If they don't match, the
        bytes are swapped after reading.

        Input:
        - bandnr : band number to read
        """

        #logging.debug('Iread_band: band {}'.format(bandnr))
        if bandnr < 0 or bandnr >= self.nbands:
            raise ValueError("Band number {} is out of range (0..{}".format(bandnr,
                self.nbands-1))

        # Calculate start of band in the file
        start = self.ninfoblks * 512
        for bn in range(bandnr):
            start += self.num_blocks_band(bn) * 512

        # Set file pointer to start of band
        #logging.debug('Iread_band: seek file at {}'.format(start))
        try:
            self.f.seek(start)
        except Exception as e:
            logging.debug('Iread_band: seek file: {}'.format(e))
            raise

        # Create band
        #logging.debug('Iread_band: create band {}'.format(bandnr))
        if bandnr not in self.bands:
            logging.debug('Iread_band: self.bands.keys()={}'.format(self.bands.keys()))
            raise ValueError('Band {} is not in self.bands'.format(bandnr))
        binfo = self.bands[bandnr]
        pt = binfo['pixtyp'] & self.Ipixtyp_mask
        dtype,dformat = self.dtype_from_biff(pt)
        if dtype == np.complex64:
            dfloat = np.float32
        elif dtype == np.complex128:
            dfloat = np.float64
        band = np.empty([binfo['ysize'], binfo['xsize']], dtype=dtype)

        # Determine endian-ness of system and input file
        if self.Ipixsize(binfo['pixtyp']) // 8 > 1:
            endian = '<'
            #logging.debug('Iread_band: little-endian input file')
        else:
            endian = '>'
            #logging.debug('Iread_band: big-endian input file')
        binfo['status'] == 'normal'

        # Read the band
        ny,nx = binfo['ysize'], binfo['xsize']
        buffer = self.f.read(self.Ibandsize(bandnr))
        if len(dformat) == 2:
            # Unpack as floats, then view as complex
            arr = np.asarray(struct.unpack(endian+str(2*ny*nx)+dformat[0],buffer), dtype=dfloat)
            band[...] = arr.view(dtype=dtype).reshape(band.shape)
        else:
            band[...] = np.asarray(
                struct.unpack(endian+str(ny*nx)+dformat[0],buffer), dtype=dtype).reshape(band.shape)
        return band

    def Ibandsize(self, bn):
        """Return BIFF band size in bytes"""

        if bn is None: return None
        binfo = self.bands[bn]
        bitsize = self.Ipixsize(binfo['pixtyp']) * binfo['xsize'] * binfo['ysize']
        return (bitsize+7) // 8

    def Ipixsize(self, pixtyp):
        """Return the BIFF pixel size in bits"""

        pt = pixtyp & self.Ipixtyp_mask
        dtype,_ = self.dtype_from_biff(pt)
        arr = np.array(0, dtype=dtype)
        #logging.debug('Ipixsize: for {} ({}) is size {} bits'.format(pt, pixtyp, arr.dtype.itemsize * 8))
        return arr.dtype.itemsize * 8

    def Iendian(self, bandnr):
        binfo = self.bands[bandnr]
        if binfo['pixtyp'] & self.Ilittle_endian_mask:
            logging.debug('Iendian: little')
            return 'little'
        else:
            logging.debug('Iendian: big')
            return 'big'

    def Iysize(self):
        if self.nbands < 1:
            raise NoBands('No bands')
        ysize = self.bands[0]['ysize']
        for bandnr in range(1, self.nbands):
            if self.bands[bandnr]['ysize'] != ysize:
                raise VaryingImageSize('Varying ysize of the bands not supported.')
        return ysize

    def Ixsize(self):
        if self.nbands < 1:
            raise NoBands('No bands')
        xsize = self.bands[0]['xsize']
        for bandnr in range(1, self.nbands):
            if self.bands[bandnr]['xsize'] != xsize:
                raise VaryingImageSize('Varying xsize of the bands not supported.')
        return xsize

    def Ipixtyp(self):
        if self.nbands < 1:
            raise NoBands('No bands')
        pixtyp = self.bands[0]['pixtyp']
        for bandnr in range(1, self.nbands):
            if self.bands[bandnr]['pixtyp'] != pixtyp:
                raise VaryingImageSize('Varying pixtyp of the bands not supported.')
        return pixtyp

    def num_blocks_band(self, bn):
        return (self.Ibandsize(bn) + 511) // 512

    def Iset_info(self):
        """Write BIFF image header
        """
        format_image = '>4s4s4i32s10i'
        format_band  = '>8i'
        magic = b'BIFF'
        col = b'-UIO'

        # Prepare self.bands structure
        if self.output_dir == 'single':
            self.nbands = self.slices * len(self.tags[0])
        else: # self.output_dir == 'multi'
            if self.output_sort == imagedata.formats.SORT_ON_TAG:
                self.nbands = len(self.tags[0])
            else: # self.output_sort == imagedata.formats.SORT_ON_SLICE:
                self.nbands = self.slices
        #logging.debug('Iset_info: self.nbands {}'.format(self.nbands))
        self.bands = {}
        for bandnr in range(self.nbands):
            pt = self.pixtyp &  self.Ipixtyp_mask
            ysz,xsz = self.arr.shape[2:]
            self.bands[bandnr] = self.Init_band(pt, xsz, ysz)

        # Calculate BIFF image header
        self.ntextblks = len(self.text) // 512
        self.nblocks = 0 # Not implemented
        title = self.descr.encode('utf-8')
        #logging.debug('Iset_info: title {}'.format(title))
        self.param = [0,0,0,0,0,0,0,0]
        nfreechars = self.ntextblks * 512 - len(self.text)
        #logging.debug('Iset_info: nfreechars {} self.nchars {}'.format(nfreechars, len(self.text)))

        # 13 bands in first block, 16 in later ones
        self.ninfoblks = (((self.nbands + 2) // 16) + 1)
        size = 0
        for bandnr in range(self.nbands):
            size += self.num_blocks_band(bandnr)
        self.nbandblks = size
        #logging.debug('Iset_info: ninfoblks {} nbandblks {} ntextblks {} nblocks {}'.format(self.ninfoblks, self.nbandblks, self.ntextblks, self.nblocks))

        # Write BIFF image header
        try:
            self.f.write(
                struct.pack(format_image,
                    magic, col,
                    self.ninfoblks, self.nbandblks, self.ntextblks, self.nblocks,
                    title,
                    self.param[0], self.param[1], self.param[2], self.param[3], 
                    self.param[4], self.param[5], self.param[6], self.param[7], 
                    nfreechars, self.nbands)
                )
        except Exception as e:
            logging.debug('Iset_info: exception\n{}'.format(e))
            raise ImageTypeError('{}'.format(e))

        # Write data concerning each band
        for bandnr in range(self.nbands):
            pt = self.pixtyp &  self.Ipixtyp_mask
            pt = pt ^ self.Ilittle_endian_mask
            ysz,xsz = self.arr.shape[2:]
            (yst,xst,ymg,xmg) = (1,1,1,1)
            dummy = 0
            #logging.debug('Band {} pt {} xsz {} ysz {} xst {} yst {} xmg {} ymg {}'.format(bandnr, pt, xsz, ysz,
            #            xst, yst, xmg, ymg))
            try:
                self.f.write(
                    struct.pack(format_band,
                        pt, xsz, ysz,
                        xst, yst, xmg, ymg, dummy)
                    )
            except Exception as e:
                logging.debug('Iset_info: exception\n{}'.format(e))
                raise ImageTypeError('{}'.format(e))

        # Skip file to beginning of next 512 bytes block
        rest = 0 - struct.calcsize(format_image) - self.nbands*struct.calcsize(format_band)
        while rest < 0: rest += 512
        #logging.debug('Iset_info: skipping {} bytes'.format(rest))
        self.f.write(rest * b'\x00')

    def Imake_image(self, nbands, title, pt, xsize, ysize):
        """Create the whole BIFF image data structure, with 'nbands'
        bands, 'title' as title, every band of horizontal
        size 'xsize', vertical size 'ysize', and pixel type
        'pixtyp'.

        NOTE: The pixel values will not be initialized.
        """
        pass

    def Iset_text(self, txt):
        self.text = txt

    def Iwrite_text(self):
        """Write text field from BIFF image to file

        Input:
        - self.text - text field
        """
        self.f.write(self.text.encode('utf-8'))
        rest = 512 - len(self.text.encode('utf-8'))
        while rest < 0: rest += 512
        logging.debug('Iwrite_text: skipping {} bytes'.format(rest))
        self.f.write(rest * b'\x00')

    def Iclose_image(self):
        self.f.close()

    def Iwrite_band(self, bandnr, arr):
        """Write a BIFF band to file

        Input:
        - bandnr : band number
        - arr    : pixel data, 2D NumPy array
        """
        
        binfo = self.bands[bandnr]

        start = self.ninfoblks * 512
        for bn in range(bandnr):
            start += self.num_blocks_band(bn) * 512

        self.f.seek(start)

        dtype,dformat = self.dtype_from_biff(binfo['pixtyp'])
        if dtype == np.complex64:
            dfloat = np.float32
        elif dtype == np.complex128:
            dfloat = np.float64

        # Will write little endian output file
        #endian = '>' # big endian
        endian = '<' # little endian

        # Write the pixel data
        ny,nx = binfo['ysize'], binfo['xsize']
        n = ny*nx
        if len(dformat) == 2:
            # View as complex, pack as floats
            arr = arr.view(dtype=dfloat)
            n = n * 2
        self.f.write(
            struct.pack(endian+str(n)+dformat[0], *arr.flatten(order='C'))
            )

        rest = struct.calcsize(endian+str(n)+dformat[0]) % 512
        if rest != 0:
            logging.debug('Iwrite_band: filling {} bytes'.format(512-rest))
            self.f.write((512-rest) * b'\x00')
