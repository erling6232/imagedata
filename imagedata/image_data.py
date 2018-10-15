#!/usr/bin/env python3

"""Read/write image data to file(s). Handles DICOM, Nifti, VTI and mhd."""

# Copyright (c) 2013-2018 Erling Andersen, Haukeland University Hospital, Bergen, Norway

import sys
import os.path
import argparse
import logging
import numpy as np
import nibabel
import pydicom
import imagedata
import imagedata.cmdline
import imagedata.formats
from imagedata.series import Series

logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
       '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
#logger.setLevel(logging.DEBUG)
#logger.setLevel(logging.INFO)
logger.setLevel(logging.WARNING)
#logger.setLevel(logging.ERROR)

def dump():
    parser = argparse.ArgumentParser()
    imagedata.cmdline.add_argparse_options(parser)
    parser.add_argument("in_dirs", nargs='+',
            help="Input directories and files")
    args = parser.parse_args()
    if len(args.output_format) < 1: args.output_format=['dicom']
    print("Output format: %s, %s, in %s directory." % (args.output_format, imagedata.formats.sort_on_to_str(args.output_sort), args.output_dir))

    reader = imagedata.formats.find_plugin('dicom')
    logging.debug("in_dirs {}".format(args.in_dirs))
    urls,files = imagedata.readdata.sanitize_urls(args.in_dirs)

    hdr,shape = reader.read_headers(urls, files, args.input_order, args)

    StuInsUID = {}
    SerInsUID = {}
    SerTim = {}
    SeqNam = {}
    AcqNum = {}
    ImaTyp = {}
    Echo = {}
    f = open('files','w')
    for slice in hdr['DicomHeaderDict']:
        for tag,member_name,im in hdr['DicomHeaderDict'][slice]:
            #print('{}'.format(im))
            f.write('{}\n'.format(member_name[1]))
            if im.StudyInstanceUID not in StuInsUID:
                StuInsUID[im.StudyInstanceUID] = 0
            StuInsUID[im.StudyInstanceUID] += 1
            if im.SeriesInstanceUID not in SerInsUID:
                SerInsUID[im.SeriesInstanceUID] = 0
            SerInsUID[im.SeriesInstanceUID] += 1
            if im.SeriesTime not in SerTim:
                SerTim[im.SeriesTime] = 0
            SerTim[im.SeriesTime] += 1
            if im.SequenceName not in SeqNam:
                SeqNam[im.SequenceName] = 1
            SeqNam[im.SequenceName] += 1
            try:
                num = im.AcquisitionNumber
            except:
                num = 'None'
            if num not in AcqNum:
                AcqNum[num] = 0
            AcqNum[num] += 1
            if im.ImageType[0] not in ImaTyp:
                ImaTyp[im.ImageType[0]] = 0
            ImaTyp[im.ImageType[0]] += 1

            try:
                num = im.EchoNumbers
            except:
                num = 'None'
            if num not in Echo:
                Echo[num] = 0
            Echo[num] += 1

    f.close()

    for uid in StuInsUID:
        print("StuInsUID {}: {} images".format(uid, StuInsUID[uid]))
    for uid in SerInsUID:
        print("SerInsUID {}: {} images".format(uid, SerInsUID[uid]))
    for tm in SerTim:
        print("SerTim {}: {} images".format(tm, SerTim[tm]))
    for nam in SeqNam:
        print("SeqNam {}: {} images".format(nam, SeqNam[nam]))
    for num in AcqNum:
        print("AcqNum {}: {} images".format(num, AcqNum[num]))
    for typ in ImaTyp:
        print("ImaTyp {}: {} images".format(typ, ImaTyp[typ]))
    for num in Echo:
        print("Echo {}: {} images".format(num, Echo[num]))
    return(0)

def calculator():
    parser = argparse.ArgumentParser(description='Image calculator.',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""Expression examples:
        1  : New image, same shape, all ones
        a*2: Multiply first input by 2
        a+b: Add first and second image""")
    imagedata.cmdline.add_argparse_options(parser)
    parser.add_argument('--mask',
                help='Mask value', default=1)
    parser.add_argument("outdir", help="Output directory")
    parser.add_argument("expression", help="Expression")
    parser.add_argument("indirs", help="Input arguments", nargs="+")
    args = parser.parse_args()

    # Verify non-existing output directory, create
    if os.path.isdir(args.outdir):
        print("Output directory already exist. Aborted.")
        return(2)
    #else:
    #	os.makedirs(args.outdir)

    ch = "abcdefghijklmnopqrstuvwxyz"
    si={}
    i=0
    for indir in args.indirs:
        key = ch[i]
        try:
            print("Open {} as {}:".format(indir, key))
            si[key] = Series(indir, args.input_order, args)
        except imagedata.formats.NotImageError:
            print("Could not determine input format of {}.".format(indir))
            return(1)
        i += 1

    # si[key][tag,slice,rows,columns]
    """
    mask=mask.round()
    mask=mask.astype(int)
    """

    # Convert input to float64
    #output_dtype
    if args.dtype:
        print("Converting input...")
        for k in si.keys():
            si[k] = si[k].astype(args.dtype)

    print("before", si['a'].dtype, si['a'].shape, si['a'].min(), si['a'].max())
    out = si['a'].copy()
    for tag in range(si['a'].shape[0]):
        for key in si.keys():
            exec("""{}=si['{}'][tag]""".format(key,key))
        out[tag] = eval(args.expression)
    print("after", out.dtype, out.shape, out.min(), out.max())

    # Save masked image
    out.write(args.outdir, 'Image_%05d', opts=args)
    return(0)

def statistics():
    parser = argparse.ArgumentParser()
    imagedata.cmdline.add_argparse_options(parser)
    parser.add_argument("in_dirs", nargs='+',
            help="Input directories and files")
    args = parser.parse_args()
    if len(args.output_format) < 1: args.output_format=['dicom']

    try:
        si = Series(args.in_dirs, args.input_order, args)
    except imagedata.formats.NotImageError:
        print("Could not determine input format of %s." % args.in_dirs[0])
        import traceback
        traceback.print_exc(file=sys.stdout)
        return(1)

    print('Min: {}, max: {}, mean: {}, points: {}, shape: {}, dtype: {}'.format(si.min(),
        si.max(), si.mean(), si.size, si.shape, si.dtype))
    return(0)

def timeline():
    parser = argparse.ArgumentParser()
    imagedata.cmdline.add_argparse_options(parser)
    parser.add_argument("in_dirs", nargs='+',
            help="Input directories and files")
    args = parser.parse_args()
    if len(args.output_format) < 1: args.output_format=['dicom']

    try:
        si = Series(args.in_dirs, args.input_order, args)
    except imagedata.formats.NotImageError:
        print("Could not determine input format of %s." % args.in_dirs[0])
        import traceback
        traceback.print_exc(file=sys.stdout)
        return(1)

    print('Timeline:\n{}'.format(si.timeline))
    return(0)

def conversion():
    parser = argparse.ArgumentParser()
    imagedata.cmdline.add_argparse_options(parser)
    parser.add_argument("out_name",
            help="Output directory")
    parser.add_argument("in_dirs", nargs='+',
            help="Input directories and files")
    args = parser.parse_args()
    if len(args.output_format) < 1: args.output_format=['dicom']
    print("Output format: %s, %s, in %s directory." % (args.output_format, imagedata.formats.sort_on_to_str(args.output_sort), args.output_dir))

    try:
        si = Series(args.in_dirs, args.input_order, args)
    except imagedata.formats.NotImageError:
        print("Could not determine input format of %s." % args.in_dirs[0])
        import traceback
        traceback.print_exc(file=sys.stdout)
        return(1)

    si.write(args.out_name, 'Image_%05d', opts=args)
    return(0)

if __name__ == '__main__':
    pass
