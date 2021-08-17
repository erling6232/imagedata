#!/usr/bin/env python3

"""Read/write image data to file(s). Handles DICOM, Nifti, VTI and mhd."""

# Copyright (c) 2013-2018 Erling Andersen, Haukeland University Hospital, Bergen, Norway

import sys
import os.path
import argparse
import urllib
import logging
import numpy as np
import imagedata
import imagedata.cmdline
import imagedata.formats
import imagedata.readdata
import imagedata.transports
from imagedata.series import Series

logger = logging.getLogger()


# noinspection PyPep8Naming
def dump():
    parser = argparse.ArgumentParser()
    imagedata.cmdline.add_argparse_options(parser)
    parser.add_argument("in_dirs", nargs='+',
                        help="Input directories and files")
    args = parser.parse_args()
    logger.setLevel(args.loglevel)
    # if args.version:
    #    print('This is {} version {}'.format(sys.argv[0], __version__))
    print("Output format: %s, %s, in %s directory." % (
        args.output_format, imagedata.formats.sort_on_to_str(args.output_sort), args.output_dir))

    reader = imagedata.formats.find_plugin('dicom')
    logger.debug("in_dirs {}".format(args.in_dirs))
    # noinspection PyUnresolvedReferences
    urls, files = imagedata.readdata.sanitize_urls(args.in_dirs)

    hdr, shape = reader.read_headers(urls, files, args.input_order, args)

    StuInsUID = {}
    SerInsUID = {}
    SerTim = {}
    SeqNam = {}
    AcqNum = {}
    ImaTyp = {}
    Echo = {}
    f = open('files', 'w')
    for slice in hdr['DicomHeaderDict']:
        for tag, member_name, im in hdr['DicomHeaderDict'][slice]:
            # print('{}'.format(image))
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
            except AttributeError:
                num = 'None'
            if num not in AcqNum:
                AcqNum[num] = 0
            AcqNum[num] += 1
            if im.ImageType[0] not in ImaTyp:
                ImaTyp[im.ImageType[0]] = 0
            ImaTyp[im.ImageType[0]] += 1

            try:
                num = im.EchoNumbers
            except AttributeError:
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
    return 0


def calculator():
    # noinspection PyTypeChecker
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
    logger.setLevel(args.loglevel)
    # if args.version:
    #    print('This is {} version {}'.format(sys.argv[0], __version__))

    # Verify non-existing output directory, create
    if os.path.isdir(args.outdir):
        print("Output directory already exist. Aborted.")
        return 2

    ch = "abcdefghijklmnopqrstuvwxyz"
    si = {}
    i = 0
    for indir in args.indirs:
        key = ch[i]
        try:
            print("Open {} as {}:".format(indir, key))
            si[key] = Series(indir, args.input_order, args)
        except imagedata.formats.NotImageError:
            print("Could not determine input format of {}.".format(indir))
            return 1
        i += 1

    # si[key][tag,slice,rows,columns]
    """
    mask=mask.round()
    mask=mask.astype(int)
    """

    # Convert input to float64
    # output_dtype
    if args.dtype:
        print("Converting input...")
        for k in si.keys():
            si[k] = si[k].astype(args.dtype)

    print("before", si['a'].dtype, si['a'].shape, si['a'].min(), si['a'].max())
    out = si['a'].copy()
    for tag in range(si['a'].shape[0]):
        for key in si.keys():
            exec("""{}=si['{}'][tag]""".format(key, key))
        out[tag] = eval(args.expression)
    print("after", out.dtype, out.shape, out.min(), out.max())

    # Save masked image
    out.write(args.outdir, opts=args)
    return 0


def statistics():
    parser = argparse.ArgumentParser()
    imagedata.cmdline.add_argparse_options(parser)
    parser.add_argument('--mask',
                        help='Image mask', default=None)
    parser.add_argument('--bash', action='store_true',
                        help='Print bash commands')
    parser.add_argument("in_dirs", nargs='+',
                        help="Input directories and files")
    args = parser.parse_args()
    logger.setLevel(args.loglevel)

    try:
        si = Series(args.in_dirs, args.input_order, args)
    except imagedata.formats.NotImageError:
        print("Could not determine input format of %s." % args.in_dirs[0])
        import traceback
        traceback.print_exc(file=sys.stdout)
        return 1

    mask = None
    if args.mask is not None:
        try:
            mask = Series(args.mask) > 0
        except imagedata.formats.NotImageError:
            print("Could not determine input format of %s." % args.mask)
            return 1

    if mask is None:
        selection = si
    else:
        selection = si[mask]
    _min = np.min(selection)
    _max = np.max(selection)
    _mean = np.mean(selection)
    _std = np.std(selection)
    _median = np.median(np.array(selection))
    _size = selection.size
    _dtype = selection.dtype

    if args.bash:
        print('min={}\nmax={}\nmean={}\nstd={}\nmedian={}'.format(_min, _max, _mean, _std, _median))
        print('export min max mean std median')
    else:
        print('Min: {}, max: {}'.format(_min, _max))
        print('Mean: {} +- {}, median: {}'.format(_mean, _std, _median))
        print('Points: {}, shape: {}, dtype: {}'.format(_size, si.shape, _dtype))
    return 0


def timeline():
    parser = argparse.ArgumentParser()
    imagedata.cmdline.add_argparse_options(parser)
    parser.add_argument("in_dirs", nargs='+',
                        help="Input directories and files")
    args = parser.parse_args()
    logger.setLevel(args.loglevel)
    # if args.version:
    #    print('This is {} version {}'.format(sys.argv[0], __version__))

    try:
        si = Series(args.in_dirs, args.input_order, args)
    except imagedata.formats.NotImageError:
        print("Could not determine input format of %s." % args.in_dirs[0])
        import traceback
        traceback.print_exc(file=sys.stdout)
        return 1

    print('Timeline:\n{}'.format(si.timeline))
    return 0


def conversion():
    parser = argparse.ArgumentParser()
    imagedata.cmdline.add_argparse_options(parser)
    parser.add_argument("out_name",
                        help="Output directory")
    parser.add_argument("in_dirs", nargs='+',
                        help="Input directories and files")
    args = parser.parse_args()
    logger.setLevel(args.loglevel)
    # if args.version:
    #    print('This is {} version {}'.format(sys.argv[0], __version__))
    #print("Output format: %s, %s, in %s directory." % (
    #    args.output_format, imagedata.formats.sort_on_to_str(args.output_sort), args.output_dir))

    try:
        si = Series(args.in_dirs, args.input_order, args)
    except imagedata.formats.NotImageError:
        print("Could not determine input format of %s." % args.in_dirs[0])
        import traceback
        traceback.print_exc(file=sys.stdout)
        return 1

    si.write(args.out_name, opts=args)
    return 0


def image_list():
    parser = argparse.ArgumentParser()
    imagedata.cmdline.add_argparse_options(parser)
    parser.add_argument("-r", "--recursive", help="Descend into directory tree", action="store_true")
    parser.add_argument("input", help="Input URL")
    args = parser.parse_args()
    logger.setLevel(args.loglevel)

    print('input: {}'.format(args.input))
    url_tuple = urllib.parse.urlsplit(args.input)
    netloc = '{}://{}'.format(url_tuple.scheme, url_tuple.netloc)
    transport = imagedata.transports.Transport(args.input)
    found = False
    for root, dirs, files in transport.walk('*'):
        found = True
        for dir in dirs:
            info = transport.info('{}/{}'.format(root, dir))
            print('{}{}/{} {}'.format(netloc, root, dir, info))
        for filename in files:
            info = transport.info('{}/{}'.format(root, filename))
            print('{}{}/{} {}'.format(netloc, root, filename, info))
        if not args.recursive:
            break  # Do not descend down the tree
    transport.close()

    if found:
        return 0
    else:
        print('No matching data found.')
        return 1


if __name__ == '__main__':
    conversion()
