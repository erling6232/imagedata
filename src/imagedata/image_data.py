#!/usr/bin/env python3

"""Read/write image data to file(s). Handles DICOM, Nifti, VTI and mhd."""

# Copyright (c) 2013-2025 Erling Andersen, Haukeland University Hospital, Bergen, Norway

import sys
import os.path
import shutil
import argparse
import urllib
import logging
import pydicom
from datetime import datetime, date, time
import numpy as np
from .cmdline import add_argparse_options
from .formats import find_plugin, NotImageError, shape_to_str
from .readdata import _get_sources
from .transports import Transport
from .series import Series
from .collection import Cohort


def safe_string(str):
    str = f'{str}'.replace(' ', '_')
    keepcharacters = ('.', '_')
    return "".join(c for c in str if c.isalnum() or c in keepcharacters).rstrip()


def sort(args=sys.argv[1:]):

    parser = argparse.ArgumentParser()
    parser.add_argument("destination", help="Destination directory")
    parser.add_argument("in_dirs", nargs='+',
                        help="Input directories and files")
    args = parser.parse_args(args)

    # Let in_opts be a dict from args
    if args is None:
        in_opts = {}
    elif issubclass(type(args), dict):
        in_opts = args
    elif issubclass(type(args), argparse.Namespace):
        in_opts = vars(args)
    else:
        raise TypeError('Unknown args type ({}): {}'.format(type(args), args))
    in_opts['skip_pixels'] = True

    sources = _get_sources(args.in_dirs, mode='r')

    if os.path.exists(args.destination):
        raise FileExistsError(f'Destination directory {args.destination} already exists')

    image_dict = {}
    for source in sources:
        archive = source['archive']
        scan_files = source['files']
        if scan_files is None or len(scan_files) == 0:
            scan_files = ['*']
        for path in archive.getnames(scan_files):
            if os.path.basename(path) == 'DICOMDIR':
                continue
            try:
                im = pydicom.filereader.dcmread(path, stop_before_pixels=True)
            except pydicom.errors.InvalidDicomError as e:
                continue
            except Exception:
                raise
            if im.PatientID not in image_dict:
                image_dict[im.PatientID] = {None: safe_string(im.PatientName)}
            if im.StudyInstanceUID not in image_dict[im.PatientID]:
                try:
                    studes = safe_string(im.StudyDescription)
                except AttributeError:
                    studes = ''
                try:
                    studat = im.StudyDate
                except AttributeError:
                    studat = '00000000'
                try:
                    stutim = im.StudyTime
                except AttributeError:
                    stutim = '000000'
                image_dict[im.PatientID][im.StudyInstanceUID] = {None: f'{studes}_{studat}_{stutim}'}
            if im.SeriesInstanceUID not in image_dict[im.PatientID][im.StudyInstanceUID]:
                try:
                    serdes = safe_string(im.SeriesDescription)
                except AttributeError:
                    serdes = ''
                image_dict[im.PatientID][im.StudyInstanceUID][im.SeriesInstanceUID] = [f'{im.SeriesNumber}_{serdes}']
            image_dict[im.PatientID][im.StudyInstanceUID][im.SeriesInstanceUID].append(path)

    path = args.destination
    for pat in image_dict.keys():
        pat_path = path
        if len(image_dict) > 1:
            pat_path = os.path.join(path, f'{pat}_{image_dict[pat][None]}')
        os.makedirs(pat_path, exist_ok=True)
        for study in image_dict[pat].keys():
            if study is None:
                continue
            study_path = pat_path
            if len(image_dict) > 1 or len(image_dict[pat]) > 2:  # studes count as 1
                study_path = os.path.join(pat_path, f'{image_dict[pat][study][None]}')
            os.makedirs(study_path, exist_ok=True)
            for series in image_dict[pat][study].keys():
                if series is None:
                    continue
                series_path = study_path
                if len(image_dict) > 1 or len(image_dict[pat]) > 2 or len(image_dict[pat][study]) > 2:  # studes and serdes count as 1
                    series_path = os.path.join(study_path, f'{image_dict[pat][study][series][0]}')
                os.makedirs(series_path, exist_ok=True)
                for fname in image_dict[pat][study][series][1:]:
                    try:
                        shutil.copy(fname, series_path)
                    except Exception:
                        raise
    return 0


def dump():
    parser = argparse.ArgumentParser()
    add_argparse_options(parser)
    parser.add_argument("in_dirs", nargs='+',
                        help="Input directories and files")
    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel)
    logger = logging.getLogger()

    # Let in_opts be a dict from args
    if args is None:
        in_opts = {}
    elif issubclass(type(args), dict):
        in_opts = args
    elif issubclass(type(args), argparse.Namespace):
        in_opts = vars(args)
    else:
        raise TypeError('Unknown args type ({}): {}'.format(type(args), args))
    in_opts['skip_pixels'] = True

    reader = find_plugin('dicom')
    logger.debug("in_dirs {}".format(args.in_dirs))
    sources = _get_sources(args.in_dirs, mode='r')

    image_dict = {}
    for source in sources:
        archive = source['archive']
        scan_files = source['files']
        if scan_files is None or len(scan_files) == 0:
            scan_files = ['*']
        for path in archive.getnames(scan_files):
            if os.path.basename(path) == 'DICOMDIR':
                continue
            member = archive.getmembers([path, ])
            if len(member) != 1:
                raise IndexError('Should not be multiple files for a filename')
            member = member[0]
            try:
                with archive.open(member, mode='rb') as f:
                    reader._sort_datasets(image_dict, archive, path, f, in_opts)
            except Exception:
                raise

    StuInsUID = {}
    SerInsUID = {}
    SerTim = {}
    SeqNam = {}
    AcqNum = {}
    ImaTyp = {}
    Echo = {}
    for sloc in image_dict.keys():
        for archive, member_name, im in image_dict[sloc]:
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
                SeqNam[im.SequenceName] = 0
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

    parser = argparse.ArgumentParser(description='Image calculator.',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog="""Expression examples:
        1  : New image, same shape, all ones
        a*2: Multiply first input by 2
        a+b: Add first and second image""")
    add_argparse_options(parser)
    # parser.add_argument('--mask',
    #                    help='Mask value', default=1)
    parser.add_argument("outdir", help="Output directory")
    parser.add_argument("expression", help="Expression")
    parser.add_argument("indirs", help="Input arguments", nargs="*")
    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel)

    if len(args.output_format) == 0:
        args.output_format.append('dicom')
    # if args.version:
    #    print('This is {} version {}'.format(sys.argv[0], __version__))

    # Verify non-existing output directory, create
    if os.path.isdir(args.outdir):
        print("Output directory already exist. Aborted.")
        return 2

    ch = "abcdefghijklmnopqrstuvwxyz"
    si = {}
    i = 0
    names = {}
    for indir in args.indirs:
        key = ch[i]
        names[key] = indir
        try:
            si[key] = Series(indir, args.input_order, args)
        except NotImageError:
            print("Could not determine input format of {}.".format(indir))
            return 1
        i += 1

    # DICOM templates
    template = geometry = None
    if args.template is not None:
        template = Series(args.template, opts=args)
    if args.geometry is not None:
        geometry = Series(args.geometry, opts=args)

    # si[key][tag,slice,rows,columns]
    """
    mask=mask.round()
    mask=mask.astype(int)
    """

    # Convert input if requested
    if args.dtype:
        for key in si.keys():
            si[key] = si[key].astype(args.dtype)

    # Print input data
    for key in si.keys():
        print("{} = {} {} {}".format(key, names[key], si[key].shape, si[key].dtype))

    # Calculate
    if args.indirs:
        out = si['a'].copy()
        for tag in range(si['a'].shape[0]):
            for key in si.keys():
                exec("""{}=si['{}'][tag]""".format(key, key))
            out[tag] = eval(args.expression)
    else:
        out = Series(eval(args.expression), template=template, geometry=geometry)

    # Save output series
    print("{} = {} {} {}".format(args.outdir, args.expression, out.shape, out.dtype))
    out.write(args.outdir, opts=args)
    return 0


def statistics(cmdline=None):

    def _key_study_time(study):
        _date = study.studyDate if study.studyDate is not None else date.min
        _time = study.studyTime if study.studyTime is not None else time.min
        return datetime.combine(_date, _time)

    parser = argparse.ArgumentParser()
    add_argparse_options(parser)
    parser.add_argument('--mask',
                        help='Image mask', default=None)
    parser.add_argument('--bash', action='store_true',
                        help='Print bash commands')
    parser.add_argument("in_dirs",  # nargs='+',
                        help="Input directories and files")
    if cmdline is None:
        args = parser.parse_args()
    else:
        if not issubclass(type(cmdline), list):
            cmdline = [cmdline]
        args = parser.parse_args(cmdline)
    logging.basicConfig(level=args.loglevel)

    try:
        cohort = Cohort(args.in_dirs, opts=args)
        # si = Series(args.in_dirs, args.input_order, args)
    except NotImageError:
        print('Could not determine input format of "%s"' % args.in_dirs[0])
        import traceback
        traceback.print_exc(file=sys.stdout)
        return 1

    mask = None
    if args.mask is not None:
        try:
            mask = Series(args.mask) > 0
        except NotImageError:
            print('Could not determine input format of "%s"' % args.mask)
            return 1

    print("{}".format(cohort))
    patient_list = []
    for patientID in cohort:
        patient_list.append(cohort[patientID])
    patient_list.sort(key=lambda patient: str.lower(patient.patientName.family_comma_given()))
    for patient in patient_list:
        print("  {}".format(patient))
        for studyInstanceUID in patient:
            study_list = []
            study_list.append(patient[studyInstanceUID])
        # study_list.sort(key=lambda study: datetime.combine(study.studyDate, study.studyTime))
        study_list.sort(key=_key_study_time)
        for study in study_list:
            print("    {}".format(study))
            series_list = []
            for seriesInstanceUID in study:
                series_list.append(study[seriesInstanceUID])
            series_list.sort(key=lambda series: series.seriesNumber)
            for series in series_list:
                try:
                    _seriesDescription = series.seriesDescription
                except ValueError:
                    _seriesDescription = ''
                print("      Series [{}] {}: {}, shape: {}, dtype: {}, input order: {}".format(
                    series.seriesNumber, series.modality,
                    _seriesDescription,
                    shape_to_str(series.shape), series.dtype,
                    series.input_order
                ))
                print_statistics(series, mask, bash=args.bash)


def print_statistics(si, mask=None, bash=False):
    if mask is None:
        selection = si
    else:
        selection = si[mask]
    _min = np.min(selection)
    _max = np.max(selection)
    _mean = np.mean(selection)
    _std = np.std(selection)
    _median = np.median(np.array(selection))

    if bash:
        print('min={}\nmax={}\nmean={}\nstd={}\nmedian={}'.format(
            _min, _max, _mean, _std, _median))
        print('export min max mean std median')
    else:
        print('        Min: {}, max: {}, mean: {} +/- {}, median: {}'.format(
            _min, _max, _mean, _std, _median))
    return 0


def timeline():
    parser = argparse.ArgumentParser()
    add_argparse_options(parser)
    parser.add_argument("in_dirs", nargs='+',
                        help="Input directories and files")
    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel)
    # if args.version:
    #    print('This is {} version {}'.format(sys.argv[0], __version__))

    try:
        si = Series(args.in_dirs, args.input_order, args)
    except NotImageError:
        print("Could not determine input format of %s." % args.in_dirs[0])
        import traceback
        traceback.print_exc(file=sys.stdout)
        return 1

    print('Timeline:\n{}'.format(si.timeline))
    return 0


def show():
    parser = argparse.ArgumentParser()
    add_argparse_options(parser)
    parser.add_argument("in_dirs", nargs='+',
                        help="Input directories and files")
    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel)
    # if args.version:
    #    print('This is {} version {}'.format(sys.argv[0], __version__))

    try:
        si = Series(args.in_dirs, args.input_order, args)
    except NotImageError:
        print("Could not determine input format of %s." % args.in_dirs[0])
        import traceback
        traceback.print_exc(file=sys.stdout)
        return 1

    si.show()
    return 0


def _reduce(cohort):
    """Reduce cohort level to the lowest level.
    """
    if len(cohort) > 1:
        return cohort
    try:
        for patientID in cohort:
            pass
    except IndexError:
        raise IndexError('No patient in cohort')
    patient = cohort[patientID]
    if len(patient) > 1:
        return patient
    try:
        for studyInsUID in patient:
            pass
        # studyInsUID = patient.keys()[0]
    except IndexError:
        raise IndexError('No study for patient')
    study = patient[studyInsUID]
    if len(study) > 1:
        return study
    try:
        for seriesInsUID in study:
            pass
    except IndexError:
        raise IndexError('No series in study')
    series = study[seriesInsUID]
    return series


def conversion():
    parser = argparse.ArgumentParser()
    add_argparse_options(parser)
    parser.add_argument("out_name",
                        help="Output directory")
    # parser.add_argument("in_dirs", nargs='+',
    parser.add_argument("in_dirs",
                        help="Input directories and files")
    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel)
    # if args.version:
    #    print('This is {} version {}'.format(sys.argv[0], __version__))
    # print("Output format: %s, %s, in %s directory." % (
    #    args.output_format, sort_on_to_str(args.output_sort), args.output_dir))

    try:
        cohort = Cohort(args.in_dirs, opts=args)
        # si = Series(args.in_dirs, args.input_order, args)
    except NotImageError:
        print("Could not determine input format of %s." % args.in_dirs[0])
        import traceback
        traceback.print_exc(file=sys.stdout)
        return 1

    selection = _reduce(cohort)

    selection.write(args.out_name, opts=args)
    return 0


def image_list():
    parser = argparse.ArgumentParser()
    add_argparse_options(parser)
    parser.add_argument("-r", "--recursive", help="Descend into directory tree",
                        action="store_true")
    parser.add_argument("input", help="Input URL")
    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel)
    logger = logging.getLogger()

    print('input: {}'.format(args.input))
    url_tuple = urllib.parse.urlsplit(args.input)
    netloc = '{}://{}'.format(url_tuple.scheme, url_tuple.netloc)
    logger.debug("image_list: url_tuple {}".format(url_tuple))
    transport = Transport(args.input)
    found = False
    for root, dirs, files in transport.walk('*'):
        logger.debug("image_list: root: {}, dirs: {}, files: {}".format(root, dirs, files))
        found = True
        for dir in dirs:
            info = transport.info('{}/{}'.format(root, dir))
            print('{}{}/{} {}'.format(netloc, root, dir, info))
        for filename in files:
            info = transport.info('{}/{}'.format(root, filename))
            print('{}{}/{} {}'.format(netloc, root, filename, info))
        if not args.recursive:
            if root == url_tuple.path:
                break  # Do not descend further down the tree
    transport.close()

    if found:
        return 0
    else:
        print('No matching data found.')
        return 1


if __name__ == '__main__':
    conversion()
