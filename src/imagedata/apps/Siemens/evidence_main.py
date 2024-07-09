#!/usr/bin/env python3

import os
import argparse
import logging
import pydicom
import imagedata.formats
from imagedata.cmdline import add_argparse_options
# from imagedata.readdata import read as r_read
# from imagedata.apps.Siemens.evidence2mask import evidence2roi
from evidence2mask import evidence2roi
from imagedata import Series


logger = logging.getLogger()


def read_uid_map(filename):
    uidmap = {}
    with open(filename, 'r') as f:
        # line = f.readline()
        for line in f.readlines():
            # orig, pseudo = f.readline().strip().split('\t')
            orig, pseudo = line.strip().split('\t')
            uidmap[orig] = pseudo
    return uidmap


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_argparse_options(parser)
    parser.add_argument("output")
    parser.add_argument("evidence")
    parser.add_argument("in_dirs", nargs='+',
                        help="Input directories and files")
    parser.add_argument("--uid_table",
                        help="UID translate table",
                        default=None)
    parser.add_argument("-l", "--laterality",
                        help="[--laterality separate|combined] - Separate left and right ROIs "
                             "in separate files. Otherwise combine both (default).",
                        default="combined")
    args = parser.parse_args()
    logger.setLevel(args.loglevel)

    out_name = args.output
    evidence_name = args.evidence

    uidmap = None
    if args.uid_table is not None:
        uidmap = read_uid_map(args.uid_table)
    si = Series(args.in_dirs, imagedata.formats.INPUT_ORDER_TIME)

    reading = pydicom.read_file(args.evidence)

    # mask = {}
    # mask, content = evidence2roi(mask, reading, si, args.laterality)
    mask, content = evidence2roi(reading, uidmap)
    # print(content)

    serNum = 5000
    imageType = ['DERIVED', 'SECONDARY', 'MASK']

    f = open(os.path.join(out_name, "timepoint.txt"), "w")
    f.write("%d" % content['timepoint'])
    f.close()

    for laterality in mask.keys():
        mask[laterality].removePrivateTags()
        mask[laterality].setSeriesNumber(serNum)
        mask[laterality].setSeriesDescription("Mask %s" % laterality)
        mask[laterality].setImageType(imageType)
        # Acquisition Date
        mask[laterality].setDicomAttribute((0x0008, 0x0022), content['date'])
        # Acquisition Time
        mask[laterality].setDicomAttribute((0x0008, 0x0032), content['time'])
        # Operator's Name
        mask[laterality].setDicomAttribute((0x0008, 0x1070), content['creator'])
        # Image Laterality
        mask[laterality].setDicomAttribute((0x0020, 0x0062), laterality)
        mask[laterality].write(os.path.join(out_name, "%s_%%05d" % laterality))
        serNum += 1
