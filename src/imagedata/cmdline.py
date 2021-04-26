"""Add standard command line options."""

# Copyright (c) 2013-2018 Erling Andersen, Haukeland University Hospital, Bergen, Norway

import sys
import argparse
import copy
import logging
import imagedata
import imagedata.formats


class DictAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(DictAction, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, _parser, namespace, values, option_string=None):
        try:
            k, v = values.split("=", 1)
        except ValueError:
            raise argparse.ArgumentError(self, "Format must be key=value")

        # noinspection PyProtectedMember
        items = copy.copy(getattr(namespace, self.dest, {}))  # Default mutables, use copy!
        # items = copy.copy(argparse._ensure_value(namespace, self.dest, {}))  # Default mutables, use copy!
        items[k] = v
        setattr(namespace, self.dest, items)


class CommaSepAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(CommaSepAction, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, _parser, namespace, values, option_string=None):
        # print('%r %r %r' % (namespace, values, option_string))
        commasep = values.split(',')
        setattr(namespace, self.dest, commasep)


class OutputFormatAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(OutputFormatAction, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, _parser, namespace, values, option_string=None):
        # print('%r %r %r' % (namespace, values, option_string))
        of = getattr(namespace, self.dest)
        if values not in of:
            of.append(values)
            setattr(namespace, self.dest, of)


class SortOnAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(SortOnAction, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, _parser, namespace, values, option_string=None):
        # print('%r %r %r' % (namespace, values, option_string))
        sort = imagedata.formats.str_to_sort_on(values)
        setattr(namespace, self.dest, sort)


class InputOrderAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(InputOrderAction, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, _parser, namespace, values, option_string=None):
        # print('%r %r %r' % (namespace, values, option_string))
        input_order = imagedata.formats.str_to_input_order(values)
        setattr(namespace, self.dest, input_order)


class DtypeAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(DtypeAction, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, _parser, namespace, values, option_string=None):
        # print('%r %r %r' % (namespace, values, option_string))
        output_dtype = imagedata.formats.str_to_dtype(values)
        setattr(namespace, self.dest, output_dtype)


def add_argparse_options(parser):
    parser.add_argument('--of', dest="output_format", action=OutputFormatAction,
                        help="Output format [dicom|itk|nifti|biff|mat|ps] (default: dicom). Replaces %%p in output "
                             "path.",
                        choices=['dicom', 'itk', 'nifti', 'biff', 'mat', 'ps'],
                        default=[])
    parser.add_argument('--sort', dest="output_sort", action=SortOnAction,
                        help="Sort output file on slice or input order 'tag' (default: slice)",
                        choices=['slice', 'tag'], default=imagedata.formats.SORT_ON_SLICE)
    parser.add_argument('--order', dest="input_order",
                        action=InputOrderAction,
                        help="How to sort input file (time, b-value, fa, te) (default: none)",
                        choices=['none', 'time', 'b', 'fa', 'te', 'faulty'],
                        default=imagedata.formats.INPUT_ORDER_NONE)
    # readdata.str_to_dtype() will convert choice to numpy dtype
    parser.add_argument('--dtype', action=DtypeAction,
                        help="Specify output datatype. Otherwise keep input datatype",
                        choices=['uint8', 'uint16', 'int16', 'int', 'float', 'float32', 'float64', 'double'],
                        default=None)
    parser.add_argument('--psopt',
                        help="Postscript options (opt=value,opt=value) where opt can be: " +
                             "dpi= (Resolution in pixels/inch of image, default=150); " +
                             "driver=png16m/pnggray (default=png16m); "
                             "rotate=90 (default=0)",
                        default=None)
    parser.add_argument('--odir', dest="output_dir",
                        help="Store all images in a single or multiple directories (default: single)",
                        choices=['single', 'multi'], default='single')
    parser.add_argument('--template',
                        help="Directory to get DICOM template from")
    parser.add_argument('--geometry',
                        help="Second DICOM template for geometry attributes")
    parser.add_argument('--frame',
                        help="Set DICOM frame of reference uid")
    parser.add_argument('--SOPClassUID', help="Set DICOM SOP Class UID", default=None)
    parser.add_argument('--sernum', type=int,
                        help="Set DICOM series number")
    parser.add_argument('--serdes',
                        help="Set DICOM series description")
    parser.add_argument('--input_serinsuid',
                        help="Only read images with specified Series Instance UID",
                        default=None)
    parser.add_argument('--input_echo',
                        help="Only read images with specified Echo Number",
                        default=None)
    parser.add_argument('--imagetype', action=CommaSepAction,
                        help="Set DICOM image type, comma-separated list")
    parser.add_argument('--input_options', action=DictAction,
                        help="Set input options (e.g. time=TriggerTime)",
                        default={
                            'time': 'AcquisitionTime',
                            'fa': 'FlipAngle',
                            'te': 'EchoTime'
                        })
    parser.add_argument('--calling_aet',
                        help='Calling AEtitle',
                        default=None)
    parser.add_argument('--correct_acq', action='store_true',
                        help="Correct acquisition time on dynamic series (DICOM only) (implies --order time)")
    parser.add_argument('--headers_only', action='store_true',
                        help="Read headers only")
    parser.add_argument('--input_shape',
                        help="How to shape input data (t)x(z), e.g. 10x30 for 10 tags, 30 slices",
                        default=None)
    parser.add_argument('--input_sort', action=SortOnAction,
                        help="Sort input file on slice or input order 'tag' (default: slice)",
                        choices=['slice', 'tag'], default=imagedata.formats.SORT_ON_SLICE)
    parser.add_argument('-d', '--debug',
                        help="Print lots of debugging statements",
                        action="store_const", dest="loglevel",
                        const=logging.DEBUG,
                        default=logging.WARNING)
    parser.add_argument('-v', '--verbose',
                        help="Be verbose",
                        action="store_const", dest="loglevel",
                        const=logging.INFO)
    parser.add_argument('-V', '--version',
                        help='Show program version',
                        action='version',
                        version='{} {}'.format(sys.argv[0], imagedata.__version__))
