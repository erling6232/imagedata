"""Image study

The Study class is a collection of Series objects.

  Typical example usage:

  study = Study('input')

"""

import collections
import logging
import argparse

from .series import Series
from .readdata import read as r_read

logger = logging.getLogger(__name__)


class UnknownOptionType(Exception):
    pass


class Study(collections.OrderedDict):
    """Study -- A collection of Series objects.
    """

    name = "Study"
    description = "Image study"
    authors = "Erling Andersen"
    version = "1.0.0"
    url = "www.helse-bergen.no"

    def __init__(self, url, opts=None):
        super(Study, self).__init__()

        if opts is None:
            in_opts = {}
        elif issubclass(type(opts), dict):
            in_opts = opts
        elif issubclass(type(opts), argparse.Namespace):
            in_opts = vars(opts)
        else:
            raise UnknownOptionType('Unknown opts type ({}): {}'.format(type(opts),
                                                                        opts))

        # Read input, hdr is dict of attributes
        in_opts['separate_series'] = True
        hdr, si = r_read(url, order='auto', opts=in_opts)

        for uid in hdr:
            self[uid] = Series(si[uid])
            self[uid].header = hdr[uid]

        # dicomplugin:
        #   read_files
        #   get_dicom_files
        #     process_member
        #     sort_images
        #   construct_pixel_array
        #   extractDicomAttributes
