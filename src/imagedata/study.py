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

    _attributes = [
        'studyDate', 'studyTime', 'studyDescription', 'studyID', 'studyInstanceUID',
        'referringPhysiciansName'
    ]

    def __init__(self, url, opts=None):
        super(Study, self).__init__()
        for _attr in self._attributes:
            setattr(self, _attr, None)

        if opts is None:
            _in_opts = {}
        elif issubclass(type(opts), dict):
            _in_opts = opts
        elif issubclass(type(opts), argparse.Namespace):
            _in_opts = vars(opts)
        else:
            raise UnknownOptionType('Unknown opts type ({}): {}'.format(type(opts),
                                                                        opts))

        # Read input, hdr is dict of attributes
        _in_opts['separate_series'] = True
        _hdr, _si = r_read(url, order='auto', opts=_in_opts)

        for _uid in _hdr:
            self[_uid] = Series(_si[_uid])
            self[_uid].header = _hdr[_uid]

            for _attr in self._attributes:
                _dicom_attribute = _attr[0].upper() + _attr[1:]
                # TODO: Consider what to do when study attributes differ in series.
                # Update self property if None from series
                if getattr(self, _attr, None) is None:
                    setattr(self, _attr,
                            self[_uid].getDicomAttribute(_dicom_attribute))

        # dicomplugin:
        #   read_files
        #   get_dicom_files
        #     process_member
        #     sort_images
        #   construct_pixel_array
        #   extractDicomAttributes
