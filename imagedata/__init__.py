"""imagedata"""

import imagedata.formats
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

__version__ = "unknown" 
try: 
    from ._version import __version__ 
except (ImportError,SystemError): 
    # We're running in a tree that didn't come with a 
    # _version.py, so we don't know what our version is. This
    # should not happen very often. 
    pass 

__author__ = 'Erling Andersen, Haukeland University Hospital, Bergen, Norway'
__email__ = 'Erling.Andersen@Helse-Bergen.NO'
