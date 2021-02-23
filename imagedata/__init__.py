"""imagedata"""

import imagedata.formats
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

import importlib.metadata
__version__ = importlib.metadata.version('imagedata')

__author__ = 'Erling Andersen, Haukeland University Hospital, Bergen, Norway'
__email__ = 'Erling.Andersen@Helse-Bergen.NO'
