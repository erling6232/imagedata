"""imagedata"""

# import imagedata.formats
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

try:
    import importlib.metadata
    __version__ = importlib.metadata.version('imagedata')
except Exception:
    import imagedata as _
    from os.path import join
    with open(join(_.__path__[0],"..","..","VERSION.txt"), 'r') as fh:
        __version__ = fh.readline()

__author__ = 'Erling Andersen, Haukeland University Hospital, Bergen, Norway'
__email__ = 'Erling.Andersen@Helse-Bergen.NO'
