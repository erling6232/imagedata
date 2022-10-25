"""imagedata"""

import logging
# from imagedata.formats.abstractplugin import AbstractPlugin

logging.getLogger(__name__).addHandler(logging.NullHandler())

try:
    from importlib.metadata import version, entry_points
    __version__ = version('imagedata')
except ModuleNotFoundError:
    from importlib_metadata import version, entry_points
    __version__ = version('imagedata')
except Exception:
    # import imagedata as _
    from . import __path__ as _path
    from os.path import join
    # with open(join(_.__path__[0], "..", "VERSION.txt"), 'r') as fh:
    with open(join(_path[0], "..", "VERSION.txt"), 'r') as fh:
        __version__ = fh.readline().strip()

__author__ = 'Erling Andersen, Haukeland University Hospital, Bergen, Norway'
__email__ = 'Erling.Andersen@Helse-Bergen.NO'

plugins = {}
try:
    _plugins = entry_points(group='imagedata_plugins')
except TypeError:
    _plugins = entry_points()['imagedata_plugins']
for _plugin in _plugins:
    _class = _plugin.load()
    if _class.plugin_type not in plugins:
        plugins[_class.plugin_type] = []
    if (_plugin.name, _class.name, _class) not in plugins[_class.plugin_type]:
        plugins[_class.plugin_type].append((_plugin.name, _class.name, _class))
