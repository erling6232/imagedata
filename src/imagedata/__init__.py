"""imagedata"""

import logging
import importlib
# from imagedata.formats.abstractplugin import AbstractPlugin
from .series import Series
from .collections import Study, Patient, Cohort

logging.getLogger(__name__).addHandler(logging.NullHandler())

try:
    from importlib.metadata import version, entry_points
    try:
        __version__ = version('imagedata')
    except Exception:
        __version__ = None
except ModuleNotFoundError:
    from importlib_metadata import version, entry_points
    try:
        __version__ = version('imagedata')
    except Exception:
        __version__ = None
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
_plugins = {}
try:
    _plugins = entry_points().select(group='imagedata_plugins')
except AttributeError:
    try:
        _plugins = entry_points(group='imagedata_plugins')
    except TypeError:
        try:
            _plugins = entry_points()['imagedata_plugins']
        except KeyError:
            pass
except TypeError:
    try:
        _plugins = entry_points()['imagedata_plugins']
    except KeyError:
        pass
for _plugin in _plugins:
    _class = _plugin.load()
    try:
        importlib.import_module(_plugin.module)
    except AttributeError:
        _module, _classname = _plugin.value.split(':')
        importlib.import_module(_module)
    except Exception as e:
        raise ImportError("Cannot import plugin: {}\n{}: {}: {}\n{}".format(
            e, _plugin.name, _plugin.group, _plugin.value, _plugin.__dir__()))
    if _class.plugin_type not in plugins:
        plugins[_class.plugin_type] = []
    if (_plugin.name, _class.name, _class) not in plugins[_class.plugin_type]:
        plugins[_class.plugin_type].append((_plugin.name, _class.name, _class))
