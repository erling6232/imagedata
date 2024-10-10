"""imagedata"""

import logging
from importlib import import_module
from importlib.metadata import version, entry_points
from .series import Series
from .collection import Study, Patient, Cohort

logging.getLogger(__name__).addHandler(logging.NullHandler())

try:
    __version__ = version('imagedata')
except Exception:
    __version__ = None

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

if len(_plugins) == 0:
    # Fallback to known built-in plugins
    try:
        from src.imagedata.archives.filesystemarchive import FilesystemArchive
        from src.imagedata.archives.zipfilearchive import ZipfileArchive
        plugins['archive'] = [
            ('filesystemarchive', FilesystemArchive.name, FilesystemArchive),
            ('zipfilearchive', ZipfileArchive.name, ZipfileArchive)
        ]
        from src.imagedata.transports.filetransport import FileTransport
        from src.imagedata.transports.dicomtransport import DicomTransport
        from src.imagedata.transports.xnattransport import XnatTransport
        plugins['transport'] = [
            ('filetransport', FileTransport.name, FileTransport),
            ('dicomtransport', DicomTransport.name, DicomTransport),
            ('xnattransport', XnatTransport.name, XnatTransport)
        ]
        from src.imagedata.formats.dicomplugin import DICOMPlugin
        from src.imagedata.formats.itkplugin import ITKPlugin
        from src.imagedata.formats.matplugin import MatPlugin
        from src.imagedata.formats.niftiplugin import NiftiPlugin
        plugins['format'] = [
            ('dicomformat', DICOMPlugin.name, DICOMPlugin),
            ('itkformat', ITKPlugin.name, ITKPlugin),
            ('matformat', MatPlugin.name, MatPlugin),
            ('niftiformat', NiftiPlugin.name, NiftiPlugin)
        ]
    except ModuleNotFoundError:
        pass

for _plugin in _plugins:
    _class = _plugin.load()
    try:
        import_module(_plugin.module)
    except AttributeError:
        _module, _classname = _plugin.value.split(':')
        import_module(_module)
    except Exception as e:
        raise ImportError("Cannot import plugin: {}\n{}: {}: {}\n{}".format(
            e, _plugin.name, _plugin.group, _plugin.value, _plugin.__dir__()))
    if _class.plugin_type not in plugins:
        plugins[_class.plugin_type] = []
    if (_plugin.name, _class.name, _class) not in plugins[_class.plugin_type]:
        plugins[_class.plugin_type].append((_plugin.name, _class.name, _class))
