"""Read/Write image files, calling appropriate transport, archive and format plugins
"""

# Copyright (c) 2013-2022 Erling Andersen, Haukeland University Hospital, Bergen, Norway

import os.path
import logging
import mimetypes
import argparse
import fnmatch
import pathlib
import urllib.parse
import traceback as tb
from typing import Dict, List, Tuple, Union
from .formats import INPUT_ORDER_NONE, find_plugin, get_plugins_list
from .formats import CannotSort, NotImageError, UnknownInputError, WriteNotImplemented
from .transports import RootIsNotDirectory
from .archives import find_mimetype_plugin, ArchivePluginNotFound


class NoTransportError(Exception):
    pass


class NoArchiveError(Exception):
    pass


class UnknownOptionType(Exception):
    pass


logger = logging.getLogger(__name__)


def read(urls, order=None, opts=None, input_format=None):
    """Read image data, calling appropriate transport, archive and format plugins

    Args:
        urls: list of urls or url to read (list of str, or str)
        order: determine how to sort the images (default: auto-detect)
        opts: input options (argparse.Namespace or dict)
        input_format: specify a particular input format (default: auto-detect)

    Returns:
        tuple of
            - hdr: header instance
            - si[tag,slice,rows,columns]: numpy array

    Raises:
        ValueError: When no sources are given.
        UnknownOptionType: When opts cannot be made into a dict.
        FileNotFoundError: When specified URL cannot be opened.
        UnknownInputError: When the input format could not be determined.
        CannotSort: When input data cannot be sorted.
    """

    _name: str = '{}.{}'.format(__name__, read.__name__)

    logger.debug("{}: urls {}".format(_name, urls))
    #    transport,my_urls,files = sanitize_urls(urls)
    #    if len(my_urls) < 1:
    #        raise ValueError("No URL(s) where given")
    #    logger.debug("reader.read: transport {} my_urls {}".format(transport,my_urls))
    sources = _get_sources(urls, mode='r', opts=opts)
    if len(sources) < 1:
        raise ValueError("No source(s) where given")
    logger.debug("{}: sources {}".format(_name, sources))

    # Let in_opts be a dict from opts
    if opts is None:
        in_opts = {}
    elif issubclass(type(opts), dict):
        in_opts = opts
    elif issubclass(type(opts), argparse.Namespace):
        in_opts = vars(opts)
    else:
        raise UnknownOptionType('Unknown opts type ({}): {}'.format(type(opts),
                                                                    opts))

    if input_format is None and 'input_format' in in_opts:
        input_format = in_opts['input_format']

    # Let the calling party override a default input order
    input_order = INPUT_ORDER_NONE
    if 'input_order' in in_opts:
        input_order = in_opts['input_order']
    if order != 'none':
        input_order = order
    logger.info("{}: Input order: {}.".format(
        _name, input_order))

    # Pre-fetch DICOM template
    pre_hdr = None
    if 'template' in in_opts and in_opts['template']:
        logger.debug("{}: template {}".format(_name, in_opts['template']))
        template_source = _get_sources(in_opts['template'], mode='r', opts=in_opts)
        reader = find_plugin('dicom')
        pre_hdr, _ = reader.read(template_source, None, input_order, in_opts)
        if len(pre_hdr) != 1:
            raise ValueError('Template is not a single series')
        pre_hdr = pre_hdr[next(iter(pre_hdr))]

    # Pre-fetch DICOM geometry
    geom_hdr = None
    if 'geometry' in in_opts and in_opts['geometry']:
        logger.debug("{}: geometry {}".format(_name, in_opts['geometry']))
        geometry_source = _get_sources(in_opts['geometry'], mode='r', opts=in_opts)
        reader = find_plugin('dicom')
        geom_hdr, _ = reader.read(geometry_source, None, input_order, in_opts)
        if len(geom_hdr) != 1:
            raise ValueError('Geometry template is not a single series')
        geom_hdr = geom_hdr[next(iter(geom_hdr))]
        # if pre_hdr is None:
        #    pre_hdr = {}
        # _add_dicom_geometry(pre_hdr, geom_hdr)

    # Call reader plugins in turn to read the image data
    plugins = sorted_plugins_dicom_first(get_plugins_list(), input_format)
    logger.debug("{}: plugins length {}".format(_name, len(plugins)))
    summary = 'Summary of read plugins:'
    for pname, ptype, pclass in plugins:
        logger.debug("{}: {:20s} ({:8s}) {}".format(
            _name, pname, ptype, pclass.description))
        reader = pclass()
        try:
            hdr, si = reader.read(sources, None, input_order, in_opts)
            del reader

            for source in sources:
                logger.debug("{}: close archive {}".format(_name, source['archive']))
                source['archive'].close()
            # if 'headers_only' in in_opts and in_opts['headers_only']:
            #     pass
            for seriesUID in hdr:
                hdr[seriesUID].add_template(pre_hdr)
                hdr[seriesUID].add_geometry(geom_hdr)
            return hdr, si
        except (FileNotFoundError, CannotSort):
            if 'skip_broken_series' in opts and opts['skip_broken_series']:
                pass
            else:
                raise
        except NotImageError as e:
            logger.info("{}: Giving up {}: {}".format(_name, ptype, e))
            summary = summary + '\n  {}: {}'.format(ptype, e)
        except Exception as e:
            logger.info("{}: Giving up (OTHER) {}: {}".format(_name, ptype, e))
            summary = summary + '\n  {}: {}'.format(ptype, e)
            # import traceback, sys
            # traceback.print_exc(file=sys.stdout)
            # exit(1)

    for source in sources:
        logger.debug("{}: close archive {}".format(_name, source['archive']))
        source['archive'].close()

    # All reader plugins failed - report
    if issubclass(type(urls), list):
        raise UnknownInputError('Could not determine input format of "{}": {}'.format(
            urls[0], summary))
    else:
        raise UnknownInputError('Could not determine input format of "{}": {}'.format(
            urls, summary))


# def _add_template(hdr, pre_hdr):
#    if pre_hdr is not None:
#        for key in pre_hdr:
#            hdr[key] = copy.copy(pre_hdr[key])

# def _add_dicom_geometry(pre_hdr, geometry):
#        """For each slice in geometry, use most of pre_hdr, adding a few attributes from geometry
#        """
#
#        #logger.debug("_add_dicom_geometry template %s geometry %s" % (
#        #    imagedata.formats.shape_to_str(self.shape),
#        #    imagedata.formats.shape_to_str(geometry.shape)))
#        pre_hdr['sliceLocations'] = geometry['sliceLocations'].copy()
#        pre_hdr['spacing']        = geometry['spacing'].copy()
#        pre_hdr['orientation']    = geometry['orientation'].copy()
#        pre_hdr['imagePositions'] = {}
#        logger.debug("_add_dicom_geometry:")
#        logger.debug("_add_dicom_geometry: geometry.imagePositions {}".format(
#            geometry['imagePositions'].keys()))
#        for k in geometry['imagePositions'].keys():
#            pre_hdr['imagePositions'][k] = geometry['imagePositions'][k].copy()
#        pre_hdr['axes'] = geometry['axes'].copy()

def write(si, url, opts=None, formats=None):
    """Write image data, calling appropriate format plugins

    Args:
        si[tag,slice,rows,columns]: Series array
        url: output destination url
        opts: Output options (argparse.Namespace or dict)
        formats: list of output formats, overriding opts.output_format (list or str)
    Raises:
        UnknownOptionType: When opts cannot be made into a dict.
        TypeError: List of output format is not list().
        ValueError: Wrong number of destinations given, or no way to write multidimensional image.
        imagedata.formats.WriteNotImplemented: Cannot write this image format.
    """

    _name: str = '{}.{}'.format(__name__, write.__name__)

    def _replace_url(url, pattern, value):
        if isinstance(url, str):
            url = url.replace(pattern, value)
        elif issubclass(type(url), pathlib.PurePath):
            _pl = []
            for _p in url.parts:
                _pl.append(_p.replace(pattern, value))
            url = pathlib.Path(*_pl)
        return url

    # logger.debug("write: directory_name(si): {}".format(directory_name(si)))

    # Let out_opts be a dict from opts
    if opts is None:
        out_opts = {}
    elif issubclass(type(opts), dict):
        out_opts = opts
    elif issubclass(type(opts), argparse.Namespace):
        out_opts = vars(opts)
    else:
        raise UnknownOptionType('Unknown opts type ({}): {}'.format(type(opts),
                                                                    opts))

    if 'sernum' in out_opts and out_opts['sernum']:
        si.seriesNumber = out_opts['sernum']
    if 'serdes' in out_opts and out_opts['serdes']:
        si.seriesDescription = out_opts['serdes']
    if 'imagetype' in out_opts and out_opts['imagetype']:
        si.imageType = out_opts['imagetype']
    if 'frame' in out_opts and out_opts['frame']:
        si.frameOfReferenceUID = out_opts['frame']
    if 'SOPClassUID' in out_opts and out_opts['SOPClassUID']:
        si.SOPClassUID = out_opts['SOPClassUID']

    # Default output format is input format
    try:
        output_formats = [si.input_format]
    except AttributeError:
        output_formats = None
    logger.debug("{}: Default    output format : {}".format(_name, output_formats))
    logger.debug("{}: Overriding output formats: {}".format(_name, formats))
    logger.debug("{}: Options: {}".format(_name, out_opts))
    if formats is not None:
        if isinstance(formats, list):
            output_formats = formats
        elif isinstance(formats, str):
            output_formats = [formats]
        else:
            raise TypeError("List of output format is not list() ({})".format(type(formats)))
    elif 'output_format' in out_opts and len(out_opts['output_format']):
        output_formats = out_opts['output_format']
    if output_formats is None:
        output_formats = ['dicom']  # Fall-back to dicom output
    logger.info("{}: Output formats: {}".format(_name, output_formats))

    # Determine output dtype
    write_si = si
    if 'dtype' in out_opts and out_opts['dtype'] is not None:
        if out_opts['dtype'] != si.dtype:
            # write_si = si.astype(str_to_dtype(out_opts['dtype']))
            write_si = si.astype(out_opts['dtype'])

    # Verify there is one destination only
    # destinations = _get_sources(url, mode='w')
    # if len(destinations) != 1:
    #    raise ValueError('Wrong number of destinations (%d) given' %
    #        len(destinations))

    # Call plugin writers in turn to store the data
    logger.debug("{}: Available plugins {}".format(_name, len(get_plugins_list())))
    written = False
    msg = ''
    for pname, ptype, pclass in get_plugins_list():
        if ptype in output_formats:
            logger.debug("{}: Attempt plugin {}".format(_name, ptype))
            # Create plugin to write data in specified format
            writer = pclass()
            logger.debug("{}: Created writer plugin of type {}".format(
                _name, type(writer)))
            # local_url = url.replace('%p', ptype)
            local_url = _replace_url(url, '%p', ptype)
            destinations = _get_sources(local_url, mode='w', opts=out_opts)
            if len(destinations) != 1:
                raise ValueError('Wrong number of destinations (%d) given' %
                                 len(destinations))
            destination = destinations[0]
            logger.debug('{}: destination {}'.format(_name, destination))
            try:
                if write_si.ndim >= 4 and write_si.shape[0] > 1:
                    # 4D data
                    writer.write_4d_numpy(write_si, destination, out_opts)
                elif write_si.ndim >= 2:
                    # 2D-3D data
                    writer.write_3d_numpy(write_si, destination, out_opts)
                else:
                    raise ValueError("Don't know how to write image of shape {}".format(
                        write_si.shape))
                written = True
                del writer
            except WriteNotImplemented:
                raise
            except Exception as e:
                logger.info("{}: Giving up (OTHER) {}: {}".format(
                    _name, ptype, e))
                msg = msg + '\n{}: {}'.format(ptype, e)
                msg = msg + '\n' + ''.join(tb.format_exception(None, e, e.__traceback__))
                pass
            destination['archive'].close()
    if not written:
        if len(msg) > 0:
            raise IOError("Failed writing: {}".format(msg))
        raise ValueError("No writer plugin was found for {}".format(output_formats))
    if len(msg) > 0:
        logger.error("{}: {}".format(_name, msg))
    # destination['archive'].close()


def sorted_plugins_dicom_first(plugins, input_format):
    """Sort plugins such that any Nifti plugin is used early."""
    if input_format is not None:
        for pname, ptype, pclass in plugins:
            if ptype == input_format:
                return [(pname, ptype, pclass)]
    for pname, ptype, pclass in plugins:
        if ptype == 'nifti':
            plugins.remove((pname, ptype, pclass))
            plugins.insert(0, (pname, ptype, pclass))
            break
    """Sort plugins such that any DICOM plugin is used first."""
    for pname, ptype, pclass in plugins:
        if ptype == 'dicom':
            plugins.remove((pname, ptype, pclass))
            plugins.insert(0, (pname, ptype, pclass))
            break
    return plugins


def _get_location_part(url):
    """Get location part of URL: scheme, netloc and path"""

    _name: str = '{}.{}'.format(__name__, _get_location_part.__name__)

    if os.name == 'nt' and fnmatch.fnmatch(url, '[A-Za-z]:\\*'):
        # Windows: Parse without x:, then reattach drive letter
        url_tuple = urllib.parse.urlsplit(url[2:], scheme="file")
        _path = url[:2] + url_tuple.path
    elif os.name == 'nt' and fnmatch.fnmatch(url, '//*'):
        # Windows: Parse UNC without leading /, then reattach
        url_tuple = urllib.parse.urlsplit(url[1:], scheme="file")
        _path = url[:1] + url_tuple.path
    else:
        url_tuple = urllib.parse.urlsplit(url, scheme="file")
        _path = url_tuple.path
    # url_tuple = urllib.parse.urlsplit(url, scheme='file')
    # Strip off query and fragment parts
    location = urllib.parse.urlunsplit((url_tuple.scheme, url_tuple.netloc, _path, None, None))
    if url_tuple.scheme == 'file' and url[0] != '/':
        _path = os.path.abspath(_path)
        location = urllib.parse.urlunsplit((url_tuple.scheme, url_tuple.netloc, _path, None, None))
    logger.debug('{}: scheme {}'.format(_name, url_tuple.scheme))
    logger.debug('{}: netloc {}'.format(_name, url_tuple.netloc))
    logger.debug('{}: path {}'.format(_name, _path))
    logger.debug('{}: location {}'.format(_name, location))
    return location


def _get_query_part(url):
    """Get query part of URL. This may contain file name"""

    url_tuple = urllib.parse.urlsplit(url, scheme='file')
    return url_tuple.query


def _get_archive(url, mode='r', opts=None):
    """Get archive plugin for given URL."""

    _name: str = '{}.{}'.format(__name__, _get_archive.__name__)

    if opts is None:
        opts = {}
    logger.debug('{}: url {}'.format(_name, url))
    url_tuple = urllib.parse.urlsplit(url, scheme="file")
    if os.name == 'nt' and \
            url_tuple.scheme == 'file' and \
            fnmatch.fnmatch(url_tuple.netloc, '[A-Za-z]:\\*'):
        # Windows: Parse without /x:, then re-attach drive letter
        _path = url_tuple.netloc
    else:
        _path = url_tuple.path
    # url_tuple = urllib.parse.urlsplit(url, scheme='file')
    mimetype = mimetypes.guess_type(_path)[0]
    archive = find_mimetype_plugin(
        mimetype,
        url,
        mode,
        # read_directory_only=mode[0] == 'r',
        read_directory_only=False,
        opts=opts)
    logger.debug('{}: _mimetypes {}'.format(_name, mimetype))
    logger.debug('{}: archive {}'.format(_name, archive.name))
    return archive


def _common_prefix(level):
    """This unlike the os.path.commonprefix version
    always returns path prefixes as it compares
    path component wise
    https://stackoverflow.com/questions/21498939
    """

    cp = []
    ls = [p.split(os.sep) for p in level]
    ml = min(len(p) for p in ls)

    for i in range(ml):

        s = set(p[i] for p in ls)
        if len(s) != 1:
            break

        cp.append(s.pop())

    return os.sep.join(cp)


def _simplify_locations(locations):
    """Simplify locations by joining file:/// locations to a common prefix."""

    _name: str = '{}.{}'.format(__name__, _simplify_locations.__name__)

    logger.debug('{}: locations {}'.format(_name, locations))
    new_locations = {}
    paths = []
    for location in locations:
        # On Windows, any backslash (os.sep) will be replaced by slash in URL
        # url_tuple = urllib.parse.urlsplit(location.replace(os.sep, '/'), scheme='file')
        if os.name == 'nt' and fnmatch.fnmatch(location, '[A-Za-z]:\\*'):
            # Windows: Parse without x:, then reattach drive letter
            url_tuple = urllib.parse.urlsplit(location[2:], scheme='file')
            _path = location[:2] + url_tuple.path
        else:
            url_tuple = urllib.parse.urlsplit(location, scheme='file')
            _path = url_tuple.path if len(url_tuple.path) > 0 else url_tuple.netloc
        if url_tuple.scheme == 'file':
            paths.append(_path)
        else:
            new_locations[location] = True
    logger.debug('{}: paths {}'.format(_name, paths))
    if len(paths) > 0:
        prefix = _common_prefix(paths)
        logger.debug('{}: prefix {}'.format(_name, prefix))
        prefix_url = urllib.parse.urlunsplit((
            'file',
            '',
            prefix,
            None,
            None))
        # urlunsplit prepends file:/// when a Windows drive is present. Simplify to file://
        if os.name == 'nt' and fnmatch.fnmatch(prefix_url, 'file:///[A-Za-z]:\\*'):
            prefix_url = 'file://' + prefix_url[8:]
        new_locations[prefix_url] = True
    logger.debug('{}: new_locations {}'.format(_name, new_locations))
    return new_locations


def _get_sources(
        urls: Union[List, Tuple, str],
        mode: str, opts: dict = None) -> List[Dict]:
    """Determine transport, archive and file from each url.

    Handle both single url, a url tuple, and a url list

    Args:
        urls: list, tuple or single string, e.g.:
            file://dicom
                transport: file, archive: fs, url: dicom
            file://dicom.zip?query
                transport: file, archive: zip, files: query
            file://dicom.tar.gz?query
                transport: file, archive: tgz, files: query
            http://server:port/dicom.zip
                transport: http, archive: zip
            dicom://server:port/AET
                transport: dicom, archive: fs
            xnat://server:port/project/subject/experiment/scan
               transport: xnat, archive: zip
        mode: 'r' or 'w' for Read or Write
            When mode = 'r', the urls must exist.
    Returns:
        sources: list of dict for each url
            - 'archive'  : archive plugin
            - 'files'    : list of file names or regexp. May be empty list.
    """

    _name: str = '{}.{}'.format(__name__, _get_sources.__name__)

    # Ensure the input is a list: my_urls
    if opts is None:
        opts = {}
    if isinstance(urls, list):
        source_urls = urls
    elif isinstance(urls, tuple):
        source_urls = list(urls)
    else:
        source_urls = [urls]
    my_urls = []
    for url in source_urls:
        if issubclass(type(url), pathlib.PurePath):
            my_urls.append(str(url.resolve()))
        else:
            my_urls.append(url)

    # Scan my_urls to determine the locations of the inputs
    locations = {}
    for url in my_urls:
        locations[_get_location_part(url)] = True
    locations = _simplify_locations(locations)

    # Set up sources for each location, and possibly add files
    sources = []
    for location in locations:
        logger.debug('{}: location {}'.format(_name, location))
        source_location = location
        source = {'files': []}
        try:
            source['archive'] = _get_archive(source_location, mode=mode, opts=opts)
        except (RootIsNotDirectory,
                ArchivePluginNotFound) as e:
            # Retry with parent directory
            source_location, filename = os.path.split(source_location)
            logger.debug('{}: retry location {}'.format(_name, source_location))
            source['archive'] = _get_archive(source_location, mode=mode, opts=opts)
        for url in my_urls:
            location_part = _get_location_part(url)
            logger.debug('{}: compare _get_location_part {} location {}'.format(
                         _name, location_part, source_location))
            query = _get_query_part(url)
            logger.debug('{}: query {}'.format(_name, query))
            if location_part.startswith(source_location):
                if source['archive'].use_query():
                    fname = query
                else:
                    if query:
                        fname = query
                    else:
                        fname = location_part[len(source_location) + 1:]
                # _get_query_part(url)
                if len(fname) > 0:
                    source['files'].append(fname)
        sources.append(source)
    for source in sources:
        logger.debug('{}: sources {}'.format(_name, source))
    return sources


def str_to_dtype(s):
    """Convert dtype string to numpy dtype."""
    return eval('np.' + s)
