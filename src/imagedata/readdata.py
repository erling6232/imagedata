"""Read/Write image files, calling appropriate transport, archive and format plugins
"""

# Copyright (c) 2013-2019 Erling Andersen, Haukeland University Hospital, Bergen, Norway

import os.path
import logging
import mimetypes
import argparse
import urllib.parse
from imagedata.header import add_template, add_geometry
import imagedata.formats
import imagedata
import imagedata.transports
import imagedata.archives


class NoTransportError(Exception):
    pass


class NoArchiveError(Exception):
    pass


class UnknownOptionType(Exception):
    pass


def read(urls, order=None, opts=None):
    """Read image data, calling appropriate transport, archive and format plugins

    Args:
        urls: list of urls or url to read (list of str, or str)
        order: determine how to sort the images (default: auto-detect)
        opts: input options (argparse.Namespace or dict)

    Returns:
        tuple of
            - hdr: header instance
            - si[tag,slice,rows,columns]: numpy array

    Raises:
        ValueError: When no sources are given.
        UnknownOptionType: When opts cannot be made into a dict.
        FileNotFoundError: When specified URL cannot be opened.
        imagedata.formats.UnknownInputError: When the input format could not be determined.
        imagedata.formats.CannotSort: When input data cannot be sorted.
    """

    logging.debug("reader.read: urls {}".format(urls))
    #    transport,my_urls,files = sanitize_urls(urls)
    #    if len(my_urls) < 1:
    #        raise ValueError("No URL(s) where given")
    #    logging.debug("reader.read: transport {} my_urls {}".format(transport,my_urls))
    sources = _get_sources(urls, mode='r')
    if len(sources) < 1:
        raise ValueError("No source(s) where given")
    logging.debug("reader.read: sources {}".format(sources))

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

    # Let the calling party override a default input order
    input_order = imagedata.formats.INPUT_ORDER_NONE
    if 'input_order' in in_opts:
        input_order = in_opts['input_order']
    if order != 'none':
        input_order = order
    logging.info("Input order: {}.".format(imagedata.formats.input_order_to_str(input_order)))

    # Pre-fetch DICOM template
    pre_hdr = None
    if 'template' in in_opts and in_opts['template']:
        logging.debug("readdata.read template {}".format(in_opts['template']))
        template_source = _get_sources(in_opts['template'], mode='r')
        reader = imagedata.formats.find_plugin('dicom')
        pre_hdr, _ = reader.read_headers(template_source, input_order, in_opts)

    # Pre-fetch DICOM geometry
    geom_hdr = None
    if 'geometry' in in_opts and in_opts['geometry']:
        logging.debug("readdata.read geometry {}".format(in_opts['geometry']))
        geometry_source = _get_sources(in_opts['geometry'], mode='r')
        reader = imagedata.formats.find_plugin('dicom')
        geom_hdr, _ = reader.read_headers(geometry_source, input_order, in_opts)
        # if pre_hdr is None:
        #    pre_hdr = {}
        # _add_dicom_geometry(pre_hdr, geom_hdr)

    # Call reader plugins in turn to read the image data
    plugins = sorted_plugins_dicom_first(imagedata.formats.get_plugins_list())
    logging.debug("readdata.read plugins length {}".format(len(plugins)))
    for pname, ptype, pclass in plugins:
        logging.debug("%20s (%8s) %s" % (pname, ptype, pclass.description))
        reader = pclass()
        try:
            hdr, si = reader.read(sources, None, input_order, in_opts)

            for source in sources:
                logging.debug("readdata.read: close archive {}".format(source['archive']))
                source['archive'].close()
            add_template(hdr, pre_hdr)
            add_geometry(hdr, pre_hdr, geom_hdr)
            return hdr, si
        except (FileNotFoundError, imagedata.formats.CannotSort):
            raise
        except imagedata.formats.NotImageError as e:
            logging.info("Giving up {}: {}".format(ptype, e))
            pass
        except Exception as e:
            logging.info("Giving up (OTHER) {}: {}".format(ptype, e))
            pass

    for source in sources:
        logging.debug("readdata.read: close archive {}".format(source['archive']))
        source['archive'].close()

    if issubclass(type(urls), list):
        raise imagedata.formats.UnknownInputError("Could not determine input format of %s." % urls[0])
    else:
        raise imagedata.formats.UnknownInputError("Could not determine input format of %s." % urls)


# def _add_template(hdr, pre_hdr):
#    if pre_hdr is not None:
#        for key in pre_hdr:
#            hdr[key] = copy.copy(pre_hdr[key])

# def _add_dicom_geometry(pre_hdr, geometry):
#        """For each slice in geometry, use most of pre_hdr, adding a few attributes from geometry
#        """
#
#        #logging.debug("_add_dicom_geometry template %s geometry %s" % (
#        #    imagedata.formats.shape_to_str(self.shape),
#        #    imagedata.formats.shape_to_str(geometry.shape)))
#        pre_hdr['sliceLocations'] = geometry['sliceLocations'].copy()
#        pre_hdr['spacing']        = geometry['spacing'].copy()
#        pre_hdr['orientation']    = geometry['orientation'].copy()
#        pre_hdr['imagePositions'] = {}
#        logging.debug("_add_dicom_geometry:")
#        logging.debug("_add_dicom_geometry: geometry.imagePositions {}".format(geometry['imagePositions'].keys()))
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

    # logging.debug("write: directory_name(si): {}".format(directory_name(si)))

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
    logging.debug("Default    output format : {}".format(output_formats))
    logging.debug("Overriding output formats: {}".format(formats))
    logging.debug("Options: {}".format(out_opts))
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
    logging.info("Output formats: {}".format(output_formats))

    # Determine output dtype
    write_si = si
    if 'dtype' in out_opts and out_opts['dtype'] is not None:
        if out_opts['dtype'] != si.dtype:
            # write_si = si.astype(str_to_dtype(out_opts['dtype']))
            write_si = si.astype(out_opts['dtype'])
    _color = 0
    if write_si.color:
        _color = 1

    # Verify there is one destination only
    # destinations = _get_sources(url, mode='w')
    # if len(destinations) != 1:
    #    raise ValueError('Wrong number of destinations (%d) given' %
    #        len(destinations))

    # Call plugin writers in turn to store the data
    logging.debug("Available plugins {}".format(len(imagedata.formats.get_plugins_list())))
    written = False
    msg = ''
    for pname, ptype, pclass in imagedata.formats.get_plugins_list():
        if ptype in output_formats:
            logging.debug("Attempt plugin {}".format(ptype))
            # Create plugin to write data in specified format
            writer = pclass()
            logging.debug("readdata.write: Created writer plugin of type {}".format(type(writer)))
            local_url = url.replace('%p', ptype)
            destinations = _get_sources(local_url, mode='w', opts=out_opts)
            if len(destinations) != 1:
                raise ValueError('Wrong number of destinations (%d) given' %
                                 len(destinations))
            destination = destinations[0]
            logging.debug('readdata.write: destination {}'.format(destination))
            try:
                if write_si.ndim - _color == 4 and write_si.shape[0] > 1:
                    # 4D data
                    writer.write_4d_numpy(write_si, destination, out_opts)
                elif write_si.ndim - _color >= 2:
                    # 2D-3D data
                    writer.write_3d_numpy(write_si, destination, out_opts)
                else:
                    raise ValueError("Don't know how to write image of shape {}".format(write_si.shape))
                written = True
            except imagedata.formats.WriteNotImplemented:
                raise
            except Exception as e:
                logging.info("Giving up (OTHER) {}: {}".format(ptype, e))
                msg = msg + '\n{}: {}'.format(ptype, e)
                pass
            destination['archive'].close()
    if not written:
        if len(msg) > 0:
            raise IOError("Failed writing: {}".format(msg))
        raise ValueError("No writer plugin was found for {}".format(output_formats))
    if len(msg) > 0:
        logging.error(msg)
    # destination['archive'].close()


def sorted_plugins_dicom_first(plugins):
    """Sort plugins such that any Nifti plugin is used early."""
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

    url_tuple = urllib.parse.urlsplit(url, scheme='file')
    # Strip off query and fragment parts
    location = urllib.parse.urlunsplit((
        url_tuple.scheme,
        url_tuple.netloc,
        url_tuple.path,
        None,
        None))
    if location[:8] == 'file:///' and url_tuple.path[0] != '/':
        location = 'file://' + os.path.abspath(location[8:])
    logging.debug('readdata._get_location_part: scheme %s' % url_tuple.scheme)
    logging.debug('readdata._get_location_part: netloc %s' % url_tuple.netloc)
    logging.debug('readdata._get_location_part: path %s' % url_tuple.path)
    logging.debug('readdata._get_location_part: location %s' % location)
    return location


def _get_query_part(url):
    """Get query part of URL. This may contain file name"""

    url_tuple = urllib.parse.urlsplit(url, scheme='file')
    return url_tuple.query


def _get_archive(url, mode='r', opts=None):
    """Get archive plugin for given URL."""

    if opts is None:
        opts = {}
    logging.debug('readdata._get_archive: url %s' % url)
    url_tuple = urllib.parse.urlsplit(url, scheme='file')
    mimetype = mimetypes.guess_type(url_tuple.path)[0]
    archive = imagedata.archives.find_mimetype_plugin(
        mimetype,
        url,
        mode,
        opts=opts)
    logging.debug('readdata._get_archive: _mimetypes %s' % mimetype)
    logging.debug('readdata._get_archive: archive %s' % archive.name)
    return archive


def _common_prefix(level):
    """This unlike the os.path.commonprefix version
    always returns path prefixes as it compares
    path component wise
    https://stackoverflow.com/questions/21498939
    """

    cp = []
    ls = [p.split('/') for p in level]
    ml = min(len(p) for p in ls)

    for i in range(ml):

        s = set(p[i] for p in ls)
        if len(s) != 1:
            break

        cp.append(s.pop())

    return '/'.join(cp)


def _simplify_locations(locations):
    """Simplify locations by joining file:/// locations to a common prefix."""

    logging.debug('readdata._simplify_locations: locations {}'.format(locations))
    new_locations = {}
    paths = []
    for location in locations:
        # On Windows, any backslash (os.sep) will be replaced by slash in URL
        url_tuple = urllib.parse.urlsplit(location.replace(os.sep, '/'), scheme='file')
        if url_tuple.scheme == 'file':
            paths.append(url_tuple.path)
        else:
            new_locations[location] = True
    logging.debug('readdata._simplify_locations: paths {}'.format(paths))
    if len(paths) > 0:
        prefix = _common_prefix(paths)
        logging.debug('readdata._simplify_locations: prefix {}'.format(prefix))
        prefix_url = urllib.parse.urlunsplit((
            'file',
            '',
            prefix,
            None,
            None))
        new_locations[prefix_url] = True
    logging.debug('readdata._simplify_locations: new_locations {}'.format(new_locations))
    return new_locations


def _get_sources(urls, mode, opts=None):
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

    # Ensure the input is a list: my_urls
    if opts is None:
        opts = {}
    if isinstance(urls, list):
        my_urls = urls
    elif isinstance(urls, tuple):
        my_urls = list(urls)
    else:
        my_urls = [urls]

    # Scan my_urls to determine the locations of the inputs
    locations = {}
    for url in my_urls:
        locations[_get_location_part(url)] = True
    locations = _simplify_locations(locations)

    # Set up sources for each location, and possibly add files
    sources = []
    for location in locations:
        logging.debug('readdata._get_sources: location %s' % location)
        source_location = location
        source = {'files': []}
        try:
            source['archive'] = _get_archive(source_location, mode=mode, opts=opts)
        except (imagedata.transports.RootIsNotDirectory,
                imagedata.archives.ArchivePluginNotFound):
            # Retry with parent directory
            source_location, filename = os.path.split(source_location)
            logging.debug('readdata._get_sources: retry location %s' % source_location)
            source['archive'] = _get_archive(source_location, mode=mode, opts=opts)
        for url in my_urls:
            location_part = _get_location_part(url)
            logging.debug('readdata._get_sources: compare _get_location_part %s location %s' %
                          (location_part, source_location))
            query = _get_query_part(url)
            logging.debug('readdata._get_sources: query %s' % query)
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
        logging.debug('readdata._get_sources: sources %s' % source)
    return sources


def str_to_dtype(s):
    """Convert dtype string to numpy dtype."""
    return eval('np.' + s)
